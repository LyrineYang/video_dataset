from __future__ import annotations

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable

import requests
from PIL import Image

from .config import CaptionConfig

log = logging.getLogger(__name__)

try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log.warning("Decord not available for caption frame extraction: %s", exc)


def generate_captions(videos: Iterable[Path], cfg: CaptionConfig) -> Dict[str, str]:
    """
    Generate captions for a batch of videos.

    Returns a mapping of path string -> caption. Failures are logged and omitted.
    """
    if not cfg.enabled:
        return {}
    paths = list(videos)
    if not paths:
        return {}
    generator = CaptionGenerator(cfg)
    return generator.generate(paths)


class CaptionGenerator:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg
        if cfg.provider not in {"api", "openrouter"}:
            raise ValueError(f"Unsupported caption provider: {cfg.provider}")
        if cfg.provider == "api" and not cfg.api_url:
            raise ValueError("caption.api_url is required when provider=api")

    def generate(self, paths: list[Path]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        workers = max(int(self.cfg.max_workers or 1), 1)
        if workers > 1 and len(paths) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self._generate_single, p): p for p in paths}
                for fut in as_completed(futures):
                    path = futures[fut]
                    caption = self._safe_result(fut, path)
                    if caption:
                        results[str(path)] = caption
        else:
            for path in paths:
                try:
                    caption = self._generate_single(path)
                    if caption:
                        results[str(path)] = caption
                except Exception as exc:  # noqa: BLE001
                    log.warning("Caption failed for %s: %s", path, exc)
        return results

    def _safe_result(self, fut, path: Path) -> str | None:
        try:
            return fut.result()
        except Exception as exc:  # noqa: BLE001
            log.warning("Caption failed for %s: %s", path, exc)
            return None

    def _generate_single(self, path: Path) -> str | None:
        if self.cfg.provider == "api":
            return self._call_api(path)
        if self.cfg.provider == "openrouter":
            return self._call_openrouter(path)
        raise ValueError(f"Unsupported caption provider: {self.cfg.provider}")

    def _call_api(self, path: Path) -> str | None:
        headers = {}
        if self.cfg.api_key:
            headers[self.cfg.api_key_header or "Authorization"] = self.cfg.api_key
        timeout = max(float(self.cfg.timeout or 60.0), 0.1)
        file_field = self.cfg.file_field or "file"
        retries = max(int(self.cfg.retry or 0), 0) + 1
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                with path.open("rb") as f:
                    files = {file_field: (path.name, f, "video/mp4")}
                    resp = requests.post(
                        self.cfg.api_url,
                        headers=headers,
                        data=self.cfg.extra_fields or {},
                        files=files,
                        timeout=timeout,
                    )
                if resp.status_code >= 500 and attempt < retries - 1:
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                caption = self._parse_response(resp)
                return caption.strip() if isinstance(caption, str) else caption
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(1.0)
        if last_exc:
            raise last_exc
        return None

    def _parse_response(self, resp: requests.Response) -> str | None:
        try:
            data = resp.json()
        except ValueError:
            text = resp.text.strip()
            return text or None

        if isinstance(data, dict):
            if self.cfg.response_field and data.get(self.cfg.response_field) is not None:
                return str(data[self.cfg.response_field])
            for key in ("caption", "text", "message"):
                if key in data and data[key] is not None:
                    return str(data[key])
            for val in data.values():
                if isinstance(val, (str, int, float)):
                    return str(val)
            return None

        if isinstance(data, str):
            return data
        return None

    def _call_openrouter(self, path: Path) -> str | None:
        api_url = self.cfg.api_url or "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.cfg.api_key:
            headers[self.cfg.api_key_header or "Authorization"] = f"Bearer {self.cfg.api_key}"
        if self.cfg.openrouter_referer:
            headers["HTTP-Referer"] = self.cfg.openrouter_referer
        if self.cfg.openrouter_title:
            headers["X-Title"] = self.cfg.openrouter_title

        content = [{"type": "text", "text": self._build_text_prompt(path)}]
        image_b64 = self._extract_image_b64(path) if self.cfg.include_image else None
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

        body = {
            "model": self.cfg.model or "gpt-4o",
            "messages": [],
            "max_tokens": int(self.cfg.max_tokens or 120),
            "temperature": float(self.cfg.temperature),
        }
        if self.cfg.system_prompt:
            body["messages"].append({"role": "system", "content": [{"type": "text", "text": self.cfg.system_prompt}]})
        body["messages"].append({"role": "user", "content": content})

        retries = max(int(self.cfg.retry or 0), 0) + 1
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.post(api_url, json=body, headers=headers, timeout=self.cfg.timeout)
                if resp.status_code >= 500 and attempt < retries - 1:
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                return self._parse_openrouter(resp)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(1.0)
        if last_exc:
            raise last_exc
        return None

    def _parse_openrouter(self, resp: requests.Response) -> str | None:
        data = resp.json()
        try:
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                return None
            msg = choices[0]["message"]
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
                ]
                if texts:
                    return "\n".join(texts)
            return None
        except Exception:  # noqa: BLE001
            return None

    def _build_text_prompt(self, path: Path) -> str:
        base_prompt = self.cfg.user_prompt or "请为视频生成简洁描述。"
        return f"{base_prompt}\n文件名: {path.name}"

    def _extract_image_b64(self, path: Path) -> str | None:
        if decord is None:
            return None
        try:
            vr = decord.VideoReader(str(path))
            total = len(vr)
            if total == 0:
                return None
            idx = total // 2
            frame = vr[idx]
            if hasattr(frame, "asnumpy"):
                frame_np = frame.asnumpy()
            else:
                frame_np = frame
            img = Image.fromarray(frame_np).convert("RGB")
            max_side = max(int(self.cfg.image_max_side or 512), 64)
            w, h = img.size
            if max(w, h) > max_side:
                if w >= h:
                    new_w = max_side
                    new_h = int(h * max_side / w)
                else:
                    new_h = max_side
                    new_w = int(w * max_side / h)
                img = img.resize((max(new_w, 1), max(new_h, 1)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to extract frame for %s: %s", path, exc)
            return None
