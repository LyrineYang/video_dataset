from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .config import OCRConfig

log = logging.getLogger(__name__)

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception as exc:  # noqa: BLE001
    RapidOCR = None  # type: ignore
    log.warning("RapidOCR not available: %s", exc)

try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log.warning("Decord not available for OCR: %s", exc)

# 控制重复告警
_ocr_unavailable_warned = False
_ocr_init_warned = False

def has_text(video_path: Path, cfg: OCRConfig) -> bool:
    if not cfg.enabled:
        return False
    global _ocr_unavailable_warned, _ocr_init_warned
    if RapidOCR is None or decord is None:
        if not _ocr_unavailable_warned:
            log.warning("OCR skipped because RapidOCR/decord not available")
            _ocr_unavailable_warned = True
        return False
    try:
        ocr = _get_ocr(cfg.use_gpu)
    except Exception as exc:  # noqa: BLE001
        if not _ocr_init_warned:
            log.warning("OCR init failed: %s; skipping OCR (further warnings suppressed)", exc)
            _ocr_init_warned = True
        return False
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return False

    stride = max(cfg.sample_stride, 1)
    hit = False
    for idx in range(0, total, stride):
        frame = vr[idx].asnumpy()  # RGB
        if _text_area_ratio(frame, ocr) >= cfg.text_area_threshold:
            hit = True
            break
    return hit


@lru_cache(maxsize=4)
def _get_ocr(use_gpu: bool = False):  # -> RapidOCR
    if RapidOCR is None:
        raise RuntimeError("RapidOCR not available")
    kwargs = {
        "det_use_cuda": use_gpu,
        "cls_use_cuda": False,
        "rec_use_cuda": use_gpu,
    }
    return RapidOCR(**kwargs)


def _text_area_ratio(frame_rgb: np.ndarray, ocr) -> float:
    h, w, _ = frame_rgb.shape
    if h == 0 or w == 0:
        return 0.0
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = _run_ocr(frame_bgr, ocr)
    boxes = _extract_boxes(result)
    total_area = h * w
    text_area = 0.0
    for item in boxes:
        if not item:
            continue
        poly = item[0] if isinstance(item, (list, tuple)) else None
        area = _polygon_area(poly)
        text_area += max(area, 0.0)
    return float(text_area / total_area) if total_area > 0 else 0.0


def _run_ocr(frame_bgr: np.ndarray, ocr) -> list:
    """
    Normalize OCR call across RapidOCR (callable) and Paddle-style objects (ocr method).
    """
    # RapidOCR: __call__ returns (result, elapse)
    if callable(getattr(ocr, "__call__", None)) and not hasattr(ocr, "ocr"):
        result, _ = ocr(frame_bgr)
        return result if isinstance(result, list) else []
    # Fallback to Paddle-style API if present
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(frame_bgr, det=True, rec=False, cls=False)  # type: ignore[attr-defined]
        except Exception:
            return ocr.ocr(frame_bgr)  # type: ignore[attr-defined]
    return []


def _extract_boxes(result: list) -> list:
    if not result:
        return []
    # RapidOCR: list of [poly, text, score]
    first = result[0]
    if isinstance(first, (list, tuple)) and len(first) >= 3 and isinstance(first[0], (list, tuple)):
        return result
    # Paddle style: list per image
    if isinstance(first, list):
        return first
    return []


def _polygon_area(points: List[List[float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
