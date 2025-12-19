import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from .models import ScoreResult


def materialize_results(
    shard: str,
    results: Iterable[ScoreResult],
    output_root: Path,
    extras: dict[str, dict] | None = None,
    resize_720p: bool = False,
) -> Path:
    shard_out = output_root / shard
    videos_out = shard_out / "videos"
    shard_out.mkdir(parents=True, exist_ok=True)
    videos_out.mkdir(parents=True, exist_ok=True)

    metadata_path = shard_out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for res in results:
            target = videos_out / res.path.name if res.keep and res.path.exists() else None
            extra = (extras or {}).get(str(res.path), {})
            record = {
                "source_path": str(res.path),
                "output_path": str(target) if target else None,
                "size_bytes": res.path.stat().st_size if res.path.exists() else 0,
                "scores": res.scores,
                "keep": res.keep,
                "reason": res.reason,
                **extra,
            }
            f.write(json.dumps(record) + "\n")
            if res.keep and res.path.exists() and target is not None:
                if resize_720p:
                    _transcode_to_720p(res.path, target)
                else:
                    _link_or_copy(res.path, target)
    return metadata_path


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _transcode_to_720p(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale=-2:720",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
