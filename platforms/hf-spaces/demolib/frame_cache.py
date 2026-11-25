"""Atomic, bounded local cache for per-step DIRE image frames.

Cache keys bind the input bytes, checkpoint identity, DDIM step count and
pipeline version. Uploaded originals are not copied into cache; the saved x_t
and x0-hat frames can still reveal image content, so cache stays on local disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

CACHE_VERSION = "dire-ddim-frames-v1"


def cache_root() -> Path:
    return Path(os.environ.get("DIRE_FRAME_CACHE_DIR", Path(tempfile.gettempdir()) / "dire-frame-cache")).expanduser()


def make_key(image_bytes: bytes, checkpoint_id: str, steps: int) -> str:
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(b"\0")
    h.update(str(steps).encode())
    h.update(b"\0")
    h.update(checkpoint_id.encode())
    h.update(b"\0")
    h.update(image_bytes)
    return h.hexdigest()


def load(key: str, root: Path | None = None) -> dict[str, Any] | None:
    folder = (root or cache_root()) / key
    manifest = folder / "result.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for field in ("map_png", "recon_png"):
            if field in data:
                data[field] = (folder / data[field]).read_bytes()
        frames = []
        for item in data.pop("frames"):
            frame = dict(item)
            for field in ("x_t", "x0_hat"):
                frame[f"{field}_png"] = (folder / item[field]).read_bytes()
            frames.append(frame)
        data["frames"] = frames
        if data.get("key") != key:
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store(key: str, result: dict[str, Any], root: Path | None = None, max_entries: int = 8) -> Path:
    base = root or cache_root()
    base.mkdir(parents=True, exist_ok=True)
    destination = base / key
    if load(key, base) is not None:
        return destination
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:10]}-", dir=base))
    manifest: dict[str, Any] = {k: v for k, v in result.items() if k not in ("frames", "map_png", "recon_png")}
    for field in ("map_png", "recon_png"):
        if field in result:
            name = f"{field}.png"
            (temp / name).write_bytes(result[field])
            manifest[field] = name
    manifest["key"] = key
    serialized_frames = []
    try:
        for i, frame in enumerate(result.get("frames", [])):
            # `live.run_steps` also exposes x0_png as a compatibility alias of
            # x0_hat_png. Store one copy only; binary payloads live in PNG files.
            item = {k: v for k, v in frame.items() if k not in ("x_t_png", "x0_hat_png", "x0_png")}
            for field in ("x_t", "x0_hat"):
                name = f"{field}-{i:03d}.png"
                (temp / name).write_bytes(frame[f"{field}_png"])
                item[field] = name
            serialized_frames.append(item)
        manifest["frames"] = serialized_frames
        (temp / "result.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            temp.rename(destination)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
        # Keep cache bounded. Never remove the just-written entry.
        entries = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in entries[max(1, max_entries):]:
            if stale != destination:
                shutil.rmtree(stale, ignore_errors=True)
        return destination
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def clear(root: Path | None = None) -> int:
    """Remove only entries within this cache directory; return entry count."""
    base = root or cache_root()
    if not base.is_dir():
        return 0
    entries = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for p in entries:
        shutil.rmtree(p, ignore_errors=True)
    return len(entries)
