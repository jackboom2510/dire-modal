#!/usr/bin/env python3
"""Compute real ADM/DDIM reconstructions for selected saved gallery images.

The tool can process every validated original in a gallery (`--all`) or selected
sample names. It skips reconstructions that already exist unless `--force` is
set. One S=20 reconstruction is 40 ADM forwards (~404 s on the measured 2-vCPU
CPU). It never estimates x_hat from an error map; it runs the same ADM/DDIM path
as the demo on each original image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "platforms" / "hf-spaces"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SPACE))

from demolib import live, provenance  # noqa: E402

def default_gallery() -> Path:
    configured = os.environ.get("DIRE_GALLERY_DIR")
    if configured:
        folder = Path(configured).expanduser()
        return folder if folder.is_absolute() else SPACE / folder
    repo_gallery = ROOT / "modal-runs" / "gallery"
    return repo_gallery if any(repo_gallery.glob("*.json")) else SPACE / "gallery"


DEFAULT_GALLERY = default_gallery()
DEFAULT_CKPT = SPACE / "ckpt" / "adm_lsun.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def selected_samples(gallery: Path, names: list[str] | None = None, *, all_samples: bool = False):
    samples, problems = provenance.discover(gallery)
    if all_samples:
        return samples, problems
    names = names or []
    by_name = {sample.name: sample for sample in samples}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(
            f"Không thấy mẫu trong {gallery}: {missing}\n"
            f"Mẫu hợp lệ hiện có: {len(samples)}; truyền đúng stem từ file .json, không thêm hậu tố _orig."
        )
    return [by_name[name] for name in names], problems


def make_recons(
    gallery: Path,
    checkpoint: Path,
    names: list[str] | None = None,
    *,
    all_samples: bool = False,
    dry_run: bool,
    force: bool,
) -> int:
    samples, problems = selected_samples(gallery, names, all_samples=all_samples)
    if not samples:
        raise SystemExit(f"Không tìm thấy mẫu gallery hợp lệ trong {gallery}.")
    for problem in problems:
        print(f"[gallery warning] {problem}", file=sys.stderr)

    todo = []
    for sample in samples:
        destination = gallery / f"{sample.name}_recon.png"
        if destination.exists() and not force:
            print(f"[skip] {sample.name}: đã có ảnh tái tạo; dùng --force để tính lại")
            continue
        seconds = live.estimate_seconds(sample.steps)
        print(f"[{'dry-run' if dry_run else 'queue'}] {sample.name} · S={sample.steps} · "
              f"~{seconds:.0f}s CPU · {sample.orig.name} → {destination.name}")
        todo.append((sample, destination))

    total_seconds = sum(live.estimate_seconds(sample.steps) for sample, _ in todo)
    print(
        f"[summary] {len(todo)} recon cần tính · "
        f"~{total_seconds / 3600:.2f} giờ CPU theo benchmark 2-vCPU · "
        f"{len(samples) - len(todo)} đã có/bỏ qua"
    )
    if dry_run or not todo:
        return 0
    if not checkpoint.is_file():
        raise SystemExit(
            f"Không có ADM checkpoint tại {checkpoint}. Cần `256x256_diffusion_uncond.pt` "
            "(ADM U-Net, khoảng 2,5 GB), không phải `lsun_adm.pth` (ResNet detector). "
            "Đặt đúng checkpoint tại đường dẫn trên hoặc dùng --checkpoint."
        )

    # Load once, run the shared, parity-tested DDIM implementation, and keep a
    # separate, auditable display PNG plus provenance for each selected sample.
    model, checkpoint_digest = live.load_checkpoint(checkpoint)
    for sample, destination in todo:
        result = live.run_steps(model, sample.orig, steps=sample.steps)
        atomic_write(destination, result["recon_png"])

        meta_path = gallery / f"{sample.name}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        hashes = dict(meta.get("sha256", {}))
        hashes["png_recon"] = sha256_file(destination)
        meta["sha256"] = hashes
        meta["reconstruction"] = {
            "method": "ADM U-Net · DDIM inversion + reconstruction",
            "steps": sample.steps,
            "checkpoint_sha256_prefix": checkpoint_digest,
            "png_is_display_only": True,
        }
        payload = (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        atomic_write(meta_path, payload)
        print(f"[ok] {sample.name}: wrote {destination.name}; NFE={result['nfe']} · {result['seconds']:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", dest="all_samples", action="store_true",
                           help="tính recon cho mọi ảnh gốc hợp lệ trong gallery; mẫu đã có recon sẽ được bỏ qua")
    selection.add_argument("--name", action="append",
                           help="gallery sample stem (repeat for each real/generated sample)")
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--dry-run", action="store_true", help="list source/output/time without loading weights")
    parser.add_argument("--force", action="store_true", help="recompute and replace existing reconstruction PNGs")
    args = parser.parse_args(argv)
    return make_recons(
        args.gallery, args.checkpoint, args.name,
        all_samples=args.all_samples, dry_run=args.dry_run, force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
