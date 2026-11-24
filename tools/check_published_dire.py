#!/usr/bin/env python3
r"""Compare a small sample of published DIRE maps with this repo's DIRE pipeline.

This is a LOCAL verification tool. It never extracts archives to disk, writes maps,
starts Modal, or modifies input files. It only reads the image/map tarballs and ADM
checkpoint; an optional JSON report is created with exclusive (no-overwrite) mode.

Pairing is deliberately strict: by default image and map members must have unique
matching filename stems. Use --pairing order only when independent evidence shows
that tar order is the intended correspondence and both archives have equal counts.

Published map storage is not assumed. .npy maps are interpreted as native float
values in [0, 2]. Image maps require --image-map-encoding linear-0-2, meaning each
uint8 channel encodes [0, 2] linearly (pixel * 2 / 255). Do not select that encoding
unless archive metadata/source confirms it.

Example (PowerShell):
  python tools/check_published_dire.py `
    --images 'F:\DiffusionForensics\images\train\lsun_bedroom\adm.tar.gz' `
    --maps 'F:\DiffusionForensics\dire\train\lsun_bedroom\adm.tar.gz' `
    --checkpoint 'F:\DiffusionForensics\checkpoints\256x256_diffusion_uncond.pt' `
    --sample 8 --steps 20 --device cuda

Install local optional dependencies first, e.g. `pip install -e '.[zero]'`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

# Allow running this file from a checkout without requiring `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAP_EXTS = IMAGE_EXTS | {".npy"}


def _stem(member: str) -> str:
    return PurePosixPath(member).stem.casefold()


def _image_members(tf: tarfile.TarFile, allowed: set[str]) -> list[tarfile.TarInfo]:
    return [m for m in tf if m.isfile() and PurePosixPath(m.name).suffix.casefold() in allowed]


def _pair_members(images: list[tarfile.TarInfo], maps: list[tarfile.TarInfo], pairing: str):
    if pairing == "order":
        if len(images) != len(maps):
            raise ValueError(
                f"pairing=order cần cùng số member: ảnh={len(images)}, map={len(maps)}"
            )
        return list(zip(images, maps)), {"method": "tar-order", "n_images": len(images), "n_maps": len(maps)}

    image_by_stem: dict[str, list[tarfile.TarInfo]] = {}
    map_by_stem: dict[str, list[tarfile.TarInfo]] = {}
    for member in images:
        image_by_stem.setdefault(_stem(member.name), []).append(member)
    for member in maps:
        map_by_stem.setdefault(_stem(member.name), []).append(member)
    duplicate_images = sorted(k for k, v in image_by_stem.items() if len(v) != 1)
    duplicate_maps = sorted(k for k, v in map_by_stem.items() if len(v) != 1)
    if duplicate_images or duplicate_maps:
        raise ValueError(
            "stem pairing không đơn trị; có stem trùng. "
            f"Ảnh (tối đa 10): {duplicate_images[:10]}; map: {duplicate_maps[:10]}"
        )
    shared = sorted(image_by_stem.keys() & map_by_stem.keys())
    pairs = [(image_by_stem[k][0], map_by_stem[k][0]) for k in shared]
    return pairs, {
        "method": "unique-casefolded-filename-stem",
        "n_images": len(images),
        "n_maps": len(maps),
        "n_matched": len(pairs),
        "unmatched_images": len(images) - len(pairs),
        "unmatched_maps": len(maps) - len(pairs),
    }


def _member_bytes(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fp = tf.extractfile(member)
    if fp is None:
        raise OSError(f"Không đọc được tar member: {member.name}")
    with fp:
        return fp.read()


def _decode_map(data: bytes, member_name: str, image_encoding: str):
    import numpy as np

    suffix = PurePosixPath(member_name).suffix.casefold()
    if suffix == ".npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        original_shape = list(arr.shape)
        stored_dtype = str(arr.dtype)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"map .npy cần HxWx3, 3xHxW hoặc HxW; nhận {arr.shape}")
        arr = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("map .npy có NaN hoặc Inf")
        lo, hi = float(arr.min()), float(arr.max())
        if lo < -1e-3 or hi > 2.001:
            raise ValueError(
                f"map .npy nằm ngoài [0,2] (min={lo:.6g}, max={hi:.6g}); "
                "không tự đoán scale. Cần xác minh định dạng/thang lưu."
            )
        return arr, {"format": "npy", "shape_on_disk": original_shape, "stored_dtype": stored_dtype,
                     "stored_range": [lo, hi], "conversion": "native float values [0,2]"}

    if image_encoding != "linear-0-2":
        raise ValueError(
            f"map ảnh ({member_name}) chưa có encoding được xác nhận. "
            "Sau khi kiểm tra metadata, truyền --image-map-encoding linear-0-2 "
            "nếu uint8 mã hóa tuyến tính thang [0,2]."
        )
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    arr = rgb * (2.0 / 255.0)
    return arr, {
        "format": "image",
        "shape_on_disk": list(rgb.shape),
        "stored_range": [int(rgb.min()), int(rgb.max())],
        "conversion": "uint8 * (2/255), assuming linear encoding of [0,2]",
    }


def _metrics(reference, actual, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    abs_delta = np.abs(delta)
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() and b.std() else None
    close = np.isclose(actual, reference, atol=atol, rtol=rtol)
    return {
        "mae": float(abs_delta.mean()),
        "rmse": float(math.sqrt(float(np.square(delta).mean()))),
        "max_abs_error": float(abs_delta.max()),
        "cosine_similarity": float(np.dot(a, b) / denom) if denom else None,
        "pearson_correlation": corr,
        "fraction_within_atol_rtol": float(close.mean()),
        "all_within_atol_rtol": bool(close.all()),
        "atol": atol,
        "rtol": rtol,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from dire_zero.checkpoint import load_adm
    from dire_zero.diffusion import ModelCounter, Schedule, dire_map, from_uint8

    for key in ("images", "maps", "checkpoint"):
        p = Path(getattr(args, key)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Không thấy --{key}: {p}")
        setattr(args, key, p)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng torch.cuda.is_available() là False")
    dtype = "float16" if device == "cuda" else "float32"
    started = time.time()

    with tarfile.open(args.images, "r:*") as image_tar, tarfile.open(args.maps, "r:*") as map_tar:
        image_members = _image_members(image_tar, IMAGE_EXTS)
        map_members = _image_members(map_tar, MAP_EXTS)
        pairs, pairing_info = _pair_members(image_members, map_members, args.pairing)
        if not pairs:
            raise ValueError("Không có cặp member nào để kiểm tra")
        if args.sample > len(pairs):
            raise ValueError(f"sample={args.sample} lớn hơn số cặp ghép được ({len(pairs)})")
        chosen = random.Random(args.seed).sample(pairs, args.sample)

        model = load_adm(str(args.checkpoint), device=device, dtype=dtype, strict=True)
        counter = ModelCounter(model)
        schedule = Schedule(steps=args.steps, device=device)
        records = []
        for index, (image_member, map_member) in enumerate(chosen, start=1):
            image_bytes = _member_bytes(image_tar, image_member)
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                if im.size != (256, 256):
                    raise ValueError(
                        f"Ảnh {image_member.name} có size {im.size}; pipeline ::dire bỏ qua ảnh không phải 256x256"
                    )
                pixels = np.asarray(im, dtype=np.uint8).copy()
            image_tensor = from_uint8(pixels).to(device)
            map_arr, map_info = _decode_map(_member_bytes(map_tar, map_member), map_member.name,
                                             args.image_map_encoding)
            if map_arr.shape != (256, 256, 3):
                raise ValueError(f"map {map_member.name} có shape {map_arr.shape}, cần 256x256x3")

            counter.reset()
            with torch.inference_mode():
                computed, _recon = dire_map(counter, image_tensor, schedule)
            nfe = counter.nfe
            computed_np = computed.squeeze(0).detach().float().cpu().permute(1, 2, 0).numpy()
            # Repository stores maps as float16 after DIRE computation.
            computed_np = computed_np.astype(np.float16).astype(np.float32)
            archived_np = map_arr
            record = {
                "image_member": image_member.name,
                "map_member": map_member.name,
                "map_storage": map_info,
                "nfe": nfe,
                "expected_nfe": 2 * args.steps,
                "comparison": _metrics(computed_np, archived_np, args.atol, args.rtol),
            }
            records.append(record)
            print(
                f"[{index}/{args.sample}] {image_member.name} ↔ {map_member.name} | "
                f"MAE={record['comparison']['mae']:.6g}, "
                f"max={record['comparison']['max_abs_error']:.6g}, "
                f"within={record['comparison']['fraction_within_atol_rtol']:.2%}, NFE={nfe}"
            )

    report = {
        "tool": "tools/check_published_dire.py",
        "scope_warning": (
            "Kết quả chỉ áp dụng cho các archive, checkpoint, steps, encoding và mẫu đã kiểm tra; "
            "không chứng minh toàn bộ archive khớp hoặc tái lập protocol Table 4/5."
        ),
        "images_archive": str(args.images),
        "maps_archive": str(args.maps),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "steps": args.steps,
        "device": device,
        "dtype": dtype,
        "seed": args.seed,
        "sample_requested": args.sample,
        "pairing": pairing_info,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "samples": records,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="archive ảnh gốc, ví dụ images/train/lsun_bedroom/adm.tar.gz")
    parser.add_argument("--maps", required=True, help="archive DIRE tương ứng, ví dụ dire/train/lsun_bedroom/adm.tar.gz")
    parser.add_argument("--checkpoint", required=True, help="ADM-LSUN checkpoint 256x256 dùng với ::dire")
    parser.add_argument("--sample", type=int, default=8, help="số cặp ngẫu nhiên có seed (mặc định: 8)")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=20, help="phải trùng protocol của map đã phát hành; mặc định ::dire=20")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pairing", choices=("stem", "order"), default="stem",
                        help="stem duy nhất (mặc định), hoặc thứ tự tar nếu đã xác minh độc lập")
    parser.add_argument("--image-map-encoding", choices=("unspecified", "linear-0-2"), default="unspecified",
                        help="bắt buộc chọn linear-0-2 cho map ảnh sau khi xác minh encoding")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--report", help="JSON report riêng; không ghi đè file đã tồn tại")
    args = parser.parse_args()
    if args.sample <= 0 or args.steps <= 0:
        parser.error("--sample và --steps phải lớn hơn 0")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol và --rtol không được âm")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
        print("\nTÓM TẮT")
        print(f"  Cặp ghép: {len(report['samples'])}/{report['pairing'].get('n_matched', report['pairing'].get('n_images'))}")
        print(f"  Checkpoint SHA-256: {report['checkpoint_sha256']}")
        print(f"  Scope: {report['scope_warning']}")
        if args.report:
            path = Path(args.report).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"  JSON report: {path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
