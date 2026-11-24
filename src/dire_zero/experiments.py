"""Protocol-locked helpers for the paper's Table 4, Table 5 and Figure 4.

These helpers only define deterministic inputs and experiment grids. The Modal
entrypoints own I/O, training, checkpoint provenance and result ledgers.
"""
from __future__ import annotations

import io
import re
from typing import Mapping

import torch
from torch import Tensor

TABLE4_STEPS = (5, 10, 20, 50)
TABLE5_REPRESENTATIONS = ("REC", "RGB", "RGB_DIRE", "DIRE")
ROBUSTNESS_LEVELS = (
    {"name": "clean", "kind": "clean", "value": None},
    {"name": "blur_sigma1", "kind": "gaussian_blur", "value": 1.0},
    {"name": "blur_sigma2", "kind": "gaussian_blur", "value": 2.0},
    {"name": "blur_sigma3", "kind": "gaussian_blur", "value": 3.0},
    {"name": "jpeg_q65", "kind": "jpeg", "value": 65},
    {"name": "jpeg_q30", "kind": "jpeg", "value": 30},
)
EXPERIMENT_TEST_GENERATORS = ("adm", "ddpm", "iddpm", "pndm", "sdv2")
TEST_GENERATOR_SCOPE_NOTE = (
    "LSUN-Bedroom test data in this run uses sdv2; sdv1 is not present in that dataset tree."
)


def validate_steps(steps: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Require the complete paper grid; do not silently produce a partial Table 4."""
    values = tuple(int(s) for s in steps)
    if len(set(values)) != len(values):
        raise ValueError("Table 4 có S trùng lặp")
    if set(values) != set(TABLE4_STEPS):
        raise ValueError(f"Table 4 yêu cầu đủ S={TABLE4_STEPS}; nhận {values}")
    return TABLE4_STEPS


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id or ""):
        raise ValueError("run_id cần 1–64 ký tự ASCII chữ/số, '.', '_' hoặc '-'")
    return run_id


def _as_chw(image: Tensor, label: str) -> Tensor:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"{label}: chỉ nhận một ảnh mỗi lần")
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{label}: cần tensor (3,H,W), nhận {tuple(image.shape)}")
    return image.float()


def _canonical_representation(name: str) -> str:
    key = name.upper().replace("&", "_").replace(" ", "")
    if key == "RGBDIRE":
        key = "RGB_DIRE"
    if key not in TABLE5_REPRESENTATIONS:
        raise ValueError(f"representation phải thuộc {TABLE5_REPRESENTATIONS}, nhận {name!r}")
    return key


def make_representation(rgb_minus1_1: Tensor, reconstruction_minus1_1: Tensor,
                        dire_0_2: Tensor, name: str) -> Tensor:
    """Create exact model input tensor for one Table 5 representation.

    RGB and REC are converted from the DIRE implementation's [-1,1] image domain
    to [0,1], matching the detector's ImageNet normalization. DIRE retains its
    native [0,2] residual scale. RGB_DIRE is channel concatenation (RGB first,
    then DIRE); the resulting six-channel contract is recorded in run metadata.
    """
    key = _canonical_representation(name)
    rgb = _as_chw(rgb_minus1_1, "RGB")
    rec = _as_chw(reconstruction_minus1_1, "REC")
    dire = _as_chw(dire_0_2, "DIRE")
    if rgb.shape != rec.shape or rgb.shape != dire.shape:
        raise ValueError("RGB, REC và DIRE phải cùng kích thước")
    if not torch.isfinite(rgb).all() or not torch.isfinite(rec).all() or not torch.isfinite(dire).all():
        raise ValueError("representation có NaN/Inf")
    rgb01 = ((rgb.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    rec01 = ((rec.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    if key == "RGB":
        return rgb01
    if key == "REC":
        return rec01
    if key == "DIRE":
        return dire.clamp(0, 2)
    return torch.cat((rgb01, dire.clamp(0, 2)), dim=0)


def corrupt_image(rgb_minus1_1: Tensor, condition: Mapping[str, object]) -> Tensor:
    """Apply one Figure 4 perturbation to an RGB image and return [-1,1] CHW.

    Gaussian blur uses Pillow's GaussianBlur radius (the paper's sigma levels).
    JPEG conditions perform a single encode/decode at the specified quality.
    Every call starts from the clean source, never a previously corrupted image.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    image = _as_chw(rgb_minus1_1, "RGB").clamp(-1, 1)
    u8 = ((image + 1.0) * 127.5).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
    pil = Image.fromarray(u8, mode="RGB")
    kind, value = condition.get("kind"), condition.get("value")
    if kind == "clean":
        pass
    elif kind == "gaussian_blur":
        sigma = float(value)
        if sigma not in (1.0, 2.0, 3.0):
            raise ValueError("Figure 4 chỉ định Gaussian blur σ=1,2,3")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif kind == "jpeg":
        quality = int(value)
        if quality not in (65, 30):
            raise ValueError("Figure 4 chỉ định JPEG quality=65,30")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as decoded:
            pil = decoded.convert("RGB").copy()
    else:
        raise ValueError(f"perturbation không hỗ trợ: {kind!r}")
    arr = np.asarray(pil, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def image_variant_metadata(name: str) -> dict:
    """Machine-readable contract persisted with each variant's model and scores."""
    key = _canonical_representation(name)
    channels = 6 if key == "RGB_DIRE" else 3
    return {
        "name": key,
        "channels": channels,
        "rgb_range": "[0,1]" if key in ("RGB", "RGB_DIRE") else None,
        "rec_range": "[0,1]" if key == "REC" else None,
        "dire_range": "[0,2]" if key in ("DIRE", "RGB_DIRE") else None,
        "rgb_dire_order": ["RGB", "DIRE"] if key == "RGB_DIRE" else None,
        "fusion": "channel_concatenation" if key == "RGB_DIRE" else None,
        "normalization": "ImageNet mean/std per RGB-like channel group",
    }


def robustness_metadata() -> dict:
    return {
        "conditions": [dict(x) for x in ROBUSTNESS_LEVELS],
        "input_stage": "source image before feature extraction; clean source is used for every condition",
        "dire_policy": "recompute DIRE from each perturbed image (not perturb a cached clean DIRE map)",
        "metric": "average_precision",
    }
