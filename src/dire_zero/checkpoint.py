"""Nối tên tham số của mình với tên trong checkpoint ADM.

Vì sao cần một lớp riêng
------------------------
Checkpoint `256x256_diffusion_uncond.pt` có 566 tensor với tên do bản gốc đặt
(`input_blocks.7.0.in_layers.2.weight`). Mô hình ở :mod:`dire_zero.unet` đặt tên
theo ý mình (`enc.3.1.res.conv_in.weight`) để đọc code là hiểu cấu trúc.

Hai cách nối, và cách đã chọn:

* Đổi tên mô hình cho giống checkpoint — nhanh, nhưng mất luôn thông tin "đây
  là bản viết lại": không ai phân biệt được với bản chép.
* Giữ tên riêng + **một bảng ánh xạ tường minh**, kiểm bằng test song ánh trên
  file `tests/fixtures/adm_lsun_state_keys.json` (566 cặp tên → shape, sinh từ
  kiến trúc tham chiếu, không chứa trọng số).

Cách thứ hai có thêm một lợi ích thực dụng: bảng ánh xạ là chỗ **duy nhất** có
thể làm rơi một tensor. `strict=True` của PyTorch bắt được điều đó, nhưng chỉ
khi có checkpoint thật trong tay — tức là chỉ trên GPU, sau khi đã trả tiền.
Test song ánh bắt được ngay trên laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from torch import nn

from .data import sha256_file
from .unet import ADM_LSUN, UNetConfig, ZeroUNet

__all__ = [
    "RES_SUFFIX",
    "ATTN_SUFFIX",
    "iter_name_pairs",
    "check_bijection",
    "map_state_dict",
    "load_adm",
    "sha256_file",
]

#: hậu tố tham số trong ResBlock của mình → của checkpoint
RES_SUFFIX = {
    "norm_in": "in_layers.0",
    "conv_in": "in_layers.2",
    "emb_mlp.1": "emb_layers.1",
    "norm_out": "out_layers.0",
    "conv_out": "out_layers.3",   # out_layers = [norm, SiLU, Dropout, conv]
    "skip": "skip_connection",
}

#: hậu tố trong Attention của mình → của checkpoint
ATTN_SUFFIX = {"norm": "norm", "qkv": "qkv", "proj": "proj_out"}


def _params(module: nn.Module, own: str, ref: str) -> Iterator[tuple[str, str]]:
    """Sinh cặp tên cho mọi tham số/buffer của `module`, giữ nguyên hậu tố."""
    for name, _ in module.named_parameters(recurse=True):
        yield f"{own}.{name}", f"{ref}.{name}"
    for name, buf in module.named_buffers(recurse=True):
        if buf is not None:
            yield f"{own}.{name}", f"{ref}.{name}"


def _res(module: nn.Module, own: str, ref: str) -> Iterator[tuple[str, str]]:
    for my_suffix, ref_suffix in RES_SUFFIX.items():
        sub = module
        for part in my_suffix.split("."):
            sub = getattr(sub, part, None)
            if sub is None:
                break
        if sub is None:
            continue
        if isinstance(sub, nn.Identity):
            continue
        for name, _ in sub.named_parameters(recurse=True):
            yield f"{own}.{my_suffix}.{name}", f"{ref}.{ref_suffix}.{name}"


def _attn(module: nn.Module, own: str, ref: str) -> Iterator[tuple[str, str]]:
    for my_suffix, ref_suffix in ATTN_SUFFIX.items():
        sub = getattr(module, my_suffix)
        for name, _ in sub.named_parameters(recurse=True):
            yield f"{own}.{my_suffix}.{name}", f"{ref}.{ref_suffix}.{name}"


def iter_name_pairs(model: ZeroUNet) -> Iterator[tuple[str, str]]:
    """Duyệt mô hình theo đúng thứ tự cấu trúc, sinh (tên của mình, tên checkpoint)."""
    yield from _params(model.time_mlp, "time_mlp", "time_embed")
    yield from _params(model.stem, "stem", "input_blocks.0.0")

    ref_i = 1
    for level, entries in enumerate(model.enc):
        for k, entry in enumerate(entries):
            base = f"enc.{level}.{k}"
            yield from _res(entry.res, f"{base}.res", f"input_blocks.{ref_i}.0")
            if entry.attn is not None:
                yield from _attn(entry.attn, f"{base}.attn", f"input_blocks.{ref_i}.1")
            ref_i += 1

    for j, block in enumerate(model.mid):
        if j == 1:
            yield from _attn(block, f"mid.{j}", f"middle_block.{j}")
        else:
            yield from _res(block, f"mid.{j}", f"middle_block.{j}")

    ref_i = 0
    for level, entries in enumerate(model.dec):
        for k, entry in enumerate(entries):
            base = f"dec.{level}.{k}"
            yield from _res(entry.res, f"{base}.res", f"output_blocks.{ref_i}.0")
            j = 1
            if entry.attn is not None:
                yield from _attn(entry.attn, f"{base}.attn", f"output_blocks.{ref_i}.{j}")
                j += 1
            if entry.up is not None:
                yield from _res(entry.up, f"{base}.up", f"output_blocks.{ref_i}.{j}")
            ref_i += 1

    yield from _params(model.head_norm, "head_norm", "out.0")
    yield from _params(model.head_conv, "head_conv", "out.2")


def check_bijection(model: ZeroUNet, ref_keys: dict[str, list[int]]) -> dict[str, str]:
    """Đối chiếu bảng ánh xạ với bộ key tham chiếu. Sai là `AssertionError`.

    Ba điều kiện, mỗi điều bắt một kiểu lỗi khác nhau:

    1. mọi tên của checkpoint đều xuất hiện **đúng một lần** (ánh xạ toàn phần,
       không trùng) — bắt lỗi đánh số block;
    2. mọi tham số của mình đều có đích — bắt lỗi thêm/bớt khối;
    3. shape khớp từng cặp — bắt lỗi sai số kênh, sai kernel, sai stride.
    """
    pairs = list(iter_name_pairs(model))
    mapping = dict(pairs)
    assert len(mapping) == len(pairs), "có tên của mình bị sinh hai lần"

    mine = dict(model.state_dict())
    got = set(mapping.values())
    missing = sorted(set(ref_keys) - got)
    extra = sorted(got - set(ref_keys))
    assert not missing, f"{len(missing)} key của checkpoint không được nạp, vd {missing[:3]}"
    assert not extra, f"{len(extra)} key không tồn tại trong checkpoint, vd {extra[:3]}"
    assert len(got) == len(pairs), "hai tham số của mình cùng trỏ về một key"

    for my_name, ref_name in pairs:
        assert tuple(mine[my_name].shape) == tuple(ref_keys[ref_name]), (
            f"shape lệch: {my_name}{tuple(mine[my_name].shape)} "
            f"vs {ref_name}{tuple(ref_keys[ref_name])}"
        )
    return mapping


def map_state_dict(
    state: dict, model: ZeroUNet, ref_keys: dict[str, list[int]] | None = None
) -> dict[str, "object"]:
    """Đổi `{tên checkpoint: tensor}` thành `{tên của mình: tensor}`."""
    mapping = dict(iter_name_pairs(model))
    if ref_keys is not None:
        check_bijection(model, ref_keys)
    keys = set(state)
    if ("backbone.fc.weight" in keys or "backbone.fc.bias" in keys
            or any(k.startswith("backbone.stages.") for k in keys)):
        raise ValueError(
            "checkpoint nhận được là ResNet/DIRE detector (`backbone.*`), không phải ADM diffusion U-Net. "
            "`lsun_adm.pth` của tác giả là detector để chấm DIRE map; không thể chuyển nó thành ADM bằng cách đổi tên hoặc copy. "
            "Local inference cần trọng số bộ sinh `256x256_diffusion_uncond.pt` (~2,5 GB). "
            "Đặt đúng tệp tại `platforms/hf-spaces/ckpt/adm_lsun.pt` hoặc trỏ `DIRE_CKPT` tới nó."
        )
    unknown = sorted(keys - set(mapping.values()))
    assert not unknown, f"checkpoint có {len(unknown)} key lạ, vd {unknown[:3]}"
    out = {}
    for my_name, ref_name in mapping.items():
        if ref_name in state:
            out[my_name] = state[ref_name]
    return out


def load_adm(
    path: str | Path,
    cfg: UNetConfig = ADM_LSUN,
    device: str = "cpu",
    dtype: str = "float32",
    strict: bool = True,
) -> ZeroUNet:
    """Nạp `256x256_diffusion_uncond.pt` vào U-Net của mình.

    `strict=True` là **chốt chặn của cả package**: nếu kiến trúc của mình lệch
    kiến trúc thật dù chỉ một tensor, hàm này nổ ngay tại đây thay vì cho ra
    những bức ảnh trông hợp lý mà vô nghĩa.
    """
    import torch  # noqa: PLC0415  (import trễ: `python -m dire_zero --help` không cần torch)

    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "ema"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    model = ZeroUNet(cfg)
    model.load_state_dict(map_state_dict(raw, model), strict=strict)
    model.to(device=device)
    if dtype in ("float16", "half"):
        model.half()
    else:
        model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
