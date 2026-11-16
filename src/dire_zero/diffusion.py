"""DDIM đảo ngược + tái tạo, và DIRE map — phần cần torch.

Định nghĩa, đúng ba dòng:

::

    ε̂_t  = f_θ(x_t, t)[:3]                        (3 kênh đầu; 3 kênh sau là log-σ²)
    x̂₀   = (x_t − √(1−ᾱ_t)·ε̂_t) / √ᾱ_t            ∈ [−1, 1] (clip_denoised)
    đảo ngược  x_{t+1} = √ᾱ_{t+1}·x̂₀ + √(1−ᾱ_{t+1})·ε̂_t
    tái tạo    x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1−ᾱ_{t−1})·ε̂_t
    DIRE       |x₀ − R(I(x₀))|                    ∈ [0, 2], 3 kênh

Hai điều bắt buộc, và đây là chỗ khác bản cũ:

1. **NFE đếm thật.** `Counter` tăng ngay trong `_eval`, không đếm bằng công
   thức. Hàm trả về số đếm được, và `dire_map` assert `= 2S` trước khi trả.
2. **Không chia cho 0 ở mốc 0.** `ᾱ[0] = 0,9999` (0-based như ADM) nên
   `√(1−ᾱ[0]) ≈ 0,01`. Nếu dùng `ᾱ[0] = 1,0` thì `ε̂ = 0/0 = NaN` ngay bước đầu
   và **toàn bộ map thành NaN** mà không có exception nào — lỗi này có thật
   trong đường cũ, xem `docs/from-zero-clean-room.md` §"Ba lỗi tìm được".
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from . import schedule as sch

__all__ = ["Schedule", "ModelCounter", "invert", "reconstruct", "sample", "dire_map", "to_uint8"]


class Schedule:
    """Bảng ``ᾱ`` + tập mốc, giữ dạng tensor để chỉ số hoá rẻ."""

    def __init__(self, t_train: int = sch.T_TRAIN, steps: int = 20, device: str = "cpu") -> None:
        self.t_train = t_train
        self.steps = steps
        gamma = sch.alphas_cumprod(sch.linear_betas(t_train))
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.timesteps = sch.timesteps(t_train, steps)
        self.device = device

    def a(self, t: int, dtype: torch.dtype = torch.float32) -> Tensor:
        """``ᾱ_t`` dạng scalar tensor broadcast được với ảnh."""
        return self.gamma[t].to(dtype=dtype, device=self.device).view(1, 1, 1, 1)

    def __repr__(self) -> str:  # pragma: no cover - chỉ để đọc log
        return f"Schedule(T={self.t_train}, S={self.steps}, t_max={self.timesteps[-1]})"


class ModelCounter:
    """Bọc mô hình, đếm số lượt chạy thật."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.nfe = 0

    def __call__(self, x: Tensor, t: int) -> tuple[Tensor, Tensor]:
        ts = torch.full((x.shape[0],), int(t), dtype=torch.long, device=x.device)
        with torch.inference_mode():
            out = self.model(x, ts)
        self.nfe += x.shape[0]
        eps = out[:, :3] if out.shape[1] == 6 else out
        return eps, ts

    def reset(self) -> int:
        n, self.nfe = self.nfe, 0
        return n


def _x0(eps: Tensor, x_t: Tensor, a: Tensor, clip: bool = True) -> Tensor:
    x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt()
    return x0.clamp(-1.0, 1.0) if clip else x0


def _eval(f: ModelCounter, x: Tensor, t: int, sched: Schedule, clip: bool) -> tuple[Tensor, Tensor]:
    """Một lượt mạng tại mốc ``t``: trả ``(x̂₀, ε̂)``."""
    eps, _ = f(x, t)
    a = sched.a(t, dtype=x.dtype)
    x0 = _x0(eps, x, a, clip)
    # ε̂ nhất quán với x̂₀ sau khi clip (đúng như bản gốc: clip rồi suy lại ε)
    eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
    return x0, eps


def invert(f: ModelCounter, x0_img: Tensor, sched: Schedule, clip: bool = True) -> Tensor:
    """Ảnh sạch → trạng thái nhiễu ở mốc cao nhất. Đúng ``S`` lượt mạng."""
    x = x0_img
    ts = sched.timesteps
    for i, t in enumerate(ts):
        x0h, eps = _eval(f, x, t, sched, clip)
        t_next = ts[min(i + 1, len(ts) - 1)]
        a_next = sched.a(t_next, dtype=x.dtype)
        x = a_next.sqrt() * x0h + (1 - a_next).sqrt() * eps
    return x


def reconstruct(f: ModelCounter, x_top: Tensor, sched: Schedule, clip: bool = True) -> Tensor:
    """Trạng thái nhiễu → ảnh tái tạo. Đúng ``S`` lượt mạng.

    Bước cuối dùng ``ᾱ = 1`` (ảnh sạch hoàn toàn), không dùng ``ᾱ[0] = 0,9999``:
    ``√1·x̂₀ + √0·ε̂ = x̂₀``. Nếu dùng ``ᾱ[0]``, đầu ra còn ~0,5 % nhiễu và DIRE
    map bị "nhiễu hoá" đúng bằng phần đó — sai lệch nhỏ nhưng **có hệ thống**,
    và nó làm reconstruction trông tệ hơn thực tế ở mọi ảnh như nhau.
    """
    x = x_top
    ts = sched.timesteps
    for i in range(len(ts) - 1, -1, -1):
        x0h, eps = _eval(f, x, ts[i], sched, clip)
        if i == 0:
            x = x0h
            continue
        a_prev = sched.a(ts[i - 1], dtype=x.dtype)
        x = a_prev.sqrt() * x0h + (1 - a_prev).sqrt() * eps
    return x


def sample(f: ModelCounter, shape, sched: Schedule, seed: int = 0, clip: bool = True) -> Tensor:
    """Sinh ảnh từ nhiễu thuần (DDIM, η = 0). Đúng ``S`` lượt mạng."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(shape, generator=g, dtype=torch.float32).to(sched.device)
    return reconstruct(f, x, sched, clip)


def dire_map(
    f: ModelCounter,
    x0_img: Tensor,
    sched: Schedule | None = None,
    clip: bool = True,
) -> tuple[Tensor, Tensor]:
    """``(|x₀ − x̂₀|, x̂₀)``; assert NFE ``= 2·S`` trước khi trả."""
    sched = sched or Schedule(steps=20, device=str(x0_img.device))
    before = f.nfe              # không reset: bộ đếm là của cả lô, reset ở đây là mất số
    x_top = invert(f, x0_img, sched, clip)
    recon = reconstruct(f, x_top, sched, clip)
    sch.check_nfe(f.nfe - before, x0_img.shape[0], sched.steps)
    return (x0_img - recon).abs(), recon


def to_uint8(x: Tensor) -> Tensor:
    """``[−1, 1]`` → uint8 thang **1×**: ``(x+1)·127,5``. Không nhân 4, không clip 2 lần."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


def from_uint8(img: "object") -> Tensor:
    """uint8 ``[0, 255]`` → float ``[−1, 1]``, shape ``(1, 3, H, W)``."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def sigma_floor(sched: Schedule) -> float:
    """``min √(1 − ᾱ_t)`` trên tập mốc — phải > 0, nếu không là NaN chờ sẵn."""
    return min(math.sqrt(max(0.0, 1.0 - float(sched.gamma[t]))) for t in sched.timesteps)
