"""Lịch nhiễu và tập mốc — toán thuần, không cần torch.

Đây là chỗ một lỗi chỉ số đã từng làm hỏng cả đường tái lập, nên nó được viết
lại cẩn thận và có test khoá từng mốc.

**Chỉ số 0-based, giống bản gốc ADM.** ``ᾱ[0] = 1 − β₀ = 0,9999`` — không phải
``1,0``. Nghe như chuyện vụn, nhưng:

* ``ᾱ[0] = 1`` ⇒ ``√(1 − ᾱ[0]) = 0`` ⇒ phép đảo ngược ở bước đầu chia cho 0 để
  suy ra ``ε̂`` ⇒ **toàn bộ DIRE map thành NaN**, không một exception nào;
* ``ᾱ[0] = 0,9999`` ⇒ mọi mẫu số ≥ 0,01, và mốc ``t = 0`` vẫn là "gần sạch".

**Mốc cuối là 950, không phải 999.** ``[round(i·1000/20)] = [0, 50, …, 950]``,
đúng tập mà ``timestep_respacing="ddim20"`` dùng. Dùng trọn 1000 bước cho ra map
khác hẳn, trông vẫn hợp lý, và tốn 25 lần tiền.
"""

from __future__ import annotations

__all__ = [
    "T_TRAIN",
    "linear_betas",
    "alphas_cumprod",
    "timesteps",
    "nfe_per_image",
    "check_nfe",
]

#: Số bước huấn luyện của ADM 256×256 (linear schedule).
T_TRAIN = 1000
BETA_MIN = 1e-4
BETA_MAX = 2e-2


def linear_betas(t_train: int = T_TRAIN, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX) -> list[float]:
    """``β_t`` tuyến tính, ``t = 0 … T−1`` (0-based, như ADM)."""
    if t_train < 2:
        raise ValueError("t_train phải ≥ 2")
    if not 0 < beta_min < beta_max < 1:
        raise ValueError("cần 0 < beta_min < beta_max < 1")
    return [beta_min + (t / (t_train - 1)) * (beta_max - beta_min) for t in range(t_train)]


def alphas_cumprod(betas: list[float]) -> list[float]:
    """``ᾱ_t = ∏_{s ≤ t} (1 − β_s)``, dài đúng ``len(betas)``, ``ᾱ_0 = 1 − β₀``."""
    out: list[float] = []
    cur = 1.0
    for b in betas:
        cur *= 1.0 - b
        out.append(cur)
    return out


def timesteps(t_train: int = T_TRAIN, steps: int = 20) -> list[int]:
    """Tập mốc **tăng dần**, ``S`` phần tử, mốc cuối ``round((S−1)·T/S)``."""
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    if steps > t_train:
        raise ValueError("steps không được vượt t_train")
    return [round(i * t_train / steps) for i in range(steps)]


def nfe_per_image(steps: int) -> int:
    """S lượt chạy mạng cho phép đảo ngược + S lượt cho tái tạo."""
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    return 2 * steps


def check_nfe(counted: int, n_images: int, steps: int = 20) -> None:
    """Chặn cứng nếu NFE khác ``2·S·N`` — cổng tiền, không phải cổng kỹ thuật."""
    want = n_images * nfe_per_image(steps)
    if counted != want:
        raise AssertionError(
            f"NFE sai: đếm {counted}, phải là {want} ({n_images} ảnh × 2 × {steps} bước). "
            "Dừng trước khi trả tiền GPU."
        )
