"""Toán của lịch nhiễu + DDIM, tách khỏi torch để test được ở mọi máy.

Vì sao tách riêng
-----------------
Ba thứ quyết định **đúng/sai** và **tiền** của đường tái lập from-scratch đều là
số học thuần, không cần GPU:

* `ᾱ_t` — quyết định mô hình "thấy" bao nhiêu tín hiệu ở mốc bắt đầu tái tạo;
* tập mốc respacing — quyết định `t` cuối là 950 hay 999 (khác biểu diễn là
  khác DIRE map, dù trông giống nhau);
* `NFE` — quyết định hoá đơn. Lỗi cũ (chạy 1020 thay vì 40 NFE) nhân chi phí
  lên 25 lần mà **không** có exception nào.

Đặt ở `dire_core` (Python thuần) nên `tests/test_scratch.py` chạy được trên máy
không có torch, và `ddim.py` (phần torch) chỉ việc gọi lại các hàm dưới đây.
"""

from __future__ import annotations

#: Số bước huấn luyện của ADM LSUN-Bedroom (`256x256_diffusion_uncond.pt`).
T_TRAIN = 1000
BETA_MIN = 1e-4
BETA_MAX = 2e-2


def linear_betas(
    t_train: int = T_TRAIN, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX
) -> list[float]:
    """Lịch tuyến tính của DDPM/ADM: ``β_t`` với ``t = 1..T``.

    Công thức (giáo trình Phần 5 §5.7):

        β_t = β_min + (t / (T-1)) · (β_max − β_min)

    Chú ý chỉ số: dùng ``t`` (1-based) chứ không phải ``t−1``. Hai cách lệch
    nhau rất ít ở giữa lịch nhưng lệch ~0,5% ở ``t = 250`` — đủ để bảng trong
    sách và bảng do script in ra không khớp, gây tranh cãi không cần thiết.
    """
    if t_train < 2:
        raise ValueError("t_train phải ≥ 2")
    if not 0 < beta_min < beta_max < 1:
        raise ValueError("cần 0 < beta_min < beta_max < 1")
    return [beta_min + (t / (t_train - 1)) * (beta_max - beta_min)
            for t in range(1, t_train + 1)]


def alphas_cumprod(betas: list[float]) -> list[float]:
    """``ᾱ_t = ∏_{s≤t} (1 − β_s)``, có ``ᾱ_0 = 1`` (chưa thêm nhiễu).

    Trả list dài ``len(betas)+1`` để ``a[t]`` đúng nghĩa "sau t bước".
    """
    out = [1.0]
    for b in betas:
        out.append(out[-1] * (1.0 - b))
    return out


def ddim_timesteps(t_train: int = T_TRAIN, steps: int = 20) -> list[int]:
    """Tập mốc của `SpacedDiffusion(..., "ddim20")` trên lịch ``T`` bước.

    Với ``T = 1000, steps = 20`` ⇒ ``[0, 50, …, 950]``: **20 phần tử, mốc lớn
    nhất 950**, không phải 999. Đây đúng là điều làm DIRE map của bản cài sai
    (dùng trọn 1000 bước) khác map phát hành, dù cả hai "20 bước".
    """
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    if steps > t_train:
        raise ValueError("steps không được vượt t_train")
    return [round(i * t_train / steps) for i in range(steps)]


def nfe_per_image(steps: int) -> int:
    """Số lượt chạy mạng cho MỘT ảnh: ``S`` bước đảo ngược + ``S`` bước tái tạo."""
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    return 2 * steps


def nfe_total(n_images: int, steps: int = 20) -> int:
    return n_images * nfe_per_image(steps)


def inversion_pairs(timesteps: list[int]) -> list[tuple[int, int]]:
    """Cặp ``(t_hiện_tại, t_kế_tiếp)`` khi đi **ngược** từ ảnh sạch lên nhiễu.

    Đảo ngược DDIM dùng ``ᾱ_{t+1}`` (không phải ``ᾱ_{t−1}``): từ ``x_t`` suy ra
    ``x_{t+1}``. Với tập ``[0, 50, …, 950]`` cặp đầu là ``(0, 50)``.
    """
    return list(zip(timesteps[:-1], timesteps[1:]))


def reconstruction_pairs(timesteps: list[int]) -> list[tuple[int, int]]:
    """Cặp ``(t, t_trước)`` khi đi **xuôi** từ nhiễu về ảnh, kết thúc ở ``(50, 0)``."""
    # Đi XUÔI theo thời gian đã respacing: 800→600 → … → 200→0.
    return list(zip(timesteps[:0:-1], timesteps[-2::-1]))


def check_nfe(counted: int, n_images: int, steps: int = 20) -> None:
    """Chặn cứng nếu số lượt chạy mạng khác ``2·S·N``.

    Đây là "điều kiện vàng G5" ở dạng chạy được: đường cũ chạy 1020 NFE/ảnh vì
    gọi cả quỹ đạo 1000 bước; sai 25 lần mà kết quả vẫn in ra bình thường. Một
    assert rẻ hơn một hoá đơn $5.650.
    """
    want = nfe_total(n_images, steps)
    if counted != want:
        raise AssertionError(
            f"NFE sai: đếm {counted}, phải là {want} "
            f"({n_images} ảnh × 2 × {steps} bước). Dừng trước khi trả tiền GPU."
        )
