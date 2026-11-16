"""DDIM inversion + reconstruction, và DIRE map — bản dùng cho đường scratch.

Vì sao viết lại thay vì sửa `compute_dire.py`
---------------------------------------------
`compute_dire.py` là bản port sát mã tác giả, có ba lỗi đã ghi nhận (F1 truyền
`x0` vào vị trí `shape`, F2 chạy 1020 NFE thay vì 40, F3 `|Δ|·4` rồi clip).
Sửa nó là việc riêng, có người phải duyệt — và nếu sửa thì mất luôn khả năng
đối chiếu với bản port cũ.

Đường scratch cần một bản **đúng theo định nghĩa**, nên ở đây viết mới, ngắn, và
mỗi bước đối chiếu được với công thức trong giáo trình:

    đảo ngược :  x_{t+1} = √ᾱ_{t+1}·x̂₀ + √(1−ᾱ_{t+1})·ε̂      (dùng ᾱ KẾ tiếp)
    tái tạo   :  x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1−ᾱ_{t−1})·ε̂
    DIRE      :  |x₀ − R(I(x₀))|, giữ 3 kênh, ở [-1, 1]

Ba điểm khác bản port cũ, có chủ ý:

1. **Đếm NFE thật.** Mỗi lần gọi mô hình tăng bộ đếm; hàm trả về `nfe` và
   `dire_for_image` assert `nfe == 2·S`. Không dựa vào "chắc là 40".
2. **Không khuếch đại.** Không nhân `|Δ|` với 4, không clip trước khi lưu.
   `|Δ|` nằm trong `[0, 2]`; việc đổi sang `[0, 255]` là `·255/2`, làm đúng
   **một lần** ở chỗ ghi file, và ghi rõ trong metadata.
3. **Mốc lấy từ `ddim_schedule.ddim_timesteps`** ⇒ `{0, 50, …, 950}`. Đây là
   chỗ dễ sai nhất: dùng trọn 1000 bước cho ra map khác hẳn mà trông vẫn "ổn".

Phụ thuộc: `guided-diffusion` (clone trong image, xem `common.gpu_image`) để
**dựng kiến trúc** U-Net; trọng số là checkpoint công khai của OpenAI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dire_core.ddim_schedule import (
    alphas_cumprod,
    check_nfe,
    ddim_timesteps,
    linear_betas,
)

#: Nơi `common.gpu_image` clone repo tác giả (chỉ dùng `guided_diffusion`).
DIRE_SRC = "/opt/dire/guided-diffusion"


def _import_guided():
    if DIRE_SRC not in sys.path:
        sys.path.insert(0, DIRE_SRC)
    try:
        from guided_diffusion.script_util import (  # noqa: PLC0415
            create_model,
            model_and_diffusion_defaults,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Không import được guided_diffusion từ {DIRE_SRC}. "
            "Trong image GPU nó được clone sẵn; ở máy local hãy "
            "`git clone https://github.com/openai/guided-diffusion` và đặt "
            "biến DIRE_GUIDED_DIFFUSION trỏ tới thư mục đó."
        ) from exc
    return create_model, model_and_diffusion_defaults


#: Siêu tham số của `256x256_diffusion_uncond.pt` (đọc từ `compute_dire.sh`).
ADM_ARGS = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
    learn_sigma=True,
    diffusion_steps=1000,
    noise_schedule="linear",
)


def load_adm(ckpt: str | Path, device: str = "cuda"):
    """Nạp ADM ở chế độ eval, trả `(model, device)`.

    `torch.load(..., map_location=...)` là bắt buộc: checkpoint lưu trên GPU,
    máy không có CUDA sẽ chết nếu thiếu. Bản port cũ thiếu tham số này.
    """
    import torch  # noqa: PLC0415

    create_model, defaults = _import_guided()
    args = dict(defaults())
    args.update(ADM_ARGS)
    model = create_model(**args)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "state_dict" not in state:
        state = state["model"]  # một số checkpoint bọc thêm {"model": ...}
    model.load_state_dict(state)
    if ADM_ARGS["use_fp16"] and device.startswith("cuda"):
        model.convert_to_fp16()
    model.to(device)
    model.eval()
    return model, device


class Schedule:
    """Bảng `ᾱ_t` + tập mốc, tính một lần rồi tra cứu."""

    def __init__(self, t_train: int = 1000, steps: int = 20):
        self.t_train = t_train
        self.steps = steps
        self.abar = alphas_cumprod(linear_betas(t_train))
        self.timesteps = ddim_timesteps(t_train, steps)

    def _tensor(self, t: int, device, dtype):
        import torch  # noqa: PLC0415

        return torch.full((1,), int(t), device=device, dtype=torch.long)

    def _ab(self, t: int, device, dtype):
        """``ᾱ`` tại **mốc** ``t``, dùng chỉ số 0-based như ADM.

        `alphas_cumprod` trả ``[1,0, ᾱ₁, …]`` (phần tử 0 = "chưa thêm nhiễu"),
        nên mốc ``t`` ứng với ``abar[t + 1]``, tức ``ᾱ`` **sau** ``t + 1`` bước
        nhiễu. Bản cũ lấy ``abar[t]``: tại ``t = 0`` ra ``ᾱ = 1,0`` ⇒
        ``√(1−ᾱ) = 0`` ⇒ ``ε̂ = 0/0 = NaN``, và NaN đó lan qua cả map. Xem
        `tests/test_ddim_protocol.py`.
        """
        import torch  # noqa: PLC0415

        v = self.abar[min(int(t) + 1, len(self.abar) - 1)]
        return torch.tensor(v, device=device, dtype=dtype).view(1, 1, 1, 1)

    def eps_from_x0(self, x_t, t: int, x0, device):
        """ε̂ = (x_t − √ᾱ_t·x̂₀) / √(1−ᾱ_t) — đảo lại `q_sample` để lấy ε từ x̂₀."""
        a = self._ab(t, device, x_t.dtype)
        return (x_t - a.sqrt() * x0) / (1 - a).sqrt()


def _predict(model, x_t, t, schedule: Schedule, device):
    """Một lượt chạy mạng: trả `(x̂₀, ε̂, nfe)`.

    ADM có `learn_sigma=True` nên đầu ra 6 kênh; 3 kênh đầu là ε, 3 kênh sau là
    phương sai. Lấy `[:, :3]` — lấy cả 6 là lỗi im lặng kinh điển.
    """
    import torch  # noqa: PLC0415

    ts = schedule._tensor(t, device, x_t.dtype)
    with torch.inference_mode():
        out = model(x_t, ts)
    eps = out[:, :3] if out.shape[1] == 6 else out
    a = schedule._ab(t, device, x_t.dtype)
    x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt()
    x0 = x0.clamp(-1.0, 1.0)  # clip_denoised=True, đúng như script tác giả
    eps = schedule.eps_from_x0(x_t, t, x0, device)
    return x0, eps, 1


def invert(model, x0_img, schedule: Schedule, device):
    """x₀ → mốc cao nhất, **đúng S lượt mạng**; trả `(x_last, nfe)`.

    `S − 1` cặp mốc chỉ cho `S − 1` lượt, nên bước cuối chiếu lại trạng thái tại
    chính mốc cao nhất (``x = √ᾱ·x̂₀ + √(1−ᾱ)·ε̂`` là phép chiếu idempotent, chỉ
    khác khi ``x̂₀`` bị clip). Thiếu bước này thì NFE = `2(S−1)` chứ không phải
    `2S`, và con số chi phí bị báo thấp hơn thực tế.
    """
    x = x0_img
    nfe = 0
    for t_cur, t_next in zip(schedule.timesteps[:-1], schedule.timesteps[1:]):
        x0_hat, eps_hat, used = _predict(model, x, t_cur, schedule, device)
        a_next = schedule._ab(t_next, device, x.dtype)
        x = a_next.sqrt() * x0_hat + (1 - a_next).sqrt() * eps_hat
        nfe += used
    x0_hat, eps_hat, used = _predict(model, x, schedule.timesteps[-1], schedule, device)
    a_max = schedule._ab(schedule.timesteps[-1], device, x.dtype)
    x = a_max.sqrt() * x0_hat + (1 - a_max).sqrt() * eps_hat
    nfe += used
    return x, nfe


def reconstruct(model, x_t, schedule: Schedule, device):
    """Mốc cao nhất → x̂₀, **đúng S lượt mạng**; trả `(recon, nfe)`.

    Bước cuối chạy mạng tại mốc 0 rồi **trả thẳng x̂₀** (ảnh sạch), thay vì cộng
    thêm ``√(1−ᾱ₀)·ε̂ ≈ 0,01·ε̂``. Bản cũ dừng ở cặp ``(50, 0)`` nên không có
    lượt nào tại mốc 0, và đầu ra còn ~0,5 % nhiễu — sai lệch nhỏ nhưng **có hệ
    thống** trên mọi ảnh, tức là nó dịch cả phân bố DIRE.
    """
    x = x_t
    nfe = 0
    for t_cur, t_prev in zip(schedule.timesteps[:0:-1], schedule.timesteps[-2::-1]):
        x0_hat, eps_hat, used = _predict(model, x, t_cur, schedule, device)
        a_prev = schedule._ab(t_prev, device, x.dtype)
        x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps_hat
        nfe += used
    x0_hat, _eps, used = _predict(model, x, 0, schedule, device)
    nfe += used
    return x0_hat, nfe


def dire_map(model, x0_img, schedule: Schedule | None = None, device: str | None = None):
    """Tính DIRE map cho một ảnh `[-1, 1]`, shape `(1, 3, H, W)`.

    Trả `(dire, recon, nfe)` với `dire = |x₀ − recon|` trong `[0, 2]` và
    `nfe` **đếm thật** — hàm assert `nfe == 2·S` trước khi trả.
    """
    schedule = schedule or Schedule()
    device = device or str(x0_img.device)
    x_last, n1 = invert(model, x0_img, schedule, device)
    recon, n2 = reconstruct(model, x_last, schedule, device)
    nfe = n1 + n2
    check_nfe(nfe, 1, schedule.steps)
    return (x0_img - recon).abs(), recon, nfe


def to_uint8(x):
    """`[-1, 1]` → `[0, 255]` uint8. Thang **1×**: `(x+1)·127,5`, không nhân 4."""
    import torch  # noqa: PLC0415

    return ((x.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()


def sample(model, shape, schedule: Schedule, device: str, seed: int, noise=None):
    """Sinh một ảnh từ nhiễu thuần (DDIM, η = 0), trả `(x0, nfe)`.

    Dùng cho R3: ảnh "giả" do **mình** sinh, seed do mình chọn. `noise` cho phép
    truyền nhiễu cố định để test tất định.
    """
    import torch  # noqa: PLC0415

    if noise is None:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(shape, generator=g, dtype=torch.float32)
    x = noise.to(device)
    nfe = 0
    steps = schedule.timesteps
    for t_cur, t_prev in zip(steps[:0:-1], steps[-2::-1]):
        x0_hat, eps_hat, used = _predict(model, x, t_cur, schedule, device)
        a_prev = schedule._ab(t_prev, device, x.dtype)
        x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps_hat
        nfe += used
    # bước cuối: t = 0
    x0_hat, _eps, used = _predict(model, x, 0, schedule, device)
    nfe += used
    return x0_hat, nfe
