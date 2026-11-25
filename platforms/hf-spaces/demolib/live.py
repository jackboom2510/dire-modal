"""Tính DIRE **thật** ngay tại chỗ — chậm, và nói rõ chậm bao nhiêu.

Vì sao module này tồn tại
-------------------------
Demo mặc định chỉ hiện map tính sẵn, nên câu hỏi "DIRE được tính thế nào" bị trả
lời bằng một tấm ảnh. Module này cho chạy đúng hai vòng lặp (đảo ngược rồi tái
tạo) và **nhìn thấy từng bước**: ở mỗi mốc `t`, `x̂₀` và `ε̂` hiện ra, nên người
xem thấy `|x₀ − x̂₀|` lớn lên hay nhỏ đi vì lý do gì.

Giá phải trả, đo trên 2 vCPU (Xeon 2,6 GHz, AVX-512, fp32) — không phải suy đoán:

===========  ==================  ==========================
việc         thời gian           ghi chú
===========  ==================  ==========================
1 forward    10,1 s              ADM 256², 552,8 M tham số
S = 5        101 s (1,7 phút)    10 forward
S = 20       404 s (6,7 phút)    40 forward — cấu hình của bài báo
1 forward @128²  2,6 s           **không dùng được**: ADM chỉ học ở 256²
===========  ==================  ==========================

Trong Gradio, local là backend mặc định nhưng inference chỉ chạy sau thao tác rõ ràng
của người dùng (chọn ảnh rồi bấm nút); app không tự nạp checkpoint khi khởi động.
Một lượt S=5 vẫn mất khoảng 101 s trên CPU đo được, vì vậy queue chỉ cho một job
đồng thời và UI báo rõ giới hạn thời gian trước khi chạy.

Số học không được tự viết lại
-----------------------------
Hai vòng lặp ở đây là **bản sao** của `dire_zero.diffusion.invert/reconstruct`,
cần thiết vì bản thật không trả về trạng thái trung gian. Để bản sao không trôi,
`tests/test_hf_spaces_live.py::test_stepwise_loop_matches_dire_zero` chạy cả hai
trên cùng mô hình tí hon và đòi map **bằng nhau từng phần tử**.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "MEASURED_CPU_FORWARD_SEC",
    "MEASURED_CPU_NOTE",
    "LiveStatus",
    "status",
    "load_checkpoint",
    "estimate_seconds",
    "run_steps",
]

#: Một lượt forward ADM 256² trên 2 vCPU, fp32 — đo, không suy ra.
#: Máy đo: Xeon 2,6 GHz, 2 vCPU, AVX-512, `torch==2.4.0`, `torch.set_num_threads(2)`.
MEASURED_CPU_FORWARD_SEC = 10.1

MEASURED_CPU_NOTE = (
    "2 vCPU Xeon 2,6 GHz (AVX-512), torch 2.4, fp32, batch 1. "
    "bf16 chậm hơn 3,4× (CPU không có AVX-512-BF16); channels_last không đổi; "
    "torch.compile chưa đo được ở máy đó."
)

#: Kích thước ADM học. Đưa ảnh khác cỡ vào là đo một mô hình ngoài phân bố của nó.
NATIVE = 256

#: Chặn cứng: quá số này thì không phải demo, là một buổi chờ.
MAX_STEPS = 20


@dataclass(frozen=True)
class LiveStatus:
    ok: bool
    torch: str
    dire_zero: bool
    ckpt: str
    reason: str
    hints: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.ok


def _try_import():
    try:
        import torch  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - phụ thuộc môi trường
        return None, None, f"không import được torch ({e})"
    try:
        from dire_zero import diffusion  # noqa: PLC0415
        from dire_zero.checkpoint import load_adm  # noqa: PLC0415
        from dire_zero.pipeline import to_hwc_uint8  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        return torch, None, f"không import được dire_zero ({e})"
    # `Schedule` nằm trong `diffusion` (cần torch), không phải trong `schedule`
    # (module thuần toán của `dire_zero`). Nhầm hai chỗ này là AttributeError.
    return torch, (diffusion, diffusion.Schedule, load_adm, to_hwc_uint8), ""


def ckpt_path() -> Path:
    """`DIRE_CKPT` nếu có; nếu không, `ckpt/adm_lsun.pt` cạnh `app.py`."""
    raw = os.environ.get("DIRE_CKPT", "")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p
    return Path(__file__).resolve().parent.parent / "ckpt" / "adm_lsun.pt"


def status() -> LiveStatus:
    """Mọi thứ cần để trả lời "bật được chưa" — không bao giờ ném exception."""
    torch, mods, err = _try_import()
    if torch is None:
        return LiveStatus(False, "—", False, str(ckpt_path()), err,
                          ["pip install torch==2.4.0", "hoặc bật chế độ tính sẵn (gallery)"])
    ver = torch.__version__
    if mods is None:
        return LiveStatus(False, ver, False, str(ckpt_path()), err,
                          ["cài repo: pip install \"dire-modal[zero]\"",
                           "Space cần thêm dòng này vào requirements.txt"])
    p = ckpt_path()
    if not p.exists():
        return LiveStatus(
            False, ver, True, str(p),
            f"chưa có checkpoint ADM ở {p}",
            ["gắn checkpoint 256×256 vào ổ đĩa mà Space đọc được rồi đặt DIRE_CKPT",
             "hoặc để tệp ở platforms/hf-spaces/ckpt/adm_lsun.pt khi chạy từ monorepo",
             "checkpoint ~2,5 GB: không commit; không tự tải xuống khi khởi động"],
        )
    return LiveStatus(True, ver, True, str(p), "sẵn sàng", [])


def estimate_seconds(steps: int, sec_per_forward: float = MEASURED_CPU_FORWARD_SEC) -> float:
    """`2S` forward (một lượt đảo ngược, một lượt tái tạo)."""
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    return 2 * steps * sec_per_forward


def load_checkpoint(path: Path | None = None):
    """Nạp ADM trên CPU, fp32. Trả `(model, sha256[:12])`."""
    torch, mods, err = _try_import()
    if mods is None:
        raise RuntimeError(err or "thiếu torch/dire_zero")
    _diffusion, _schedule, load_adm, _to_hwc = mods
    p = Path(path) if path else ckpt_path()
    if not p.exists():
        raise FileNotFoundError(f"không thấy checkpoint: {p}")
    model = load_adm(str(p), device="cpu", dtype="float32", strict=True)
    model.eval()
    try:
        from dire_zero.checkpoint import sha256_file  # noqa: PLC0415

        digest = sha256_file(str(p))
    except Exception:  # pragma: no cover
        digest = "?"
    return model, digest


def _to_uint8_hwc(t) -> "object":
    """Tensor ảnh → mảng uint8, qua **đúng** hàm của pipeline (một nguồn duy nhất)."""
    _torch, mods, err = _try_import()
    if mods is None:
        raise RuntimeError(err or "thiếu dire_zero")
    return mods[3](t)


def _png_bytes(hwc, max_side: int = 128) -> bytes:
    from PIL import Image  # noqa: PLC0415

    im = Image.fromarray(hwc)
    if max(im.size) > max_side:
        im = im.resize((max_side, max_side), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _load_image_hwc(image) -> "object":
    """Đường dẫn / PIL / mảng uint8 → mảng HWC uint8 cỡ 256²."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im = im.convert("RGB")
            if im.size != (NATIVE, NATIVE):
                im = im.resize((NATIVE, NATIVE), Image.LANCZOS)
            return np.asarray(im)
    if isinstance(image, Image.Image):
        im = image.convert("RGB")
        if im.size != (NATIVE, NATIVE):
            im = im.resize((NATIVE, NATIVE), Image.LANCZOS)
        return np.asarray(im)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[:2] != (NATIVE, NATIVE):
        raise ValueError(f"ảnh phải {NATIVE}×{NATIVE}, nhận {arr.shape[:2]}")
    return arr.astype("uint8")


def run_steps(model, image, steps: int = 5, on_frame=None, seed: int = 0) -> dict:
    """Chạy đúng hai vòng lặp DDIM, gọi `on_frame(dict)` sau **mỗi** forward.

    `on_frame` nhận ``{"phase": "invert"|"reconstruct", "i", "t", "sec",
    "x0_png", "eps_rms", "map_mean"}`` — đủ để vẽ tiến trình thật, không phải
    thanh chạy giả.

    Trả ``{"steps", "nfe", "seconds", "map_png", "recon_png", "stats"}``.
    """
    torch, mods, err = _try_import()
    if mods is None:
        raise RuntimeError(err or "thiếu torch/dire_zero")
    diffusion, Schedule, _load_adm, _to_hwc = mods
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps phải trong [1, {MAX_STEPS}] — quá số đó không còn là demo")
    if getattr(model, "training", False):
        model.eval()

    arr = _load_image_hwc(image)
    x0_img = diffusion.from_uint8(arr)
    sched = Schedule(steps=steps, device="cpu")
    ts = sched.timesteps
    frames: list[dict] = []
    t_start = time.time()
    nfe = 0

    def frame(phase: str, i: int, t: int, xt, x0h, eps, sec: float) -> None:
        nonlocal nfe
        nfe += 1
        xt_png = _png_bytes(_to_uint8_hwc(xt))
        x0_png = _png_bytes(_to_uint8_hwc(x0h))
        d = {
            "phase": phase,
            "i": i,
            "t": int(t),
            "sec": round(sec, 2),
            "x_t_png": xt_png,
            "x0_hat_png": x0_png,
            "x0_png": x0_png,
            "eps_rms": round(float(eps.pow(2).mean().sqrt()), 5),
            "map_mean": round(float((x0_img - x0h).abs().mean()), 5),
        }
        frames.append(d)
        if on_frame is not None:
            on_frame(d)

    with torch.inference_mode():
        # ---- đảo ngược: tăng t ----
        x = x0_img
        for i, t in enumerate(ts):
            t0 = time.time()
            x0h, eps = _eval(sched, model, x, t)
            frame("invert", i, t, x, x0h, eps, time.time() - t0)
            t_next = ts[min(i + 1, len(ts) - 1)]
            a_next = sched.a(t_next, dtype=x.dtype)
            x = a_next.sqrt() * x0h + (1 - a_next).sqrt() * eps
        x_top = x

        # ---- tái tạo: giảm t, bước cuối lấy x̂₀ ----
        x = x_top
        recon = None
        for i in range(len(ts) - 1, -1, -1):
            t0 = time.time()
            x0h, eps = _eval(sched, model, x, ts[i])
            frame("reconstruct", i, ts[i], x, x0h, eps, time.time() - t0)
            if i == 0:
                recon = x0h
                break
            a_prev = sched.a(ts[i - 1], dtype=x.dtype)
            x = a_prev.sqrt() * x0h + (1 - a_prev).sqrt() * eps

    dmap = (x0_img - recon).abs()
    stats = {
        "mean": round(float(dmap.mean()), 5),
        "max": round(float(dmap.max()), 5),
        "p99": round(float(dmap.flatten().kthvalue(max(1, int(0.99 * dmap.numel()))).values), 5),
    }
    return {
        "steps": steps,
        "nfe": nfe,
        "seconds": round(time.time() - t_start, 2),
        "map_png": _map_png(dmap),
        "map_values": dict(stats),
        "recon_png": _png_bytes(_to_uint8_hwc(recon), max_side=256),
        "frames": frames,
        "timings": [f["sec"] for f in frames],
    }


def _map_png(dmap, max_side: int = 256) -> bytes:
    """Map `[0, 2]` → ảnh xám `[0, 255]` để **xem**. Dữ liệu thật là tensor, không phải PNG.

    Cùng quy ước với `tools/make_gallery.py` để map tính tại chỗ và map trong
    gallery trông giống nhau ở cùng một sai số — nếu hai bên chuẩn hoá khác nhau,
    người xem sẽ tưởng một trong hai sai.
    """
    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    # `** 0.5` = gamma 0.5, đúng tham số mặc định của `tools/make_gallery.py`:
    # map nằm phần lớn ở biên độ nhỏ, hiển thị tuyến tính thì gần như đen tuyền.
    v = ((dmap.squeeze(0).mean(dim=0).clamp(0, 2) / 2.0) ** 0.5).numpy()
    im = Image.fromarray((v * 255.0 + 0.5).astype(np.uint8), mode="L")
    if max_side and max(im.size) > max_side:
        im = im.resize((max_side, max_side), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _eval(sched, model, x, t: int):
    """Một lượt mạng — **đúng** công thức `dire_zero.diffusion._eval`.

    Viết lại thay vì gọi hàm private (nó không nằm trong `__all__`), và bản sao
    được khoá bằng test parity với `dire_map`.
    """
    out = model(x, _timestep_tensor(x, t))
    eps = out[:, :3] if out.shape[1] == 6 else out
    a = sched.a(t, dtype=x.dtype)
    x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1.0, 1.0)
    eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
    return x0, eps


def _timestep_tensor(x, t: int):
    import torch  # noqa: PLC0415

    return torch.full((x.shape[0],), int(t), dtype=torch.long, device=x.device)
