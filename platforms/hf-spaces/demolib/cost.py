"""Chi phí GPU — **chép** công thức của `dire_zero.budget`, khoá bằng test.

Vì sao chép thay vì import
--------------------------
Space chỉ chứa thư mục `platforms/hf-spaces/`; `dire_zero` nằm ngoài. Bê cả package
lên Space nghĩa là bê theo torch (để `import dire_zero` không chết), tức là đánh
đổi vài giây cold start lấy một dòng import. Không đáng.

Cái đáng là **không để hai nơi hai số**. Nên `tests/test_hf_spaces_demo.py` so
từng trường của `estimate()` với `budget.Estimate` trên một lưới tham số, và so cả
chuỗi `estimate_line()` với `Estimate.line()`. Đổi một bên mà không đổi bên kia là
test đỏ, không phải một buổi họp.

.. math::

    \\text{NFE} = N \\cdot 2S, \\qquad
    \\text{GPU-h} = \\frac{\\text{NFE} \\cdot s_{\\text{step}}}{3600}, \\qquad
    \\text{USD} = \\text{GPU-h} \\cdot \\text{giá}
"""

from __future__ import annotations

__all__ = ["PRICES", "SPREAD", "SEC_PER_STEP_MEASURED", "estimate", "estimate_line"]

#: USD/GPU-giờ, giá niêm yết — chép từ `dire_zero.budget.PRICES`.
PRICES = {"A10G": 1.10, "A100-40GB": 2.50, "A100-80GB": 3.00, "L4": 0.80, "T4": 0.60}

#: Hệ số dao động, chép từ `dire_zero.budget.SPREAD`.
SPREAD = (0.8, 1.6)

#: `sec/step` của ADM 256² trên A10G (fp16, batch 1), do `::bench` ghi vào
#: `bench.json`. Là **số đo**, không phải hằng số khoa học — mọi ô nhập cho sửa.
#:
#: Kiểm chéo bằng FLOP, độc lập với lần đo đó: một lượt forward của ADM 256² là
#: **829 GFLOP** (415 GMAC; 552,8 M tham số). 829 GFLOP / 0,0917 s = **9,0 TFLOPS**,
#: tức **29 %** đỉnh fp16 của A10G (31,2 TFLOPS) — đúng mức một U-Net batch 1
#: thường đạt. Nếu ai đó đo ra 0,3 s/step (2,8 TFLOPS, 9 %) thì mới đáng nghi.
SEC_PER_STEP_MEASURED = 0.0917


def estimate(n_images: int, steps: int, sec_per_step: float, gpu: str = "A10G") -> dict:
    if n_images < 0:
        raise ValueError("n_images phải ≥ 0")
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    if sec_per_step <= 0:
        raise ValueError("sec_per_step phải > 0")
    if gpu not in PRICES:
        raise ValueError(f"GPU lạ: {gpu} — có {sorted(PRICES)}")
    nfe_per_image = 2 * steps
    nfe_total = n_images * nfe_per_image
    gpu_h = nfe_total * sec_per_step / 3600.0
    usd = gpu_h * PRICES[gpu]
    return {
        "n_images": n_images,
        "steps": steps,
        "nfe_per_image": nfe_per_image,
        "nfe_total": nfe_total,
        "sec_per_step": sec_per_step,
        "gpu": gpu,
        "gpu_h": gpu_h,
        "usd": usd,
        "usd_band": (usd * SPREAD[0], usd * SPREAD[1]),
    }


def estimate_line(e: dict) -> str:
    """Cùng chuỗi với `budget.Estimate.line()` — test so nguyên văn."""
    lo, hi = e["usd_band"]
    return (
        f"{e['n_images']} ảnh × {e['nfe_per_image']} NFE = {e['nfe_total']} NFE · "
        f"{e['gpu_h']:.2f} GPU-h · {e['usd']:.2f} USD (khoảng {lo:.2f}–{hi:.2f}) "
        f"@ {e['gpu']} {PRICES[e['gpu']]:.2f}/h"
    )
