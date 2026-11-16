"""Chi phí — tính trước khi chạy, và **từ chối** khi vượt trần.

Vì sao module này tồn tại riêng
-------------------------------
Ba lần trong dự án, con số tiền bị sai vì một giả định không ai kiểm:

1. chạy 1 020 NFE thay vì 40 ⇒ **25×**;
2. ``sec_per_step = 0,35`` chưa đo, trong khi thực tế có thể 2,5 s ⇒ **7×**;
3. ước lượng theo số ảnh thay vì theo NFE ⇒ bỏ sót rằng mỗi ảnh cần *hai* lượt.

Nên ở đây mọi thứ quy về **một** công thức, và có hai cổng cứng: chưa đo
`sec_per_step` thì không chạy GPU, vượt `max_usd` thì không chạy GPU.

.. math::

    \\text{GPU-giờ} = \\frac{N \\cdot \\text{NFE/ảnh} \\cdot s_{\\text{step}}}{3600}

    \\text{USD} = \\text{GPU-giờ} \\times \\text{giá/giờ}

Giá mặc định là A10G của Modal (1,10 USD/giờ). Mọi con số in ra kèm **khoảng**:
`s` đo được là trung bình, nên thời gian thật dao động ±20 % vì nhiệt và
batch. Báo một con số duy nhất là giả vờ chính xác.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["PRICES", "Estimate", "gpu_hours", "usd", "load_sec_per_step", "save_sec_per_step", "GateError", "require_estimate"]

#: USD/GPU-giờ, giá niêm yết. A10G là mức rẻ nhất còn đủ 24 GB cho ADM batch 1.
PRICES = {
    "A10G": 1.10,
    "A100-40GB": 2.50,
    "A100-80GB": 3.00,
    "L4": 0.80,
    "T4": 0.60,
}

#: Hệ số dao động quanh ước lượng (thời gian thật / ước lượng).
SPREAD = (0.8, 1.6)


class GateError(RuntimeError):
    """Cổng chi phí đóng. Không phải cảnh báo — chương trình dừng."""


@dataclass(frozen=True)
class Estimate:
    n_images: int
    nfe_per_image: int
    sec_per_step: float
    gpu: str = "A10G"

    @property
    def nfe_total(self) -> int:
        return self.n_images * self.nfe_per_image

    @property
    def gpu_h(self) -> float:
        return gpu_hours(self.nfe_total, self.sec_per_step)

    @property
    def usd(self) -> float:
        return self.gpu_h * PRICES[self.gpu]

    def band(self) -> tuple[float, float]:
        return (self.usd * SPREAD[0], self.usd * SPREAD[1])

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            nfe_total=self.nfe_total,
            gpu_h=round(self.gpu_h, 4),
            usd=round(self.usd, 4),
            usd_band=[round(v, 4) for v in self.band()],
        )
        return d

    def line(self) -> str:
        lo, hi = self.band()
        return (
            f"{self.n_images} ảnh × {self.nfe_per_image} NFE = {self.nfe_total} NFE · "
            f"{self.gpu_h:.2f} GPU-h · {self.usd:.2f} USD (khoảng {lo:.2f}–{hi:.2f}) "
            f"@ {self.gpu} {PRICES[self.gpu]:.2f}/h"
        )


def gpu_hours(nfe_total: int, sec_per_step: float) -> float:
    if sec_per_step <= 0:
        raise ValueError("sec_per_step phải > 0")
    if nfe_total < 0:
        raise ValueError("nfe_total không âm")
    return nfe_total * sec_per_step / 3600.0


def usd(nfe_total: int, sec_per_step: float, gpu: str = "A10G") -> float:
    return gpu_hours(nfe_total, sec_per_step) * PRICES[gpu]


def save_sec_per_step(path: str | Path, sec: float, *, source: str, gpu: str, batch: int = 1) -> None:
    payload = {
        "sec_per_step": sec,
        "source": source,
        "gpu": gpu,
        "batch": batch,
        "note": "số ĐO tại chỗ, không phải giả định; mọi ước lượng tiền dùng số này",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_sec_per_step(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        raise GateError(
            f"chưa có {p} — chưa ai ĐO sec/step trên GPU này.\n"
            "Chạy `modal run -m dire_zero.modal_app::bench` trước. Mọi con số tiền "
            "trong tài liệu đang dựa vào 0,35 s/step chưa kiểm; đo sai 7 lần là "
            "chuyện đã xảy ra."
        )
    sec = float(json.loads(p.read_text(encoding="utf-8"))["sec_per_step"])
    if sec <= 0:
        raise GateError(f"{p} có sec_per_step = {sec}, không dùng được")
    return sec


def require_estimate(est: Estimate, max_usd: float | None) -> None:
    """Cổng cuối trước khi xin GPU: vượt trần thì dừng, in rõ vượt bao nhiêu."""
    if max_usd is None:
        return
    if est.usd > max_usd:
        raise GateError(
            f"ước tính {est.usd:.2f} USD > trần {max_usd:.2f} USD.\n  {est.line()}\n"
            "Giảm số ảnh, giảm số bước, hoặc nâng trần bằng --max-usd."
        )
