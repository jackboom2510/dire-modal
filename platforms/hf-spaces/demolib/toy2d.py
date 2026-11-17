"""Thí nghiệm 2 chiều: DIRE trên một mô hình khuếch tán có **score chính xác**.

Vì sao cần một thí nghiệm "đồ chơi"
-----------------------------------
Trên ảnh, không ai biết score thật của phân phối dữ liệu, nên mọi kết luận về
DIRE đều lẫn hai nguồn: (a) cơ chế hình học của phép đảo ngược, và (b) mô hình
đã học phân phối tốt đến đâu. Ở đây chỉ còn (a):

* dữ liệu là **hỗn hợp Gauss 2 chiều** ⇒ phân phối sau khi thêm nhiễu cũng là
  hỗn hợp Gauss, nên **score có công thức đóng**;
* "mô hình" là score đó, tức giả định **học hoàn hảo** `p_g`. Không huấn luyện,
  không sai số học, không ngẫu nhiên ngoài phép lấy mẫu;
* vì vậy mọi khác biệt giữa DIRE(ảnh sinh) và DIRE(ảnh thật) đo **đúng** phần
  hình học mà bài báo gọi là cơ chế.

Điều thí nghiệm này KHÔNG phải
------------------------------
Không phải ADM, không phải ảnh, không phải Table 2. Nó là một mô hình lý thuyết
để kiểm lập luận, và phải được dán nhãn như vậy ở mọi chỗ hiển thị.

Toán
----
Lịch và mốc lấy đúng như `dire_zero.schedule` (0-based, `ᾱ₀ = 1 − β₀`).

Với `p_g = Σ_k w_k 𝒩(μ_k, Σ_k)`, phân phối đã thêm nhiễu ở mức `ᾱ` là
`Σ_k w_k 𝒩(√ᾱ·μ_k, ᾱΣ_k + (1−ᾱ)I)`. Score của nó:

.. math::

    \\nabla \\log p_t(x) = \\frac{\\sum_k w_k \\mathcal{N}(x; \\mu_k(t), \\sigma_k^2(t))
                                 \\cdot (-(x-\\mu_k(t))/\\sigma_k^2(t))}
                                {\\sum_k w_k \\mathcal{N}(x; \\mu_k(t), \\sigma_k^2(t))}

và mạng dự đoán nhiễu tương ứng `ε̂ = −√(1−ᾱ)·∇log p_t(x)`.

Sau đó chạy đúng hai vòng lặp của repo (`invert` / `reconstruct`), kể cả phép
cắt `[−1, 1]` sau mỗi bước — chính phép cắt này là một trong hai ứng viên cơ
chế, nên nó phải bật/tắt được.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "T_TRAIN",
    "linear_betas",
    "alphas_cumprod",
    "timesteps",
    "Mixture",
    "P_G",
    "P_R",
    "dire",
    "run",
]

T_TRAIN = 1000
BETA_MIN = 1e-4
BETA_MAX = 2e-2


def linear_betas(t_train: int = T_TRAIN) -> list[float]:
    return [BETA_MIN + (t / (t_train - 1)) * (BETA_MAX - BETA_MIN) for t in range(t_train)]


def alphas_cumprod(betas: list[float]) -> list[float]:
    out, cur = [], 1.0
    for b in betas:
        cur *= 1.0 - b
        out.append(cur)
    return out


def timesteps(t_train: int = T_TRAIN, steps: int = 20) -> list[int]:
    """Trùng `dire_zero.schedule.timesteps`: tăng dần, `[0, 50, …, 950]` khi S=20."""
    if steps < 1:
        raise ValueError("steps phải ≥ 1")
    if steps > t_train:
        raise ValueError("steps không được vượt t_train")
    return [round(i * t_train / steps) for i in range(steps)]


A = alphas_cumprod(linear_betas())


class Mixture:
    """Hỗn hợp Gauss đẳng hướng 2 chiều, trọng số đều."""

    def __init__(self, comps: tuple[tuple[float, float, float], ...]) -> None:
        self.comps = comps

    def sample(self, rng: random.Random) -> tuple[float, float]:
        mx, my, sd = self.comps[rng.randrange(len(self.comps))]
        return (rng.gauss(mx, sd), rng.gauss(my, sd))

    def _density(self, x: tuple[float, float], a: float) -> list[tuple[float, float, float]]:
        """Trọng số chuẩn hoá + trung bình + nghịch đảo phương sai, tại mức `a`."""
        s, v = math.sqrt(a), 1.0 - a
        out = []
        for mx, my, sd in self.comps:
            var = a * sd * sd + v
            prec = 1.0 / var
            d0, d1 = x[0] - s * mx, x[1] - s * my
            w = math.exp(-0.5 * (d0 * d0 + d1 * d1) * prec) / (2.0 * math.pi * var)
            out.append((w, d0, d1, prec))
        return out


#: `p_g` — phân phối mà "mô hình" đã học (ba cụm, gọn, trong `[−1, 1]`).
P_G = Mixture(((-0.30, -0.15, 0.16), (0.32, -0.20, 0.14), (0.05, 0.34, 0.13)))

#: `p_r` — phân phối thật, **lệch khỏi** `p_g` nhưng cùng thang đo.
P_R = Mixture(((0.42, 0.30, 0.20), (-0.38, 0.33, 0.18), (0.10, -0.40, 0.17)))


def _eps(mix: Mixture, x: tuple[float, float], a: float) -> tuple[float, float]:
    """`ε̂ = −√(1−ᾱ)·∇log p_t(x)` — mạng dự đoán nhiễu của mô hình hoàn hảo."""
    rows = mix._density(x, a)
    den = sum(r[0] for r in rows)
    if den <= 0.0:  # pragma: no cover - chỉ khi mật độ dưới tràn số
        return (0.0, 0.0)
    g0 = sum(r[0] * (-r[1] * r[3]) for r in rows) / den
    g1 = sum(r[0] * (-r[2] * r[3]) for r in rows) / den
    v = math.sqrt(1.0 - a)
    return (-v * g0, -v * g1)


def _eval(
    mix: Mixture, x: tuple[float, float], t: int, clip: bool
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Một lượt "mạng": trả `(x̂₀, ε̂)` — cùng hợp đồng với `diffusion._eval`."""
    a = A[t]
    eps = _eps(mix, x, a)
    sa, sv = math.sqrt(a), math.sqrt(1.0 - a)
    x0 = ((x[0] - sv * eps[0]) / sa, (x[1] - sv * eps[1]) / sa)
    if clip:
        x0 = (max(-1.0, min(1.0, x0[0])), max(-1.0, min(1.0, x0[1])))
    if sv > 1e-12:
        eps = ((x[0] - sa * x0[0]) / sv, (x[1] - sa * x0[1]) / sv)
    return x0, eps


def invert(mix: Mixture, x0_img, ts: list[int], clip: bool = True):
    x = x0_img
    for i, t in enumerate(ts):
        x0h, eps = _eval(mix, x, t, clip)
        tn = ts[min(i + 1, len(ts) - 1)]
        an = A[tn]
        x = (math.sqrt(an) * x0h[0] + math.sqrt(1 - an) * eps[0],
             math.sqrt(an) * x0h[1] + math.sqrt(1 - an) * eps[1])
    return x


def reconstruct(mix: Mixture, x_top, ts: list[int], clip: bool = True):
    x = x_top
    for i in range(len(ts) - 1, -1, -1):
        x0h, eps = _eval(mix, x, ts[i], clip)
        if i == 0:
            return x0h
        ap = A[ts[i - 1]]
        x = (math.sqrt(ap) * x0h[0] + math.sqrt(1 - ap) * eps[0],
             math.sqrt(ap) * x0h[1] + math.sqrt(1 - ap) * eps[1])
    return x  # pragma: no cover


def dire(
    mix: Mixture,
    x0_img: tuple[float, float],
    sched: list[int] | None = None,
    clip: bool = True,
) -> tuple[float, tuple[float, float]]:
    """`(|x₀ − x̂₀|, x̂₀)` cho một điểm — bản 2 chiều của `diffusion.dire_map`."""
    ts = sched if sched is not None else timesteps(T_TRAIN, 20)
    recon = reconstruct(mix, invert(mix, x0_img, ts, clip), ts, clip)
    return math.hypot(x0_img[0] - recon[0], x0_img[1] - recon[1]), recon


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:  # pragma: no cover
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def _hist(vals: list[float], edges: int, hi: float) -> list[int]:
    out = [0] * edges
    if hi <= 0:
        return out
    for v in vals:
        k = min(edges - 1, int(v / hi * edges))
        out[k] += 1
    return out


def run(
    steps: int = 20,
    n: int = 200,
    clip: bool = True,
    seed: int = 3407,
    bins: int = 18,
) -> dict:
    """Một lượt đo: DIRE của `n` mẫu từ `p_g` và `n` mẫu từ `p_r`."""
    steps = max(1, min(int(steps), T_TRAIN))
    n = max(10, min(int(n), 400))
    ts = timesteps(T_TRAIN, steps)
    rng = random.Random(seed)

    gen, real, pairs = [], [], []
    # **Một** mô hình duy nhất (đã học p_g) chấm cả hai lớp — đúng như DIRE:
    # bộ tái tạo không biết ảnh nào là ảnh sinh. Dùng p_r làm mô hình cho lớp
    # thật là một lỗi dễ mắc và nó làm kết quả đẹp giả tạo (mỗi lớp được chấm
    # bằng đúng mô hình đã học lớp đó).
    for k in range(n):
        xg = P_G.sample(rng)
        dg, rg = dire(P_G, xg, ts, clip)
        gen.append(dg)
        xr = P_R.sample(rng)
        dr, rr = dire(P_G, xr, ts, clip)
        real.append(dr)
        if k < 40:
            pairs.append({"x": [xg[0], xg[1]], "r": [rg[0], rg[1]], "class": "gen"})
            pairs.append({"x": [xr[0], xr[1]], "r": [rr[0], rr[1]], "class": "real"})

    hi = max(max(gen), max(real)) * 1.02 or 1.0
    sg, sr = sorted(gen), sorted(real)
    mg = sum(gen) / len(gen)
    mr = sum(real) / len(real)
    return {
        "steps": steps,
        "n": n,
        "clip": clip,
        "seed": seed,
        "nfe": 2 * steps,
        "t_marks": ts,
        "alpha_at_marks": [round(A[t], 6) for t in ts],
        "gen": {
            "mean": round(mg, 5),
            "median": round(_quantile(sg, 0.5), 5),
            "p90": round(_quantile(sg, 0.9), 5),
            "max": round(sg[-1], 5),
        },
        "real": {
            "mean": round(mr, 5),
            "median": round(_quantile(sr, 0.5), 5),
            "p90": round(_quantile(sr, 0.9), 5),
            "max": round(sr[-1], 5),
        },
        "ratio": round(mr / mg, 3) if mg > 0 else None,
        "hist": {"edges": [round(i * hi / bins, 5) for i in range(bins + 1)],
                 "gen": _hist(gen, bins, hi), "real": _hist(real, bins, hi)},
        "pairs": pairs,
        "note": (
            "Mô hình 2 chiều có score chính xác (hỗn hợp Gauss). "
            "Đây là minh hoạ cơ chế, KHÔNG phải ADM và không phải Table 2."
        ),
    }
