"""Hình vẽ bằng SVG thuần — không matplotlib, không JavaScript.

Vì sao không dùng matplotlib
----------------------------
Space này chạy trên CPU free: mỗi phụ thuộc thêm là một tầng có thể hỏng lúc
build, và matplotlib kéo theo ~50 MB chỉ để vẽ ba biểu đồ. SVG do Python sinh ra
thì hiển thị được bằng một thẻ ``<img>`` với ảnh dữ liệu nhúng (data-URI) hoặc
bằng ``st.markdown(..., unsafe_allow_html=True)``, không cần gói nào.

Ba hình, đúng ba câu hỏi của buổi thuyết trình:

* `histogram` — DIRE của ảnh sinh và ảnh thật có tách được không;
* `scatter` — điểm đi đâu khi đảo ngược rồi tái tạo (đây là "vì sao" của DIRE);
* `sweep` — tăng `S` thì tách tốt hơn bao nhiêu, và tới đâu thì bão hoà.

Mọi hàm trả **chuỗi SVG**, không vẽ ra màn hình — nên test được mà không cần
trình duyệt.
"""

from __future__ import annotations

__all__ = ["histogram", "scatter", "sweep", "COLOR_GEN", "COLOR_REAL"]

COLOR_GEN = "#c2410c"   # ảnh do generator sinh (nhãn 1)
COLOR_REAL = "#1d4ed8"  # ảnh thật (nhãn 0)
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def _txt(x: float, y: float, s: str, size: int = 11, anchor: str = "start",
         color: str = MUTED, weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif">{s}</text>')


def _frame(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {w} {h}" '
            f'role="img" style="max-width:{w}px;background:#fff;border-radius:8px">'
            f'{body}</svg>')


def _legend_right(w: int, y: float, gen_label: str, real_label: str) -> str:
    return (
        f'<rect x="{w - 220}" y="{y - 9}" width="11" height="11" fill="{COLOR_GEN}" rx="2"/>'
        + _txt(w - 203, y, gen_label, 11)
        + f'<rect x="{w - 110}" y="{y - 9}" width="11" height="11" fill="{COLOR_REAL}" rx="2"/>'
        + _txt(w - 93, y, real_label, 11)
    )


def histogram(edges: list[float], gen: list[int], real: list[int],
              width: int = 720, height: int = 300) -> str:
    """Cột ghép đôi: số mẫu theo khoảng DIRE, hai lớp cạnh nhau."""
    n = len(gen)
    if n == 0 or len(real) != n or len(edges) != n + 1:
        raise ValueError("cần len(edges) == len(counts) + 1 và hai lớp cùng số khoảng")
    ml, mr, mt, mb = 52, 16, 26, 44
    w, h = width - ml - mr, height - mt - mb
    top = max(max(gen), max(real), 1)
    out = []
    for frac in (0.0, 0.5, 1.0):
        y = mt + h * (1 - frac)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + w}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(_txt(ml - 6, y + 4, f"{int(round(top * frac))}", anchor="end"))
    bw = w / n
    inner = bw * 0.72
    for i in range(n):
        x0 = ml + i * bw
        for j, (counts, color) in enumerate(((gen, COLOR_GEN), (real, COLOR_REAL))):
            hh = counts[i] / top * h
            out.append(
                f'<rect x="{x0 + bw * 0.14 + j * inner / 2:.2f}" y="{mt + h - hh:.2f}" '
                f'width="{inner / 2:.2f}" height="{hh:.2f}" fill="{color}" opacity="0.85" rx="1"/>'
            )
    step = max(1, n // 7)
    for i in range(0, n, step):
        out.append(_txt(ml + (i + 0.5) * bw, mt + h + 16, f"{edges[i]:.3f}", 10, "middle"))
    out.append(_txt(ml + w, mt + h + 34, f"DIRE (tới {edges[-1]:.3f})", 10, "end"))
    out.append(_txt(ml - 40, mt + 12, "số mẫu", 10))
    out.append(_legend_right(width, 14, "sinh", "thật"))
    return _frame(width, height, "".join(out))


def scatter(pairs: list[dict], width: int = 380, height: int = 380) -> str:
    """Mũi tên `x₀ → x̂₀` trong mặt phẳng `[−1, 1]²`; chấm là `x̂₀`."""
    m, top = 26, 26
    size = min(width, height) - 2 * m

    def px(v: float) -> float:
        return m + (max(-1.0, min(1.0, v)) + 1.0) / 2.0 * size

    out = [f'<rect x="{m}" y="{top}" width="{size}" height="{size}" fill="#fafafa" '
           f'stroke="{GRID}"/>',
           f'<line x1="{m + size / 2}" y1="{top}" x2="{m + size / 2}" y2="{top + size}" '
           f'stroke="{GRID}"/>',
           f'<line x1="{m}" y1="{top + size / 2}" x2="{m + size}" y2="{top + size / 2}" '
           f'stroke="{GRID}"/>']
    for p in pairs:
        color = COLOR_GEN if p.get("class") == "gen" else COLOR_REAL
        x0, y0 = px(p["x"][0]), px(p["x"][1])
        x1, y1 = px(p["r"][0]), px(p["r"][1])
        out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                   f'stroke="{color}" stroke-width="1" opacity="0.45"/>')
        out.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2.6" fill="{color}" opacity="0.9"/>')
        out.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="1.6" fill="none" stroke="{color}" '
                   f'stroke-width="0.8" opacity="0.55"/>')
    out.append(_txt(m, top - 8, "x₀ (vòng tròn) → x̂₀ (chấm), mặt phẳng [−1,1]²", 10))
    out.append(_legend_right(width, top + size + 18, "sinh", "thật"))
    return _frame(width, height, "".join(out))


def sweep(points: list[dict], width: int = 720, height: int = 260) -> str:
    """Tỉ lệ `DIRE(ảnh thật) / DIRE(ảnh sinh)` theo `S` — đo độ tách lớp."""
    if not points:
        raise ValueError("cần ít nhất một điểm")
    ml, mr, mt, mb = 56, 18, 22, 40
    w, h = width - ml - mr, height - mt - mb
    xs = [p["steps"] for p in points]
    ys = [p["ratio"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 1.0, max(1.1, max(ys) * 1.1)

    def px(v):
        return ml + ((v - x0) / (x1 - x0) if x1 > x0 else 0.5) * w

    def py(v):
        return mt + h - (v - y0) / (y1 - y0) * h

    out = [f'<line x1="{ml}" y1="{py(1.0):.1f}" x2="{ml + w}" y2="{py(1.0):.1f}" '
           f'stroke="{MUTED}" stroke-dasharray="4 3"/>',
           _txt(ml - 6, py(1.0) + 4, "1,0", 10, "end")]
    for frac in (0.0, 0.5, 1.0):
        v = y0 + (y1 - y0) * frac
        out.append(f'<line x1="{ml}" y1="{py(v):.1f}" x2="{ml + w}" y2="{py(v):.1f}" stroke="{GRID}"/>')
        out.append(_txt(ml - 6, py(v) + 4, f"{v:.1f}", 10, "end"))
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(p['steps']):.1f},{py(p['ratio']):.1f}"
                    for i, p in enumerate(points))
    out.append(f'<path d="{path}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    for p in points:
        out.append(f'<circle cx="{px(p["steps"]):.1f}" cy="{py(p["ratio"]):.1f}" r="3.2" '
                   f'fill="{INK}"/>')
        out.append(_txt(px(p["steps"]), py(p["ratio"]) - 8, f"{p['ratio']:.2f}", 10, "middle", INK))
    for p in points:
        out.append(_txt(px(p["steps"]), mt + h + 16, f"S={p['steps']}", 10, "middle"))
    out.append(_txt(ml, mt - 6, "tỉ lệ DIRE thật / DIRE sinh", 10))
    return _frame(width, height, "".join(out))
