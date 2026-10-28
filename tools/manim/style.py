# tools/manim/style.py
"""Bảng màu, font, và các "khối" dùng lại — một chỗ duy nhất cho hình thức.

Vì sao cần: 5 scene dùng chung một ngôn ngữ thị giác. Nếu mỗi scene tự đặt
`font_size=28` và tự chọn màu thì video trông lắp ghép, và sửa một chỗ phải sửa
năm nơi. Ở đây chỉ có ba thứ: **màu**, **font**, và các hàm dựng khối.

Font tiếng Việt
---------------
``DejaVu Sans`` phủ đủ dấu tiếng Việt và có sẵn trên Linux/Windows. Đổi bằng
``DIRE_FONT`` nếu máy thiếu. Docstring/dòng chữ trong scene đều là tiếng Việt nên
đây không phải chi tiết phụ: font thiếu glyph sẽ ra ô vuông, mà lỗi đó chỉ thấy
sau khi render.

API manim dùng ở đây được giữ trong một danh sách ngắn (xem
``tests/test_manim_scenes.py``) để (a) test chạy được khi KHÔNG có manim, và
(b) nâng manim lên bản mới thì biết ngay chỗ nào cần xem lại.
"""

from __future__ import annotations

import os

from manim import (  # noqa: F401 — các tên này được scenes.py dùng lại
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    Rectangle,
    RoundedRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
FONT = os.environ.get("DIRE_FONT", "DejaVu Sans")
MONO = os.environ.get("DIRE_FONT_MONO", "DejaVu Sans Mono")

# ---------------------------------------------------------------------------
# Màu — tên theo VAI TRÒ, không theo sắc thái ("REAL" chứ không "BLUE_2")
# ---------------------------------------------------------------------------
C = {
    "bg": "#0E1116",
    "ink": "#E8EAED",
    "dim": "#9AA0A6",
    "real": "#4C9AFF",      # ảnh thật
    "fake": "#FF6B6B",      # ảnh do diffusion sinh
    "map": "#FFC857",       # DIRE map / residual
    "net": "#8B7CF6",       # ResNet
    "ok": "#4ADE80",        # số đo của mình / đã kiểm
    "warn": "#FBBF24",      # nhãn "chưa đo", "demo"
    "bad": "#F87171",
    "line": "#3C4043",
}

TITLE_SIZE = 40
BODY_SIZE = 26
SMALL_SIZE = 20
MONO_SIZE = 20


# ---------------------------------------------------------------------------
# Khối chữ
# ---------------------------------------------------------------------------
def title(text: str) -> Text:
    return Text(text, font=FONT, font_size=TITLE_SIZE, color=C["ink"], weight="BOLD")


def body(text: str, size: int = BODY_SIZE, color: str | None = None) -> Text:
    return Text(text, font=FONT, font_size=size, color=color or C["ink"])


def mono(text: str, size: int = MONO_SIZE, color: str | None = None) -> Text:
    return Text(text, font=MONO, font_size=size, color=color or C["dim"])


def badge(text: str, color: str | None = None) -> VGroup:
    """Nhãn nguồn/nhãn demo — LUÔN nằm trên khung hình, không phải phụ lục.

    Vì sao là một khối riêng: mọi scene có số đều phải nói số đó từ đâu ra. Một
    câu rời rạc dễ bị bỏ khi sửa scene; một hàm thì không.
    """
    t = Text(text, font=FONT, font_size=SMALL_SIZE, color=C["bg"])
    pad = Rectangle(width=t.width + 0.36, height=t.height + 0.24,
                    color=color or C["warn"], fill_color=color or C["warn"],
                    fill_opacity=1.0, stroke_width=0)
    return VGroup(pad, t.move_to(pad.get_center()))


def source_note(text: str) -> Text:
    """Dòng nguồn cỡ nhỏ, góc dưới — 'số của paper', 'đo 2026-10-25'…"""
    return Text(text, font=FONT, font_size=SMALL_SIZE - 4, color=C["dim"])


# ---------------------------------------------------------------------------
# Khối hình
# ---------------------------------------------------------------------------
def box(text: str, color: str, width: float = 2.6, height: float = 0.95,
        size: int = BODY_SIZE) -> VGroup:
    """Hộp có nhãn — dùng cho pipeline (ảnh → map → ResNet → logit)."""
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12,
                            color=color, stroke_width=2,
                            fill_color=color, fill_opacity=0.10)
    label = Text(text, font=FONT, font_size=size, color=C["ink"])
    label.move_to(rect.get_center())
    return VGroup(rect, label)


def bar(label: str, value: float, max_value: float, color: str,
        width: float = 4.2, height: float = 0.30) -> VGroup:
    """Một thanh ngang tỉ lệ + nhãn trái + số phải.

    Tự vẽ thay vì ``BarChart`` của manim: ở đây cần đúng một kiểu thanh và cần
    nhãn tiếng Việt canh trái; dùng BarChart thì vẫn phải viết lại phần nhãn.
    """
    w = max(width * (float(value) / float(max_value)), 0.02)
    fill = Rectangle(width=w, height=height, color=color,
                     fill_color=color, fill_opacity=0.85, stroke_width=0)
    track = Rectangle(width=width, height=height, color=C["line"],
                      stroke_width=1, fill_opacity=0.0)
    fill.align_to(track, LEFT)
    name = Text(label, font=FONT, font_size=SMALL_SIZE, color=C["ink"])
    num = Text(f"{value:.1f}", font=MONO, font_size=SMALL_SIZE, color=C["ink"])
    name.next_to(track, LEFT, buff=0.16)
    num.next_to(track, RIGHT, buff=0.16)
    return VGroup(name, track, fill, num)


def chip_row(items: list[str], color: str = "dim") -> VGroup:
    """Dãy nhãn nhỏ nằm ngang (dùng cho 'real | fake', 'gzip -t | sha256')."""
    parts = VGroup(*[Text(s, font=MONO, font_size=SMALL_SIZE - 2,
                          color=C.get(color, C["dim"])) for s in items])
    parts.arrange(RIGHT, buff=0.28)
    return parts


def curve_from_points(points: list[tuple[float, float]], origin, size: tuple[float, float],
                      color: str) -> VGroup:
    """Polyline qua ``points`` đã chuẩn hoá về ``[0,1]²``.

    Vẽ bằng các ``Line`` nối tiếp thay vì ``Axes.plot``: kiểm soát được khung,
    không phụ thuộc API ``Axes``/``plot`` (thứ đổi giữa các bản manim), và nối
    được cả khi chỉ có 3–5 điểm như một lần train demo.
    """
    sx, sy = size
    ox, oy = origin[0], origin[1]
    segs = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        a = (ox + x0 * sx, oy + y0 * sy)
        b = (ox + x1 * sx, oy + y1 * sy)
        segs.append(Line([a[0], a[1], 0], [b[0], b[1], 0], color=color,
                         stroke_width=3))
    if not segs:      # một điểm duy nhất ⇒ vẽ chấm thay vì polyline rỗng
        x, y = points[0]
        segs.append(Dot(point=[ox + x * sx, oy + y * sy, 0], radius=0.05, color=color))
    return VGroup(*segs)


def frame(width: float = 12.2, height: float = 6.6) -> Rectangle:
    """Khung mờ để canh bố cục (không bắt buộc dùng)."""
    return Rectangle(width=width, height=height, color=C["line"], stroke_width=1,
                     fill_opacity=0.0)


__all__ = [
    "C", "FONT", "MONO", "TITLE_SIZE", "BODY_SIZE", "SMALL_SIZE",
    "title", "body", "mono", "badge", "source_note", "box", "bar", "chip_row",
    "curve_from_points", "frame",
    # manim re-export
    "AnimationGroup", "Arrow", "Circle", "Create", "Dot", "FadeIn", "FadeOut",
    "Indicate", "Line", "Rectangle", "RoundedRectangle", "Text", "Transform",
    "VGroup", "Write", "UP", "DOWN", "LEFT", "RIGHT", "ORIGIN",
]
