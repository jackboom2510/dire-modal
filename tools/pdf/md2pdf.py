#!/usr/bin/env python3
"""Dựng PDF từ Markdown + LaTeX (công thức) — không cần LaTeX, không cần trình duyệt.

Vì sao có script này: sandbox (và máy người dùng) không có `pandoc`/`xelatex`,
cũng không tải được Chromium để in HTML. Bộ dựng này chỉ dùng **thư viện Python
thuần**: `reportlab` (dàn trang) + `matplotlib.mathtext` (kết xuất công thức ra
ảnh vector-quality) + `pygments` (tô sáng mã) + `fontTools` (kiểm phủ glyph).

Cài đặt::

    python3 -m venv /tmp/pdfenv
    /tmp/pdfenv/bin/pip install reportlab matplotlib pygments fonttools

Chạy::

    /tmp/pdfenv/bin/python tools/pdf/md2pdf.py \\
        --src docs/book --out docs/dire-tu-so-0.pdf

Phông chữ:
    * Thân bài  — DejaVu Serif (đi kèm matplotlib, phủ đủ tiếng Việt + ký hiệu toán).
    * Tiêu đề   — DejaVu Sans.
    * Mã nguồn  — JetBrains Mono (giấy phép OFL). Tự tải từ codeload.github.com
      vào thư mục cache nếu chưa có; **không** commit file phông vào repo.
    * Công thức  — STIX (đi kèm matplotlib), chất lượng như LaTeX Computer Modern.

Cú pháp Markdown được hỗ trợ (tập con có chủ đích — đủ cho tài liệu kỹ thuật):

    # Tiêu đề phần        ## Mục        ### Mục con
    đoạn văn, **đậm**, *nghiêng*, `mã nội dòng`, [chữ](https://url)
    [1] trích dẫn  →  [[1]]  (hiện thành chỉ số trên, bấm được nếu có link)
    - gạch đầu dòng     1. danh sách số
    > trích dẫn khối
    ```python ... ```      khối mã có tô sáng + số dòng tuỳ chọn (```python linenums)
    $$[12] \\frac{a}{b} $$  công thức hiển thị, [12] là số hiệu (tuỳ chọn)
    | a | b |              bảng (dòng thứ hai là |---|---|)
    ::: note Tiêu đề ... :::   hộp ghi chú (kind: note, warn, key)
    <!-- pagebreak -->
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Phông chữ
# --------------------------------------------------------------------------- #

JB_URL = (
    "https://codeload.github.com/JetBrains/JetBrainsMono/tar.gz/refs/heads/master"
)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dire-pdf"


def _mpl_font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def ensure_mono_font() -> tuple[Path, Path]:
    """Trả về (regular, bold) của JetBrains Mono, tải nếu thiếu."""
    CACHE.mkdir(parents=True, exist_ok=True)
    reg = CACHE / "JetBrainsMono-Regular.ttf"
    bold = CACHE / "JetBrainsMono-Bold.ttf"
    if reg.exists() and bold.exists():
        return reg, bold
    if not shutil_which("curl"):
        sys.exit("[!] Thiếu phông JetBrains Mono và không có curl để tải.")
    print("[..] tải JetBrains Mono (OFL) từ codeload.github.com ...", flush=True)
    tar = CACHE / "jbm.tar.gz"
    subprocess.run(
        ["curl", "-sSL", "-o", str(tar), JB_URL, "-w", "%{http_code}"],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    with __import__("tarfile").open(tar) as tf:
        for member in tf.getmembers():
            if member.name.endswith("fonts/ttf/JetBrainsMono-Regular.ttf"):
                member.name = "JetBrainsMono-Regular.ttf"
                tf.extract(member, CACHE)
            elif member.name.endswith("fonts/ttf/JetBrainsMono-Bold.ttf"):
                member.name = "JetBrainsMono-Bold.ttf"
                tf.extract(member, CACHE)
    tar.unlink(missing_ok=True)
    if not reg.exists():
        sys.exit("[!] Không giải nén được JetBrains Mono.")
    return reg, bold


def shutil_which(name: str):
    from shutil import which

    return which(name)


# --------------------------------------------------------------------------- #
# Màu sắc và kích thước
# --------------------------------------------------------------------------- #

INK = "#1A1A1A"
MUTED = "#5B6673"
RULE = "#D5DCE4"
ACCENT = "#0E5A6B"      # xanh mòng két đậm — tiêu đề
ACCENT_2 = "#12657A"
CODE_BG = "#F6F8FA"
CODE_EDGE = "#D9E1E8"
LINK = "#1F5FA8"
HEAD_BG = "#12414F"
STRIPE = "#F4F7F9"

PAGE_W, PAGE_H = A4 = (595.276, 841.89) if False else (595.276, 841.89)
M_LEFT = M_RIGHT = 54          # ~19 mm
M_TOP, M_BOTTOM = 62, 58

BODY_SIZE = 10.3
BODY_LEAD = 15.9
MONO_SIZE = 8.1


# --------------------------------------------------------------------------- #
# Parser Markdown → danh sách block
# --------------------------------------------------------------------------- #

BLOCK_H1 = "h1"
BLOCK_H2 = "h2"
BLOCK_H3 = "h3"
BLOCK_P = "p"
BLOCK_CODE = "code"
BLOCK_MATH = "math"
BLOCK_TABLE = "table"
BLOCK_UL = "ul"
BLOCK_OL = "ol"
BLOCK_QUOTE = "quote"
BLOCK_CALLOUT = "callout"
BLOCK_HR = "hr"
BLOCK_PAGEBREAK = "pagebreak"


def parse(src_text: str) -> list[tuple]:
    lines = src_text.replace("\r\n", "\n").split("\n")
    blocks: list[tuple] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        if not line.strip():
            i += 1
            continue

        # khối mã
        m = re.match(r"^```(\w*)(.*)$", line.strip())
        if m:
            lang = m.group(1) or "text"
            opts = m.group(2)
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # bỏ ``` đóng
            blocks.append((BLOCK_CODE, lang, "\n".join(buf), opts))
            continue

        # công thức hiển thị
        if line.strip().startswith("$$"):
            body = line.strip()[2:]
            num = None
            mn = re.match(r"^\[(\d+)\]\s*", body)
            if mn:
                num = mn.group(1)
                body = body[mn.end():]
            if body.endswith("$$"):
                body = body[:-2]
                i += 1
            else:
                i += 1
                parts = [body] if body.strip() else []
                while i < n and not lines[i].strip().endswith("$$"):
                    parts.append(lines[i])
                    i += 1
                if i < n:
                    parts.append(lines[i].strip()[:-2])
                    i += 1
                body = " ".join(p.strip() for p in parts)
            blocks.append((BLOCK_MATH, body.strip(), num))
            continue

        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            inner = line.strip()[4:-3].strip().lower()
            if inner == "pagebreak":
                blocks.append((BLOCK_PAGEBREAK,))
            i += 1
            continue

        if re.match(r"^ {0,3}(---+|\*\*\*+|___+)\s*$", line):
            blocks.append((BLOCK_HR,))
            i += 1
            continue

        # hộp ghi chú
        if line.strip().startswith(":::"):
            head = line.strip()[3:].strip()
            kind, _, title = head.partition(" ")
            kind = kind.lower() or "note"
            i += 1
            buf = []
            while i < n and lines[i].strip() != ":::":
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append((BLOCK_CALLOUT, kind, title.strip(), "\n".join(buf).strip()))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            blocks.append(
                (BLOCK_H1, text) if level == 1 else
                (BLOCK_H2, text) if level == 2 else
                (BLOCK_H3, text)
            )
            i += 1
            continue

        # bảng
        if line.strip().startswith("|") and i + 1 < n and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            rows = []
            header = _split_row(line)
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append((BLOCK_TABLE, header, rows))
            continue

        # trích dẫn khối
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].lstrip())
                i += 1
            blocks.append((BLOCK_QUOTE, " ".join(buf)))
            continue

        # danh sách
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            items = []  # list[[indent, text]]
            while i < n:
                cur = lines[i]
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", cur)
                if mm:
                    if bool(re.match(r"\d+\.$", mm.group(2))) != ordered:
                        break
                    items.append([len(mm.group(1)) // 2, mm.group(3).strip()])
                    i += 1
                elif items and cur.strip() and cur.startswith("  "):
                    items[-1][1] += " " + cur.strip()
                    i += 1
                elif items and not cur.strip():
                    # dòng trống bên trong danh sách: dừng nếu dòng sau không phải item
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[j]):
                        i = j
                    else:
                        break
                else:
                    break
            blocks.append((BLOCK_OL if ordered else BLOCK_UL, items))
            continue

        # đoạn văn
        buf = []
        while i < n:
            cur = lines[i]
            if (
                not cur.strip()
                or re.match(r"^(#{1,4})\s+", cur)
                or cur.strip().startswith(("```", "$$", ":::", ">", "|"))
                or re.match(r"^ {0,3}(---+|\*\*\*+|___+)\s*$", cur)
                or re.match(r"^\s*([-*]|\d+\.)\s+", cur)
            ):
                break
            buf.append(cur.strip())
            i += 1
        blocks.append((BLOCK_P, " ".join(buf)))
    return blocks


def keep_spaces(body: str) -> str:
    """Đổi khoảng trắng đầu dòng (và cụm ≥2 dấu cách) thành ``&#160;``.

    ``Paragraph`` gộp mọi khoảng trắng liên tiếp, nên không làm vậy thì khối mã
    mất hết thụt lề — đúng lỗi đã gặp. Chỉ giữ nguyên **khoảng trắng**, không
    đụng tới thẻ ``<font>`` của pygments.
    """
    if not body:
        return body
    lead = len(body) - len(body.lstrip(" "))
    out = "&#160;" * lead + body[lead:]
    return re.sub(r" {2,}", lambda m: "&#160;" * len(m.group(0)), out)


def _split_row(line: str) -> list[str]:
    """Tách ô của một hàng bảng; tôn trọng ``\\|`` (dấu gạch đứng trong ô)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, buf, i = [], [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            buf.append("|")
            i += 2
            continue
        if s[i] == "|":
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(s[i])
        i += 1
    cells.append("".join(buf).strip())
    return cells


# --------------------------------------------------------------------------- #
# Inline markdown → markup reportlab
# --------------------------------------------------------------------------- #

def fix_marks(t: str) -> str:
    """Thay ký tự dựng sẵn không có glyph bằng dạng tổ hợp.

    ``ᾱ`` (U+1FB1) không có trong DejaVu Serif/Sans và JetBrains Mono ⇒ ra ô
    "tofu". ``α`` + U+0304 (combining macron) thì cả ba phông đều có.
    """
    return t.replace("\u1fb1", "\u03b1\u0304")


MONO_MISSING = set("\u1eab\u1ea3\u1ebf\u1ec1\u1ec3\u1ec5\u1edf")

_MATH_CACHE: dict = {}


def math_renderer(fontsize: float = 9.8):
    """Một MathRenderer cho mỗi cỡ chữ (nội dòng ~9,8 pt, hiển thị ~12,4 pt)."""
    key = round(fontsize, 1)
    if key not in _MATH_CACHE:
        _MATH_CACHE[key] = MathRenderer(CACHE / "math", fontsize=fontsize, dpi=600)
    return _MATH_CACHE[key]


def inline_math(tex: str, fontsize: float = 9.8,
                max_h: float | None = None, max_w: float | None = None) -> str:
    """``$...$`` → thẻ ``<img>``, tự thu nhỏ để không tràn dòng / tràn ô bảng.

    ``Paragraph`` của reportlab tính chiều cao dòng theo cỡ chữ, **không** theo
    ảnh, nên một công thức cao (phân số, tích, căn) sẽ đè lên dòng trên. Vì vậy
    chiều cao bị chặn ở ``max_h`` (mặc định 1,5 lần cỡ chữ) và chiều rộng ở
    ``max_w``; vượt thì thu nhỏ theo tỉ lệ.
    """
    max_h = max_h if max_h is not None else fontsize * 1.5
    try:
        path, w, h = math_renderer(fontsize).render(tex)
    except Exception:                                  # công thức sai cú pháp
        return f'<font name="Body-It">{esc(tex)}</font>'
    scale = min(1.0, max_h / h if h else 1.0)
    if max_w and w * scale > max_w:
        scale = max_w / w
    w, h = w * scale, h * scale
    return f'<img src="{path}" width="{w:.2f}" height="{h:.2f}" valign="-2.4"/>'

TOKEN = re.compile(
    r"("
    r"\\\$"
    r"|\$[^$]+\$"
    r"|`[^`]+`"
    r"|\*\*[^*]+\*\*"
    r"|\*[^*]+\*"
    r"|\[\[[\d,\s]+\]\]"
    r"|\[[^\]]+\]\([^)\s]+\)"
    r"|</?(?:sub|sup|b|i|br|font|link)[^>]*>"
    r")"
)


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str, math_fontsize: float = 9.8,
           math_max_h: float | None = None, math_max_w: float | None = None,
           base_size: float | None = None) -> str:
    """Markdown nội dòng → markup reportlab (Paragraph)."""
    text = fix_marks(text)
    out = []
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok == "\\$":
            out.append("$")
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            out.append(inline_math(tok[1:-1], math_fontsize, math_max_h, math_max_w))
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 1:
            # JetBrains Mono thiếu một số dấu tiếng Việt dựng sẵn (ẫ ả ế ề ể ễ ở)
            # — gặp thì dùng phông thân bài để không ra ô "tofu".
            combining = any(0x300 <= ord(c) <= 0x36F for c in tok[1:-1])
            face = "Body" if (set(tok[1:-1]) & MONO_MISSING or combining) else "Mono"
            base = base_size or BODY_SIZE
            size = round(base - (0.6 if face == "Body" else 1.1), 2)
            out.append(
                f'<font name="{face}" size="{size}" color="#8A3A1E">'
                f"{esc(tok[1:-1])}</font>"
            )
        elif tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            out.append(f"<b>{inline(tok[2:-2])}</b>")
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            out.append(f"<i>{inline(tok[1:-1])}</i>")
        elif tok.startswith("[[") and tok.endswith("]]"):
            nums = re.findall(r"\d+", tok)
            for k, num in enumerate(nums):
                if k:
                    out.append(", ")
                out.append(
                    f'<super><font size="6.6" color="{LINK}">[{num}]</font></super>'
                )
        elif tok.startswith("[") and "](" in tok and tok.endswith(")"):
            label, _, url = tok[1:-1].partition("](")
            out.append(
                f'<link href="{esc(url)}"><font color="{LINK}">'
                f"{inline(label)}</font></link>"
            )
        elif re.match(r"</?(?:sub|sup|b|i|br|font|link)", tok):
            out.append(tok)
        else:
            out.append(esc(tok))
    return "".join(out)


def plain(text: str) -> str:
    """Bỏ hết markup để lấy chữ thuần (dùng cho TOC, header)."""
    text = fix_marks(text)
    t = re.sub(r"`([^`]+)`", r"\1", text)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"\[\[([\d,\s]+)\]\]", r"", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"</?(?:sub|sup|b|i|br|font|link)[^>]*>", "", t)
    t = re.sub(r"\$([^$]+)\$", r"\1", t)
    return t.strip()


# --------------------------------------------------------------------------- #
# Kết xuất công thức
# --------------------------------------------------------------------------- #


class MathRenderer:
    def __init__(self, cache_dir: Path, fontsize: float = 13.6, dpi: int = 600):
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fontsize = fontsize
        self.dpi = dpi

    @staticmethod
    def sanitize(tex: str) -> str:
        """Đưa LaTeX về tập con mà mathtext hiểu được.

        mathtext **không** có ``\\big``/``\\Big``/``\\lvert``/``\\text``, và
        ``\\left|`` phải viết thành ``\\left\\vert``. Thay vì bắt người viết
        nhớ, bộ dựng tự chuẩn hoá.
        """
        t = tex
        t = re.sub(r"\\(?:bigg|Bigg|big|Big)(?=\s*[()\[\]|.\\])", "", t)
        t = t.replace("\\left|", "\\left\\vert").replace("\\right|", "\\right\\vert")
        t = t.replace("\\lvert", "\\vert").replace("\\rvert", "\\vert")
        t = re.sub(r"\\text\{", r"\\mathrm{", t)
        # mathtext không có các lệnh quan hệ/bao hàm dài; đổi sang glyph tương đương
        for a, b in (("ge", "geq"), ("le", "leq"), ("gets", "leftarrow"),
                     ("to", "rightarrow"), ("implies", "Longrightarrow")):
            # chỉ đổi khi KHÔNG dính chữ cái phía sau (tránh "\le" khớp "\left")
            t = re.sub(r"\\" + a + r"(?![A-Za-z])", "\\\\" + b, t)
        t = t.replace("\\nonumber", "").replace("\\tag{", "\\mathrm{(")
        return t

    def render(self, tex: str):
        """tex → (đường dẫn png, rộng pt, cao pt)."""
        tex = self.sanitize(tex)
        from matplotlib import mathtext
        from matplotlib.font_manager import FontProperties
        from PIL import Image

        key = hashlib.sha1(f"{tex}|{self.fontsize}|{self.dpi}".encode()).hexdigest()[:16]
        path = self.cache / f"eq_{key}.png"
        if not path.exists():
            prop = FontProperties(size=self.fontsize)
            mathtext.math_to_image(
                f"${tex}$", str(path), prop=prop, dpi=self.dpi, format="png"
            )
        with Image.open(path) as im:
            px_w, px_h = im.size
        scale = 72.0 / self.dpi
        return path, px_w * scale, px_h * scale


# --------------------------------------------------------------------------- #
# Tài liệu
# --------------------------------------------------------------------------- #


def build(src_dir: Path, out_pdf: Path, title: str, subtitle: str,
          toc_pages: dict | None = None) -> dict:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    # ---- phông ---------------------------------------------------------- #
    fd = _mpl_font_dir()
    mono_reg, mono_bold = ensure_mono_font()

    reg = {
        "Body": fd / "DejaVuSerif.ttf",
        "Body-Bd": fd / "DejaVuSerif-Bold.ttf",
        "Body-It": fd / "DejaVuSerif-Italic.ttf",
        "Body-BdIt": fd / "DejaVuSerif-BoldItalic.ttf",
        "Sans": fd / "DejaVuSans.ttf",
        "Sans-Bd": fd / "DejaVuSans-Bold.ttf",
        "Mono": mono_reg,
        "Mono-Bd": mono_bold,
    }
    for name, path in reg.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bd", italic="Body-It", boldItalic="Body-BdIt"
    )
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bd", italic="Sans", boldItalic="Sans-Bd")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bd", italic="Mono", boldItalic="Mono-Bd")

    # ---- style ---------------------------------------------------------- #
    S = {}
    S["body"] = ParagraphStyle(
        "body", fontName="Body", fontSize=BODY_SIZE, leading=BODY_LEAD,
        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5.2,
        hyphenationLang="", embeddedHyphenation=0,
    )
    S["h1"] = ParagraphStyle(
        "h1", fontName="Sans-Bd", fontSize=19.5, leading=24, textColor=ACCENT,
        spaceBefore=2, spaceAfter=4, alignment=TA_LEFT, keepWithNext=1,
    )
    S["h1sub"] = ParagraphStyle(
        "h1sub", fontName="Body-It", fontSize=10.3, leading=15, textColor=MUTED,
        spaceAfter=11, alignment=TA_LEFT, keepWithNext=1,
    )
    S["h2"] = ParagraphStyle(
        "h2", fontName="Sans-Bd", fontSize=12.6, leading=16.4, textColor="#123B4A",
        spaceBefore=13, spaceAfter=5, keepWithNext=1,
    )
    S["h3"] = ParagraphStyle(
        "h3", fontName="Sans-Bd", fontSize=10.6, leading=14, textColor="#27505F",
        spaceBefore=9, spaceAfter=3.5, keepWithNext=1,
    )
    S["bullet"] = ParagraphStyle(
        "bullet", parent=S["body"], leftIndent=15, bulletIndent=3,
        spaceAfter=2.6, alignment=TA_LEFT,
    )
    S["quote"] = ParagraphStyle(
        "quote", parent=S["body"], fontName="Body-It", textColor="#3E4B57",
        leftIndent=14, rightIndent=8, spaceBefore=4, spaceAfter=7,
    )
    S["caption"] = ParagraphStyle(
        "caption", fontName="Sans", fontSize=8.2, leading=11, textColor=MUTED,
        alignment=TA_LEFT, spaceBefore=3.5, spaceAfter=9,
    )
    S["cell"] = ParagraphStyle(
        "cell", fontName="Body", fontSize=8.3, leading=11.2, textColor=INK,
        alignment=TA_LEFT,
    )
    S["cellh"] = ParagraphStyle(
        "cellh", fontName="Sans-Bd", fontSize=8.3, leading=11.2,
        textColor="#FFFFFF", alignment=TA_LEFT,
    )
    S["code"] = ParagraphStyle(
        "code", fontName="Mono", fontSize=MONO_SIZE, leading=MONO_SIZE * 1.42,
        textColor="#1F2328",
    )
    S["eqnum"] = ParagraphStyle(
        "eqnum", fontName="Body", fontSize=8.6, textColor=MUTED, alignment=TA_RIGHT
    )
    S["toc1"] = ParagraphStyle(
        "toc1", fontName="Sans-Bd", fontSize=10.2, leading=15.5,
        textColor=ACCENT, spaceBefore=7,
    )
    S["toc2"] = ParagraphStyle(
        "toc2", fontName="Body", fontSize=9.4, leading=13.4,
        textColor="#2C3742", leftIndent=14, firstLineIndent=-14,
    )
    S["title"] = ParagraphStyle(
        "title", fontName="Sans-Bd", fontSize=31, leading=36.5, textColor=ACCENT,
        alignment=TA_LEFT, spaceAfter=6,
    )
    S["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Sans", fontSize=13.4, leading=19, textColor="#33505C",
        alignment=TA_LEFT, spaceAfter=4,
    )

    math = MathRenderer(CACHE / "math")
    story: list = []

    # ---- helpers -------------------------------------------------------- #
    def para(text: str, style="body"):
        return Paragraph(inline(text), S[style])

    def code_block(code: str, lang: str, opts: str):
        from pygments import lex
        from pygments.lexers import get_lexer_by_name
        from pygments.styles import get_style_by_name

        show_nums = "linenums" in opts
        try:
            lexer = get_lexer_by_name(lang if lang != "text" else "text")
        except Exception:
            lexer = get_lexer_by_name("text")
        style = get_style_by_name("friendly")

        lines: list[list[str]] = [[]]
        try:
            toks = list(lex(code, lexer))
        except Exception:
            toks = [(None, code)]
        for ttype, value in toks:
            color = None
            bold = False
            if ttype is not None:
                st = style.style_for_token(ttype)
                if st.get("color"):
                    color = "#" + st["color"]
                bold = bool(st.get("bold"))
            parts = value.split("\n")
            for k, part in enumerate(parts):
                if part:
                    frag = esc(part)
                    if color or bold:
                        attrs = []
                        if color:
                            attrs.append(f'color="{color}"')
                        if bold:
                            attrs.append('face="Mono-Bd"')
                        frag = f"<font {' '.join(attrs)}>{frag}</font>"
                    lines[-1].append(frag)
                if k < len(parts) - 1:
                    lines.append([])

        # thu nhỏ nếu dòng quá dài để không tràn lề
        width_avail = PAGE_W - M_LEFT - M_RIGHT - 24
        longest = max((len(l) for l in code.split("\n")), default=1)
        size = MONO_SIZE
        while size > 6.2 and longest * size * 0.601 > width_avail:
            size -= 0.15

        # Mỗi dòng là một hàng của Table: Paragraph không bị gộp dòng, và Table
        # được phép ngắt trang giữa các hàng (splitByRow).
        line_style = ParagraphStyle(
            f"cl{size}", parent=S["code"], fontSize=size, leading=round(size * 1.52, 2),
        )
        rows = []
        for idx, frags in enumerate(lines, start=1):
            body = keep_spaces("".join(frags))
            if not body.strip():
                body = "&#160;"
            if show_nums:
                body = f'<font color="#9AA6B2">{idx:>3}</font>  {body}'
            rows.append([Paragraph(body, line_style)])

        t = Table(rows, colWidths=[PAGE_W - M_LEFT - M_RIGHT], splitByRow=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CODE_BG)),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(CODE_EDGE)),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor("#9DB7C4")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 0.2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0.2),
                    ("TOPPADDING", (0, 0), (-1, 0), 7),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
                ]
            )
        )
        if len(rows) <= 14:
            from reportlab.platypus import KeepTogether
            return [KeepTogether(t), Spacer(1, 8)]
        return [t, Spacer(1, 8)]

    def math_block(tex: str, num: str | None):
        path, w, h = math.render(tex)
        maxw = PAGE_W - M_LEFT - M_RIGHT - 70
        if w > maxw:
            h *= maxw / w
            w = maxw
        img = Image(str(path), width=w, height=h)
        if num:
            t = Table(
                [[Para_center(img), Paragraph(f"({num})", S["eqnum"])]],
                colWidths=[PAGE_W - M_LEFT - M_RIGHT - 40, 40],
            )
            t.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            return [Spacer(1, 3), t, Spacer(1, 8)]
        return [Spacer(1, 3), Para_center(img), Spacer(1, 8)]

    def _col_widths(header, rows, ncol: int, avail: float,
                    size: float) -> tuple[list[float], float]:
        """Chia bề rộng cột: mỗi cột đủ chỗ cho **token dài nhất** (đo bằng đúng
        phông sẽ vẽ), phần dư chia theo tổng độ dài. Nhờ vậy `utils/config.py`
        không bị bẻ đôi giữa từ."""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        def segments(cell: str) -> list[tuple[str, str]]:
            """Tách ô thành (đoạn, phông): trong `...` là Mono, còn lại là Body."""
            out, in_code = [], False
            for chunk in cell.split("`"):
                if chunk:
                    out.append((plain(chunk) if not in_code else chunk,
                                "Mono" if in_code else "Body"))
                in_code = not in_code
            return out or [("", "Body")]

        def measure(cell: str) -> tuple[float, float]:
            """(bề rộng cả ô, bề rộng token dài nhất) — token được phép ngắt sau
            `/` `_` `-` để đường dẫn dài vẫn xuống dòng đẹp."""
            full, tok = 0.0, 0.0
            for text, font in segments(cell):
                full += stringWidth(text, font, size)
                for w in text.split():
                    parts = [w] if len(w) < 22 else re.split(r"(?<=[/_\-])", w)
                    wmax = max(stringWidth(x, font, size) for x in parts)
                    tok = max(tok, wmax)
            return full, tok

        need, want = [], []
        for c in range(ncol):
            cells = [header[c]] + [r[c] for r in rows]
            tok = full = 0.0
            for cell in cells:
                f, t = measure(cell)
                full, tok = max(full, f), max(tok, t)
            need.append(tok + 11)
            want.append(full + 11)
        if sum(need) > avail and size > 7.0:
            # không đủ chỗ cho các token dài: thu nhỏ chữ một chút (tối thiểu 7 pt)
            return _col_widths(header, rows, ncol, avail, round(size - 0.3, 2))
        leftover = avail - sum(need)
        t = sum(want) or 1.0
        return [need[i] + leftover * want[i] / t for i in range(ncol)], size

    def table_block(header: list[str], rows: list[list[str]]):
        ncol = len(header)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        avail = PAGE_W - M_LEFT - M_RIGHT
        # bảng nhiều cột: giảm cỡ chữ
        size = 8.3 if ncol <= 6 else 7.5
        widths, size = _col_widths(header, rows, ncol, avail, size)
        cell = ParagraphStyle(f"c{ncol}", parent=S["cell"], fontSize=size, leading=size * 1.35)
        cellh = ParagraphStyle(f"ch{ncol}", parent=S["cellh"], fontSize=size, leading=size * 1.35)
        data = [[Paragraph(inline(h, size, size * 1.25, avail / ncol, size), cellh)
                 for h in header]]
        for r in rows:
            data.append([Paragraph(inline(c, size, size * 1.25, avail / ncol, size), cell)
                         for c in r])
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        st = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(RULE)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C7D2DA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ]
        for idx in range(1, len(data)):
            if idx % 2 == 0:
                st.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor(STRIPE)))
        t.setStyle(TableStyle(st))
        return [t, Spacer(1, 8)]

    def callout_block(kind: str, title: str, body: str):
        palette = {
            "note": ("#EAF2F6", "#7FA8B8", ACCENT),
            "warn": ("#FDF3E7", "#D9A55B", "#8A4B08"),
            "key": ("#EDF5EC", "#8FBA8A", "#2C5E30"),
        }
        bg, edge, ink = palette.get(kind, palette["note"])
        label = title or {"note": "Ghi chú", "warn": "Cảnh báo", "key": "Điểm cốt lõi"}[
            kind if kind in ("note", "warn", "key") else "note"
        ]
        inner = [
            Paragraph(
                f'<font name="Sans-Bd" size="8.4" color="{ink}">{esc(label.upper())}</font>',
                ParagraphStyle("cl", leading=12, spaceAfter=3),
            )
        ]
        for chunk in body.split("\n\n"):
            chunk = " ".join(l.strip() for l in chunk.split("\n") if l.strip())
            if chunk:
                inner.append(
                    Paragraph(
                        inline(chunk),
                        ParagraphStyle(
                            "clb", parent=S["body"], fontSize=BODY_SIZE - 0.6,
                            leading=BODY_LEAD - 1.2, spaceAfter=3.5, textColor="#22303A",
                        ),
                    )
                )
        t = Table([[inner]], colWidths=[PAGE_W - M_LEFT - M_RIGHT])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.6, colors.HexColor(edge)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 11),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )
        return [Spacer(1, 3), t, Spacer(1, 9)]

    def Para_center(flowable):
        t = Table([[flowable]], colWidths=[PAGE_W - M_LEFT - M_RIGHT - 40])
        t.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return t

    def list_block(items, ordered: bool):
        out = []
        counters: dict[int, int] = {}
        for indent, text in items:
            st = ParagraphStyle(
                f"b{indent}{ordered}", parent=S["bullet"],
                leftIndent=16 + 14 * indent, bulletIndent=2 + 14 * indent,
            )
            if ordered:
                counters[indent] = counters.get(indent, 0) + 1
                for k in list(counters):
                    if k > indent:
                        del counters[k]
                bullet = f"{counters[indent]}."
                st.bulletFontName = "Sans-Bd"
                st.bulletFontSize = BODY_SIZE - 1.2
                st.bulletColor = ACCENT
            else:
                bullet = "•" if indent == 0 else "–"
                st.bulletFontName = "Body"
                st.bulletColor = "#54626E" if indent == 0 else "#8593A0"
            out.append(Paragraph(inline(text), st, bulletText=bullet))
        out.append(Spacer(1, 4.5))
        return out

    # ---- đọc nguồn ------------------------------------------------------- #
    parts = sorted(src_dir.glob("*.md"))
    if not parts:
        sys.exit(f"[!] Không có .md trong {src_dir}")
    blocks: list[tuple] = []
    for p in parts:
        blocks.extend(parse(p.read_text(encoding="utf-8")))

    # ---- bìa ------------------------------------------------------------ #
    from reportlab.platypus import Flowable

    class Rule(Flowable):
        def __init__(self, width, thickness=1.1, color=ACCENT, space=6):
            super().__init__()
            self.width, self.thickness, self.color, self.space = (
                width, thickness, color, space,
            )
            self.height = thickness + space

        def draw(self):
            self.canv.setStrokeColor(self.color)
            self.canv.setLineWidth(self.thickness)
            self.canv.line(0, self.space, self.width, self.space)

    story.append(Spacer(1, 96))
    story.append(Paragraph("GIÁO TRÌNH · THỊ GIÁC MÁY TÍNH", ParagraphStyle(
        "kick", fontName="Sans-Bd", fontSize=9.2, leading=12, textColor="#7A8A94",
        spaceAfter=18)))
    story.append(Paragraph(title, S["title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(subtitle, S["subtitle"]))
    story.append(Spacer(1, 14))
    story.append(Rule(PAGE_W - M_LEFT - M_RIGHT, 2.2, ACCENT, 0))
    story.append(Spacer(1, 18))

    abstract = (
        "Tài liệu này giải thích **DIRE** — một biểu diễn dùng mô hình khuếch tán "
        "(diffusion model) để phát hiện ảnh do AI sinh — đi từ nền tảng xử lý ảnh "
        "và học máy, qua lý thuyết diffusion, tới từng hàm và từng dòng mã của "
        "bài báo gốc và của kho mã này. Mỗi công thức đều được nối với chỗ cài đặt "
        "tương ứng; mỗi con số đều có nguồn. Tài liệu viết cho người đã biết xử lý "
        "ảnh truyền thống và học máy cơ bản, **chưa** biết diffusion model."
    )
    box = Table(
        [[Paragraph(
            inline(abstract),
            ParagraphStyle("abs", parent=S["body"], fontSize=10.2, leading=16.2,
                           textColor="#22313B"))]],
        colWidths=[PAGE_W - M_LEFT - M_RIGHT],
    )
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F6F8")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(ACCENT)),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))
    story.append(box)
    story.append(Spacer(1, 26))

    meta = [
        ["Đối tượng", "Sinh viên / kỹ sư đã biết xử lý ảnh số và học máy cơ bản"],
        ["Nguồn gốc", "arXiv:2303.09295 (DIRE) [[1]] · github.com/ZhendongWang6/DIRE [[2]]"],
        ["Phạm vi", "Lý thuyết nền → diffusion → DIRE → mã nguồn → hạn chế & hướng tiếp"],
        ["Ngày dựng", "2026-10-14"],
    ]
    mt = Table(
        [[Paragraph(f'<font name="Sans-Bd" size="8.4" color="#5B6673">{a.upper()}</font>',
                    ParagraphStyle("m1", leading=12)),
          Paragraph(inline(b), ParagraphStyle("m2", fontName="Body", fontSize=8.9,
                                              leading=13, textColor="#243039"))]
         for a, b in meta],
        colWidths=[92, PAGE_W - M_LEFT - M_RIGHT - 92],
    )
    mt.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(mt)
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ---- mục lục -------------------------------------------------------- #
    story.append(Paragraph("Mục lục", ParagraphStyle(
        "tochead", fontName="Sans-Bd", fontSize=18, leading=22, textColor=ACCENT,
        spaceAfter=6)))
    story.append(Rule(PAGE_W - M_LEFT - M_RIGHT, 1.0, RULE, 0))
    story.append(Spacer(1, 12))

    # Mục lục tự dựng (hai lượt): hàng tiêu đề | số trang. Số hàng cố định giữa
    # hai lượt nên phân trang không đổi — nhờ vậy số trang ở lượt 2 là đúng.
    heads = [
        (0 if b[0] == BLOCK_H1 else 1, plain(b[1]))
        for b in blocks
        if b[0] in (BLOCK_H1, BLOCK_H2)
    ]
    toc_rows, toc_style = [], [
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E7EDF2")),
    ]
    for r, (lvl, text) in enumerate(heads):
        page = "—" if not toc_pages else str(toc_pages.get(text, "—"))
        if lvl == 0:
            st_title, st_page = S["toc1"], ParagraphStyle(
                "tocp1", fontName="Sans-Bd", fontSize=9.6, textColor=ACCENT,
                alignment=TA_RIGHT)
            toc_style.append(("TOPPADDING", (0, r), (-1, r), 9))
            toc_style.append(("LINEABOVE", (0, r), (-1, r), 0.7, colors.HexColor(RULE)))
        else:
            st_title, st_page = S["toc2"], ParagraphStyle(
                "tocp2", fontName="Body", fontSize=9, textColor="#5B6673",
                alignment=TA_RIGHT)
        toc_rows.append([
            Paragraph(inline(text), st_title),
            Paragraph(page, st_page),
        ])
    toc_t = Table(toc_rows, colWidths=[PAGE_W - M_LEFT - M_RIGHT - 40, 40])
    toc_t.setStyle(TableStyle(toc_style))
    story.append(toc_t)
    story.append(PageBreak())

    # ---- thân ----------------------------------------------------------- #
    first_h1 = True
    for blk in blocks:
        kind = blk[0]
        if kind == BLOCK_H1:
            if not first_h1:
                story.append(PageBreak())
            first_h1 = False
            story.append(Paragraph(inline(blk[1]), S["h1"]))
            story.append(Rule(PAGE_W - M_LEFT - M_RIGHT, 0.9, RULE, 4))
            story.append(Spacer(1, 9))
        elif kind == BLOCK_H2:
            story.append(Paragraph(inline(blk[1]), S["h2"]))
        elif kind == BLOCK_H3:
            story.append(Paragraph(inline(blk[1]), S["h3"]))
        elif kind == BLOCK_P:
            story.append(para(blk[1]))
        elif kind == BLOCK_CODE:
            story.extend(code_block(blk[2], blk[1], blk[3]))
        elif kind == BLOCK_MATH:
            story.extend(math_block(blk[1], blk[2]))
        elif kind == BLOCK_TABLE:
            story.extend(table_block(blk[1], blk[2]))
        elif kind in (BLOCK_UL, BLOCK_OL):
            story.extend(list_block(blk[1], kind == BLOCK_OL))
        elif kind == BLOCK_QUOTE:
            t = Table([[Paragraph(inline(blk[1]), S["quote"])]],
                      colWidths=[PAGE_W - M_LEFT - M_RIGHT])
            t.setStyle(TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor("#C6D3DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(t)
            story.append(Spacer(1, 7))
        elif kind == BLOCK_CALLOUT:
            story.extend(callout_block(blk[1], blk[2], blk[3]))
        elif kind == BLOCK_HR:
            story.append(Spacer(1, 4))
            story.append(Rule(PAGE_W - M_LEFT - M_RIGHT, 0.6, RULE, 0))
            story.append(Spacer(1, 10))
        elif kind == BLOCK_PAGEBREAK:
            story.append(PageBreak())

    # ---- tài liệu ------------------------------------------------------- #
    class Doc(BaseDocTemplate):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.current_part = ""
            self.toc_log: list[tuple[int, str, int]] = []

        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph):
                name = flowable.style.name
                if name == "h1":
                    txt = plain(flowable.getPlainText())
                    key = f"h1-{abs(hash(txt)) % (10**8)}"
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(txt, key, level=0, closed=False)
                    self.current_part = txt
                    self.toc_log.append((0, txt, self.page))
                elif name == "h2":
                    txt = plain(flowable.getPlainText())
                    key = f"h2-{abs(hash(txt)) % (10**8)}"
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(txt, key, level=1, closed=True)
                    self.toc_log.append((1, txt, self.page))

    def on_body_header(canv, doc):
        canv.saveState()
        # header
        canv.setFont("Sans", 7.6)
        canv.setFillColor(colors.HexColor(MUTED))
        canv.drawString(M_LEFT, PAGE_H - 36, "DIRE — từ số 0")
        part = getattr(doc, "current_part", "") or ""
        canv.drawRightString(PAGE_W - M_RIGHT, PAGE_H - 36, part[:74])
        canv.setStrokeColor(colors.HexColor(RULE))
        canv.setLineWidth(0.5)
        canv.line(M_LEFT, PAGE_H - 42, PAGE_W - M_RIGHT, PAGE_H - 42)
        # footer
        canv.setFont("Body", 8)
        canv.setFillColor(colors.HexColor(MUTED))
        canv.drawCentredString(PAGE_W / 2, 30, str(canv.getPageNumber()))
        canv.restoreState()

    def on_cover_page(canv, doc):
        canv.saveState()
        canv.setFillColor(colors.HexColor("#F5F8FA"))
        canv.rect(0, PAGE_H - 12, PAGE_W, 12, stroke=0, fill=1)
        canv.setFillColor(colors.HexColor(ACCENT))
        canv.rect(0, 0, 6, PAGE_H, stroke=0, fill=1)
        canv.restoreState()

    doc = Doc(
        str(out_pdf),
        pagesize=A4,
        leftMargin=M_LEFT, rightMargin=M_RIGHT,
        topMargin=M_TOP, bottomMargin=M_BOTTOM,
        title=title, author="Arena Agent", subject=subtitle,
    )
    frame = Frame(
        M_LEFT, M_BOTTOM, PAGE_W - M_LEFT - M_RIGHT, PAGE_H - M_TOP - M_BOTTOM,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="main",
    )
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=on_cover_page),
        PageTemplate(id="body", frames=[frame], onPageEnd=on_body_header),
    ])
    doc.multiBuild(story)
    return {text: page for _lvl, text, page in doc.toc_log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/book")
    ap.add_argument("--out", default="docs/dire-tu-so-0.pdf")
    ap.add_argument("--title", default="DIRE từ số 0")
    ap.add_argument(
        "--subtitle",
        default="Lý thuyết nền, mô hình khuếch tán và phát hiện ảnh do AI sinh",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(a.src)
    tmp = out.with_name(out.stem + ".pass1.pdf")
    pages = build(src, tmp, a.title, a.subtitle)
    build(src, out, a.title, a.subtitle, toc_pages=pages)
    tmp.unlink(missing_ok=True)
    print(f"[ok] {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(pages)} mục)")


if __name__ == "__main__":
    main()
