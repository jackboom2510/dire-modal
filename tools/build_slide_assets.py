#!/usr/bin/env python3
"""Build slide-ready SVG diagrams/tables from repository facts (stdlib only).

Usage:
    python tools/build_slide_assets.py
    python tools/build_slide_assets.py --out /tmp/dire-slide-assets --per-cell 200 400

The script copies the hand-authored conceptual diagrams, then generates protocol,
subset-size, representation, status, and paper-reference tables from checked-in
sources. It never reads or fabricates experimental scores. SVG works as a crisp
vector in PowerPoint/Google Slides; CSV companions make the tables editable.
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_PY = ROOT / "src" / "dire_zero" / "experiments.py"
PAPER_TABLE = ROOT / "src" / "dire_modal" / "paper_table2.tsv"
SOURCE_DIAGRAMS = ROOT / "docs" / "slide-assets"
DEFAULT_OUT = ROOT / "docs" / "slide-assets" / "generated"

NAVY = "#0b1728"
NAVY_2 = "#14283c"
TEAL = "#81d1c2"
AMBER = "#f2c66d"
INK = "#e3edf1"
MUTED = "#a6bbc7"
GRID = "#345168"


def _literal_assignment(path: Path, name: str):
    """Read a simple literal assignment without importing torch/Modal modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"Không tìm thấy literal {name} trong {path}")


def _protocol_facts() -> dict:
    return {
        "steps": tuple(_literal_assignment(EXPERIMENTS_PY, "TABLE4_STEPS")),
        "representations": tuple(_literal_assignment(EXPERIMENTS_PY, "TABLE5_REPRESENTATIONS")),
        "conditions": tuple(_literal_assignment(EXPERIMENTS_PY, "ROBUSTNESS_LEVELS")),
        "test_generators": tuple(_literal_assignment(EXPERIMENTS_PY, "EXPERIMENT_TEST_GENERATORS")),
        "scope_note": _literal_assignment(EXPERIMENTS_PY, "TEST_GENERATOR_SCOPE_NOTE"),
    }


def _paper_table2_averages() -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    with PAPER_TABLE.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader((line for line in fh if not line.lstrip().startswith("#")), delimiter="\t"):
            if not row or row[0] == "ckpt":
                continue
            if len(row) != 4:
                raise ValueError(f"Dòng Table 2 sai cột trong {PAPER_TABLE}: {row}")
            ckpt, generator, acc, ap = row
            if generator == "avg":
                found[ckpt] = {"detector": ckpt.removeprefix("lsun_"),
                               "mean_ACC_percent": acc, "mean_AP_percent": ap}
    order = ("lsun_adm", "lsun_pndm", "lsun_iddpm", "lsun_stylegan")
    if set(found) != set(order):
        raise ValueError(f"Table 2 cần đúng bốn hàng avg {order}; nhận {sorted(found)}")
    return [found[key] for key in order]


def _wrap(value: object, width: int) -> list[str]:
    words = str(value).split()
    lines: list[str] = []
    line = ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines or [""]


def _svg_table(title: str, subtitle: str, headers: Sequence[str], rows: Sequence[Sequence[object]],
               *, widths: Sequence[int], footnote: str = "") -> str:
    if len(headers) != len(widths) or any(len(row) != len(headers) for row in rows):
        raise ValueError("Số cột bảng không khớp headers/widths")
    width = sum(widths)
    x_positions = [0]
    for w in widths:
        x_positions.append(x_positions[-1] + w)
    margin = 44
    top = 135
    header_h = 54
    font_size = 17
    pad_x = 16
    body_rows: list[tuple[Sequence[object], list[list[str]], int]] = []
    for row in rows:
        wrapped = [_wrap(cell, max(9, int((w - 2 * pad_x) / (font_size * 0.54))))
                   for cell, w in zip(row, widths)]
        max_lines = max(map(len, wrapped), default=1)
        body_rows.append((row, wrapped, max(56, 22 + max_lines * 23)))
    total_h = top + header_h + sum(item[2] for item in body_rows) + 72
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 2 * margin}" height="{total_h}" viewBox="0 0 {width + 2 * margin} {total_h}" role="img">',
        '<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b1728"/><stop offset="1" stop-color="#14283c"/></linearGradient></defs>',
        f'<rect width="100%" height="100%" rx="24" fill="url(#bg)"/>',
        f'<text x="{margin}" y="48" fill="{TEAL}" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" letter-spacing="1.4">DIRE · SLIDE DATA</text>',
        f'<text x="{margin}" y="87" fill="#fff" font-family="Inter,Arial,sans-serif" font-size="28" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{margin}" y="116" fill="{MUTED}" font-family="Inter,Arial,sans-serif" font-size="14">{html.escape(subtitle)}</text>',
        f'<rect x="{margin}" y="{top}" width="{width}" height="{header_h}" rx="8" fill="#214253"/>',
    ]
    for i, text in enumerate(headers):
        x = margin + x_positions[i] + pad_x
        out.append(f'<text x="{x}" y="{top + 34}" fill="{TEAL}" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="800">{html.escape(str(text))}</text>')
    y = top + header_h
    for ri, (_row, wrapped, row_h) in enumerate(body_rows):
        fill = "#13283a" if ri % 2 == 0 else "#182f42"
        out.append(f'<rect x="{margin}" y="{y}" width="{width}" height="{row_h}" fill="{fill}"/>')
        out.append(f'<path d="M{margin} {y + row_h} H{margin + width}" stroke="{GRID}" stroke-width="1"/>')
        for ci, lines in enumerate(wrapped):
            x = margin + x_positions[ci] + pad_x
            base_y = y + 25
            spans = "".join(f'<tspan x="{x}" dy="{0 if idx == 0 else 22}">{html.escape(line)}</tspan>'
                            for idx, line in enumerate(lines))
            out.append(f'<text x="{x}" y="{base_y}" fill="{INK}" font-family="Inter,Arial,sans-serif" font-size="{font_size}">{spans}</text>')
        y += row_h
    foot_y = y + 34
    if footnote:
        out.append(f'<text x="{margin}" y="{foot_y}" fill="{AMBER}" font-family="Inter,Arial,sans-serif" font-size="13">{html.escape(footnote)}</text>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _condition_label(condition: dict) -> str:
    name, kind, value = condition["name"], condition["kind"], condition["value"]
    if kind == "clean":
        return "Clean"
    if kind == "gaussian_blur":
        return f"Gaussian blur σ={value:g}"
    if kind == "jpeg":
        return f"JPEG quality={value}"
    return str(name)


def _build_tables(out: Path, per_cell: Sequence[int]) -> list[Path]:
    f = _protocol_facts()
    created: list[Path] = []

    # Protocol table: facts are read from the exact constants used by the runners.
    protocol_headers = ("Experiment", "Varied factor", "Protocol encoded in repo", "Output / status")
    protocol_rows = [
        ("Table 4", f"DDIM S={', '.join(map(str, f['steps']))}",
         "ADM train; test per generator: " + ", ".join(f["test_generators"]),
         "One model/cache per S; no subset result reported here"),
        ("Table 5", "Representation: " + ", ".join(f["representations"]),
         "S=20; separate representation models; same selected samples",
         "REC needs reconstruction; no subset metrics supplied"),
        ("Figure 4", "; ".join(_condition_label(c) for c in f["conditions"]),
         "AP by test generator; DIRE recomputed after each corruption",
         "Project RGB vs DIRE; paper baselines not reproduced"),
    ]
    spec = ("experiment-protocol", "Protocol matrix · implementation facts, not measurements",
            protocol_headers, protocol_rows, (130, 245, 530, 300),
            f["scope_note"])
    created.extend(_emit_table(out, *spec))

    rep_rows = (
        ("REC", "Reconstruction image", "3", "[0,1]"),
        ("RGB", "Original RGB image", "3", "[0,1]"),
        ("RGB_DIRE", "RGB channels then DIRE channels", "6", "RGB [0,1] + DIRE [0,2]"),
        ("DIRE", "Absolute residual map", "3", "[0,2]"),
    )
    rep_headers = ("Representation", "Input semantics", "Channels", "Stored value range")
    spec = ("table5-representations", "Table 5 · representation contracts from dire_zero/experiments.py",
            rep_headers, rep_rows, (190, 410, 120, 330),
            "Implementation contract; exact correspondence to every paper detail must be independently verified.")
    created.extend(_emit_table(out, *spec))

    cells = 2 + 2 + 1 + len(f["test_generators"])
    subset_rows = []
    for n in per_cell:
        subset_rows.append((f"{n} / cell", f"≤ {n * cells:,}",
                            "Nested SHA-256 ranking; same subset_id + seed",
                            "Manifest/index only; original images are not copied"))
    subset_headers = ("Stage size", "Maximum rows*", "Selection identity", "Storage / reuse")
    spec = ("subset-stages", "Incremental subset stages · nominal upper bounds",
            subset_headers, subset_rows, (180, 180, 430, 440),
            "*Upper bound assumes every required cell has at least per_cell rows; actual n_rows comes from each snapshot.")
    created.extend(_emit_table(out, *spec))

    paper_rows = _paper_table2_averages()
    paper_headers = ("Detector trained by author", "Mean ACC %", "Mean AP %")
    display_names = {"adm": "ADM", "pndm": "PNDM", "iddpm": "iDDPM", "stylegan": "StyleGAN"}
    paper_rows = [(display_names[r["detector"]], r["mean_ACC_percent"], r["mean_AP_percent"])
                  for r in paper_rows]
    spec = ("paper-table2-reference", "Reference only · author-reported Table 2 averages",
            paper_headers, paper_rows, (420, 260, 260),
            "Wang et al., ICCV 2023, Table 2 · not measured by this project or the subset workflow.")
    created.extend(_emit_table(out, *spec))

    status_headers = ("Item", "State", "Safe wording for slides")
    status_rows = (
        ("Subset manifest workflow", "Implemented", "Code is available; snapshot must still be created on Volume"),
        ("Table 4 subset metrics", "Not run", "No project result to plot"),
        ("Table 5 subset metrics", "Not run", "No project result to plot"),
        ("Figure 4 subset metrics", "Not run", "No project result to plot"),
        ("Figure 4 paper baselines", "Not implemented", "Do not claim exact Figure 4 reproduction"),
        ("Author-map numeric encoding", "Must verify", "Do not infer PNG scale from extension or peek"),
    )
    spec = ("status-and-caveats", "Project status · do not turn plans into results",
            status_headers, status_rows, (310, 230, 700),
            "This table describes the repository/workspace state represented by this script; refresh after runs or protocol changes.")
    created.extend(_emit_table(out, *spec))
    return created


def _emit_table(out: Path, slug: str, title: str, headers: Sequence[str],
                rows: Sequence[Sequence[object]], widths: Sequence[int], footnote: str) -> list[Path]:
    svg_path = out / f"{slug}.svg"
    csv_path = out / f"{slug}.csv"
    _write_csv(csv_path, headers, rows)
    svg_path.write_text(_svg_table(title, "Generated from checked-in sources · no experiment was run", headers, rows,
                                   widths=widths, footnote=footnote),
                        encoding="utf-8")
    return [svg_path, csv_path]


def _gallery(out: Path, assets: Iterable[Path]) -> None:
    rels = [p.name for p in assets if p.suffix.lower() == ".svg"]
    sections = "\n".join(
        f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in rels
    )
    page = f'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DIRE slide assets</title><style>
body{{margin:0;padding:32px;background:#0b1728;color:#e3edf1;font:16px/1.5 Inter,Arial,sans-serif}}
h1{{color:#81d1c2}}p{{color:#a6bbc7}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}}
figure{{margin:0;padding:16px;background:#14283c;border:1px solid #345168;border-radius:14px}}
img{{width:100%;height:auto;display:block}}figcaption{{padding-top:8px;color:#a6bbc7;font-size:13px}}
</style><h1>DIRE · generated slide assets</h1>
<p>Vector schematics and source-backed tables. No new experimental measurements are generated.</p><main>{sections}</main></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


def build(out: Path, per_cell: Sequence[int]) -> list[Path]:
    if not per_cell or any(n < 1 for n in per_cell) or len(set(per_cell)) != len(per_cell):
        raise ValueError("--per-cell cần số nguyên dương, không lặp")
    out.mkdir(parents=True, exist_ok=True)
    assets: list[Path] = []
    for name in ("dire-pipeline.svg", "map-provenance.svg", "nested-subsets.svg", "experiment-map.svg"):
        source = SOURCE_DIAGRAMS / name
        if not source.is_file():
            raise FileNotFoundError(f"Thiếu sơ đồ nguồn: {source}")
        target = out / name
        shutil.copyfile(source, target)
        assets.append(target)
    assets.extend(_build_tables(out, per_cell))
    _gallery(out, assets)
    assets.append(out / "index.html")
    manifest = {
        "asset_count": len(assets) + 1,
        "sources": ["src/dire_zero/experiments.py", "src/dire_modal/paper_table2.tsv",
                    "docs/slide-assets/*.svg"],
        "per_cell_stages": list(per_cell),
        "warning": "SVGs are schematics/source-backed tables, not current subset experiment measurements.",
        "files": [p.name for p in assets] + ["asset-manifest.json"],
    }
    (out / "asset-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    assets.append(out / "asset-manifest.json")
    return assets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"đầu ra (mặc định {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--per-cell", type=int, nargs="+", default=[200, 400],
                        help="các stage subset cần đưa vào bảng; chỉ là ceiling, không phải số ảnh thực tế")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    outputs = build(out, args.per_cell)
    print(f"[OK] tạo {len(outputs)} file trong {out}")
    for path in outputs:
        print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
