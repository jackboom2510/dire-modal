from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_slide_assets", ROOT / "tools" / "build_slide_assets.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_builds_source_backed_slide_pack_without_ml_dependencies(tmp_path: Path):
    outputs = mod.build(tmp_path / "slides", (200, 400))
    out = tmp_path / "slides"
    names = {p.name for p in outputs}

    assert {"dire-pipeline.svg", "map-provenance.svg", "nested-subsets.svg", "experiment-map.svg"} <= names
    assert "experiment-protocol.svg" in names
    assert "table5-representations.svg" in names
    assert "subset-stages.svg" in names
    assert "paper-table2-reference.svg" in names
    assert "status-and-caveats.svg" in names
    assert "index.html" in names

    for svg in out.glob("*.svg"):
        ET.parse(svg)
    paper_csv = (out / "paper-table2-reference.csv").read_text(encoding="utf-8-sig")
    assert "ADM,99.9,100.0" in paper_csv
    assert "PNDM,98.7,100.0" in paper_csv
    subset_svg = (out / "subset-stages.svg").read_text(encoding="utf-8")
    assert "≤ 2,000" in subset_svg and "≤ 4,000" in subset_svg
    status_svg = (out / "status-and-caveats.svg").read_text(encoding="utf-8")
    assert "Not run" in status_svg
    assert "No project result to plot" in status_svg


def test_stage_sizes_must_be_positive_and_unique(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="số nguyên dương"):
        mod.build(tmp_path / "bad", (200, 200))
    with pytest.raises(ValueError, match="số nguyên dương"):
        mod.build(tmp_path / "bad2", (0, 200))
