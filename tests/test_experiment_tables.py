from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment_tables", ROOT / "tools" / "experiment_tables.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _result():
    return {"status": "complete", "by_test_generator": {
        g: {"status": "ran", "metrics": {"acc": 0.9, "ap": 0.95}, "n_real": 10, "n_fake": 10}
        for g in mod.GENERATORS
    }}


def test_render_table4_and_table5_only_from_complete_ledgers():
    t4 = {"by_steps": {str(s): _result() for s in mod.STEPS}}
    t5 = {"by_representation": {r: _result() for r in mod.REPRESENTATIONS}}
    out = mod.build_tables(t4, t5)
    assert out["complete"] is True
    assert len(out["table4"]) == 20
    assert len(out["table5"]) == 20
    assert out["table4"][0]["acc_percent"] == "90.0"
    assert out["table5"][-1]["ap_percent"] == "95.0"


def test_incomplete_results_are_rejected_by_default_but_explicitly_marked():
    with pytest.raises(ValueError, match="chưa đủ cell"):
        mod.build_tables({"by_steps": {"5": _result()}}, None)
    partial = mod.build_tables({"by_steps": {"5": _result()}}, None, allow_partial=True)
    assert partial["complete"] is False
    assert partial["table4"][5]["acc_percent"] == "—"


def test_writer_outputs_csv_markdown_and_source_hash(tmp_path: Path):
    result = mod.build_tables({"by_steps": {str(s): _result() for s in mod.STEPS}}, None)
    md = mod.write_tables(result, tmp_path, {"table4.json": "abc"})
    assert md.exists() and (tmp_path / "table4.csv").exists()
    content = md.read_text(encoding="utf-8")
    assert "Table 4 — DDIM steps" in content and "abc" in content
