"""Tests for the read-only relabel conflict audit helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_relabel_conflicts.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("audit_relabel_conflicts", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_joins_both_paths_and_reports_run_provenance():
    tool = _load_tool()
    report = {
        "plan": {
            "unresolved": [
                {
                    "rel_src": "test/fake/a.npy",
                    "rel_dst": "test/adm/a.npy",
                    "problem": "đích đã tồn tại — không ghi đè",
                },
                {"rel_src": "test/fake/b.npy", "rel_dst": "test/adm/b.npy"},
            ]
        }
    }
    index = {
        "n": 2,
        "steps": 20,
        "steps_mixed": [20],
        "ckpt_sha256": "abc123",
        "runs": [{"created": "2026-11-18T00:00:00Z", "steps": 20, "ckpt_sha256": "abc123"}],
        "items": [
            {"path": "test/fake/a.npy", "source": "/data/test/adm/a.png", "generator": "fake"},
            {"path": "test/adm/a.npy", "source": "/data/test/adm/a.png", "generator": "adm"},
        ],
    }

    result = tool.audit_conflicts(report, index, limit=1)
    assert result["n_unresolved"] == 2
    assert result["n_shown"] == 1
    conflict = result["conflicts"][0]
    assert conflict["source_item"]["generator"] == "fake"
    assert conflict["destination_item"]["generator"] == "adm"
    assert conflict["source_item"]["source"] == conflict["destination_item"]["source"]
    assert result["runs"][0]["ckpt_sha256"] == "abc123"


def test_audit_limit_zero_includes_all_and_missing_records_are_explicit():
    tool = _load_tool()
    report = {"plan": {"unresolved": [
        {"rel_src": "s1", "rel_dst": "d1"},
        {"rel_src": "s2", "rel_dst": "d2"},
    ]}}
    result = tool.audit_conflicts(report, {"items": []}, limit=0)
    assert result["n_shown"] == 2
    assert result["conflicts"][0]["source_item"] is None
    assert result["conflicts"][1]["destination_item"] is None


def test_audit_rejects_negative_limit():
    tool = _load_tool()
    with pytest.raises(ValueError, match="limit"):
        tool.audit_conflicts({"plan": {"unresolved": []}}, {"items": []}, limit=-1)
