from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "docs/README.md",
    ROOT / "docs/agent-docs-and-skills.md",
    ROOT / "docs/adr/adr-0007-kaggle-research-program.md",
    ROOT / "GLOSSARY.md",
    ROOT / "docs/plans/README.md",
    ROOT / "docs/plans/research/full-paper-reproduction-plan.md",
    ROOT / "docs/plans/research/figures-plan.md",
    ROOT / "docs/plans/research/incremental-paper-subsets.md",
    ROOT / "docs/kaggle-celebahq-progress.md",
    ROOT / "docs/plans/demos/demo-hf-spaces-plan.md",
    ROOT / "docs/plans/demos/demo-modal-plan.md",
    ROOT / "docs/plans/demos/demo-web-plan.md",
    ROOT / "docs/plans/presentations/manim-video-plan.md",
    ROOT / "docs/history/00-index.md",
    ROOT / "docs/history/01-modal-dataset-runs.md",
    ROOT / "docs/history/90-run-record-template.md",
    ROOT / "docs/workflows/README.md",
    ROOT / "docs/workflows/reproduction.md",
    ROOT / "docs/workflows/operations.md",
    ROOT / "docs/workflows/delivery.md",
    ROOT / "AGENTS.md",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DOC_ID_RE = re.compile(r"\b(?:DOC|HIST|WF)-\d{3}\b|\b(?:ADR|SCI|EXP|OPS)-\d{3}\b")


def test_new_doc_routes_have_no_broken_local_links():
    broken: list[str] = []
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            resolved = (doc.parent / path).resolve()
            if not resolved.exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "broken local documentation links:\n" + "\n".join(broken)


def test_document_catalog_has_stable_ids_and_history_routes():
    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    ids = DOC_ID_RE.findall(index)
    assert "DOC-000" in ids
    assert "DOC-020" in ids
    assert "HIST-001" in ids
    assert {"WF-000", "WF-001", "WF-002", "WF-003"}.issubset(set(ids))
    assert "history/01-modal-dataset-runs.md" in index
    assert "history/90-run-record-template.md" in index
    assert "workflows/README.md" in index
    assert "workflows/reproduction.md" in index
    assert "workflows/operations.md" in index
    assert "workflows/delivery.md" in index
    assert "adr/adr-0001-storage.md" in index
    assert "adr/adr-0007-kaggle-research-program.md" in index
    assert "GLOSSARY.md" in index
    assert "plans/README.md" in index
    assert "plans/research/full-paper-reproduction-plan.md" in index
    assert "plans/research/incremental-paper-subsets.md" in index
    assert "EXP-008" in index
    assert "kaggle-celebahq-progress.md" in index
    assert "plans/demos/demo-web-plan.md" in index


def test_architecture_decisions_live_in_the_adr_directory():
    adr_dir = ROOT / "docs" / "adr"
    assert {p.name for p in adr_dir.glob("adr-*.md")} == {
        "adr-0001-storage.md",
        "adr-0002-iaas.md",
        "adr-0003-demo-training.md",
        "adr-0004-retrain-for-reconciliation.md",
        "adr-0005-local-first-modal-wrapper.md",
        "adr-0006-reproduction-levels.md",
        "adr-0007-kaggle-research-program.md",
    }
    assert not list((ROOT / "docs").glob("adr-*.md"))


def test_topic_plans_are_grouped_without_moving_the_root_agent_plan():
    plans = ROOT / "docs" / "plans"
    expected = {
        "research/full-paper-reproduction-plan.md",
        "research/figures-plan.md",
        "research/incremental-paper-subsets.md",
        "demos/demo-hf-spaces-plan.md",
        "demos/demo-modal-plan.md",
        "demos/demo-web-plan.md",
        "presentations/manim-video-plan.md",
    }
    actual = {
        p.relative_to(plans).as_posix()
        for p in plans.rglob("*.md")
        if p.name != "README.md"
    }
    assert actual == expected
    assert (ROOT / "plan.md").is_file()
    assert not list((ROOT / "docs").glob("*-plan.md"))
    assert not (ROOT / "docs" / "incremental-paper-subsets.md").exists()


def test_history_snapshot_stays_linked_to_recorded_source_counts():
    import json

    history = (ROOT / "docs/history/01-modal-dataset-runs.md").read_text(encoding="utf-8")
    root_report = json.loads((ROOT / "modal-runs/runs/report.json").read_text(encoding="utf-8"))
    rebuild = json.loads(
        (ROOT / "modal-runs/runs/rebuild/rebuild-2026-10-14/report.json").read_text(encoding="utf-8")
    )
    source = root_report["data"]
    assert f"{source['n_real']:,}".replace(",", ".") in history
    assert f"{source['n_fake']:,}".replace(",", ".") in history
    assert rebuild["dire"]["n_distinct_maps"] == 23868
    assert "23.868 map phân biệt" in history
    assert "5.900" in history
