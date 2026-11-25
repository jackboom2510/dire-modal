from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "docs/README.md",
    ROOT / "docs/agent-docs-and-skills.md",
    ROOT / "docs/history/00-index.md",
    ROOT / "docs/history/01-modal-dataset-runs.md",
    ROOT / "docs/history/90-run-record-template.md",
    ROOT / "AGENTS.md",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DOC_ID_RE = re.compile(r"\b(?:DOC|HIST)-\d{3}\b|\b(?:ADR|SCI|EXP|OPS)-\d{3}\b")


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
    assert "HIST-001" in ids
    assert "history/01-modal-dataset-runs.md" in index
    assert "history/90-run-record-template.md" in index


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
