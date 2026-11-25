"""Read-only loaders for the private DIRE research prototype.

All paths are fixed within this checkout: uploaded content is never used as a
path, and this module performs no network, Modal, or write operations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GALLERY = ROOT / "modal-runs" / "gallery"
TABLE4_PATH = (
    ROOT / "modal-runs/experiments/experiments/table4/table4-lsun-pilot"
    / "subsets/per-cell-000200/table4.json"
)
TABLE5_ROOT = ROOT / "modal-runs/experiments/experiments/table5/table5-lsun-pilot"
FIGURE4_ROOT = ROOT / "modal-runs/experiments/experiments/figure4"
REPORT_PATH = ROOT / "modal-runs/runs/report.json"
EVAL_PATH = ROOT / "modal-runs/runs/eval/gen-adm/eval.json"
PAPER_TABLE2_PATH = ROOT / "src/dire_modal/paper_table2.tsv"


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a checked-in JSON artifact; missing or malformed data stays missing."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def paper_table2() -> list[dict[str, str]]:
    """Paper-reported reference values, loaded from the project's sourced TSV."""
    try:
        with PAPER_TABLE2_PATH.open(encoding="utf-8", newline="") as stream:
            content = (line for line in stream if line.strip() and not line.lstrip().startswith("#"))
            return list(csv.DictReader(content, delimiter="\t"))
    except OSError:
        return []


def table4() -> dict[str, Any] | None:
    return read_json(TABLE4_PATH)


def table4_rows(ledger: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten recorded per-generator test metrics; never impute missing cells."""
    if not ledger:
        return []
    rows: list[dict[str, Any]] = []
    for steps, result in sorted(
        ledger.get("by_steps", {}).items(), key=lambda item: int(item[0])
    ):
        for generator in ledger.get("test_generators", []):
            cell = result.get("by_test_generator", {}).get(generator, {})
            metrics = cell.get("metrics", {})
            rows.append({
                "S": int(steps),
                "Generator": generator,
                "Status": cell.get("status", "missing"),
                "n": metrics.get("n"),
                "AP": metrics.get("ap"),
                "AUC": metrics.get("auc"),
                "ACC (threshold 0.5)": metrics.get("acc"),
            })
    for steps in ledger.get("missing_steps", []):
        for generator in ledger.get("test_generators", []):
            rows.append({
                "S": int(steps), "Generator": generator, "Status": "not run",
                "n": None, "AP": None, "AUC": None,
                "ACC (threshold 0.5)": None,
            })
    return rows


def gallery_samples() -> list[dict[str, Any]]:
    """Only return gallery records with both corresponding image assets."""
    samples = []
    if not GALLERY.is_dir():
        return samples
    for meta_path in sorted(GALLERY.glob("*.json")):
        meta = read_json(meta_path)
        if not meta:
            continue
        name = meta.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            continue
        orig = GALLERY / f"{name}_orig.png"
        diff = GALLERY / f"{name}_diff.png"
        if orig.is_file() and diff.is_file():
            samples.append({"meta": meta, "orig": orig, "diff": diff})
    return samples


def table5_status() -> dict[str, Any]:
    protocol = read_json(TABLE5_ROOT / "inputs/_protocol.json")
    ledgers = list(TABLE5_ROOT.rglob("table5.json")) if TABLE5_ROOT.exists() else []
    return {"protocol": protocol, "ledger_count": len(ledgers), "ledgers": ledgers}


def figure4_status() -> dict[str, Any]:
    ledgers = list(FIGURE4_ROOT.rglob("figure4.json")) if FIGURE4_ROOT.exists() else []
    return {"exists": FIGURE4_ROOT.exists(), "ledger_count": len(ledgers), "ledgers": ledgers}


def pipeline_status() -> list[dict[str, str]]:
    report = read_json(REPORT_PATH)
    evaluation = read_json(EVAL_PATH)
    ledger = table4()
    return [
        {"Stage": "R1+R2 report", "Evidence": str(REPORT_PATH.relative_to(ROOT)),
         "Status": "recorded" if report else "missing"},
        {"Stage": "ADM evaluation", "Evidence": str(EVAL_PATH.relative_to(ROOT)),
         "Status": "recorded" if evaluation else "missing"},
        {"Stage": "Table 4 pilot", "Evidence": str(TABLE4_PATH.relative_to(ROOT)),
         "Status": "partial" if ledger and ledger.get("missing_steps") else "recorded" if ledger else "missing"},
        {"Stage": "Table 5 pilot", "Evidence": "inputs protocol; no result ledger",
         "Status": "inputs only" if table5_status()["protocol"] else "missing"},
        {"Stage": "Figure 4 pilot", "Evidence": "no result ledger found",
         "Status": "not run" if not figure4_status()["ledger_count"] else "recorded"},
    ]
