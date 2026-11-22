#!/usr/bin/env python3
"""Read-only join of a relabel audit log with maps/dire.json.

Example::

    python tools/audit_relabel_conflicts.py relabel-log.json dire-index.json --limit 5

Prints the source and destination index records for unresolved relabel entries.
It does not read or modify map arrays, and never writes to either input file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ITEM_FIELDS = (
    "path",
    "source",
    "generator",
    "split",
    "label",
    "reused",
    "relabelled_from",
)
INDEX_FIELDS = ("n", "n_new", "n_reused", "steps", "steps_mixed", "scale", "ckpt_sha256")
RUN_FIELDS = ("created", "steps", "ckpt_sha256", "n", "n_new", "n_reused", "generators", "splits")


def audit_conflicts(report: dict[str, Any], index: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    """Join unresolved report paths to their records in ``dire.json``.

    ``limit=0`` includes every unresolved entry. Missing index records are
    represented as ``None`` rather than guessed or silently omitted.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")

    plan = report.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("unresolved"), list):
        raise ValueError("report must contain plan.unresolved as a list")
    items = index.get("items")
    if not isinstance(items, list):
        raise ValueError("dire.json must contain items as a list")

    by_path = {
        item["path"]: item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    unresolved = plan["unresolved"]
    selected = unresolved if limit == 0 else unresolved[:limit]
    conflicts = []
    for conflict in selected:
        if not isinstance(conflict, dict):
            continue
        src_path, dst_path = conflict.get("rel_src"), conflict.get("rel_dst")
        src_item, dst_item = by_path.get(src_path), by_path.get(dst_path)
        conflicts.append(
            {
                "rel_src": src_path,
                "rel_dst": dst_path,
                "problem": conflict.get("problem"),
                "source_item": ({k: src_item.get(k) for k in ITEM_FIELDS} if src_item else None),
                "destination_item": ({k: dst_item.get(k) for k in ITEM_FIELDS} if dst_item else None),
            }
        )

    return {
        "n_unresolved": len(unresolved),
        "n_shown": len(conflicts),
        "index_summary": {k: index.get(k) for k in INDEX_FIELDS},
        "runs": [
            {k: run.get(k) for k in RUN_FIELDS}
            for run in index.get("runs", [])
            if isinstance(run, dict)
        ],
        "conflicts": conflicts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relabel_log", type=Path, help="JSON file saved by ::relabel")
    parser.add_argument("dire_index", type=Path, help="downloaded maps/dire.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="number of unresolved entries to print; 0 prints all (default: 5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    try:
        report = json.loads(args.relabel_log.read_text(encoding="utf-8"))
        index = json.loads(args.dire_index.read_text(encoding="utf-8"))
        result = audit_conflicts(report, index, limit=args.limit)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
