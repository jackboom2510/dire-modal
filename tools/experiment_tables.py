#!/usr/bin/env python3
"""Render Table 4 and Table 5 ledgers from Modal experiment outputs.

No paper numbers are copied or filled in. By default, generation is refused until
all requested cells are present and have status=ran.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (5, 10, 20, 50)
GENERATORS = ("adm", "ddpm", "iddpm", "pndm", "sdv2")
REPRESENTATIONS = ("REC", "RGB", "RGB_DIRE", "DIRE")


def _load(path: str | Path) -> tuple[dict, str]:
    p = Path(path)
    raw = p.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _cell(row: dict | None, key: str) -> dict | None:
    if not row:
        return None
    out = (row.get("by_test_generator") or {}).get(key)
    if not out or out.get("status") != "ran":
        return None
    return out


def _metric_values(cell: dict | None) -> tuple[str, str, str, str]:
    if not cell:
        return ("—", "—", "—", "—")
    m = cell["metrics"]
    return (f"{100 * float(m['acc']):.1f}", f"{100 * float(m['ap']):.1f}",
            str(int(cell["n_real"])), str(int(cell["n_fake"])))


def build_tables(table4: dict | None, table5: dict | None, *, allow_partial: bool = False) -> dict:
    missing = []
    protocol_notes = []
    for ledger in (table4, table5):
        note = (ledger or {}).get("test_generator_scope_note")
        if note and note not in protocol_notes:
            protocol_notes.append(str(note))
    t4_rows = []
    if table4 is not None:
        by_steps = table4.get("by_steps") or {}
        for step in STEPS:
            result = by_steps.get(str(step))
            if not result or result.get("status") != "complete":
                missing.append(f"Table 4 S={step}")
            for gen in GENERATORS:
                cell = _cell(result, gen)
                if not cell:
                    missing.append(f"Table 4 S={step}, {gen}")
                t4_rows.append({"steps": step, "generator": gen,
                                "acc_percent": _metric_values(cell)[0],
                                "ap_percent": _metric_values(cell)[1],
                                "n_real": _metric_values(cell)[2],
                                "n_fake": _metric_values(cell)[3]})
    t5_rows = []
    if table5 is not None:
        by_rep = table5.get("by_representation") or {}
        for rep in REPRESENTATIONS:
            result = by_rep.get(rep)
            if not result or result.get("status") != "complete":
                missing.append(f"Table 5 {rep}")
            for gen in GENERATORS:
                cell = _cell(result, gen)
                if not cell:
                    missing.append(f"Table 5 {rep}, {gen}")
                vals = _metric_values(cell)
                t5_rows.append({"representation": rep, "generator": gen,
                                "acc_percent": vals[0], "ap_percent": vals[1],
                                "n_real": vals[2], "n_fake": vals[3]})
    if missing and not allow_partial:
        raise ValueError("chưa đủ cell để hoàn thiện bảng:\n  " + "\n  ".join(dict.fromkeys(missing)))
    return {"table4": t4_rows, "table5": t5_rows, "protocol_notes": protocol_notes,
            "missing": list(dict.fromkeys(missing)), "complete": not missing}


def _markdown(title: str, rows: list[dict], first: str) -> str:
    headers = [first, "Generator", "ACC (%)", "AP (%)", "n real", "n fake"]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in
                     ("steps" if first == "S" else "representation", "generator",
                      "acc_percent", "ap_percent", "n_real", "n_fake")) + " |")
    return "\n".join(lines)


def write_tables(result: dict, out: str | Path, sources: dict[str, str]) -> Path:
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    pieces = ["# Results from DIRE ablation ledgers", "",
              "ACC/AP are percentages. `—` marks missing cells; paper values are not inserted.", ""]
    for note in result.get("protocol_notes", []):
        pieces.extend([f"**Protocol note:** {note}", ""])
    if result["table4"]:
        pieces.append(_markdown("Table 4 — DDIM steps", result["table4"], "S"))
        pieces.append("")
        with (dest / "table4.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["table4"][0]))
            writer.writeheader(); writer.writerows(result["table4"])
    if result["table5"]:
        pieces.append(_markdown("Table 5 — Input representation", result["table5"], "Representation"))
        pieces.append("")
        with (dest / "table5.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["table5"][0]))
            writer.writeheader(); writer.writerows(result["table5"])
    pieces.extend(["## Provenance", "", "| Ledger | SHA-256 |", "|---|---|"])
    pieces.extend(f"| `{name}` | `{digest}` |" for name, digest in sources.items())
    pieces.append("")
    if result["missing"]:
        pieces.extend(["## Missing", "", *[f"- {x}" for x in result["missing"]], ""])
    if not result["complete"]:
        pieces.extend(["**Incomplete ledger — do not cite as completed paper table.**", ""])
    path = dest / "tables.md"
    path.write_text("\n".join(pieces), encoding="utf-8")
    (dest / "tables.json").write_text(
        json.dumps({**result, "sources": sources}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table4", help="Modal table4.json")
    p.add_argument("--table5", help="Modal table5.json")
    p.add_argument("--out", default="paper-tables")
    p.add_argument("--allow-partial", action="store_true", help="render — for missing cells; mark incomplete")
    args = p.parse_args(argv)
    if not args.table4 and not args.table5:
        p.error("cần --table4 và/hoặc --table5")
    tables, sources = {}, {}
    for name, path in (("table4", args.table4), ("table5", args.table5)):
        if path:
            tables[name], sources[path] = _load(path)
    result = build_tables(tables.get("table4"), tables.get("table5"), allow_partial=args.allow_partial)
    output = write_tables(result, args.out, sources)
    print(f"[xong] {output}" if result["complete"] else f"[partial] {output} · thiếu {len(result['missing'])} cell")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
