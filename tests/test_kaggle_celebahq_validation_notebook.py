from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dire_modal/kaggle_celebahq_validation.py"
NOTEBOOK = ROOT / "platforms/kaggle/celebahq-path-index.ipynb"


def test_notebook_embeds_validation_module_and_all_code_cells_parse() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    embedded = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert MODULE.read_text(encoding="utf-8") in embedded
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")


def test_stage_three_runner_uses_reported_summary_fallback() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    runner = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "Stage 3 runner — use the just-finished" in "".join(cell.get("source", []))
    )
    assert "/kaggle/working/celebahq-integrity-81dcda99aeb997b25b82-summary.json" in runner
    assert "integrity_result[\"summary_json\"]" in runner
    assert "output_dir=Path(\"/kaggle/working\")" in runner
