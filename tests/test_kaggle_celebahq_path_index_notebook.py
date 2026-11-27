from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dire_modal/kaggle_celebahq_tree.py"
NOTEBOOK = ROOT / "platforms/kaggle/celebahq-path-index.ipynb"
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_tree import REQUIRED_GROUPS  # noqa: E402


def _notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _fixture(root: Path) -> None:
    for split, generator in REQUIRED_GROUPS:
        leaf = root / split / "celebahq" / generator / generator
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / f"{split}-{generator}.png").write_bytes(b"fixture")


def test_notebook_embeds_current_adapter_and_all_code_cells_parse() -> None:
    notebook = _notebook()
    embedded_adapter = "".join(notebook["cells"][1]["source"])
    assert embedded_adapter == MODULE.read_text(encoding="utf-8")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")


def test_notebook_writes_index_outside_input_without_changing_fixture(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    output = tmp_path / "celebahq-path-index.json"
    _fixture(root)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    notebook = _notebook()
    namespace: dict[str, Any] = {}
    exec(compile("".join(notebook["cells"][1]["source"]), str(NOTEBOOK), "exec"), namespace)
    runner = "".join(notebook["cells"][2]["source"])
    runner = runner.replace(
        "Path('/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire')",
        f"Path({str(root)!r})",
    ).replace(
        "Path('/kaggle/working/celebahq-path-index.json')",
        f"Path({str(output)!r})",
    )
    exec(compile(runner, str(NOTEBOOK), "exec"), namespace)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["input_modified"] is False
    assert report["image_contents_read"] is False
    assert report["gpu_work_started"] is False
    assert report["archive_member_paths_verified"] is False
    assert report["validation_split_assigned"] is False
    assert len(report["records"]) == len(REQUIRED_GROUPS)
    assert all(row["locator_inferred"] for row in report["records"])
    assert all(row["member_locator_inferred"].endswith(".png") for row in report["records"])
    assert output.is_relative_to(tmp_path) and not output.is_relative_to(root)
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
