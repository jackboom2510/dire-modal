from __future__ import annotations

import ast
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dire_modal/kaggle_celebahq_integrity.py"
NOTEBOOK = ROOT / "platforms/kaggle/celebahq-path-index.ipynb"
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_tree import REQUIRED_GROUPS, scan_celebahq_tree  # noqa: E402


def _fixture(root: Path) -> list[Any]:
    for i, (split, generator) in enumerate(REQUIRED_GROUPS):
        leaf = root / split / "celebahq" / generator / generator
        leaf.mkdir(parents=True, exist_ok=True)
        image_path = leaf / f"sample-{split}-{generator}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = BytesIO()
        Image.new("RGB", (4, 3), (i, 10, 20)).save(buffer, format="PNG")
        image_path.write_bytes(buffer.getvalue())
    return scan_celebahq_tree(root)


def test_notebook_embeds_integrity_module_and_code_cells_parse() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert "".join(notebook["cells"][3]["source"]) == MODULE.read_text(encoding="utf-8")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")


def test_notebook_integrity_cell_uses_existing_index_and_writes_outside_input(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    workdir = tmp_path / "working"
    records = _fixture(root)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "Path": Path,
        "DIRE_ROOT": root,
        "records": records,
    }
    exec(compile("".join(notebook["cells"][3]["source"]), str(NOTEBOOK), "exec"), namespace)
    runner = "".join(notebook["cells"][4]["source"]).replace(
        "Path('/kaggle/working')",
        f"Path({str(workdir)!r})",
    )
    exec(compile(runner, str(NOTEBOOK), "exec"), namespace)

    result = namespace["integrity_result"]
    assert result["status"] == "pass"
    assert result["records_in_checkpoint"] == len(REQUIRED_GROUPS)
    assert Path(result["checkpoint_sqlite"]).is_relative_to(workdir)
    assert Path(result["file_manifest_jsonl"]).is_relative_to(workdir)
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
