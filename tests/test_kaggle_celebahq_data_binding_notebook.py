from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "platforms/kaggle/celebahq-data-binding.ipynb"


def test_data_binding_notebook_code_cells_parse() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 2
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"{NOTEBOOK}:cell-{index}")


def test_data_binding_notebook_checks_locked_counts_without_iterating_images() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "build_celebahq_datasets(" in code
    assert "sys.path.insert(0, str(CODE_DATASET_ROOT))" in code
    assert "kaggle_celebahq_data.py" in code
    assert "builder_kwargs['progress_callback'] = progress" in code
    assert "c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd" in code
    assert "bundle.train.records" in code
    assert "bundle.validation.records" in code
    assert "len(dataset)" in code
    assert "bundle.train[" not in code
    assert "bundle.validation[" not in code
    assert "cuda" not in code.lower()
    assert "Image.open" not in code
    assert "training_started': False" in code
