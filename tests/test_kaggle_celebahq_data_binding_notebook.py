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
    assert len(code_cells) == 4
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
    assert "The attached Kaggle code Dataset is outdated" in code
    assert "BUILDER_SOURCE_PATH" in code
    assert "src/dire_modal/kaggle_celebahq_data.py" in code
    assert "c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd" in code
    assert "bundle.train.records" in code
    assert "bundle.validation.records" in code
    assert "len(dataset)" in code
    assert "bundle.train[" not in code
    assert "bundle.validation[" not in code
    binding_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    binding = binding_cells[1]
    assert "cuda" not in code.lower()
    assert "Image.open" not in code
    assert "torch" not in binding.lower()
    assert "training_started': False" in code


def test_binding_notebook_discloses_metadata_fingerprint_is_skipped() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "metadata_fingerprint_check': 'not_performed'" in code
    assert "bundle.source_fingerprint" not in code
    assert "current_fingerprint = _source_fingerprint(" not in code
    assert "does not compare per-run metadata fingerprints" in markdown
    assert "Image.open" not in code
    assert "torch.load" not in code


def test_binding_notebook_checkpoint_stage_is_read_only_and_no_gpu() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    checkpoint = code_cells[2]
    assert "resnet50-19c8e357.pth" in checkpoint
    assert "hashlib.sha256()" in checkpoint
    assert "file_size_bytes" in checkpoint
    assert "CHECKPOINT_REPORT_PATH" in checkpoint
    assert "checkpoint_deserialized': False" in checkpoint
    assert "gpu_workload_started': False" in checkpoint
    assert "training_started': False" in checkpoint
    assert "torch.load" not in checkpoint
    assert "cuda" not in checkpoint.lower()
    assert "Image.open" not in checkpoint


def test_training_plan_cell_is_cpu_only_and_does_not_call_trainer() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    plan_cell = code_cells[-1]
    assert "build_training_plan(" in plan_cell
    assert "shutil.disk_usage(working).free" in plan_cell
    assert "celebahq-training-plan.json" in plan_cell
    assert "training_started': False" in plan_cell
    assert "gpu_workload_started': False" in plan_cell
    assert "run_celebahq_seed_suite(" not in plan_cell
    assert "torch" not in plan_cell.lower()
    assert "Image.open" not in plan_cell
