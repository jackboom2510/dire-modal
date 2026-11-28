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
    assert len(code_cells) == 6
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"{NOTEBOOK}:cell-{index}")


def test_data_binding_notebook_checks_locked_counts_without_iterating_images() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    code = "\n".join(code_cells)
    stage5_code = "\n".join(code_cells[:2])
    assert "build_celebahq_datasets(" in stage5_code
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
    assert "cuda" not in stage5_code.lower()
    assert "Image.open" not in code
    assert "torch" not in binding.lower()
    assert "training_started': False" in code


def test_binding_notebook_discloses_metadata_fingerprint_is_skipped() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    code = "\n".join(code_cells[:3])
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
    plan_cell = code_cells[-3]
    assert "build_training_plan(" in plan_cell
    assert "expected_execution_schedule" in plan_cell
    assert "parallel_independent_seeds" in plan_cell
    assert "waves': [[3407, 3408], [3409]]" in plan_cell
    assert "attached trainer is stale" in plan_cell
    assert "'execution_schedule': expected_execution_schedule" in plan_cell
    assert "shutil.disk_usage(working).free" in plan_cell
    assert "celebahq-training-plan.json" in plan_cell
    assert "training_started': False" in plan_cell
    assert "gpu_workload_started': False" in plan_cell
    assert "run_celebahq_seed_suite(" not in plan_cell
    assert "torch" not in plan_cell.lower()
    assert "Image.open" not in plan_cell


def test_two_t4_metadata_cell_is_read_only_and_not_training() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    gpu_cell = code_cells[-2]
    assert "celebahq-gpu-metadata-report.json" in gpu_cell
    assert "exactly two distinct NVIDIA Tesla T4 devices" in gpu_cell
    assert "nvidia-smi" in gpu_cell
    assert "gpu_workload_started': False" in gpu_cell
    assert "training_started': False" in gpu_cell
    assert "import torch" not in gpu_cell
    assert "run_celebahq_parallel_seed_schedule" not in gpu_cell
    assert "Image.open" not in gpu_cell


def test_stage9_cell_validates_gates_then_calls_parallel_training_runner() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    launch_cell = code_cells[-1]
    assert "run_celebahq_parallel_seed_schedule(" in launch_cell
    assert "execution_schedule" in launch_cell
    assert "celebahq-gpu-metadata-report.json" in launch_cell
    assert "free_bytes < required_bytes" in launch_cell
    assert "resume=True" in launch_cell
    assert "celebahq-parallel-run-report.json" in launch_cell
    assert "Stage 9 gates pass. Launching" in launch_cell
