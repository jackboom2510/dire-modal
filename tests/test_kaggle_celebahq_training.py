from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from dire_modal.kaggle_celebahq_training import (
    CelebAHQTrainingConfig,
    average_precision,
    build_training_plan,
    canonical_id_sha256,
    optimizer_steps_per_epoch,
    validate_two_t4_gpu_report,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dire_modal/kaggle_celebahq_training.py"


class _Record(NamedTuple):
    label: int
    relative_path: str


class _Sized:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _approved_bundle():
    train_records = [
        *(_Record(0, f"train/real/{index:05}.png") for index in range(25_200)),
        *(_Record(1, f"train/sdv2/{index:05}.png") for index in range(36_000)),
    ]
    validation_records = [
        *(_Record(0, f"train/real/val-{index:04}.png") for index in range(2_800)),
        *(_Record(1, f"train/sdv2/val-{index:04}.png") for index in range(4_000)),
    ]
    test_by_generator = {
        name: _Sized(size)
        for name, size in {
            "real": 1_000, "dalle2": 500, "if": 1_000,
            "midjourney": 100, "stargan": 1_000, "sdv2": 1_000,
        }.items()
    }
    return SimpleNamespace(
        train=SimpleNamespace(root=Path("/data/dire"), records=train_records),
        validation=SimpleNamespace(root=Path("/data/dire"), records=validation_records),
        test_by_generator=test_by_generator,
        validation_id_list_sha256="c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd",
    )


def test_approved_training_config_defaults_are_locked_but_amp_is_disabled() -> None:
    config = CelebAHQTrainingConfig()
    assert config.seeds == (3407, 3408, 3409)
    assert config.microbatch_size == 32
    assert config.accumulation_steps == 2
    assert config.max_epochs == 20
    assert config.patience == 5
    assert config.learning_rate == 1e-4
    assert config.betas == (0.9, 0.999)
    assert config.weight_decay == 0
    assert config.positive_class_weight == 0.7
    assert config.use_amp is False


def test_training_config_rejects_amp_until_protocol_is_changed() -> None:
    with pytest.raises(ValueError, match="FP32"):
        CelebAHQTrainingConfig(use_amp=True)


def test_plan_checks_locked_inventory_and_reports_updates_without_training() -> None:
    plan = build_training_plan(_approved_bundle())
    assert plan["status"] == "ready_for_review"
    assert plan["inventory"]["train_records"] == 61_200
    assert plan["inventory"]["validation_records"] == 6_800
    assert plan["microbatches_per_epoch"] == 1_913
    assert plan["optimizer_steps_per_epoch"] == 957
    assert plan["max_optimizer_steps_per_seed"] == 19_140
    assert plan["max_steps_all_seeds"] == 57_420
    assert plan["execution_schedule"] == {
        "mode": "parallel_independent_seeds",
        "gpu_count": 2,
        "waves": [[3407, 3408], [3409]],
        "per_seed_gpu_visibility": 1,
        "effective_batch_per_seed": 64,
    }
    assert plan["training_started"] is False
    assert plan["gpu_workload_started"] is False
    assert plan["image_contents_read"] is False


def test_parallel_gpu_gate_accepts_exactly_two_distinct_t4s() -> None:
    report = {
        "status": "pass",
        "devices": [
            {"name": "Tesla T4", "uuid": "GPU-a"},
            {"name": "Tesla T4", "uuid": "GPU-b"},
        ],
    }
    assert validate_two_t4_gpu_report(report) == ("GPU-a", "GPU-b")


@pytest.mark.parametrize("devices", [
    [{"name": "Tesla T4", "uuid": "GPU-a"}],
    [{"name": "Tesla T4", "uuid": "GPU-a"}, {"name": "Tesla T4", "uuid": "GPU-a"}],
    [{"name": "Tesla T4", "uuid": "GPU-a"}, {"name": "Tesla P100", "uuid": "GPU-b"}],
])
def test_parallel_gpu_gate_rejects_wrong_inventory(devices) -> None:
    with pytest.raises(ValueError):
        validate_two_t4_gpu_report({"status": "pass", "devices": devices})


def test_optimizer_step_count_accounts_for_partial_microbatch_and_accumulation() -> None:
    assert optimizer_steps_per_epoch(61_200, 32, 2) == 957
    assert optimizer_steps_per_epoch(64, 32, 2) == 1
    assert optimizer_steps_per_epoch(96, 32, 2) == 2
    assert optimizer_steps_per_epoch(0, 32, 2) == 0


def test_average_precision_matches_non_interpolated_tie_aware_definition() -> None:
    assert average_precision([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.4]) == pytest.approx(5 / 6)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_average_precision_rejects_invalid_or_single_class_inputs() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        average_precision([0, 1], [0.5])
    with pytest.raises(ValueError, match="without positive"):
        average_precision([0, 0], [0.1, 0.2])


def test_id_hash_uses_sorted_relative_paths_and_labels_only() -> None:
    records = [
        SimpleNamespace(label=1, relative_path="z.png", mtime_ns=100),
        SimpleNamespace(label=0, relative_path="b.png", mtime_ns=200),
    ]
    reordered_with_new_mount_times = [
        SimpleNamespace(label=0, relative_path="b.png", mtime_ns=999),
        SimpleNamespace(label=1, relative_path="z.png", mtime_ns=888),
    ]
    assert canonical_id_sha256(records) == canonical_id_sha256(reordered_with_new_mount_times)


def test_training_module_parses_without_import_time_training_or_torch_import() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in top_level_imports)
    assert all(not isinstance(node, ast.Call) for node in tree.body)
