"""Gated CelebA-HQ-SDv2 ResNet-50 training implementation for Kaggle.

Importing this module is CPU-only and does not load a model, inspect a GPU, read
images, or start training. Training happens only when ``run_seed_suite`` is
explicitly called. Its outputs must live outside the mounted input tree.

This is a research-extension trainer, not a paper-reproduction command. The
implementation consumes the Stage 5 lazy datasets and the audited ImageNet
checkpoint, trains on all approved training records, and uses validation AP for
checkpoint selection. Test-family evaluation is deliberately a later stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .kaggle_celebahq_data import CelebAHQDataBundle, CelebAHQDataset

EXPECTED_VALIDATION_ID_LIST_SHA256 = (
    "c635c38a5888ef4e0196423dd8f342f412dc15ea633c17d835330c740d7dc9fd"
)
EXPECTED_TRAIN_COUNTS = {0: 25_200, 1: 36_000}
EXPECTED_VALIDATION_COUNTS = {0: 2_800, 1: 4_000}
EXPECTED_TEST_COUNTS = {
    "real": 1_000,
    "dalle2": 500,
    "if": 1_000,
    "midjourney": 100,
    "stargan": 1_000,
    "sdv2": 1_000,
}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
CHECKPOINT_NAME = "resnet50-19c8e357.pth"
CHECKPOINT_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"


@dataclass(frozen=True)
class CelebAHQTrainingConfig:
    """User-approved plan defaults; configuration is serialized into every run."""

    seeds: tuple[int, ...] = (3407, 3408, 3409)
    microbatch_size: int = 32
    accumulation_steps: int = 2
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    positive_class_weight: float = 0.7
    num_workers: int = 2
    output_reserve_bytes: int = 2 * 1024**3
    resize_short_edge: int = 256
    crop_size: int = 224
    horizontal_flip_probability: float = 0.5
    validation_acc_threshold: float = 0.5
    use_amp: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty sequence of unique integers")
        if any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("every seed must be an integer")
        if self.microbatch_size < 1 or self.accumulation_steps < 1:
            raise ValueError("microbatch_size and accumulation_steps must be positive")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("Adam betas must contain two values in [0, 1)")
        if self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive")
        if self.num_workers < 0 or self.output_reserve_bytes < 0:
            raise ValueError("num_workers and output_reserve_bytes must be non-negative")
        if self.resize_short_edge < self.crop_size or self.crop_size < 1:
            raise ValueError("resize_short_edge must be >= positive crop_size")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if self.use_amp:
            raise ValueError("the approved first run is FP32; AMP is not enabled in this protocol")


def optimizer_steps_per_epoch(
    n_examples: int, microbatch_size: int, accumulation_steps: int
) -> int:
    """Count updates with a short final accumulation window handled correctly."""
    if n_examples < 0 or microbatch_size < 1 or accumulation_steps < 1:
        raise ValueError("n_examples must be non-negative; batch and accumulation must be positive")
    microbatches = math.ceil(n_examples / microbatch_size)
    return math.ceil(microbatches / accumulation_steps)


def average_precision(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Non-interpolated binary AP, grouping equal scores as a single threshold."""
    labels = [int(value) for value in y_true]
    scores = [float(value) for value in y_score]
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("average_precision expects binary labels 0/1")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    n_positive = sum(labels)
    if n_positive == 0:
        raise ValueError("average_precision is undefined without positive examples")
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    true_positive = false_positive = 0
    previous_recall = 0.0
    result = 0.0
    i = 0
    while i < len(order):
        score = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == score:
            if labels[order[j]] == 1:
                true_positive += 1
            else:
                false_positive += 1
            j += 1
        recall = true_positive / n_positive
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return result


def canonical_id_sha256(records: Iterable[Any]) -> str:
    """Hash an ordered list of root-relative paths and labels, not file metadata."""
    items = [
        {"label": int(record.label), "path": str(record.relative_path)}
        for record in records
    ]
    items.sort(key=lambda item: (item["label"], item["path"]))
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bundle(bundle: CelebAHQDataBundle) -> dict[str, Any]:
    """Fail closed unless the Stage 5 bundle matches the locked CelebA-HQ split."""
    train_counts = {0: 0, 1: 0}
    validation_counts = {0: 0, 1: 0}
    for record in bundle.train.records:
        train_counts[int(record.label)] += 1
    for record in bundle.validation.records:
        validation_counts[int(record.label)] += 1
    if train_counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"training counts differ from approved split: {train_counts}")
    if validation_counts != EXPECTED_VALIDATION_COUNTS:
        raise ValueError(f"validation counts differ from approved split: {validation_counts}")
    if bundle.validation_id_list_sha256 != EXPECTED_VALIDATION_ID_LIST_SHA256:
        raise ValueError("validation ID-list hash differs from the approved Stage 3 v2 list")
    train_paths = {record.relative_path for record in bundle.train.records}
    validation_paths = {record.relative_path for record in bundle.validation.records}
    if train_paths & validation_paths:
        raise ValueError("train/validation root-relative path overlap detected")
    test_counts = {name: len(dataset) for name, dataset in bundle.test_by_generator.items()}
    if test_counts != EXPECTED_TEST_COUNTS:
        raise ValueError(f"test counts differ from approved inventory: {test_counts}")
    return {
        "train_count_by_label": {str(k): v for k, v in sorted(train_counts.items())},
        "validation_count_by_label": {str(k): v for k, v in sorted(validation_counts.items())},
        "test_count_by_generator": test_counts,
        "train_records": len(bundle.train.records),
        "validation_records": len(bundle.validation.records),
        "validation_id_list_sha256": bundle.validation_id_list_sha256,
        "train_id_list_sha256": canonical_id_sha256(bundle.train.records),
    }


def build_training_plan(
    bundle: CelebAHQDataBundle,
    *,
    config: CelebAHQTrainingConfig = CelebAHQTrainingConfig(),
    weights_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a CPU-only, no-write plan; it never opens image files or loads weights."""
    inventory = validate_bundle(bundle)
    n_train = inventory["train_records"]
    microbatches = math.ceil(n_train / config.microbatch_size)
    steps = optimizer_steps_per_epoch(
        n_train, config.microbatch_size, config.accumulation_steps
    )
    return {
        "status": "ready_for_review",
        "protocol": "celebahq-content-disjoint-exact-count-hash-rank-v2",
        "training_config": asdict(config),
        "inventory": inventory,
        "microbatches_per_epoch": microbatches,
        "optimizer_steps_per_epoch": steps,
        "max_optimizer_steps_per_seed": steps * config.max_epochs,
        "max_steps_all_seeds": steps * config.max_epochs * len(config.seeds),
        "weights_report_path": str(Path(weights_report_path)) if weights_report_path else None,
        "test_data_used_for_selection": False,
        "image_contents_read": False,
        "checkpoint_deserialized": False,
        "gpu_workload_started": False,
        "training_started": False,
    }


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_output_root(output_root: Path, input_roots: Iterable[Path]) -> None:
    resolved = output_root.expanduser().resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and _inside(resolved, kaggle_input.resolve()):
        raise ValueError(f"training output must be outside read-only Kaggle input: {resolved}")
    for root in input_roots:
        if _inside(resolved, root.resolve()):
            raise ValueError(f"training output is inside mounted input root {root}: {resolved}")


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_report(report_path: str | Path, weights_path: str | Path) -> dict[str, Any]:
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    weights_file = Path(weights_path).expanduser().resolve()
    if report.get("status") != "pass":
        raise ValueError("Stage 6 checkpoint report is not a pass")
    if Path(report.get("checkpoint_path", "")).expanduser().resolve() != weights_file:
        raise ValueError("Stage 6 report refers to a different ImageNet checkpoint path")
    if report.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("Stage 6 report SHA-256 differs from the reviewed checkpoint digest")
    if weights_file.name != CHECKPOINT_NAME:
        raise ValueError(f"expected audited checkpoint filename {CHECKPOINT_NAME!r}, got {weights_file.name!r}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"ImageNet weights file not found: {weights_file}")
    actual_sha256 = _sha256_file(weights_file)
    if actual_sha256 != CHECKPOINT_SHA256:
        raise ValueError("current ImageNet checkpoint bytes differ from the Stage 6 audited digest")
    return {
        "path": weights_file,
        "sha256": actual_sha256,
        "size_bytes": weights_file.stat().st_size,
        "report_path": report_file,
    }


def _torch_and_torchvision():
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import resnet50
    except Exception as exc:  # pragma: no cover - Kaggle provides these packages
        raise RuntimeError("PyTorch and torchvision are required for the explicit training call") from exc
    if "weights_only" not in __import__("inspect").signature(torch.load).parameters:
        raise RuntimeError("This trainer requires torch.load(weights_only=True) support")
    return torch, transforms, resnet50


def _make_model(torch: Any, resnet50: Any, weights_path: Path, seed: int) -> Any:
    torch.manual_seed(seed)
    model = resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("ImageNet checkpoint is not a non-empty state_dict mapping")
    model.load_state_dict(state_dict, strict=True)
    model.fc = torch.nn.Linear(2048, 1)
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(model.fc.bias)
    return model


def _make_transforms(transforms: Any, config: CelebAHQTrainingConfig) -> tuple[Any, Any]:
    interpolation = transforms.InterpolationMode.BILINEAR
    normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    train_transform = transforms.Compose([
        transforms.Resize(config.resize_short_edge, interpolation=interpolation, antialias=True),
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_probability),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(config.resize_short_edge, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def _seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:  # pragma: no cover - PyTorch normally depends on NumPy
        pass


def _make_loaders(
    torch: Any,
    transforms: Any,
    bundle: CelebAHQDataBundle,
    config: CelebAHQTrainingConfig,
    seed: int,
) -> tuple[Any, Any]:
    train_transform, eval_transform = _make_transforms(transforms, config)
    train_dataset = CelebAHQDataset(
        bundle.train.root, bundle.train.records, transform=train_transform
    )
    validation_dataset = CelebAHQDataset(
        bundle.validation.root, bundle.validation.records, transform=eval_transform
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    validation_generator = torch.Generator()
    validation_generator.manual_seed(seed + 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.microbatch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
        worker_init_fn=_seed_worker,
        persistent_workers=False,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.microbatch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=validation_generator,
        worker_init_fn=_seed_worker,
        persistent_workers=False,
    )
    return train_loader, validation_loader


def _set_determinism(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _run_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_sum_fn: Any,
    device: Any,
    accumulation_steps: int,
) -> tuple[float, int, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_examples = 0
    updates = 0
    group_examples = 0
    total_batches = len(loader)
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True).reshape(-1)
        logits = model(images).reshape(-1)
        loss_sum = loss_sum_fn(logits, labels)
        loss_sum.backward()
        batch_count = int(labels.numel())
        total_loss_sum += float(loss_sum.detach().item())
        total_examples += batch_count
        group_examples += batch_count
        if batch_index % accumulation_steps == 0 or batch_index == total_batches:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_examples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_examples = 0
            updates += 1
    return total_loss_sum / total_examples, total_examples, updates


def _evaluate_validation(torch: Any, model: Any, loader: Any, device: Any, threshold: float) -> dict[str, float]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True)).reshape(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            labels_list = labels.to(dtype=torch.int64).reshape(-1).tolist()
            scores_all.extend(float(score) for score in scores)
            labels_all.extend(int(label) for label in labels_list)
    ap = average_precision(labels_all, scores_all)
    correct = sum(
        int((score >= threshold) == bool(label))
        for score, label in zip(scores_all, labels_all)
    )
    return {"ap": ap, "acc_at_0_5": correct / len(labels_all)}


def _atomic_torch_save(torch: Any, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _config_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _best_checkpoint_is_consistent(
    torch: Any,
    path: Path,
    *,
    seed: int,
    epoch: int,
    config_sha256: str,
    validation_id_list_sha256: str,
    weights_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return (
        isinstance(state, Mapping)
        and int(state.get("seed", -1)) == seed
        and int(state.get("epoch", -1)) == epoch
        and state.get("config_sha256") == config_sha256
        and state.get("validation_id_list_sha256") == validation_id_list_sha256
        and state.get("weights_sha256") == weights_sha256
    )


def run_celebahq_seed_suite(
    bundle: CelebAHQDataBundle,
    *,
    checkpoint_report_path: str | Path,
    output_root: str | Path = "/kaggle/working/celebahq-run-v2",
    config: CelebAHQTrainingConfig = CelebAHQTrainingConfig(),
    resume: bool = True,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Run all configured seeds sequentially; caller must explicitly invoke this.

    This function is the only entry point that trains. It requires a visible
    CUDA device, the audited checkpoint, a passing Stage 5 bundle, and free disk
    space equal to the configured reserve. It never evaluates test datasets.
    """
    torch, transforms, resnet50 = _torch_and_torchvision()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("approved training requires one visible Kaggle CUDA GPU")
    device = torch.device("cuda:0")
    inventory = validate_bundle(bundle)
    checkpoint_report = Path(checkpoint_report_path).expanduser().resolve()
    checkpoint_meta = json.loads(checkpoint_report.read_text(encoding="utf-8"))
    weights_path = Path(checkpoint_meta.get("checkpoint_path", "")).expanduser().resolve()
    audited_weights = _load_checkpoint_report(checkpoint_report, weights_path)
    root = Path(output_root).expanduser().resolve()
    _validate_output_root(root, [bundle.train.root, bundle.validation.root])
    suite_reserve = config.output_reserve_bytes * len(config.seeds)
    if _free_bytes(root.parent) < suite_reserve:
        raise OSError(
            f"free disk is below the configured suite reserve {suite_reserve} bytes "
            f"({config.output_reserve_bytes} bytes per seed) at {root.parent}"
        )
    root.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    for seed in config.seeds:
        plans.append(_run_one_seed(
            torch=torch,
            transforms=transforms,
            resnet50=resnet50,
            bundle=bundle,
            inventory=inventory,
            checkpoint_meta=audited_weights,
            output_root=root,
            config=config,
            seed=seed,
            resume=resume,
            device=device,
            progress_callback=progress_callback,
        ))
    return plans


def _run_one_seed(
    *,
    torch: Any,
    transforms: Any,
    resnet50: Any,
    bundle: CelebAHQDataBundle,
    inventory: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    output_root: Path,
    config: CelebAHQTrainingConfig,
    seed: int,
    resume: bool,
    device: Any,
    progress_callback: Any | None,
) -> dict[str, Any]:
    _set_determinism(torch, seed)
    run_dir = output_root / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / "run.json"
    last_path = run_dir / "last_resume.pt"
    best_path = run_dir / "best_model.pt"
    config_data = {"training": asdict(config), "seed": seed}
    config_hash = _config_digest(config_data)
    run_metadata = {
        "schema_version": 1,
        "status": "running",
        "seed": seed,
        "config": config_data,
        "config_sha256": config_hash,
        "inventory": inventory,
        "weights": {
            "path": str(checkpoint_meta["path"]),
            "size_bytes": checkpoint_meta["size_bytes"],
            "sha256": checkpoint_meta["sha256"],
        },
        "optimizer": "Adam",
        "learning_rate_schedule": "constant",
        "precision": "float32",
        "test_data_used_for_selection": False,
    }
    if last_path.exists():
        if not resume:
            raise FileExistsError(f"resume checkpoint exists but resume=False: {last_path}")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state.get("config_sha256") != config_hash:
            raise ValueError("resume checkpoint training config differs; refusing incompatible resume")
        if state.get("train_id_list_sha256") != inventory["train_id_list_sha256"]:
            raise ValueError("resume checkpoint train IDs differ; refusing incompatible resume")
        if state.get("validation_id_list_sha256") != inventory["validation_id_list_sha256"]:
            raise ValueError("resume checkpoint validation IDs differ; refusing incompatible resume")
        if state.get("weights_sha256") != checkpoint_meta["sha256"]:
            raise ValueError("resume checkpoint ImageNet weights digest differs")
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(2048, 1)
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, betas=config.betas,
            weight_decay=config.weight_decay,
        )
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["completed_epoch"]) + 1
        best_val_ap = float(state["best_val_ap"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        completed_epoch = int(state["completed_epoch"])
        if best_epoch == 0 or best_epoch > completed_epoch:
            raise ValueError("resume checkpoint has invalid best_epoch metadata")
        best_matches = _best_checkpoint_is_consistent(
            torch,
            best_path,
            seed=seed,
            epoch=best_epoch,
            config_sha256=config_hash,
            validation_id_list_sha256=inventory["validation_id_list_sha256"],
            weights_sha256=checkpoint_meta["sha256"],
        )
        if not best_matches:
            if best_epoch != completed_epoch:
                raise ValueError("best_model is missing/inconsistent with resume checkpoint")
            _atomic_torch_save(torch, best_path, {
                "schema_version": 1,
                "seed": seed,
                "epoch": best_epoch,
                "model_state": _state_dict_cpu(model),
                "validation_ap": best_val_ap,
                "validation_id_list_sha256": inventory["validation_id_list_sha256"],
                "weights_sha256": checkpoint_meta["sha256"],
                "config_sha256": config_hash,
            })
    else:
        if best_path.exists():
            raise FileExistsError(f"best_model exists without last_resume checkpoint: {best_path}")
        if run_json.exists():
            previous_run = json.loads(run_json.read_text(encoding="utf-8"))
            if previous_run.get("status") != "running" or previous_run.get("latest_epoch"):
                raise FileExistsError(
                    f"run metadata exists without last_resume checkpoint; refusing overwrite: {run_json}"
                )
        model = _make_model(torch, resnet50, checkpoint_meta["path"], seed)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, betas=config.betas,
            weight_decay=config.weight_decay,
        )
        start_epoch = 1
        best_val_ap = -math.inf
        best_epoch = 0
        stale_epochs = 0
        history = []
        completed_epoch = 0
    train_loader, validation_loader = _make_loaders(
        torch, transforms, bundle, config, seed
    )
    loss_sum_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.positive_class_weight], device=device),
        reduction="sum",
    )
    _atomic_json(run_json, run_metadata)
    if progress_callback is not None:
        progress_callback(f"seed {seed}: starting at epoch {start_epoch}/{config.max_epochs}")

    epoch_range = (
        range(start_epoch, config.max_epochs + 1)
        if stale_epochs < config.patience
        else range(0)
    )
    for epoch in epoch_range:
        # Re-seed the sampler and worker transform RNG per epoch for resumable order.
        train_loader.generator.manual_seed(seed + epoch)
        started = time.monotonic()
        train_loss, n_seen, updates = _run_train_epoch(
            torch, model, train_loader, optimizer, loss_sum_fn, device,
            config.accumulation_steps,
        )
        validation = _evaluate_validation(
            torch, model, validation_loader, device, config.validation_acc_threshold
        )
        improved = validation["ap"] > best_val_ap
        if improved:
            best_val_ap = validation["ap"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_ap": validation["ap"],
            "validation_acc_at_0_5": validation["acc_at_0_5"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": updates,
            "train_examples_seen": n_seen,
            "seconds": round(time.monotonic() - started, 3),
        }
        history.append(row)
        resume_state = {
            "schema_version": 1,
            "seed": seed,
            "completed_epoch": epoch,
            "best_val_ap": best_val_ap,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "model_state": _state_dict_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config_sha256": config_hash,
            "train_id_list_sha256": inventory["train_id_list_sha256"],
            "validation_id_list_sha256": inventory["validation_id_list_sha256"],
            "weights_sha256": checkpoint_meta["sha256"],
        }
        _atomic_torch_save(torch, last_path, resume_state)
        _atomic_json(run_json, {**run_metadata, "history": history, "latest_epoch": epoch})
        if improved:
            # Save resume first. If the process stops before this best-model write,
            # the resumed model is exactly the best epoch and can rebuild this file.
            _atomic_torch_save(torch, best_path, {
                "schema_version": 1,
                "seed": seed,
                "epoch": epoch,
                "model_state": _state_dict_cpu(model),
                "validation_ap": best_val_ap,
                "validation_id_list_sha256": inventory["validation_id_list_sha256"],
                "weights_sha256": checkpoint_meta["sha256"],
                "config_sha256": config_hash,
            })
        if progress_callback is not None:
            progress_callback(
                f"seed {seed}: epoch {epoch}/{config.max_epochs}, val AP={validation['ap']:.6f}, "
                f"updates={updates}, stale={stale_epochs}/{config.patience}"
            )
        if stale_epochs >= config.patience:
            break

    final_status = "complete" if history and history[-1]["epoch"] >= config.max_epochs else "early_stopped"
    final = {
        **run_metadata,
        "status": final_status,
        "best_epoch": best_epoch,
        "best_validation_ap": None if best_epoch == 0 else best_val_ap,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "history": history,
        "last_resume_checkpoint": str(last_path),
        "best_model_checkpoint": str(best_path),
    }
    _atomic_json(run_json, final)
    return final
