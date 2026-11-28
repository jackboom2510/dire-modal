"""Read-only CelebA-HQ datasets wired to the approved Stage 3 v2 split.

Index construction walks paths and reads the split JSON only; it never opens
image contents. Image decode happens lazily in ``__getitem__`` when a caller
explicitly iterates a dataset. This module does not create a DataLoader, move
samples to a device, train, or write beneath the input tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .kaggle_celebahq_integrity import _source_fingerprint
from .kaggle_celebahq_tree import CelebAHQImage, scan_celebahq_tree

SPLIT_PROTOCOL = "celebahq-content-disjoint-exact-count-hash-rank-v2"
EXPECTED_VALIDATION_COUNTS = {0: 2_800, 1: 4_000}
EXPECTED_TRAIN_COUNTS = {("real", 0): 28_000, ("sdv2", 1): 40_000}
TEST_GENERATORS = ("real", "dalle2", "if", "midjourney", "stargan", "sdv2")


class CelebAHQSplitError(ValueError):
    """The supplied split artifact cannot safely be applied to this tree."""


def _canonical_validation_id_hash(records: list[dict[str, Any]]) -> str:
    ids = [
        {
            "label": row["label"],
            "path_from_dire_root": row["path_from_dire_root"],
            "member_locator_inferred": row["member_locator_inferred"],
        }
        for row in records
    ]
    ids.sort(key=lambda row: (
        row["label"], row["member_locator_inferred"], row["path_from_dire_root"]
    ))
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CelebAHQDataset:
    """Lazy image dataset; iterating an item decodes that image as RGB."""

    root: Path
    records: tuple[CelebAHQImage, ...]
    transform: Callable[[Any], Any] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image  # noqa: PLC0415 - import only on explicit item access

        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


@dataclass(frozen=True)
class CelebAHQDataBundle:
    """Datasets for train/validation and each predeclared test generator."""

    train: CelebAHQDataset
    validation: CelebAHQDataset
    test_by_generator: dict[str, CelebAHQDataset]
    validation_id_list_sha256: str
    source_fingerprint: str


def build_celebahq_datasets(
    dire_root: str | Path,
    split_artifact_path: str | Path,
    *,
    transform_train: Callable[[Any], Any] | None = None,
    transform_eval: Callable[[Any], Any] | None = None,
    expected_validation_counts: dict[int, int] | None = None,
    expected_train_counts: dict[tuple[str, int], int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> CelebAHQDataBundle:
    """Validate Stage 3 v2 IDs and return lazy datasets over the mounted tree.

    The optional count overrides exist for small metadata-only unit fixtures;
    production callers should use the locked defaults. No file bytes are read
    while validating or constructing these datasets.
    """
    root = Path(dire_root).expanduser().resolve()
    artifact_path = Path(split_artifact_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"validation split artifact not found: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    if artifact.get("schema_version") != 2 or artifact.get("status") != "candidate_created":
        raise CelebAHQSplitError("expected a Stage 3 v2 candidate split artifact")
    if artifact.get("protocol", {}).get("name") != SPLIT_PROTOCOL:
        raise CelebAHQSplitError("split protocol is not the approved content-disjoint v2")
    overlap_audit = artifact.get("content_overlap_audit", {})
    if overlap_audit.get("status") != "pass" or overlap_audit.get("shared_sha256_count_all_labels") != 0:
        raise CelebAHQSplitError("split content-overlap audit is not a zero-overlap pass")
    source = artifact.get("source", {})
    if not source.get("root") or Path(source["root"]).expanduser().resolve() != root:
        raise CelebAHQSplitError("split source root does not match the mounted DIRE root")
    if source.get("archive_member_paths_verified") is not False:
        raise CelebAHQSplitError("unexpected archive-member path verification claim")
    source_fingerprint = source.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise CelebAHQSplitError("split artifact is missing its source fingerprint")

    validation_rows = artifact.get("validation_records")
    if not isinstance(validation_rows, list) or not validation_rows:
        raise CelebAHQSplitError("split artifact has no validation_records")
    if _canonical_validation_id_hash(validation_rows) != artifact.get("validation_id_list_sha256"):
        raise CelebAHQSplitError("validation ID-list SHA-256 does not match the artifact rows")

    expected_val = expected_validation_counts or EXPECTED_VALIDATION_COUNTS
    observed_val: dict[int, int] = {0: 0, 1: 0}
    val_paths: set[str] = set()
    for row in validation_rows:
        path_id = row.get("path_from_dire_root")
        label = row.get("label")
        if not isinstance(path_id, str) or not path_id or label not in (0, 1):
            raise CelebAHQSplitError("invalid validation record path or binary label")
        if path_id in val_paths:
            raise CelebAHQSplitError(f"duplicate validation path ID: {path_id}")
        val_paths.add(path_id)
        observed_val[label] += 1
    if observed_val != expected_val:
        raise CelebAHQSplitError(
            f"validation counts differ from expected: observed={observed_val}, expected={expected_val}"
        )

    if progress_callback is not None:
        progress_callback("Starting metadata-only tree scan; image contents will not be opened.")
    records = scan_celebahq_tree(root, progress_callback=progress_callback)
    if progress_callback is not None:
        progress_callback(f"Tree path inventory complete: {len(records):,} image paths.")
    current_fingerprint = _source_fingerprint(
        records, root, progress_callback=progress_callback
    )
    if current_fingerprint != source_fingerprint:
        raise CelebAHQSplitError(
            "mounted tree metadata fingerprint differs from the Stage 2 source; "
            "refusing stale split IDs"
        )
    by_path = {record.relative_path: record for record in records}
    if len(by_path) != len(records):
        raise CelebAHQSplitError("tree adapter produced duplicate root-relative paths")

    validation_records: list[CelebAHQImage] = []
    for row in validation_rows:
        path_id = row["path_from_dire_root"]
        record = by_path.get(path_id)
        if record is None:
            raise CelebAHQSplitError(f"validation ID is absent from current tree: {path_id}")
        if record.split != "train" or (record.generator, record.label) not in {
            ("real", 0), ("sdv2", 1)
        }:
            raise CelebAHQSplitError(f"validation ID is outside the selected training groups: {path_id}")
        if row.get("label") != record.label or row.get("generator") != record.generator:
            raise CelebAHQSplitError(f"validation metadata disagrees with tree path: {path_id}")
        if row.get("member_locator_inferred") != record.member_locator:
            raise CelebAHQSplitError(f"inferred member locator disagrees with tree path: {path_id}")
        if row.get("locator_inferred") is not True:
            raise CelebAHQSplitError(f"validation locator is not marked inferred: {path_id}")
        validation_records.append(record)

    train_pool = [
        record for record in records
        if record.split == "train" and record.generator in ("real", "sdv2")
    ]
    required_train = expected_train_counts or EXPECTED_TRAIN_COUNTS
    observed_train: dict[tuple[str, int], int] = {}
    for record in train_pool:
        key = (record.generator, record.label)
        observed_train[key] = observed_train.get(key, 0) + 1
    if observed_train != required_train:
        raise CelebAHQSplitError(
            f"training group counts differ from expected: observed={observed_train}, expected={required_train}"
        )
    validation_by_path = {record.relative_path for record in validation_records}
    training_records = [record for record in train_pool if record.relative_path not in validation_by_path]
    expected_training = {
        label: required_train[(generator, label)] - expected_val[label]
        for generator, label in (("real", 0), ("sdv2", 1))
    }
    actual_training = {0: 0, 1: 0}
    for record in training_records:
        actual_training[record.label] += 1
    if actual_training != expected_training:
        raise CelebAHQSplitError(
            f"training partition counts disagree with split: observed={actual_training}, expected={expected_training}"
        )
    if validation_by_path & {record.relative_path for record in training_records}:
        raise CelebAHQSplitError("train/validation path overlap detected")

    tests: dict[str, CelebAHQDataset] = {}
    for generator in TEST_GENERATORS:
        selected = tuple(
            record for record in records
            if record.split == "test" and record.generator == generator
        )
        if not selected:
            raise CelebAHQSplitError(f"test group is empty or missing: {generator}")
        tests[generator] = CelebAHQDataset(root, selected, transform_eval)

    return CelebAHQDataBundle(
        train=CelebAHQDataset(root, tuple(training_records), transform_train),
        validation=CelebAHQDataset(root, tuple(validation_records), transform_eval),
        test_by_generator=tests,
        validation_id_list_sha256=artifact["validation_id_list_sha256"],
        source_fingerprint=source_fingerprint,
    )
