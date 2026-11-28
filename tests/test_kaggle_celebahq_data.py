from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_data import (  # noqa: E402
    CelebAHQSplitError,
    build_celebahq_datasets,
)
from dire_modal.kaggle_celebahq_integrity import _source_fingerprint  # noqa: E402
from dire_modal.kaggle_celebahq_tree import REQUIRED_GROUPS, scan_celebahq_tree  # noqa: E402


def _tiny_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = (tmp_path / "dire").resolve()
    paths: dict[tuple[str, str], list[str]] = {}
    for split, generator in REQUIRED_GROUPS:
        count = 2 if (split, generator) in (("train", "real"), ("train", "sdv2")) else 1
        leaf = root / split / "celebahq" / generator / generator
        leaf.mkdir(parents=True, exist_ok=True)
        paths[(split, generator)] = []
        for index in range(count):
            name = f"{index:04}.png"
            path = leaf / name
            path.touch()  # metadata-only test fixture; image bytes are never opened
            paths[(split, generator)].append(
                path.relative_to(root).as_posix()
            )

    validation_rows = [
        {
            "path_from_dire_root": paths[("train", generator)][0],
            "generator": generator,
            "label": label,
            "member_locator_inferred": f"{generator}/0000.png",
            "locator_inferred": True,
            "sha256": f"{label + 1:064x}",
            "validation_rank": 1,
            "rank_position_before_duplicate_filter": 1,
            "selection_sha256": "0" * 64,
        }
        for generator, label in (("real", 0), ("sdv2", 1))
    ]
    canonical = sorted(
        [
            {
                "label": row["label"],
                "path_from_dire_root": row["path_from_dire_root"],
                "member_locator_inferred": row["member_locator_inferred"],
            }
            for row in validation_rows
        ],
        key=lambda row: (
            row["label"], row["member_locator_inferred"], row["path_from_dire_root"]
        ),
    )
    id_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    fingerprint = _source_fingerprint(scan_celebahq_tree(root), root)
    artifact = {
        "schema_version": 2,
        "status": "candidate_created",
        "protocol": {"name": "celebahq-content-disjoint-exact-count-hash-rank-v2"},
        "source": {
            "root": str(root),
            "source_fingerprint": fingerprint,
            "archive_member_paths_verified": False,
        },
        "content_overlap_audit": {"status": "pass", "shared_sha256_count_all_labels": 0},
        "validation_records": validation_rows,
        "validation_id_list_sha256": id_hash,
    }
    artifact_path = tmp_path / "split.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return root, artifact_path


def _counts() -> tuple[dict[int, int], dict[tuple[str, int], int]]:
    return {0: 1, 1: 1}, {("real", 0): 2, ("sdv2", 1): 2}


def test_optimized_source_fingerprint_matches_original_stat_recipe(tmp_path: Path) -> None:
    import hashlib as local_hashlib

    root, _artifact = _tiny_fixture(tmp_path)
    records = scan_celebahq_tree(root)
    digest = local_hashlib.sha256()
    for record in sorted(records, key=lambda row: row.relative_path):
        stat = record.path.stat()
        for value in (
            record.relative_path,
            record.split,
            record.generator,
            str(int(record.label)),
            record.member_locator,
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    digest.update(str(root).encode("utf-8"))

    assert _source_fingerprint(records, root) == digest.hexdigest()


def test_builds_disjoint_lazy_datasets_from_v2_metadata_only(tmp_path: Path) -> None:
    root, artifact = _tiny_fixture(tmp_path)
    before = {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*") if p.is_file()}
    val_counts, train_counts = _counts()
    progress: list[str] = []

    bundle = build_celebahq_datasets(
        root,
        artifact,
        expected_validation_counts=val_counts,
        expected_train_counts=train_counts,
        progress_callback=progress.append,
    )

    assert any("path scan complete" in message for message in progress)
    assert any("metadata fingerprint" in message for message in progress)
    assert len(bundle.train) == 2
    assert len(bundle.validation) == 2
    assert set(bundle.test_by_generator) == {
        "real", "dalle2", "if", "midjourney", "stargan", "sdv2"
    }
    train_paths = {row.relative_path for row in bundle.train.records}
    val_paths = {row.relative_path for row in bundle.validation.records}
    assert not train_paths & val_paths
    assert bundle.validation_id_list_sha256
    assert all(len(dataset) == 1 for dataset in bundle.test_by_generator.values())
    after = {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_rejects_wrong_split_hash_before_tree_use(tmp_path: Path) -> None:
    root, artifact = _tiny_fixture(tmp_path)
    payload = json.loads(artifact.read_text())
    payload["validation_id_list_sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    val_counts, train_counts = _counts()

    with pytest.raises(CelebAHQSplitError, match="ID-list SHA-256"):
        build_celebahq_datasets(
            root,
            artifact,
            expected_validation_counts=val_counts,
            expected_train_counts=train_counts,
        )


def test_rejects_nonpassing_overlap_audit(tmp_path: Path) -> None:
    root, artifact = _tiny_fixture(tmp_path)
    payload = json.loads(artifact.read_text())
    payload["content_overlap_audit"]["status"] = "review_required"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    val_counts, train_counts = _counts()

    with pytest.raises(CelebAHQSplitError, match="zero-overlap pass"):
        build_celebahq_datasets(
            root,
            artifact,
            expected_validation_counts=val_counts,
            expected_train_counts=train_counts,
        )


def test_rejects_stale_tree_metadata_fingerprint(tmp_path: Path) -> None:
    root, artifact = _tiny_fixture(tmp_path)
    payload = json.loads(artifact.read_text())
    payload["source"]["source_fingerprint"] = "f" * 64
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    val_counts, train_counts = _counts()

    with pytest.raises(CelebAHQSplitError, match="metadata fingerprint differs"):
        build_celebahq_datasets(
            root,
            artifact,
            expected_validation_counts=val_counts,
            expected_train_counts=train_counts,
        )


def test_rejects_validation_id_absent_from_mounted_tree(tmp_path: Path) -> None:
    root, artifact = _tiny_fixture(tmp_path)
    payload = json.loads(artifact.read_text())
    payload["validation_records"][0]["path_from_dire_root"] = "train/celebahq/real/real/missing.png"
    canonical = sorted(
        [
            {
                "label": row["label"],
                "path_from_dire_root": row["path_from_dire_root"],
                "member_locator_inferred": row["member_locator_inferred"],
            }
            for row in payload["validation_records"]
        ],
        key=lambda row: (
            row["label"], row["member_locator_inferred"], row["path_from_dire_root"]
        ),
    )
    payload["validation_id_list_sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    val_counts, train_counts = _counts()

    with pytest.raises(CelebAHQSplitError, match="absent from current tree"):
        build_celebahq_datasets(
            root,
            artifact,
            expected_validation_counts=val_counts,
            expected_train_counts=train_counts,
        )
