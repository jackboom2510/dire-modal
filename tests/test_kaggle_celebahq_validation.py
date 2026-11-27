from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_validation import create_celebahq_validation_split  # noqa: E402


def _write_stage2_bundle(
    base: Path,
    *,
    duplicate_content: bool = False,
    status: str = "pass",
) -> tuple[Path, Path]:
    root = base / "dire"
    root.mkdir()
    work = base / "working"
    work.mkdir()
    manifest = work / "integrity.jsonl"
    fingerprint = "a" * 64
    rows: list[dict[str, object]] = []
    for generator, label in (("real", 0), ("sdv2", 1)):
        for index in range(10):
            row_hash = "0" * 64 if duplicate_content else hashlib.sha256(
                f"{generator}-{index}".encode()
            ).hexdigest()
            rows.append(
                {
                    "path_from_dire_root": f"train/celebahq/{generator}/{generator}/{index}.png",
                    "split": "train",
                    "generator": generator,
                    "label": label,
                    "member_locator_inferred": f"{generator}/{index}.png",
                    "locator_inferred": True,
                    "sha256": row_hash,
                    "hash_status": "pass",
                    "decode_status": "pass",
                }
            )
    rows.append(
        {
            "path_from_dire_root": "test/celebahq/dalle2/dalle2/heldout.png",
            "split": "test",
            "generator": "dalle2",
            "label": 1,
            "member_locator_inferred": "dalle2/heldout.png",
            "locator_inferred": True,
            "sha256": hashlib.sha256(b"test").hexdigest(),
            "hash_status": "pass",
            "decode_status": "pass",
        }
    )
    metadata = {
        "record_type": "metadata",
        "records": len(rows),
        "source_fingerprint": fingerprint,
        "archive_member_paths_verified": False,
    }
    manifest.write_text(
        "\n".join(json.dumps(item) for item in [metadata, *rows]) + "\n",
        encoding="utf-8",
    )
    summary_path = work / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": status,
                "root": str(root),
                "source_fingerprint": fingerprint,
                "input_modified": False,
                "gpu_work_started": False,
                "records_expected": len(rows),
                "records_in_checkpoint": len(rows),
                "hash_failures": 0,
                "missing_hashes": 0,
                "decode_failures": 0,
                "file_manifest_jsonl": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    return summary_path, work


def _small_expected_counts() -> dict[tuple[str, int], int]:
    return {("real", 0): 10, ("sdv2", 1): 10}


def test_exact_count_split_is_deterministic_per_label_and_excludes_test(tmp_path: Path) -> None:
    summary_path, work = _write_stage2_bundle(tmp_path)
    kwargs = {
        "expected_train_counts": _small_expected_counts(),
        "expected_total_records": 21,
    }
    first = create_celebahq_validation_split(summary_path, work, **kwargs)
    second = create_celebahq_validation_split(summary_path, work, **kwargs)

    assert first["protocol"]["salt"] == "dire-celebahq-val-v1"
    assert first["counts_by_label"] == {
        "0": {"total": 10, "validation": 1, "training": 9},
        "1": {"total": 10, "validation": 1, "training": 9},
    }
    assert len(first["validation_records"]) == 2
    assert all("test/" not in row["path_from_dire_root"] for row in first["validation_records"])
    assert first["validation_records"] == second["validation_records"]
    assert first["validation_id_list_sha256"] == second["validation_id_list_sha256"]
    assert first["content_overlap_audit"]["status"] == "pass"
    assert Path(first["artifact_path"]).is_file()
    artifact = json.loads(Path(first["artifact_path"]).read_text(encoding="utf-8"))
    assert len(artifact["validation_records"]) == 2


def test_split_reports_exact_duplicate_bytes_across_partition(tmp_path: Path) -> None:
    summary_path, work = _write_stage2_bundle(tmp_path, duplicate_content=True)
    result = create_celebahq_validation_split(
        summary_path,
        work,
        expected_train_counts=_small_expected_counts(),
        expected_total_records=21,
    )
    audit = result["content_overlap_audit"]
    assert audit["status"] == "review_required"
    assert audit["shared_sha256_count_by_label"] == {"0": 1, "1": 1}


def test_split_refuses_nonpassing_integrity_run(tmp_path: Path) -> None:
    summary_path, work = _write_stage2_bundle(tmp_path, status="fail")
    with pytest.raises(ValueError, match="status is not pass"):
        create_celebahq_validation_split(
            summary_path,
            work,
            expected_train_counts=_small_expected_counts(),
            expected_total_records=21,
        )


def test_split_refuses_ambiguous_duplicate_locators(tmp_path: Path) -> None:
    summary_path, work = _write_stage2_bundle(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = Path(summary["file_manifest_jsonl"])
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    rows[2]["member_locator_inferred"] = rows[1]["member_locator_inferred"]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate inferred member locator"):
        create_celebahq_validation_split(
            summary_path,
            work,
            expected_train_counts=_small_expected_counts(),
            expected_total_records=21,
        )
