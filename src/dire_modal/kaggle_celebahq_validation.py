"""Create the approved deterministic CelebA-HQ train/validation ID split.

The splitter only reads the integrity summary/JSONL produced by Stage 2 and
writes a candidate split manifest beneath a caller-provided directory outside
the mounted input. It never opens images or changes the data tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALIDATION_SALT = "dire-celebahq-val-v1"
EXPECTED_TRAIN_COUNTS = {("real", 0): 28_000, ("sdv2", 1): 40_000}
EXPECTED_TOTAL_RECORDS = 72_600


def _read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        try:
            metadata = json.loads(next(stream))
        except StopIteration as exc:
            raise ValueError("integrity JSONL manifest is empty") from exc
        rows = [json.loads(line) for line in stream if line.strip()]
    if metadata.get("record_type") != "metadata":
        raise ValueError("integrity JSONL first line is not metadata")
    return metadata, rows


def _round_half_up_ratio(n: int, numerator: int, denominator: int) -> int:
    """Round ``n * numerator / denominator`` to nearest integer, ties upward."""
    return (2 * n * numerator + denominator) // (2 * denominator)


def _rank_digest(salt: str, locator: str) -> str:
    return hashlib.sha256(salt.encode("utf-8") + b"\0" + locator.encode("utf-8")).hexdigest()


def _canonical_list_sha256(values: list[Any]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_celebahq_validation_split(
    integrity_summary_path: str | Path,
    output_dir: str | Path,
    *,
    integrity_manifest_path: str | Path | None = None,
    salt: str = VALIDATION_SALT,
    expected_train_counts: dict[tuple[str, int], int] | None = None,
    expected_total_records: int | None = EXPECTED_TOTAL_RECORDS,
) -> dict[str, Any]:
    """Select exactly 10% of each training label using the locked hash-rank rule.

    The integrity summary and its JSONL manifest must describe the same complete,
    failure-free Stage 2 run. ``integrity_manifest_path`` can override the path
    embedded in the summary, allowing artifacts mounted read-only from a separate
    Kaggle Dataset. Only ``train/real`` and ``train/sdv2`` are split; all test
    records are excluded. Content duplicates crossing train/validation are
    reported for review and do not silently alter the deterministic picks.
    """
    summary_path = Path(integrity_summary_path).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"integrity summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass":
        raise ValueError("integrity summary status is not pass; refusing to create validation IDs")
    if summary.get("input_modified") is not False or summary.get("gpu_work_started") is not False:
        raise ValueError("integrity summary does not attest to read-only, no-GPU Stage 2")
    for key in ("hash_failures", "missing_hashes", "decode_failures"):
        if summary.get(key) != 0:
            raise ValueError(f"integrity summary has {key}={summary.get(key)!r}; refusing split")

    record_count = summary.get("records_in_checkpoint")
    if not isinstance(record_count, int) or record_count < 1:
        raise ValueError("integrity summary has an invalid records_in_checkpoint")
    if summary.get("records_expected") != record_count:
        raise ValueError("integrity summary expected/checkpoint record counts differ")
    if expected_total_records is not None and record_count != expected_total_records:
        raise ValueError(
            f"expected {expected_total_records} integrity records, got {record_count}"
        )

    manifest_value = integrity_manifest_path or summary.get("file_manifest_jsonl")
    if not isinstance(manifest_value, (str, Path)) or not str(manifest_value):
        raise ValueError("integrity summary does not name file_manifest_jsonl")
    manifest_path = Path(manifest_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"integrity JSONL manifest not found: {manifest_path}")
    manifest_meta, records = _read_jsonl(manifest_path)
    if len(records) != record_count or manifest_meta.get("records") != record_count:
        raise ValueError("integrity JSONL record count does not match its summary")
    if manifest_meta.get("source_fingerprint") != summary.get("source_fingerprint"):
        raise ValueError("integrity JSONL and summary source fingerprints differ")
    if manifest_meta.get("archive_member_paths_verified") is not False:
        raise ValueError("unexpected archive-member path verification claim")

    seen_paths: set[str] = set()
    classes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    observed_train_counts: Counter[tuple[str, int]] = Counter()
    for record in records:
        if record.get("hash_status") != "pass" or record.get("decode_status") != "pass":
            raise ValueError(f"integrity failure in manifest row: {record.get('path_from_dire_root')}")
        image_hash = record.get("sha256")
        if not isinstance(image_hash, str) or len(image_hash) != 64:
            raise ValueError(f"missing/invalid SHA-256 for {record.get('path_from_dire_root')}")
        try:
            int(image_hash, 16)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 for {record.get('path_from_dire_root')}") from exc
        path_id = record.get("path_from_dire_root")
        if not isinstance(path_id, str) or not path_id or path_id in seen_paths:
            raise ValueError(f"missing or duplicate root-relative path: {path_id!r}")
        seen_paths.add(path_id)

        if record.get("split") != "train":
            continue
        generator = record.get("generator")
        label = record.get("label")
        if (generator, label) not in EXPECTED_TRAIN_COUNTS:
            raise ValueError(f"unexpected CelebA-HQ training group: {(generator, label)!r}")
        locator = record.get("member_locator_inferred")
        if not isinstance(locator, str) or not locator:
            raise ValueError(f"missing inferred member locator for {path_id}")
        if record.get("locator_inferred") is not True:
            raise ValueError(f"expected locator_inferred=true for {path_id}")
        observed_train_counts[(generator, label)] += 1
        classes[label].append(record)

    required_counts = expected_train_counts or EXPECTED_TRAIN_COUNTS
    if dict(observed_train_counts) != required_counts:
        raise ValueError(
            f"training group counts differ from locked inventory: "
            f"observed={dict(observed_train_counts)}, expected={required_counts}"
        )
    if set(classes) != {0, 1}:
        raise ValueError(f"expected binary train labels 0 and 1, got {sorted(classes)}")
    for label, class_records in classes.items():
        locators = [row["member_locator_inferred"] for row in class_records]
        if len(locators) != len(set(locators)):
            raise ValueError(f"duplicate inferred member locator within label {label}")

    validation_records: list[dict[str, Any]] = []
    class_counts: dict[str, dict[str, int]] = {}
    validation_path_ids: set[str] = set()
    for label in sorted(classes):
        ranked = sorted(
            classes[label],
            key=lambda row: (
                _rank_digest(salt, row["member_locator_inferred"]),
                row["member_locator_inferred"],
            ),
        )
        selected_count = _round_half_up_ratio(len(ranked), 1, 10)
        chosen = ranked[:selected_count]
        for rank, row in enumerate(chosen, start=1):
            path_id = row["path_from_dire_root"]
            validation_path_ids.add(path_id)
            validation_records.append(
                {
                    "path_from_dire_root": path_id,
                    "generator": row["generator"],
                    "label": label,
                    "member_locator_inferred": row["member_locator_inferred"],
                    "locator_inferred": True,
                    "sha256": row["sha256"],
                    "validation_rank": rank,
                    "selection_sha256": _rank_digest(salt, row["member_locator_inferred"]),
                }
            )
        class_counts[str(label)] = {
            "total": len(ranked),
            "validation": selected_count,
            "training": len(ranked) - selected_count,
        }

    validation_records.sort(key=lambda row: (row["label"], row["validation_rank"]))
    validation_id_list = sorted(
        [
            {
                "label": row["label"],
                "path_from_dire_root": row["path_from_dire_root"],
                "member_locator_inferred": row["member_locator_inferred"],
            }
            for row in validation_records
        ],
        key=lambda item: (item["label"], item["member_locator_inferred"], item["path_from_dire_root"]),
    )
    all_train_records = [record for class_records in classes.values() for record in class_records]
    validation_hashes_all = {
        row["sha256"] for row in all_train_records
        if row["path_from_dire_root"] in validation_path_ids
    }
    training_hashes_all = {
        row["sha256"] for row in all_train_records
        if row["path_from_dire_root"] not in validation_path_ids
    }
    shared_hashes_all = sorted(validation_hashes_all & training_hashes_all)
    shared_hashes: dict[str, list[str]] = {}
    for label in sorted(classes):
        validation_hashes = {
            row["sha256"] for row in classes[label]
            if row["path_from_dire_root"] in validation_path_ids
        }
        training_hashes = {
            row["sha256"] for row in classes[label]
            if row["path_from_dire_root"] not in validation_path_ids
        }
        shared_hashes[str(label)] = sorted(validation_hashes & training_hashes)

    content_overlap = {
        "status": "review_required" if shared_hashes_all else "pass",
        "shared_sha256_count_all_labels": len(shared_hashes_all),
        "shared_sha256_all_labels": shared_hashes_all,
        "shared_sha256_count_by_label": {label: len(values) for label, values in shared_hashes.items()},
        "shared_sha256_by_label": shared_hashes,
        "note": "Exact duplicate bytes crossing train/validation are reported across both labels; the locked rank selection is not silently changed.",
    }
    output_path = Path(output_dir).expanduser().resolve()
    root_path = Path(summary.get("root", "")).expanduser().resolve()
    if not summary.get("root") or output_path == root_path or root_path in output_path.parents:
        raise ValueError("validation manifest output must be outside the read-only input root")
    output_path.mkdir(parents=True, exist_ok=True)
    validation_ids_digest = _canonical_list_sha256(validation_id_list)
    artifact = {
        "schema_version": 1,
        "status": "candidate_created",
        "protocol": {
            "name": "celebahq-exact-count-hash-rank-v1",
            "salt": salt,
            "hash": "SHA-256",
            "preimage": "UTF-8(salt) + 0x00 + UTF-8(member_locator_inferred)",
            "sort": "ascending lowercase hex digest, then member_locator_inferred",
            "fraction": {"numerator": 1, "denominator": 10},
            "rounding": "half-up",
        },
        "source": {
            "root": summary["root"],
            "source_fingerprint": summary["source_fingerprint"],
            "integrity_summary": str(summary_path),
            "integrity_manifest": str(manifest_path),
            "integrity_status": summary["status"],
            "records": record_count,
            "archive_member_paths_verified": False,
        },
        "counts_by_label": class_counts,
        "validation_id_list_sha256": validation_ids_digest,
        "content_overlap_audit": content_overlap,
        "validation_records": validation_records,
        "notes": [
            "Only train/real and train/sdv2 records are partitioned; all test records remain untouched.",
            "Locators are inferred from the extracted tree and have not been verified against original tar member paths.",
            "Candidate split creation does not authorize training; review content_overlap_audit and complete the compute preflight first.",
        ],
    }
    destination = output_path / f"celebahq-validation-split-v1-{summary['source_fingerprint'][:12]}.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    artifact["artifact_path"] = str(destination)
    return artifact
