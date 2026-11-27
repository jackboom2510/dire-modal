"""Read-only SHA-256 and image-decode audit for the selected CelebA-HQ tree.

The mounted input is never changed. Checkpoint database and exported manifests
are written only beneath a caller-provided writable directory outside the input.
The SQLite checkpoint can resume completed rows for the same file-tree metadata
fingerprint (root-relative paths, sizes, and mtimes).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

CHUNK_SIZE = 1024 * 1024


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _source_fingerprint(records: list[Any], root: Path) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row.relative_path):
        path = Path(record.path).expanduser().resolve()
        if not _inside(path, root):
            raise ValueError(f"indexed input path escapes DIRE root: {record.relative_path}")
        stat = path.stat()
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
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_db(db: sqlite3.Connection, *, root: Path, fingerprint: str, count: int) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute(
        """CREATE TABLE IF NOT EXISTS files (
            relative_path TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            generator TEXT NOT NULL,
            label INTEGER NOT NULL,
            member_locator TEXT NOT NULL,
            locator_inferred INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT,
            hash_status TEXT NOT NULL,
            decode_status TEXT NOT NULL,
            image_format TEXT,
            width INTEGER,
            height INTEGER,
            mode TEXT,
            error TEXT
        )"""
    )
    expected = {
        "schema_version": "1",
        "root": str(root),
        "source_fingerprint": fingerprint,
        "record_count": str(count),
    }
    existing = dict(db.execute("SELECT key, value FROM meta"))
    if existing and existing != expected:
        raise ValueError("checkpoint metadata mismatch; refusing to reuse another input manifest")
    if not existing:
        db.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", expected.items())


def verify_celebahq_records(
    root: str | Path,
    records: Iterable[Any],
    workdir: str | Path,
    *,
    checkpoint_every: int = 250,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Hash and fully decode each indexed file, with resumable external checkpoint.

    Records are expected to expose ``split``, ``generator``, ``label``, ``path``,
    ``relative_path``, ``member_locator``, and ``locator_inferred`` (as emitted by
    :func:`scan_celebahq_tree`). Hashing and Pillow decoding are separate reads.
    This verifies the currently mounted bytes, not original archive provenance.
    """
    if checkpoint_every < 1 or progress_every < 1:
        raise ValueError("checkpoint_every and progress_every must be positive")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Kaggle normally provides Pillow
        raise RuntimeError("Pillow is required for image decode verification") from exc

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"DIRE root is not a directory: {root_path}")
    rows = list(records)
    if not rows:
        raise ValueError("no indexed CelebA-HQ records were supplied")
    work_path = Path(workdir).expanduser().resolve()
    if _inside(work_path, root_path):
        raise ValueError("checkpoint/output directory must be outside the read-only input dataset")

    fingerprint = _source_fingerprint(rows, root_path)
    run_id = fingerprint[:20]
    work_path.mkdir(parents=True, exist_ok=True)
    db_path = work_path / f"celebahq-integrity-{run_id}.sqlite3"
    manifest_path = work_path / f"celebahq-integrity-{run_id}.jsonl"
    summary_path = work_path / f"celebahq-integrity-{run_id}-summary.json"

    db = sqlite3.connect(db_path)
    try:
        _init_db(db, root=root_path, fingerprint=fingerprint, count=len(rows))
        db.commit()
        hashed_now = 0
        reused = 0
        pending = 0
        started = time.time()

        for index, record in enumerate(rows, start=1):
            path = Path(record.path)
            stat = path.stat()
            previous = db.execute(
                "SELECT size_bytes, mtime_ns, sha256, hash_status, decode_status "
                "FROM files WHERE relative_path = ?",
                (record.relative_path,),
            ).fetchone()
            if (
                previous
                and previous[0] == stat.st_size
                and previous[1] == stat.st_mtime_ns
                and previous[2]
                and previous[3] == "pass"
                and previous[4] in {"pass", "fail"}
            ):
                reused += 1
            else:
                hash_status = "pass"
                sha256 = None
                decode_status = "fail"
                image_format = None
                width = height = None
                mode = None
                errors: list[str] = []
                try:
                    sha256 = _sha256_file(path)
                except OSError as exc:
                    hash_status = "fail"
                    errors.append(f"sha256: {exc}")
                try:
                    with Image.open(path) as image:
                        image_format = image.format
                        width, height = image.size
                        mode = image.mode
                        image.load()  # force full pixel decode; do not retain pixel data
                    decode_status = "pass"
                except Exception as exc:  # record bad files and continue through the inventory
                    errors.append(f"decode: {type(exc).__name__}: {exc}")

                db.execute(
                    """INSERT OR REPLACE INTO files(
                        relative_path, split, generator, label, member_locator,
                        locator_inferred, size_bytes, mtime_ns, sha256, hash_status,
                        decode_status, image_format, width, height, mode, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.relative_path,
                        record.split,
                        record.generator,
                        int(record.label),
                        record.member_locator,
                        int(record.locator_inferred),
                        stat.st_size,
                        stat.st_mtime_ns,
                        sha256,
                        hash_status,
                        decode_status,
                        image_format,
                        width,
                        height,
                        mode,
                        " | ".join(errors)[:1000] or None,
                    ),
                )
                hashed_now += 1
                pending += 1
                if pending >= checkpoint_every:
                    db.commit()
                    pending = 0

            if index % progress_every == 0 or index == len(rows):
                db.commit()
                print(
                    f"integrity {index}/{len(rows)}; hashed this run={hashed_now}; "
                    f"reused checkpoint={reused}; elapsed={time.time() - started:.1f}s"
                )

        db.commit()
        total = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        total_file_bytes = db.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM files").fetchone()[0]
        hash_failures = db.execute("SELECT COUNT(*) FROM files WHERE hash_status != 'pass'").fetchone()[0]
        decode_failures = db.execute("SELECT COUNT(*) FROM files WHERE decode_status != 'pass'").fetchone()[0]
        missing_hashes = db.execute("SELECT COUNT(*) FROM files WHERE sha256 IS NULL").fetchone()[0]
        group_counts = [
            {
                "split": split,
                "generator": generator,
                "label": label,
                "files": count,
                "hash_failures": hash_errors,
                "decode_failures": decode_errors,
            }
            for split, generator, label, count, hash_errors, decode_errors in db.execute(
                """SELECT split, generator, label, COUNT(*),
                          SUM(CASE WHEN hash_status != 'pass' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN decode_status != 'pass' THEN 1 ELSE 0 END)
                   FROM files GROUP BY split, generator, label
                   ORDER BY split, generator"""
            )
        ]

        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        metadata = {
            "record_type": "metadata",
            "schema_version": 1,
            "root": str(root_path),
            "source_fingerprint": fingerprint,
            "source_fingerprint_basis": "relative path + stat size + mtime_ns; no content identity",
            "archive_member_paths_verified": False,
            "input_modified": False,
            "gpu_work_started": False,
            "records": total,
        }
        with temporary_manifest.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            cursor = db.execute(
                """SELECT relative_path, split, generator, label, member_locator,
                          locator_inferred, size_bytes, sha256, hash_status,
                          decode_status, image_format, width, height, mode, error
                   FROM files ORDER BY split, generator, relative_path"""
            )
            for row in cursor:
                payload = dict(zip((
                    "path_from_dire_root", "split", "generator", "label",
                    "member_locator_inferred", "locator_inferred", "size_bytes",
                    "sha256", "hash_status", "decode_status", "image_format",
                    "width", "height", "mode", "error",
                ), row))
                payload["locator_inferred"] = bool(payload["locator_inferred"])
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        os.replace(temporary_manifest, manifest_path)

        summary: dict[str, Any] = {
            "schema_version": 1,
            "status": "pass" if total == len(rows) and not hash_failures and not decode_failures else "fail",
            "root": str(root_path),
            "source_fingerprint": fingerprint,
            "source_fingerprint_basis": "relative path + stat size + mtime_ns; no content identity",
            "input_modified": False,
            "image_contents_read": True,
            "gpu_work_started": False,
            "archive_member_paths_verified": False,
            "records_expected": len(rows),
            "records_in_checkpoint": total,
            "total_file_bytes_by_stat": total_file_bytes,
            "hashed_this_run": hashed_now,
            "reused_from_checkpoint": reused,
            "hash_failures": hash_failures,
            "missing_hashes": missing_hashes,
            "decode_failures": decode_failures,
            "groups": group_counts,
            "checkpoint_sqlite": str(db_path),
            "file_manifest_jsonl": str(manifest_path),
            "elapsed_seconds_this_run": round(time.time() - started, 2),
            "notes": [
                "SHA-256 covers the currently mounted file bytes only; it does not prove original archive or source provenance.",
                "Images are fully decoded one at a time after hashing; the input tree is never written.",
                "Checkpoint reuse assumes an immutable, identical mounted Dataset when path, size, and mtime match.",
            ],
        }
        temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary_summary, summary_path)
        summary["summary_json"] = str(summary_path)
        return summary
    finally:
        db.close()
