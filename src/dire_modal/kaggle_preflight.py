"""Read-only preflight for a mounted CelebA-HQ DIRE maps Kaggle Dataset.

This module uses only the standard library. It validates the archive bytes and
member counts against the local inventory manifest, checks a writable output
area, and optionally reports ``nvidia-smi`` information. It never imports torch,
starts training, writes to the input dataset, or checks account quota.

Run on Kaggle before any smoke/training cell::

    python -m dire_modal.kaggle_preflight \
      --root /kaggle/input/dire-celebahq-maps \
      --manifest /kaggle/input/dire-celebahq-maps/kaggle-map-manifest.json \
      --workdir /kaggle/working/dire-preflight \
      --gpu-info
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_ARCHIVES: tuple[tuple[str, str, str], ...] = (
    ("dire/train/celebahq/real.tar.gz", "train", "real"),
    ("dire/train/celebahq/sdv2.tar.gz", "train", "sdv2"),
    ("dire/test/celebahq/real.tar.gz", "test", "real"),
    ("dire/test/celebahq/dalle2.tar.gz", "test", "dalle2"),
    ("dire/test/celebahq/if.tar.gz", "test", "if"),
    ("dire/test/celebahq/midjourney.tar.gz", "test", "midjourney"),
    ("dire/test/celebahq/stargan.tar.gz", "test", "stargan"),
    ("dire/test/celebahq/sdv2.tar.gz", "test", "sdv2"),
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".npy", ".npz", ".png", ".webp"}
CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_tar(path: Path) -> tuple[int, int, dict[str, int]]:
    regular = 0
    image_like = 0
    suffixes: Counter[str] = Counter()
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            regular += 1
            suffix = PurePosixPath(member.name).suffix.lower() or "<none>"
            suffixes[suffix] += 1
            image_like += suffix in IMAGE_SUFFIXES
    if regular == 0:
        raise tarfile.TarError("archive contains no regular file members")
    return regular, image_like, dict(sorted(suffixes.items()))


def _gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": str(exc)}
    if result.returncode != 0:
        return {"available": False, "detail": result.stderr.strip() or "nvidia-smi returned nonzero"}
    return {"available": bool(result.stdout.strip()), "devices": result.stdout.strip().splitlines()}


def _write_test(workdir: Path) -> tuple[bool, str | None]:
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".dire-preflight-", dir=workdir, delete=True):
            pass
    except OSError as exc:
        return False, str(exc)
    return True, None


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def preflight_dataset(
    root: str | Path,
    manifest_path: str | Path,
    workdir: str | Path,
    *,
    output_path: str | Path | None = None,
    check_gpu: bool = False,
) -> dict[str, Any]:
    """Verify a mounted dataset against its manifest, writing only outside input."""
    root_path = Path(root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    work_path = Path(workdir).expanduser().resolve()
    report_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else work_path / "preflight-report.json"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "dire-celebahq-maps",
        "mode": "read-only preflight; no model/GPU workload",
        "input_root": str(root_path),
        "manifest_path": str(manifest_file),
        "gpu_work_started": False,
        "account_quota_verified": False,
        "archives": [],
        "errors": [],
        "warnings": [
            "Archive SHA-256 and member counts are verified; member pixels/source provenance are not.",
            "Kaggle account/session quota is not available to this local filesystem preflight.",
            "The working directory is session-local; export the report/artifacts to persistent storage.",
        ],
        "gpu_info": _gpu_info() if check_gpu else {"checked": False},
    }

    if _inside(report_path, root_path):
        report["errors"].append("report output must be outside the input dataset (read-only source)")
    if _inside(work_path, root_path):
        report["errors"].append("workdir must be outside the read-only input dataset")

    manifest: dict[str, Any] = {}
    if not root_path.is_dir():
        report["errors"].append(f"input root is not a directory: {root_path}")
    if not manifest_file.is_file():
        report["errors"].append(f"manifest does not exist: {manifest_file}")
    else:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            report["errors"].append(f"cannot read manifest: {exc}")

    if not isinstance(manifest, dict) or not manifest:
        report["errors"].append("manifest must be a non-empty JSON object")
    else:
        if manifest.get("schema_version") != 1:
            report["errors"].append(f"unsupported inventory manifest schema: {manifest.get('schema_version')!r}")
        if manifest.get("dataset") != "dire-celebahq-maps":
            report["errors"].append(f"unexpected dataset label: {manifest.get('dataset')!r}")
        rows = manifest.get("archives")
        if not isinstance(rows, list):
            rows = []
            report["errors"].append("manifest has no archive list")
        manifest_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                if row["path"] in manifest_rows:
                    report["errors"].append(f"duplicate manifest archive row: {row['path']}")
                manifest_rows[row["path"]] = row
        expected_paths = {rel for rel, _, _ in REQUIRED_ARCHIVES}
        missing_rows = sorted(expected_paths - manifest_rows.keys())
        if missing_rows:
            report["errors"].append(f"manifest lacks required archives: {missing_rows}")

        for relative, split, generator in REQUIRED_ARCHIVES:
            row = manifest_rows.get(relative)
            result: dict[str, Any] = {
                "path": relative,
                "split": split,
                "generator": generator,
                "status": "fail",
                "errors": [],
            }
            relpath = PurePosixPath(relative)
            if relpath.is_absolute() or ".." in relpath.parts:
                result["errors"].append("unsafe archive path in preflight specification")
                report["archives"].append(result)
                continue
            path = root_path.joinpath(*relpath.parts)
            if not path.is_file():
                result["errors"].append("missing archive")
            elif row is None:
                result["errors"].append("manifest row missing")
            else:
                try:
                    archive_bytes = path.stat().st_size
                    archive_sha = _sha256_file(path)
                    regular, image_like, suffix_counts = _inspect_tar(path)
                    result.update({
                        "archive_bytes": archive_bytes,
                        "archive_sha256": archive_sha,
                        "regular_members": regular,
                        "image_members": image_like,
                        "suffix_counts": suffix_counts,
                    })
                    expected_bytes = row.get("archive_bytes")
                    expected_sha = row.get("archive_sha256")
                    if archive_bytes != expected_bytes:
                        result["errors"].append(
                            f"archive size mismatch: expected {expected_bytes}, observed {archive_bytes}"
                        )
                    if archive_sha != expected_sha:
                        result["errors"].append("archive sha256 mismatch")
                    expected_members = row.get("regular_members")
                    if regular != expected_members:
                        result["errors"].append(
                            f"regular member count mismatch: expected {expected_members}, observed {regular}"
                        )
                    expected_images = row.get("image_members")
                    if image_like != expected_images:
                        result["errors"].append(
                            f"image-like member count mismatch: expected {expected_images}, observed {image_like}"
                        )
                    if row.get("suffix_counts") is not None and suffix_counts != row["suffix_counts"]:
                        result["errors"].append("suffix counts mismatch")
                except (tarfile.TarError, OSError, EOFError) as exc:
                    result["errors"].append(f"cannot inspect archive: {exc}")
            result["status"] = "pass" if not result["errors"] else "fail"
            report["archives"].append(result)
            report["errors"].extend(f"{relative}: {error}" for error in result["errors"])

    if work_path.exists() and _inside(work_path, root_path):
        writable, detail = False, "workdir is inside input dataset"
    else:
        writable, detail = _write_test(work_path)
    usage = shutil.disk_usage(work_path.parent if work_path.parent.exists() else Path.cwd())
    report["working_storage"] = {
        "workdir": str(work_path),
        "writable": writable,
        "write_error": detail,
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }
    if not writable:
        report["errors"].append(f"working artifact path is not writable: {detail}")
    report["status"] = "pass" if not report["errors"] else "fail"

    if not _inside(report_path, root_path):
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report["report_path"] = str(report_path)
            temp_path = report_path.with_name(report_path.name + ".tmp")
            temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temp_path, report_path)
        except OSError as exc:
            report["status"] = "fail"
            report["errors"].append(f"cannot write report outside input dataset: {exc}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="mounted dataset root containing `dire/`")
    parser.add_argument("--manifest", required=True, type=Path, help="uploaded local inventory JSON")
    parser.add_argument("--workdir", required=True, type=Path, help="writable Kaggle working directory")
    parser.add_argument("--output", type=Path, help="report path (default: WORKDIR/preflight-report.json)")
    parser.add_argument("--gpu-info", action="store_true", help="read-only nvidia-smi query; does no GPU work")
    args = parser.parse_args(argv)
    report = preflight_dataset(
        args.root, args.manifest, args.workdir,
        output_path=args.output, check_gpu=args.gpu_info,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
