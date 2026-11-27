"""Read-only Kaggle resource preflight for the CelebA-HQ training gate.

This performs metadata/path checks, tiny write/delete probes outside the mounted
inputs, and a metadata-only GPU query via ``nvidia-smi`` with a PyTorch device
property fallback. It does not open images, allocate tensors, run a GPU workload,
or start training.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_DIRE_ROOT = Path("/kaggle/input/datasets/j4cknguyn/diffusionforensics/dire")
EXPECTED_ARTIFACT_INPUT = Path(
    "/kaggle/input/datasets/j4cknguyn/dire-celeba-hq-stage-2-artifacts-v1"
)
EXPECTED_SPLIT_COUNTS = {
    "0": {"total": 28_000, "validation": 2_800, "training": 25_200},
    "1": {"total": 40_000, "validation": 4_000, "training": 36_000},
}
EXPECTED_SPLIT_PROTOCOL = "celebahq-content-disjoint-exact-count-hash-rank-v2"


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _probe_writable(directory: Path) -> dict[str, Any]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".celebahq-preflight-", dir=directory)
        probe = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(b"dire-celebahq-preflight\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            probe.unlink(missing_ok=True)
        return {"status": "pass", "path": str(directory), "probe_bytes": 24, "probe_removed": True}
    except OSError as exc:
        return {"status": "fail", "path": str(directory), "error": f"{type(exc).__name__}: {exc}"}


def _torch_gpu_fallback(nvidia_smi_error: dict[str, Any]) -> dict[str, Any]:
    """Query CUDA device metadata through PyTorch without allocating tensors."""
    try:
        import torch
    except Exception as exc:  # PyTorch may be absent or fail to import on CPU-only images.
        return {
            "status": "unavailable",
            "fallback": "torch.cuda metadata",
            "nvidia_smi": nvidia_smi_error,
            "error": f"{type(exc).__name__}: {exc}",
            "devices": [],
        }
    try:
        if not torch.cuda.is_available():
            return {
                "status": "no_gpu_visible",
                "fallback": "torch.cuda metadata",
                "nvidia_smi": nvidia_smi_error,
                "devices": [],
                "note": "No CUDA tensor/model was allocated; device discovery found no visible GPU.",
            }
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except Exception:
                free_bytes = total_bytes = None
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": getattr(properties, "uuid", None),
                    "memory_total_bytes": total_bytes or getattr(properties, "total_memory", None),
                    "memory_free_bytes": free_bytes,
                    "cuda_runtime_version": getattr(torch.version, "cuda", None),
                }
            )
        return {
            "status": "pass" if devices else "no_gpu_visible",
            "fallback": "torch.cuda metadata",
            "nvidia_smi": nvidia_smi_error,
            "devices": devices,
            "note": "Metadata query only; no tensor/model allocation or GPU computation was started.",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "fallback": "torch.cuda metadata",
            "nvidia_smi": nvidia_smi_error,
            "error": f"{type(exc).__name__}: {exc}",
            "devices": [],
        }


def _query_gpu(nvidia_smi: str = "nvidia-smi", timeout_seconds: int = 10) -> dict[str, Any]:
    command = [
        nvidia_smi,
        "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        error = {"command": command, "error": f"{type(exc).__name__}: {exc}"}
        return _torch_gpu_fallback(error)
    if result.returncode != 0:
        error = {"command": command, "returncode": result.returncode, "stderr": result.stderr.strip()[:1000]}
        return _torch_gpu_fallback(error)

    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        values = [value.strip() for value in row]
        if not values or not any(values):
            continue
        if len(values) != 5:
            error = {
                "command": command,
                "error": f"unexpected nvidia-smi row with {len(values)} fields",
                "raw_output": result.stdout[:2000],
            }
            return _torch_gpu_fallback(error)
        devices.append(
            {
                "name": values[0],
                "uuid": values[1],
                "memory_total_mib": values[2],
                "memory_free_mib": values[3],
                "driver_version": values[4],
            }
        )
    return {
        "status": "pass" if devices else "no_gpu_visible",
        "command": command,
        "devices": devices,
        "note": "Metadata query only; no GPU workload was started.",
    }


def _find_artifact(directory: Path, filename: str) -> Path | None:
    direct = directory / filename
    if direct.is_file():
        return direct.resolve()
    if not directory.is_dir():
        return None
    matches = list(directory.rglob(filename))
    return matches[0].resolve() if len(matches) == 1 else None


def _inspect_split(path: Path, dire_root: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "fail", "path": str(path), "error": "split artifact not found"}
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
        source_root = Path(split["source"]["root"]).expanduser().resolve()
        valid = (
            split.get("status") == "candidate_created"
            and split.get("protocol", {}).get("name") == EXPECTED_SPLIT_PROTOCOL
            and split.get("content_overlap_audit", {}).get("status") == "pass"
            and split.get("counts_by_label") == EXPECTED_SPLIT_COUNTS
            and source_root == dire_root.resolve()
        )
        return {
            "status": "pass" if valid else "fail",
            "path": str(path.resolve()),
            "protocol": split.get("protocol", {}).get("name"),
            "validation_id_list_sha256": split.get("validation_id_list_sha256"),
            "counts_by_label": split.get("counts_by_label"),
            "content_overlap_status": split.get("content_overlap_audit", {}).get("status"),
            "source_root_matches_config": source_root == dire_root.resolve(),
            "error": None if valid else "split metadata/counts/source/audit do not match the accepted v2 gate",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"status": "fail", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def run_celebahq_training_preflight(
    *,
    dire_root: str | Path,
    stage2_artifact_dir: str | Path,
    split_manifest_path: str | Path,
    working_dir: str | Path,
    artifact_dir: str | Path,
    checkpoint_dir: str | Path,
    report_path: str | Path | None = None,
    nvidia_smi: str = "nvidia-smi",
) -> dict[str, Any]:
    """Collect a no-training resource/read-write preflight and save its report.

    The only writes are small, temporary probe files and the JSON report beneath
    ``working_dir``. Mounted input trees are inspected by path existence only and
    are never opened for writing or traversed recursively (other than the small
    Stage 2 artifact directory when looking up its two filenames).
    """
    work = Path(working_dir).expanduser().resolve()
    outputs = {
        "artifact_dir": Path(artifact_dir).expanduser().resolve(),
        "checkpoint_dir": Path(checkpoint_dir).expanduser().resolve(),
    }
    report = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else work / "celebahq-training-preflight.json"
    )
    input_mount = Path("/kaggle/input").resolve()
    for destination in (work, *outputs.values(), report):
        if _inside(destination, input_mount):
            raise ValueError(f"preflight output must not be written beneath Kaggle input: {destination}")
        if not _inside(destination, work):
            raise ValueError(f"preflight outputs must remain beneath working_dir: {destination}")

    dire = Path(dire_root).expanduser().resolve()
    stage2_dir = Path(stage2_artifact_dir).expanduser().resolve()
    input_checks = {
        "dire_root_exists": dire.is_dir(),
        "train_real_group_exists": (dire / "train/celebahq/real/real").is_dir(),
        "train_sdv2_group_exists": (dire / "train/celebahq/sdv2/sdv2").is_dir(),
        "stage2_artifact_dataset_exists": stage2_dir.is_dir(),
    }
    summary_file = _find_artifact(
        stage2_dir, "celebahq-integrity-81dcda99aeb997b25b82-summary.json"
    )
    manifest_file = _find_artifact(
        stage2_dir, "celebahq-integrity-81dcda99aeb997b25b82.jsonl"
    )
    input_checks["stage2_summary_exists"] = summary_file is not None
    input_checks["stage2_jsonl_exists"] = manifest_file is not None

    split_check = _inspect_split(Path(split_manifest_path).expanduser().resolve(), dire)
    write_probes = {name: _probe_writable(path) for name, path in outputs.items()}
    try:
        usage = shutil.disk_usage(work)
        disk = {
            "status": "pass",
            "path": str(work),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "note": "Reports current capacity only; sufficiency depends on the later approved training/checkpoint budget.",
        }
    except OSError as exc:
        disk = {"status": "fail", "path": str(work), "error": f"{type(exc).__name__}: {exc}"}

    gpu = _query_gpu(nvidia_smi=nvidia_smi)
    checks_passed = all(input_checks.values()) and split_check["status"] == "pass"
    checks_passed = checks_passed and all(probe["status"] == "pass" for probe in write_probes.values())
    checks_passed = checks_passed and disk["status"] == "pass" and gpu["status"] == "pass"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if checks_passed else "needs_attention",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Kaggle resource/path preflight only; no model compute",
        "inputs": {
            "dire_root": str(dire),
            "stage2_artifact_dir": str(stage2_dir),
            "stage2_summary": str(summary_file) if summary_file else None,
            "stage2_jsonl": str(manifest_file) if manifest_file else None,
        },
        "input_checks": input_checks,
        "split_check": split_check,
        "disk": disk,
        "write_probes": write_probes,
        "gpu_query": gpu,
        "execution_flags": {
            "image_contents_read": False,
            "input_write_attempted": False,
            "gpu_workload_started": False,
            "training_started": False,
        },
        "next_gate": "review this report, then decide seed/budget/checkpoint policy; this report does not authorize training",
        "report_path": str(report),
    }
    work.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    return result
