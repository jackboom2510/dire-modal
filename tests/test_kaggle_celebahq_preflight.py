from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_preflight import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_PROTOCOL,
    _query_gpu,
    run_celebahq_training_preflight,
)


def _fake_nvidia_smi(path: Path, *, output: str = "Test GPU, GPU-uuid, 24576, 24000, 555.1\n") -> Path:
    path.write_text(f"#!/bin/sh\nprintf '%s' {output!r}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _preflight_fixture(base: Path) -> dict[str, Path]:
    dire = base / "input" / "dire"
    for relative in (
        "train/celebahq/real/real",
        "train/celebahq/sdv2/sdv2",
    ):
        (dire / relative).mkdir(parents=True, exist_ok=True)
    artifacts = base / "input" / "stage2-artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "celebahq-integrity-81dcda99aeb997b25b82-summary.json").write_text("{}")
    (artifacts / "celebahq-integrity-81dcda99aeb997b25b82.jsonl").write_text("{}\n")
    work = base / "working"
    work.mkdir()
    split_path = work / "celebahq-validation-split-v2.json"
    split_path.write_text(
        json.dumps(
            {
                "status": "candidate_created",
                "protocol": {"name": EXPECTED_SPLIT_PROTOCOL},
                "content_overlap_audit": {"status": "pass"},
                "counts_by_label": EXPECTED_SPLIT_COUNTS,
                "validation_id_list_sha256": "a" * 64,
                "source": {"root": str(dire)},
            }
        ),
        encoding="utf-8",
    )
    return {"dire": dire, "artifacts": artifacts, "work": work, "split": split_path}


def test_preflight_is_read_only_except_temporary_working_probes(tmp_path: Path) -> None:
    fixture = _preflight_fixture(tmp_path)
    gpu_binary = _fake_nvidia_smi(tmp_path / "fake-nvidia-smi")
    input_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "input").rglob("*")
        if path.is_file()
    }
    result = run_celebahq_training_preflight(
        dire_root=fixture["dire"],
        stage2_artifact_dir=fixture["artifacts"],
        split_manifest_path=fixture["split"],
        working_dir=fixture["work"],
        artifact_dir=fixture["work"] / "artifacts",
        checkpoint_dir=fixture["work"] / "checkpoints",
        nvidia_smi=str(gpu_binary),
    )

    assert result["status"] == "pass"
    assert result["gpu_query"]["devices"][0]["name"] == "Test GPU"
    assert result["split_check"]["status"] == "pass"
    assert result["disk"]["status"] == "pass"
    assert result["disk"]["free_bytes"] > 0
    assert result["execution_flags"] == {
        "image_contents_read": False,
        "input_write_attempted": False,
        "gpu_workload_started": False,
        "training_started": False,
    }
    assert all(probe["status"] == "pass" for probe in result["write_probes"].values())
    assert Path(result["report_path"]).is_file()
    assert not list(fixture["work"].rglob(".celebahq-preflight-*"))
    input_after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "input").rglob("*")
        if path.is_file()
    }
    assert input_after == input_before


def test_preflight_reports_missing_gpu_without_gpu_work(tmp_path: Path) -> None:
    fixture = _preflight_fixture(tmp_path)
    gpu_binary = _fake_nvidia_smi(tmp_path / "fake-nvidia-smi-empty", output="")
    result = run_celebahq_training_preflight(
        dire_root=fixture["dire"],
        stage2_artifact_dir=fixture["artifacts"],
        split_manifest_path=fixture["split"],
        working_dir=fixture["work"],
        artifact_dir=fixture["work"] / "artifacts",
        checkpoint_dir=fixture["work"] / "checkpoints",
        nvidia_smi=str(gpu_binary),
    )
    assert result["status"] == "needs_attention"
    assert result["gpu_query"]["status"] == "no_gpu_visible"
    assert result["execution_flags"]["gpu_workload_started"] is False


def test_preflight_rejects_outputs_under_input(tmp_path: Path) -> None:
    fixture = _preflight_fixture(tmp_path)
    with pytest.raises(ValueError, match="beneath Kaggle input|beneath working_dir"):
        run_celebahq_training_preflight(
            dire_root=fixture["dire"],
            stage2_artifact_dir=fixture["artifacts"],
            split_manifest_path=fixture["split"],
            working_dir=fixture["work"],
            artifact_dir=Path("/kaggle/input/forbidden-artifacts"),
            checkpoint_dir=fixture["work"] / "checkpoints",
            nvidia_smi="missing",
        )


def test_gpu_query_uses_torch_metadata_fallback_without_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(name="Fallback GPU", total_memory=24 * 1024**3, uuid="GPU-fallback")

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            assert index == 0
            return 20 * 1024**3, 24 * 1024**3

    fake_torch = SimpleNamespace(cuda=FakeCuda(), version=SimpleNamespace(cuda="12.4"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = _query_gpu(nvidia_smi=str(Path("/missing/nvidia-smi")))

    assert result["status"] == "pass"
    assert result["fallback"] == "torch.cuda metadata"
    assert result["devices"][0]["name"] == "Fallback GPU"
    assert "no tensor/model allocation" in result["note"]


def test_gpu_query_fallback_reports_no_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = _query_gpu(nvidia_smi=str(Path("/missing/nvidia-smi")))
    assert result["status"] == "no_gpu_visible"
    assert result["fallback"] == "torch.cuda metadata"
