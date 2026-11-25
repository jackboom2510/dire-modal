from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_preflight import REQUIRED_ARCHIVES, preflight_dataset  # noqa: E402


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def _fixture_dataset(root: Path) -> Path:
    rows = []
    for i, (rel, _split, _generator) in enumerate(REQUIRED_ARCHIVES):
        members = [(f"sample-{i}.png", b"map-content-" + str(i).encode())]
        path = root / rel
        _write_tar(path, members)
        rows.append({
            "path": rel,
            "archive_bytes": path.stat().st_size,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "regular_members": 1,
            "image_members": 1,
            "suffix_counts": {".png": 1},
        })
    manifest = {
        "schema_version": 1,
        "dataset": "dire-celebahq-maps",
        "archives": rows,
    }
    manifest_path = root / "kaggle-map-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_preflight_validates_all_archives_without_modifying_input(tmp_path: Path) -> None:
    root = tmp_path / "input"
    workdir = tmp_path / "working"
    manifest = _fixture_dataset(root)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    report = preflight_dataset(root, manifest, workdir)

    assert report["status"] == "pass"
    assert report["gpu_work_started"] is False
    assert len(report["archives"]) == len(REQUIRED_ARCHIVES)
    assert all(row["status"] == "pass" for row in report["archives"])
    assert report["working_storage"]["writable"] is True
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert (workdir / "preflight-report.json").is_file()


def test_preflight_fails_on_archive_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "input"
    workdir = tmp_path / "working"
    manifest = _fixture_dataset(root)
    target = root / REQUIRED_ARCHIVES[1][0]
    target.write_bytes(target.read_bytes() + b"tampered")

    report = preflight_dataset(root, manifest, workdir)

    assert report["status"] == "fail"
    item = next(row for row in report["archives"] if row["path"] == REQUIRED_ARCHIVES[1][0])
    assert "sha256 mismatch" in " ".join(item["errors"])


def test_preflight_reports_missing_archives_and_member_count_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "input"
    workdir = tmp_path / "working"
    manifest = _fixture_dataset(root)
    target = root / REQUIRED_ARCHIVES[2][0]
    target.unlink()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["archives"][3]["regular_members"] = 9
    manifest.write_text(json.dumps(data), encoding="utf-8")

    report = preflight_dataset(root, manifest, workdir)

    assert report["status"] == "fail"
    missing = next(row for row in report["archives"] if row["path"] == REQUIRED_ARCHIVES[2][0])
    assert "missing archive" in " ".join(missing["errors"])
    count_mismatch = next(row for row in report["archives"] if row["path"] == REQUIRED_ARCHIVES[3][0])
    assert "regular member count mismatch" in " ".join(count_mismatch["errors"])


def test_preflight_refuses_report_path_inside_read_only_dataset(tmp_path: Path) -> None:
    root = tmp_path / "input"
    manifest = _fixture_dataset(root)

    report = preflight_dataset(root, manifest, tmp_path / "working", output_path=root / "report.json")

    assert report["status"] == "fail"
    assert any("outside the input dataset" in error for error in report["errors"])
    assert not (root / "report.json").exists()


def test_kaggle_preflight_notebook_runs_preflight_before_optional_smoke_cells() -> None:
    notebook = json.loads((ROOT / "platforms/kaggle/notebook.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    titles = [cell for cell in cells if "Read-only research data preflight" in cell]
    assert titles
    preflight_positions = [i for i, cell in enumerate(cells) if "dire_modal.kaggle_preflight" in cell]
    pack_positions = [i for i, cell in enumerate(cells) if "Pack 1 archive" in cell]
    train_positions = [i for i, cell in enumerate(cells) if "DataLoader + 1 bước train giả" in cell]
    assert len(preflight_positions) == 1
    assert pack_positions and train_positions
    assert preflight_positions[0] < pack_positions[0] < train_positions[0]
    assert "Accelerator → Off" in cells[0]
