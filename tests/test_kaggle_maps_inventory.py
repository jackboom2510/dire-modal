from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "kaggle_maps_inventory.py"

EXPECTED = [
    "dire/train/celebahq/real.tar.gz",
    "dire/train/celebahq/sdv2.tar.gz",
    "dire/test/celebahq/real.tar.gz",
    "dire/test/celebahq/dalle2.tar.gz",
    "dire/test/celebahq/if.tar.gz",
    "dire/test/celebahq/midjourney.tar.gz",
    "dire/test/celebahq/stargan.tar.gz",
    "dire/test/celebahq/sdv2.tar.gz",
]


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _dataset(root: Path) -> None:
    for i, rel in enumerate(EXPECTED):
        # Keep one common real member ID between train/test to exercise leak audit.
        sample = "real-shared.png" if rel.endswith("/real.tar.gz") else f"sample-{i}.png"
        _write_tar(root / rel, {f"0_real/{sample}": f"payload-{i}".encode()})


def _run(root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(root), "--out", str(out), *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_inventory_writes_read_only_manifest_with_hashes_and_families(tmp_path: Path) -> None:
    root = tmp_path / "source"
    out = tmp_path / "manifest.json"
    _dataset(root)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.tar.gz")}

    result = _run(root, out)

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [row["path"] for row in data["archives"]] == EXPECTED
    assert data["schema_version"] == 1
    assert data["split_policy"]["validation"] == "not assigned; decide after inventory"
    assert data["id_policy"]["kind"] == "candidate basename, not verified identity"
    train_sd = next(row for row in data["archives"] if row["path"] == EXPECTED[1])
    assert train_sd["generator"] == "sdv2"
    assert train_sd["generator_family"] == "stable-diffusion"
    assert train_sd["archive_sha256"] == hashlib.sha256((root / EXPECTED[1]).read_bytes()).hexdigest()
    assert train_sd["regular_members"] == 1
    assert train_sd["image_members"] == 1
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.tar.gz")} == before
    assert "read-only" in result.stdout.lower()


def test_inventory_flags_candidate_id_overlap_without_claiming_proven_leak(tmp_path: Path) -> None:
    root = tmp_path / "source"
    out = tmp_path / "manifest.json"
    _dataset(root)

    result = _run(root, out)

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    overlaps = data["candidate_id_overlaps"]
    assert any(
        row["candidate_id"] == "real-shared"
        and row["left_path"] == EXPECTED[0]
        and row["right_path"] == EXPECTED[2]
        and row["map_bytes_identical"] is None
        for row in overlaps
    )


def test_hash_members_can_confirm_exact_duplicate_bytes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    out = tmp_path / "manifest.json"
    _dataset(root)
    payload = b"same real map bytes"
    _write_tar(root / EXPECTED[0], {"0_real/same.png": payload})
    _write_tar(root / EXPECTED[2], {"0_real/same.png": payload})

    result = _run(root, out, "--hash-members")

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert any(
        row["candidate_id"] == "same"
        and row["map_bytes_identical"] is True
        and row["content_sha256"] == hashlib.sha256(payload).hexdigest()
        for row in data["candidate_id_overlaps"]
    )


def test_inventory_records_repeated_candidate_basenames(tmp_path: Path) -> None:
    root = tmp_path / "source"
    out = tmp_path / "manifest.json"
    _dataset(root)
    _write_tar(root / EXPECTED[0], {
        "0_real/camera-a/duplicate.png": b"first",
        "0_real/camera-b/duplicate.png": b"second",
    })

    result = _run(root, out)

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    train_real = next(row for row in data["archives"] if row["path"] == EXPECTED[0])
    assert train_real["duplicate_candidate_ids"] == ["duplicate"]


def test_manifest_output_must_not_modify_source_tree(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _dataset(root)
    out = root / "kaggle-manifest.json"

    result = _run(root, out)

    assert result.returncode != 0
    assert "outside --root" in result.stderr
    assert not out.exists()


def test_missing_or_invalid_archive_fails_without_manifest(tmp_path: Path) -> None:
    root = tmp_path / "source"
    out = tmp_path / "manifest.json"
    _dataset(root)
    (root / EXPECTED[4]).write_bytes(b"not a tar archive")

    result = _run(root, out)

    assert result.returncode != 0
    assert "if.tar.gz" in result.stdout + result.stderr
    assert not out.exists()
