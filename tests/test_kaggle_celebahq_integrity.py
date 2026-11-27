from __future__ import annotations

import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_integrity import verify_celebahq_records  # noqa: E402
from dire_modal.kaggle_celebahq_tree import REQUIRED_GROUPS, scan_celebahq_tree  # noqa: E402


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (3, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def _fixture(root: Path) -> list[Any]:
    for i, (split, generator) in enumerate(REQUIRED_GROUPS):
        leaf = root / split / "celebahq" / generator / generator
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / f"sample-{split}-{generator}.png").write_bytes(_png_bytes((i, 20, 30)))
    return scan_celebahq_tree(root)


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_integrity_hashes_decodes_and_writes_only_outside_input(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    workdir = tmp_path / "working"
    records = _fixture(root)
    before = _snapshot(root)

    result = verify_celebahq_records(root, records, workdir, checkpoint_every=2, progress_every=4)

    assert result["status"] == "pass"
    assert result["records_expected"] == len(REQUIRED_GROUPS)
    assert result["records_in_checkpoint"] == len(REQUIRED_GROUPS)
    assert result["hashed_this_run"] == len(REQUIRED_GROUPS)
    assert result["reused_from_checkpoint"] == 0
    assert result["hash_failures"] == 0
    assert result["decode_failures"] == 0
    assert result["total_file_bytes_by_stat"] == sum(len(v) for v in before.values())
    assert Path(result["checkpoint_sqlite"]).is_file()
    assert Path(result["file_manifest_jsonl"]).is_file()
    assert Path(result["summary_json"]).is_file()
    lines = Path(result["file_manifest_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(REQUIRED_GROUPS) + 1
    manifest_rows = [json.loads(line) for line in lines]
    assert manifest_rows[0]["record_type"] == "metadata"
    by_path = {row["path_from_dire_root"]: row for row in manifest_rows[1:]}
    for record in records:
        row = by_path[record.relative_path]
        assert row["sha256"] == hashlib.sha256(record.path.read_bytes()).hexdigest()
        assert row["decode_status"] == "pass"
        assert row["width"] == 3 and row["height"] == 2
        assert row["locator_inferred"] is True
    assert _snapshot(root) == before


def test_integrity_rerun_reuses_checkpoint_for_same_immutable_tree(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    records = _fixture(root)
    workdir = tmp_path / "working"

    first = verify_celebahq_records(root, records, workdir)
    second = verify_celebahq_records(root, records, workdir)

    assert first["source_fingerprint"] == second["source_fingerprint"]
    assert second["hashed_this_run"] == 0
    assert second["reused_from_checkpoint"] == len(REQUIRED_GROUPS)
    assert second["status"] == "pass"


def test_integrity_records_corrupt_image_but_keeps_its_hash(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    records = _fixture(root)
    broken = records[0]
    broken.path.write_bytes(b"not an image")

    result = verify_celebahq_records(root, records, tmp_path / "working")

    assert result["status"] == "fail"
    assert result["hash_failures"] == 0
    assert result["decode_failures"] == 1
    rows = [json.loads(line) for line in Path(result["file_manifest_jsonl"]).read_text().splitlines()][1:]
    bad = next(row for row in rows if row["path_from_dire_root"] == broken.relative_path)
    assert bad["sha256"] == hashlib.sha256(b"not an image").hexdigest()
    assert bad["decode_status"] == "fail"
    assert "decode:" in bad["error"]


def test_integrity_refuses_output_inside_read_only_input(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    records = _fixture(root)

    with pytest.raises(ValueError, match="outside the read-only input dataset"):
        verify_celebahq_records(root, records, root / "report")
