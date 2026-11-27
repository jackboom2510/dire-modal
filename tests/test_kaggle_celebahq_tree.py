from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dire_modal.kaggle_celebahq_tree import (  # noqa: E402
    CelebAHQTreeLayoutError,
    REQUIRED_GROUPS,
    scan_celebahq_tree,
)


def _fixture(root: Path, *, nested: bool = True) -> None:
    for split, generator in REQUIRED_GROUPS:
        outer = root / split / "celebahq" / generator
        leaf = outer / generator if nested else outer
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / f"sample-{split}-{generator}.png").write_bytes(b"fixture-image")


def test_scans_repeated_generator_leaf_without_modifying_input(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    _fixture(root)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    rows = scan_celebahq_tree(root)

    assert len(rows) == len(REQUIRED_GROUPS)
    assert all(row.leaf_layout == "repeated-generator-child" for row in rows)
    assert all(row.locator_inferred is True for row in rows)
    assert all(row.member_locator == f"{row.generator}/sample-{row.split}-{row.generator}.png" for row in rows)
    assert all(row.relative_path == (
        f"{row.split}/celebahq/{row.generator}/{row.generator}/sample-{row.split}-{row.generator}.png"
    ) for row in rows)
    assert {row.generator: row.label for row in rows if row.split == "train"} == {
        "real": 0,
        "sdv2": 1,
    }
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_scans_flat_leaf_layout_with_same_inferred_locator(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    _fixture(root, nested=False)

    rows = scan_celebahq_tree(root)

    assert len(rows) == len(REQUIRED_GROUPS)
    assert all(row.leaf_layout == "flat-class-directory" for row in rows)
    assert all(row.member_locator == f"{row.generator}/sample-{row.split}-{row.generator}.png" for row in rows)


def test_rejects_mixed_direct_and_repeated_images(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    _fixture(root)
    outer = root / "train/celebahq/real"
    (outer / "direct.png").write_bytes(b"direct")

    with pytest.raises(CelebAHQTreeLayoutError, match="ambiguous class directory"):
        scan_celebahq_tree(root)


def test_rejects_unexpected_nested_image_folder(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    _fixture(root)
    extra = root / "train/celebahq/real/backup"
    extra.mkdir()
    (extra / "other.png").write_bytes(b"unexpected")

    with pytest.raises(CelebAHQTreeLayoutError, match="unexpected nested image directories"):
        scan_celebahq_tree(root)


def test_reports_all_missing_required_groups(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    root.mkdir()

    with pytest.raises(CelebAHQTreeLayoutError, match="missing or empty required") as exc:
        scan_celebahq_tree(root)
    assert "train/celebahq/real" in str(exc.value)
    assert "test/celebahq/stargan" in str(exc.value)


def test_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="DIRE root is not a directory"):
        scan_celebahq_tree(tmp_path / "missing")
