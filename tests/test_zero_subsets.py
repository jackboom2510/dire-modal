from __future__ import annotations

from pathlib import Path

import pytest

from dire_zero.subsets import ALGORITHM, select_nested_rows

CELLS = [
    ("train", "real"), ("train", "adm"),
    ("val", "real"), ("val", "adm"),
    ("test", "real"), ("test", "adm"), ("test", "ddpm"),
    ("test", "iddpm"), ("test", "pndm"), ("test", "sdv2"),
]


def _rows(root: Path, n: int = 500) -> list[dict]:
    rows = []
    for split, generator in CELLS:
        for i in range(n):
            path = root / split / generator / f"{i:05}.png"
            rows.append({"path": str(path), "split": split, "generator": generator,
                         "label": 0 if generator == "real" else 1})
    return rows


def test_per_cell_subset_is_exact_size_and_stores_relative_ids(tmp_path: Path):
    selected, counts = select_nested_rows(_rows(tmp_path), per_cell=200, seed=3407,
                                          source_root=tmp_path, required_cells=CELLS)
    assert len(selected) == 2_000
    assert set(counts.values()) == {200}
    assert all(not Path(row["sample_id"]).is_absolute() for row in selected)
    assert ALGORITHM == "sha256-ranked-cell-v1"


def test_larger_stage_is_nested_and_order_independent(tmp_path: Path):
    rows = _rows(tmp_path)
    small, _ = select_nested_rows(rows, per_cell=200, seed=123, source_root=tmp_path,
                                  required_cells=CELLS)
    large, _ = select_nested_rows(reversed(rows), per_cell=400, seed=123,
                                  source_root=tmp_path, required_cells=CELLS)
    small_ids = {(r["split"], r["generator"], r["sample_id"]) for r in small}
    large_ids = {(r["split"], r["generator"], r["sample_id"]) for r in large}
    assert len(large) == 4_000
    assert small_ids < large_ids


def test_subset_rejects_missing_required_cells_and_duplicates(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="thiếu cell"):
        select_nested_rows([], per_cell=200, seed=0, source_root=tmp_path,
                           required_cells=[("train", "adm")])
    duplicate = _rows(tmp_path, n=1)
    duplicate.append(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        select_nested_rows(duplicate, per_cell=1, seed=0, source_root=tmp_path,
                           required_cells=CELLS)
