"""Deterministic nested, stratified paper-experiment subsets.

Subset manifests store source paths and IDs; they do not copy or modify image files.
The same seed and per-cell ordering make a larger stage a strict superset of every
smaller stage, so cached per-image DIRE inputs can be reused safely.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

ALGORITHM = "sha256-ranked-cell-v1"


def _row_key(row: Mapping[str, object], source_root: str | Path) -> tuple[str, str, str]:
    path = Path(str(row["path"]))
    try:
        sample_id = path.resolve().relative_to(Path(source_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"subset source path is outside data root: {path}") from exc
    split, generator = str(row["split"]), str(row["generator"])
    return split, generator, sample_id


def select_nested_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    per_cell: int,
    seed: int,
    source_root: str | Path,
    required_cells: Iterable[tuple[str, str]],
) -> tuple[list[dict], dict[str, int]]:
    """Pick up to ``per_cell`` rows per (split, generator), deterministically.

    Ranking is independent per cell and stable under input iteration order. The
    rank prefix at ``per_cell=200`` is therefore contained in the prefix at 400.
    ``sample_id`` is a normalized relative path under ``source_root``.
    """
    if per_cell < 1:
        raise ValueError("per_cell phải lớn hơn 0")
    grouped: dict[tuple[str, str], list[tuple[str, str, dict]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        row = dict(raw)
        split, generator, sample_id = _row_key(row, source_root)
        identity = (split, generator, sample_id)
        if identity in seen:
            raise ValueError(f"duplicate subset sample: {identity}")
        seen.add(identity)
        rank = hashlib.sha256(f"{seed}\0{split}\0{generator}\0{sample_id}".encode("utf-8")).hexdigest()
        row["sample_id"] = sample_id
        grouped[(split, generator)].append((rank, sample_id, row))

    required = set((str(s), str(g)) for s, g in required_cells)
    absent = sorted(required - set(grouped))
    if absent:
        raise FileNotFoundError(f"subset input thiếu cell bắt buộc: {absent}")

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for cell in sorted(grouped):
        candidates = sorted(grouped[cell], key=lambda item: (item[0], item[1]))
        picked = candidates[:per_cell]
        counts[f"{cell[0]}/{cell[1]}"] = len(picked)
        selected.extend(row for _, _, row in picked)
    selected.sort(key=lambda row: (str(row["split"]), str(row["generator"]), str(row["sample_id"])))
    return selected, counts
