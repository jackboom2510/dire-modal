from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "platforms/kaggle/dire-tree-audit.ipynb"


def _audit_function() -> Any:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    definition_cell = "".join(notebook["cells"][1]["source"])
    namespace: dict[str, Any] = {}
    exec(compile(definition_cell, str(NOTEBOOK), "exec"), namespace)
    return namespace["audit_tree"]


def test_read_only_tree_audit_reports_same_name_nesting_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    (root / "train/lsun_bedroom/real/real").mkdir(parents=True)
    (root / "train/lsun_bedroom/real/direct.jpg").write_bytes(b"fake-jpg")
    (root / "train/lsun_bedroom/real/real/0001.png").write_bytes(b"fake-png")
    (root / "train/lsun_bedroom/adm/adm").mkdir(parents=True)
    (root / "train/lsun_bedroom/adm/adm/0002.tar.gz").write_bytes(b"fake-tar")
    (root / "test/imagenet/imagenet").mkdir(parents=True)
    (root / "test/imagenet/imagenet/real.png").write_bytes(b"another-png")
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    report = _audit_function()(root)

    assert report["status"] == "complete"
    assert report["input_modified"] is False
    assert report["file_contents_read"] is False
    assert report["gpu_work_started"] is False
    assert report["regular_file_count"] == 4
    assert report["image_like_file_count_by_suffix_only"] == 3
    assert report["suffix_counts"] == {".jpg": 1, ".png": 2, ".tar.gz": 1}
    candidates = {row["child_directory"]: row for row in report["same_name_nesting_candidates"]}
    assert candidates["train/lsun_bedroom/real/real"]["file_count_in_child_subtree"] == 1
    assert candidates["train/lsun_bedroom/adm/adm"]["file_count_in_child_subtree"] == 1
    assert candidates["test/imagenet/imagenet"]["file_count_in_child_subtree"] == 1
    train_domain = next(
        row for row in report["split_domain_counts"]
        if row["split"] == "train" and row["domain"] == "lsun_bedroom"
    )
    assert train_domain["file_count"] == 3
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_tree_audit_does_not_follow_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "dire"
    target = tmp_path / "outside"
    root.mkdir()
    target.mkdir()
    (target / "hidden.png").write_bytes(b"outside")
    try:
        (root / "linked").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    report = _audit_function()(root)

    assert report["regular_file_count"] == 0
    assert report["symlinks_not_followed"] == 1


def test_tree_audit_rejects_missing_root(tmp_path: Path) -> None:
    audit = _audit_function()

    try:
        audit(tmp_path / "missing")
    except NotADirectoryError as exc:
        assert "DIRE root is not a directory" in str(exc)
    else:
        raise AssertionError("expected missing audit root to be rejected")
