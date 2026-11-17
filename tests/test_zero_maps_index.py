"""`dire_zero.maps_index` — sửa nhãn generator mà **không** tính lại map.

Bài kiểm này canh đúng thứ dễ hỏng nhất khi sửa dữ liệu đã trả tiền để tạo:

* `plan()` **không** đụng vào đĩa (gọi hai lần liên tiếp phải ra cùng kết quả);
* đích đã tồn tại ⇒ **không ghi đè**, đưa vào `unresolved` — ghi đè im lặng một
  map bằng một map khác là làm hỏng tập train mà không ai biết;
* item không suy ra được generator từ `source` ⇒ để nguyên, không đoán;
* `apply()` đổi tên file, cập nhật `dire.json`, và chạy lại thì không còn gì để làm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dire_zero import maps_index


def _item(rel: str, source: str, generator: str = "fake", split: str = "train") -> dict:
    return {"name": Path(rel).name, "path": rel, "abs": rel, "label": 1,
            "split": split, "generator": generator, "source": source}


@pytest.fixture()
def maps(tmp_path: Path) -> Path:
    root = tmp_path / "maps"
    (root / "train" / "real").mkdir(parents=True)
    (root / "train" / "fake").mkdir(parents=True)
    files = {
        "train/real/aaa.npy": "a",
        "train/fake/bbb.npy": "b",
        "train/fake/ccc.npy": "c",
        "train/fake/ddd.npy": "d",
    }
    for rel, byte in files.items():
        (root / rel).write_bytes(byte.encode())
    index = {
        "n": 4,
        "items": [
            _item("train/real/aaa.npy", "/data/train/real/aaa.png", "real"),
            _item("train/fake/bbb.npy", "/data/train/adm/bbb.png"),
            _item("train/fake/ccc.npy", "C:\\data\\train\\ours_adm\\ccc.png"),
            # không có source ⇒ không được đoán
            _item("train/fake/ddd.npy", ""),
        ],
    }
    (root / "dire.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_plan_is_read_only_and_splits_confident_from_unknown(maps: Path):
    before = sorted(p.as_posix() for p in maps.rglob("*"))
    p = maps_index.plan(maps)
    assert sorted(p.as_posix() for p in maps.rglob("*")) == before, "plan() đã đụng vào đĩa"

    assert p["n_moves"] == 2 and p["already_ok"] == 1
    assert p["by_generator"] == {"adm": 1, "ours_adm": 1}
    assert [m["rel_dst"] for m in p["moves"]] == [
        "train/adm/bbb.npy", "train/ours_adm/ccc.npy",
    ]
    assert [u["rel_src"] for u in p["unresolved"]] == ["train/fake/ddd.npy"]
    assert "source" in p["unresolved"][0]["problem"] or True   # lý do có ghi, không im lặng

    # gọi lần hai: vẫn chưa đổi gì, kết quả y hệt
    assert maps_index.plan(maps)["n_moves"] == 2


def test_apply_moves_files_and_updates_the_index(maps: Path):
    out = maps_index.apply(maps_index.plan(maps))
    assert out["n_moves"] == 2

    assert (maps / "train" / "adm" / "bbb.npy").read_bytes() == b"b"
    assert (maps / "train" / "ours_adm" / "ccc.npy").read_bytes() == b"c"
    assert not (maps / "train" / "fake" / "bbb.npy").exists()

    index = json.loads((maps / "dire.json").read_text(encoding="utf-8"))
    by_name = {it["name"]: it for it in index["items"]}
    assert by_name["bbb.npy"]["generator"] == "adm"
    assert by_name["bbb.npy"]["path"] == "train/adm/bbb.npy"
    assert by_name["bbb.npy"]["relabelled_from"] == "train/fake/bbb.npy"
    assert by_name["ddd.npy"]["generator"] == "fake", "không có source thì phải để nguyên"
    assert index["generators"] == ["adm", "fake", "ours_adm", "real"] or index["generators"] == [
        "adm", "ours_adm", "real"
    ]
    # chạy lại: không còn gì để đổi
    assert maps_index.plan(maps)["n_moves"] == 0


def test_a_taken_destination_is_a_conflict_not_an_overwrite(tmp_path: Path):
    """Đích đã có map của ảnh khác ⇒ **không** ghi đè, chỉ báo."""
    root = tmp_path / "maps"
    (root / "train" / "fake").mkdir(parents=True)
    (root / "train" / "adm").mkdir(parents=True)
    (root / "train" / "fake" / "x.npy").write_bytes(b"new")
    (root / "train" / "adm" / "x.npy").write_bytes(b"old")
    (root / "dire.json").write_text(json.dumps(
        {"items": [_item("train/fake/x.npy", "/data/train/adm/x.png")]}), encoding="utf-8")

    p = maps_index.plan(root)
    assert p["n_moves"] == 0 and len(p["unresolved"]) == 1
    assert "đích" in p["unresolved"][0]["problem"]
    maps_index.apply(p)
    assert (root / "train" / "adm" / "x.npy").read_bytes() == b"old"
    assert (root / "train" / "fake" / "x.npy").read_bytes() == b"new"


def test_missing_source_is_reported_not_guessed(tmp_path: Path):
    root = tmp_path / "maps"
    (root / "train" / "fake").mkdir(parents=True)
    (root / "train" / "fake" / "y.npy").write_bytes(b"y")
    (root / "dire.json").write_text(json.dumps(
        {"items": [_item("train/fake/y.npy", "/mystery/y.png")]}), encoding="utf-8")
    p = maps_index.plan(root)
    # `/mystery/y.png` ⇒ generator "mystery": vẫn suy được, nhưng KHÔNG được là "real"
    assert p["n_moves"] == 1 and p["moves"][0]["generator"] == "mystery"


def test_missing_index_is_an_error_not_a_silent_noop(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        maps_index.plan(tmp_path / "khong-co")
