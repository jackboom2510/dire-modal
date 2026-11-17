"""App Modal — kiểm phần *khai báo* chạy được, và kiểm nó không đụng đường cũ.

`modal` là package thật trong môi trường dev (xem `requirements-dev.txt`), nên
import thật chứ không stub: stub sẽ bỏ qua đúng những lỗi decorator/tham số mà
file này dễ mắc.

Hai điều được canh:

* **Không lẫn đường dẫn** với đường scratch cũ (`/raw/scratch/**`). Dùng lẫn thì
  hai bộ số khác nhau nằm chung một chỗ, và không ai biết báo cáo đang đọc cái nào.
* **Tên hàm khớp lệnh trong tài liệu**: `modal run -m dire_zero.modal_app::gen`
  phải tồn tại, nếu không runbook sai ngay dòng đầu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")

from dire_zero import modal_app  # noqa: E402


def test_app_name_and_image():
    assert modal_app.APP_NAME == "dire-zero"
    # Một volume, một gốc — bố cục cũ hai volume đã bị thay (xem dire_zero/volume.py).
    assert modal_app.VOLUMES.keys() == {"/zero"}


def test_documented_entrypoints_exist():
    for name in ("selfcheck", "bench", "gen", "dire", "train", "report"):
        fn = getattr(modal_app, name)
        assert hasattr(fn, "local") or callable(fn), name


def test_paths_come_from_the_single_layout_module():
    """Đường dẫn phải lấy từ `volume.py`; sáu đường cũ cho cùng một checkpoint là
    đúng lỗi mà module đó sinh ra để chấm dứt."""
    from dire_zero import volume as V

    for value in (modal_app.CKPT, modal_app.DATA, modal_app.GEN, modal_app.MAPS, modal_app.OUT):
        assert value.startswith(V.MOUNT + "/"), value
        # không còn trỏ vào volume cũ nào (mount cũ là /raw và /ckpt)
        assert not value.startswith(("/raw", "/ckpt/")), value
        assert "/scratch" not in value

    assert modal_app.CKPT == V.CKPT
    assert modal_app.MANIFEST == V.MANIFEST
    assert modal_app.OUT == V.RUNS


def test_checkpoint_keeps_its_filename_so_nothing_has_to_be_downloaded_again():
    """Đổi thư mục được, đổi **tên file** thì không: 2,5 GB đã nằm trên Volume,
    và `migrate` tìm theo tên. Đổi tên là buộc người dùng tải lại."""
    assert modal_app.CKPT.endswith("adm_lsun.pt")


# --------------------------------------------------------------------------- #
# `::train --generator`: một dòng của Table 2, không phải bài toán gộp
# --------------------------------------------------------------------------- #

def _maps(tmp_path: Path, gens: dict[str, int], n_real: int = 3, split: str = "test") -> Path:
    """Cây `maps/<split>/<gen>/*.npy` giả — đủ để `collect_maps` đọc."""
    root = tmp_path / "maps"
    for gen, n in [("real", n_real), *gens.items()]:
        d = root / split / gen
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{gen}{i}.npy").write_bytes(b"npy")
    return root


def test_train_rejects_a_generator_that_has_no_maps(tmp_path: Path, monkeypatch):
    import ast
    from pathlib import Path as P

    src = P(modal_app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    body = ast.unparse(fn)
    # Lời nhắn phải chỉ đúng lệnh cần chạy trước, không chỉ "không tìm thấy".
    assert "::dire --generator" in body
    assert 'generator: str=' in body and "'auto'" in body


def test_collect_maps_labels_come_from_directory_names(tmp_path: Path):
    """Nhãn đọc từ tên thư mục generator, nên gộp 14 generator không thể lẫn nhãn."""
    pytest.importorskip("torch")        # `pipeline` import torch ở tầng module
    from dire_zero.pipeline import collect_maps

    root = _maps(tmp_path, {"adm": 2, "pndm": 4})
    maps = collect_maps(root)
    assert sum(1 for m in maps if m["label"] == 0) == 3            # real
    assert sum(1 for m in maps if m["label"] == 1) == 6
    assert {m["generator"] for m in maps} == {"real", "adm", "pndm"}
    # lọc đúng như `::train --generator adm`
    only = [m for m in maps if m["generator"] in ("real", "adm")]
    assert len(only) == 5 and {m["label"] for m in only} == {0, 1}


def test_train_payload_carries_the_scope_note(tmp_path: Path):
    """`train.json` phải tự nói bài toán là một generator hay gộp nhiều cái."""
    import ast
    from pathlib import Path as P

    src = P(modal_app.__file__).read_text(encoding="utf-8")
    body = ast.unparse(next(n for n in ast.parse(src).body
                            if isinstance(n, ast.FunctionDef) and n.name == "train"))
    assert "generators_used" in body and "note_scope" in body
    assert "Table 2" in body


def test_gpu_functions_declare_a_gpu():
    """Bước nào cần GPU phải khai GPU trong decorator — thiếu thì Modal chạy CPU
    và mọi ước lượng thời gian sai một bậc, không có lỗi nào báo."""
    import ast
    from pathlib import Path

    src = Path(modal_app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"selfcheck", "bench", "gen", "dire", "train"}
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "function":
                for kw in dec.keywords:
                    if kw.arg == "gpu":
                        found[node.name] = ast.literal_eval(kw.value)
    assert set(found) == wanted, f"thiếu: {wanted - set(found)}"
    assert set(found.values()) == {"A10G"}


# --------------------------------------------------------------------------- #
# Cổng dữ liệu: ba nguyên nhân khác nhau, ba thông báo khác nhau
# --------------------------------------------------------------------------- #

HDR = "png\tlabel\tgenerator\tsplit\tdataset\tsource_tar\tmember\tsha256\twidth\theight\n"


@pytest.fixture()
def volume_gia(tmp_path: Path, monkeypatch):
    """Trỏ `_check_data` vào một `data/` giả trong tmp_path."""
    data = tmp_path / "data"
    (data / "train" / "real").mkdir(parents=True)
    (data / "train" / "adm").mkdir(parents=True)
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MANIFEST", str(data / "manifest.tsv"))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    return tmp_path


def _run(**kw) -> dict:
    return modal_app._check_data(**kw)


def _write_ok(data: Path, n_per_class: int = 2) -> None:
    """Manifest + data_card + PNG thật, để ca "mọi thứ ổn" chạy được."""
    import hashlib

    rows = [HDR]
    for split in ("train", "test"):
        for gen, label in (("real", 0), ("adm", 1)):
            d = data / split / gen
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                png = d / f"{i:016x}.png"
                png.write_bytes(b"png" + bytes([i]))
                rows.append(f"{split}/{gen}/{png.name}\t{label}\t{gen}\t{split}\tlsun\t"
                            f"\tmember\t{hashlib.sha256(png.read_bytes()).hexdigest()}\t256\t256\n")
    (data / "manifest.tsv").write_text("".join(rows), encoding="utf-8")
    (data / "data_card.json").write_text(json.dumps({
        "label": "R1+R2", "fake_source": "author", "dataset": "lsun_bedroom",
        "license": "research-only", "comparable_to_paper": True,
    }), encoding="utf-8")


def test_missing_manifest_says_it_is_a_root_file_and_gives_the_command(volume_gia: Path):
    """Không phải "manifest rỗng": file **không lên**. Đẩy từng thư mục con là
    cách nó bị bỏ sót, nên thông báo phải nói đúng như vậy."""
    with pytest.raises(RuntimeError) as exc:
        _run(skip_sha=True)
    msg = str(exc.value)
    assert "manifest.tsv" in msg and "gốc" in msg
    assert "modal volume put dire-zero" in msg
    assert "manifest rỗng" not in msg          # ca khác, cách sửa khác


def test_empty_manifest_is_a_different_message_from_a_missing_one(volume_gia: Path):
    (volume_gia / "data" / "manifest.tsv").write_text(HDR, encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        _run(skip_sha=True)
    msg = str(exc.value)
    assert "chỉ có dòng tiêu đề" in msg
    assert "không cần đẩy lại ảnh" in msg


def test_missing_data_card_is_reported_on_its_own(volume_gia: Path):
    _write_ok(volume_gia / "data")
    (volume_gia / "data" / "data_card.json").unlink()
    with pytest.raises(RuntimeError) as exc:
        _run(skip_sha=True)
    assert "data_card.json" in str(exc.value)


def test_a_healthy_volume_passes_and_counts_png_on_disk(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=2)
    report = _run(count_png=True)
    assert report["problems"] == []
    assert report["n_rows"] == 8 and report["n_png_on_disk"] == 8
    assert report["n_real"] == 4 and report["n_fake"] == 4
    assert report["sha_scope"] == "full" and report["manifest_sha_ok"] == 8
    assert report["warnings"] == []


def test_counting_png_on_disk_is_off_by_default(volume_gia: Path):
    """Một vòng `os.walk` qua 217 585 entry trên FUSE tốn hàng phút, và nó không
    chặn bước nào — nên mặc định không đếm."""
    _write_ok(volume_gia / "data", n_per_class=1)
    assert _run()["n_png_on_disk"] is None


def test_hashing_is_parallel_and_reports_how_long_it_took(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=3)
    report = _run(workers=8)
    assert report["manifest_sha_ok"] == 12 and report["sha_workers"] == 8
    assert report["sha_checked_rows"] == 12
    assert report["sha_seconds"] >= 0.0
    # cùng kết quả với một luồng — song song không được đổi câu trả lời
    assert _run(workers=1)["manifest_sha_ok"] == report["manifest_sha_ok"]


def test_splits_filter_limits_the_work_and_says_so(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=2)
    report = _run(splits="test")
    assert report["filtered_splits"] == ["test"]
    assert report["sha_checked_rows"] == 4          # chỉ 2 lớp × 2 ảnh của test
    assert set(report["by_split_generator"]) == {"test/real", "test/adm"}


def test_sample_is_deterministic_and_labelled_as_a_sample(volume_gia: Path):
    """Mẫu phải nói rõ là mẫu — nếu không, báo cáo bị đọc thành "đã kiểm toàn bộ"."""
    _write_ok(volume_gia / "data", n_per_class=5)   # 20 dòng
    a, b = _run(sample=5), _run(sample=5)
    assert a["sha_scope"] == "sample" and a["sha_checked_rows"] == 5
    assert a["sha_checked_rows"] == b["sha_checked_rows"]
    assert a["manifest_sha_ok"] == b["manifest_sha_ok"]

    full = _run()
    assert full["sha_scope"] == "full" and full["sha_checked_rows"] == 20


def test_sample_larger_than_the_manifest_is_the_whole_manifest(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=1)   # 4 dòng
    report = _run(sample=100)
    assert report["sha_scope"] == "full" and report["sha_checked_rows"] == 4


def test_extra_pngs_outside_the_manifest_warn_but_do_not_block(volume_gia: Path):
    """`::gen` ghi `data/<split>/ours_adm` mà manifest không biết — bình thường."""
    _write_ok(volume_gia / "data", n_per_class=1)
    for i in range(3):
        (volume_gia / "data" / "train" / "ours_adm").mkdir(parents=True, exist_ok=True)
        (volume_gia / "data" / "train" / "ours_adm" / f"{i:016x}.png").write_bytes(b"x")
    assert _run()["n_png_on_disk"] is None           # không đếm trừ khi được yêu cầu
    report = _run(count_png=True)
    assert report["problems"] == []
    assert report["n_png_on_disk"] == report["n_rows"] + 3
    assert any("ours_adm" in w for w in report["warnings"])


def _write_multi(volume_gia: Path, n_real: int, gens: dict[str, int], split: str = "test") -> None:
    """`{tên generator: số ảnh}`, ảnh thật `n_real`. Đúng cấu trúc DiffusionForensics."""
    import hashlib

    data = volume_gia / "data"
    rows = [HDR]
    for gen, n in [("real", n_real), *gens.items()]:
        label = 0 if gen == "real" else 1
        d = data / split / gen
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            png = d / f"{gen}{i:012x}.png"
            png.write_bytes(b"png" + bytes([i % 251]))
            rows.append(f"{split}/{gen}/{png.name}\t{label}\t{gen}\t{split}\tlsun\t"
                        f"\tmember\t{hashlib.sha256(png.read_bytes()).hexdigest()}\t256\t256\n")
    (data / "manifest.tsv").write_text("".join(rows), encoding="utf-8")
    (data / "data_card.json").write_text(json.dumps({
        "label": "R1+R2", "fake_source": "author", "dataset": "lsun_bedroom",
        "license": "research-only", "comparable_to_paper": True,
    }), encoding="utf-8")


def test_many_generators_against_one_real_class_is_not_an_imbalance(volume_gia: Path):
    """Ca thật: 1000 thật + 12 368 giả của **14** generator (mỗi cái ~1000).

    Bản cũ kiểm tỉ lệ trên tổng ⇒ 1:12,4 ⇒ chặn, trong khi mỗi bài toán
    `real vs một generator` vẫn 1:1. Table 2 chính là từng bài toán như vậy.
    """
    _write_multi(volume_gia, 40, {"adm": 40, "pndm": 40, "stylegan": 40, "ldm": 40})
    report = _run(skip_sha=True)

    assert report["problems"] == []
    assert report["n_real"] == 40 and report["n_fake"] == 160
    assert len(report["tasks"]) == 4
    assert all(t["ok"] and t["ratio"] == 1.0 for t in report["tasks"])
    # nhưng phải cảnh báo rằng gộp tất cả là bài toán khác Table 2
    assert any("Table 2" in w for w in report["warnings"])


def test_one_unbalanced_task_is_still_caught(volume_gia: Path):
    _write_multi(volume_gia, 20, {"adm": 20, "pndm": 400})
    with pytest.raises(RuntimeError) as exc:
        _run(skip_sha=True)
    msg = str(exc.value)
    assert "test/pndm" in msg and "lệch quá 5:1" in msg
    assert "test/adm" not in msg                      # bài toán cân vẫn ổn


def test_generator_without_real_images_in_the_same_split_blocks(volume_gia: Path):
    _write_multi(volume_gia, 20, {"adm": 20}, split="test")
    ad = volume_gia / "data" / "train" / "adm"      # fixture đã tạo sẵn thư mục này
    (ad / "x.png").write_bytes(b"x")
    rows = (volume_gia / "data" / "manifest.tsv").read_text(encoding="utf-8")
    import hashlib
    (volume_gia / "data" / "manifest.tsv").write_text(
        rows + f"train/adm/x.png\t1\tadm\ttrain\tlsun\t\tmember\t"
               f"{hashlib.sha256(b'x').hexdigest()}\t256\t256\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        _run(skip_sha=True)
    assert "không có ảnh thật cùng split" in str(exc.value) and "train/adm" in str(exc.value)


def test_max_ratio_is_configurable(volume_gia: Path):
    _write_multi(volume_gia, 10, {"adm": 30})         # 3:1
    assert _run(skip_sha=True)["problems"] == []
    with pytest.raises(RuntimeError):
        _run(skip_sha=True, max_ratio=2.0)


def test_tampered_png_is_caught_by_sha(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=1)
    victim = sorted((volume_gia / "data" / "train" / "real").glob("*.png"))[0]
    victim.write_bytes(b"khac han")
    with pytest.raises(RuntimeError) as exc:
        _run()
    assert "sha256" in str(exc.value)


def test_skip_sha_skips_hashing_but_still_checks_structure(volume_gia: Path):
    _write_ok(volume_gia / "data", n_per_class=1)
    victim = sorted((volume_gia / "data" / "train" / "real").glob("*.png"))[0]
    victim.write_bytes(b"khac han")
    report = _run(skip_sha=True)
    assert report["problems"] == [] and report["sha_scope"] == "none"
