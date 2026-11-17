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
    for name in ("selfcheck", "bench", "gen", "dire", "relabel", "train", "report"):
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


def test_plan_images_caps_every_cell_by_n_and_not_the_whole_tree(tmp_path):
    """Ước lượng tiền phải đếm **đúng số ảnh sẽ chạy**.

    Lỗi thật: với `train,test` + `--generator adm`, cây có 82 000 ảnh nên ước
    lượng báo 91,90 USD, trong khi `--n 200` chỉ chạy 800 ảnh (~0,90 USD). Hệ quả
    không chỉ là con số sai: `--max-usd` mặc định 5 **chặn oan** lượt chạy rẻ.
    """
    for split, n in (("train", 5), ("test", 3)):
        for cls in ("real", "adm"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i:02d}.png").write_bytes(b"")

    n_real, n_fake = modal_app._plan_images(str(tmp_path), ["train", "test"], ["adm"], 2)
    # train: min(5,2)=2 thật + 2 giả; test: min(3,2)=2 + 2
    assert (n_real, n_fake) == (4, 4)

    # Không cắt: đếm đủ cả cây
    n_real, n_fake = modal_app._plan_images(str(tmp_path), ["train", "test"], ["adm"], 10_000)
    assert (n_real, n_fake) == (8, 8)

    # Một split, không có generator đó ⇒ 0 giả, không âm thầm đếm real của split khác
    n_real, n_fake = modal_app._plan_images(str(tmp_path), ["train"], ["pndm"], 2)
    assert (n_real, n_fake) == (2, 0)


def test_items_carry_the_generator_not_a_single_fake_bucket(tmp_path):
    """Map của hai generator khác nhau **không** được rơi vào cùng một thư mục.

    Lỗi thật: `dire` dựng item 3 phần tử `(path, label, split)`, nên `_as_item` gán
    `generator="fake"` cho mọi ảnh sinh. Hệ quả: `maps/<split>/ours_adm/` không
    bao giờ được tạo, map `ours_adm` và `adm` ghi đè nhau (tên PNG sinh là
    `seed.png`, trùng nhau), và `::train --generator ours_adm` báo "chưa có map".
    """
    for split in ("train", "test"):
        for cls in ("real", "adm", "ours_adm"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{i:02d}.png").write_bytes(b"")

    items = modal_app._items_to_map(str(tmp_path), ["train", "test"], ["adm", "ours_adm"], 2)
    gens = [it[3] for it in items]
    assert gens.count("real") == 4          # 2 mỗi split
    assert gens.count("adm") == 4
    assert gens.count("ours_adm") == 4
    assert "fake" not in gens

    # Và hai generator khác nhau phải ra hai đường dẫn khác nhau.
    from dire_zero.pipeline import map_rel_path

    a = map_rel_path((str(tmp_path / "train/adm/00.png"), 1, "train", "adm"))
    b = map_rel_path((str(tmp_path / "train/ours_adm/00.png"), 1, "train", "ours_adm"))
    assert a != b and a.as_posix().startswith("train/adm/")


def test_count_existing_only_counts_non_empty_maps(tmp_path):
    """`--skip-existing` phải đếm đúng cái *dùng lại được*, không đếm tên file rỗng."""
    maps = tmp_path / "maps"
    (maps / "train" / "real").mkdir(parents=True)
    (maps / "train" / "real" / "a.npy").write_bytes(b"x")
    (maps / "train" / "real" / "b.npy").write_bytes(b"")     # file rỗng = hỏng
    items = [("d/a.png", 0, "train", "real"), ("d/b.png", 0, "train", "real"),
             ("d/c.png", 0, "train", "real")]
    assert modal_app._count_existing(items, str(maps)) == 1


def test_map_inventory_shows_what_is_actually_on_disk(tmp_path):
    """`report` đếm map trên đĩa, không tin lời khai của `dire.json`.

    Cần cho đúng câu hỏi "đã có map chưa, có phải chạy lại không": một thư mục
    `maps/<split>/fake/` còn sót từ bản cũ (mọi generator bị gộp vào `fake`) sẽ
    hiện ở đây, và report phải nói ra thay vì để nó lẫn vào tập train.
    """
    for rel, n in (("train/real", 2), ("test/real", 1), ("train/fake", 3), ("test/ours_adm", 4)):
        d = tmp_path / rel
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i}.npy").write_bytes(b"x")
    (tmp_path / "dire-runs").mkdir()
    (tmp_path / "dire-runs" / "x.json").write_text("{}")

    inv = modal_app._map_inventory(str(tmp_path))
    assert inv["total"] == 10
    assert inv["by_split_generator"] == {
        "test/ours_adm": 4, "test/real": 1, "train/fake": 3, "train/real": 2,
    }


# ---------------------------------------------------------------------------
# Hai cổng trước khi xin GPU: tiền **và** thời gian
# ---------------------------------------------------------------------------
SEC = 0.0917          # sec/step đo được trên A10G (bench.json)
STEPS = 20


def _h(n_images: int) -> float:
    return n_images * 2 * STEPS * SEC / 3600


def test_group_generators_fits_every_group_in_the_time_budget():
    """15 generator × 1 000 ảnh = 7,6 GPU-h trong container timeout 6 h.

    Ca thật (log của người dùng): `::dire --n 500 --steps 20` với 15 generator,
    ước tính 9,88 GPU-h. Nâng `--max-usd` lên cũng **không** cứu được: container
    bị kill ở giờ thứ 6 và `vol.commit()` cuối không chạy. Cổng thời gian phải
    chia nhóm sao cho **mỗi** nhóm nằm dưới ngân sách, và không bỏ sót generator.
    """
    pending = {g: 1000 for g in ("adm", "dalle2", "ddpm", "diff-projectedgan", "diff-stylegan",
                                 "iddpm", "if", "ldm", "midjourney", "pndm", "projectedgan",
                                 "sdv2", "stylegan", "vqdiffusion", "ours_adm")}
    pending["real"] = 1000
    budget_h = 6 * 0.75                      # 4,5 h

    assert sum(pending.values()) * 2 * STEPS * SEC / 3600 > budget_h   # 1 lượt thì vượt
    groups = modal_app.group_generators(pending, SEC, STEPS, budget_h)

    assert len(groups) >= 2
    for g in groups:
        assert sum(pending[x] for x in g) * 2 * STEPS * SEC / 3600 <= budget_h
    flat = [x for g in groups for x in g]
    assert sorted(flat) == sorted(g for g in pending if g != "real")
    assert len(flat) == len(set(flat)), "một generator không được nằm hai nhóm"
    assert "real" not in flat, "`real` nằm trong mọi lệnh, không thuộc nhóm nào"


def test_group_generators_refuses_a_single_group_that_cannot_fit():
    """Một generator tự nó vượt ngân sách ⇒ báo, kèm cách sửa; không chia bừa."""
    from dire_zero.budget import GateError

    with pytest.raises(GateError) as exc:
        modal_app.group_generators({"adm": 20_000}, SEC, STEPS, 4.5)
    msg = str(exc.value)
    assert "adm" in msg and "--n" in msg


def test_group_generators_ignores_empty_and_real():
    assert modal_app.group_generators({"real": 500, "adm": 0}, SEC, STEPS, 4.5) == []
    assert modal_app.group_generators({"adm": 10}, SEC, STEPS, 4.5) == [["adm"]]


def test_select_generators_accepts_a_list_and_rejects_a_typo():
    available = ["adm", "pndm", "sdv2"]
    assert modal_app.select_generators(available, "auto") == available
    assert modal_app.select_generators(available, "pndm") == ["pndm"]
    assert modal_app.select_generators(available, "pndm, adm") == ["adm", "pndm"]
    with pytest.raises(ValueError) as exc:
        modal_app.select_generators(available, "sdv1")     # gõ nhầm: phải là lỗi
    assert "sdv1" in str(exc.value) and "sdv2" in str(exc.value)


def test_pending_by_generator_counts_only_missing_maps(tmp_path):
    """Kế hoạch theo generator phải khớp thứ `compute_maps(skip_existing=True)` sẽ làm."""
    maps = tmp_path / "maps"
    (maps / "train" / "real").mkdir(parents=True)
    (maps / "train" / "real" / "a.npy").write_bytes(b"x")
    items = [
        ("d/train/real/a.png", 0, "train", "real"),     # đã có
        ("d/train/real/b.png", 0, "train", "real"),     # còn thiếu
        ("d/train/adm/c.png", 1, "train", "adm"),
    ]
    assert modal_app._pending_by_generator(items, str(maps), skip_existing=True) == {"real": 1, "adm": 1}
    # không có cờ ⇒ tính lại tất
    assert modal_app._pending_by_generator(items, str(maps), skip_existing=False) == {"real": 2, "adm": 1}


def test_time_gate_wins_over_the_money_gate(monkeypatch):
    """Cả hai cùng vượt ⇒ phải báo **thời gian**, không báo "nâng --max-usd".

    Lời khuyên sai tệ hơn không có lời khuyên: nâng trần tiền cho một lượt chắc
    chắn bị kill ở giờ thứ 6 chỉ làm hoá đơn của phần chạy dở trở nên đắt hơn.
    """
    src = Path(modal_app.__file__).read_text(encoding="utf-8")
    body = src[src.index("def dire("):]
    body = body[: body.index("def _pending_by_generator(")]
    assert body.index("budget_h = WORK_TIMEOUT_S") < body.index("require_estimate(est, max_usd)")
    assert "Nâng `--max-usd` cũng **không** cứu được" in body
