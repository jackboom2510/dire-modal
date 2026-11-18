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
    for name in ("selfcheck", "bench", "gen", "dire", "relabel", "maps", "train",
                 "evaluate", "report"):
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


def test_train_rejects_a_generator_that_has_no_maps():
    """Lời nhắn phải chỉ đúng lệnh cần chạy trước, không chỉ "không tìm thấy"."""
    with pytest.raises(ValueError) as exc:
        modal_app._train_targets("midjourney", ["adm", "pndm"])
    msg = str(exc.value)
    assert "midjourney" in msg and "::dire --generator midjourney" in msg


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
        assert g["gpu_h"] <= budget_h, g          # đã gồm `real` của nhóm đầu
    flat = [x for g in groups for x in g["generators"]]
    assert sorted(flat) == sorted(g for g in pending if g != "real")
    assert len(flat) == len(set(flat)), "một generator không được nằm hai nhóm"
    assert all("real" not in g["generators"] for g in groups)


def test_group_generators_refuses_a_single_group_that_cannot_fit():
    """Một generator tự nó vượt ngân sách ⇒ báo, kèm cách sửa; không chia bừa."""
    from dire_zero.budget import GateError

    with pytest.raises(GateError) as exc:
        modal_app.group_generators({"adm": 20_000}, SEC, STEPS, 4.5)
    msg = str(exc.value)
    assert "adm" in msg and "--n" in msg


def test_group_generators_ignores_empty_and_real():
    groups = modal_app.group_generators({"adm": 10}, SEC, STEPS, 4.5)
    assert [g["generators"] for g in groups] == [["adm"]]
    # `real` một mình, vừa ngân sách ⇒ một nhóm, không generator, có `real`
    only_real = modal_app.group_generators({"real": 100}, SEC, STEPS, 4.5)
    assert len(only_real) == 1 and only_real[0]["generators"] == []
    assert only_real[0]["includes_real"] == 100


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


def test_the_suggested_command_actually_fits_the_budget():
    """Đúng log thật: nhóm đầu của bản cũ vẫn bị chặn lại vì bỏ quên `real`.

    `--generator adm,iddpm,dalle2,ddpm,diff-projectedgan,diff-stylegan` = 4 000 ảnh
    = 4,08 h, "vừa dưới" trần 4,50 h — nhưng lệnh đó **còn** 600 ảnh `real` chưa
    có map ⇒ 4,69 h ⇒ bị chặn y như trước. Bản gợi ý in ra mà không chạy được thì
    tệ hơn không gợi ý.
    """
    pending = {"adm": 1000, "iddpm": 1000, "dalle2": 500, "ddpm": 500,
               "diff-projectedgan": 500, "diff-stylegan": 500, "if": 500, "real": 600}
    budget_h = 4.5
    old_first = ["adm", "iddpm", "dalle2", "ddpm", "diff-projectedgan", "diff-stylegan"]
    assert (sum(pending[g] for g in old_first) + pending["real"]) * 2 * STEPS * SEC / 3600 > budget_h

    groups = modal_app.group_generators(pending, SEC, STEPS, budget_h)
    for grp in groups:
        n = grp["n_images"] + grp["includes_real"]
        assert n * 2 * STEPS * SEC / 3600 <= budget_h
    # `real` chỉ được tính **một** lần, ở lệnh đầu
    assert sum(g["includes_real"] for g in groups) == pending["real"]
    assert groups[0]["includes_real"] == pending["real"]
    # và các nhóm phải cân bằng, không phải "nhét đầy nhóm đầu"
    hours = [g["gpu_h"] for g in groups]
    assert max(hours) - min(hours) < 0.35, hours


def test_grouping_is_balanced_and_covers_everything():
    """LPT: số nhóm = ceil(tổng/trần), độ lệch giữa các nhóm nhỏ."""
    pending = {f"g{i}": 500 for i in range(9)}
    pending["real"] = 500
    groups = modal_app.group_generators(pending, SEC, STEPS, 1.5)
    # ceil(10 000 ảnh / 2 940 ảnh) = 4, nhưng LPT chia 4 nhóm thì mỗi nhóm 3 000 ảnh
    # = 1,53 h > 1,5 h ⇒ phải nới lên 5 nhóm. Không được trả về một nhóm vượt trần.
    assert len(groups) == 5, [g["gpu_h"] for g in groups]
    hours = [g["gpu_h"] for g in groups]
    assert max(hours) <= 1.5 and max(hours) - min(hours) <= 0.6, hours
    flat = [x for g in groups for x in g["generators"]]
    assert sorted(flat) == [f"g{i}" for i in range(9)]


def test_plan_commands_are_copy_paste_ready():
    """Mỗi dòng phải đủ tham số, **gồm `--skip-existing`**, và nói đúng số ảnh của nhóm.

    Thiếu `--skip-existing` thì nhóm thứ hai trả tiền lại cho `real` — đúng thứ
    vừa được chia nhỏ để tránh.
    """
    pending = {"adm": 1000, "iddpm": 1000, "dalle2": 500, "ddpm": 500,
               "diff-projectedgan": 500, "diff-stylegan": 500, "if": 500, "real": 600}
    groups = modal_app.group_generators(pending, SEC, STEPS, 4.5)
    cmds = modal_app.plan_commands(groups, n=500, steps=20, splits="train,test")

    assert len(cmds) == len(groups)
    for line, grp in zip(cmds, groups):
        assert "modal run -m dire_zero.modal_app::dire" in line
        assert "--n 500" in line and "--steps 20" in line and "--splits train,test" in line
        assert "--skip-existing" in line, line
        assert f"--generator {','.join(grp['generators'])}" in line
        assert f"{grp['n_images'] + grp['includes_real']} ảnh" in line
    # `real` chỉ được nhắc ở lệnh đầu
    assert cmds[0].count("real") == 1 and all("real" not in c for c in cmds[1:])


# ---------------------------------------------------------------------------
# Sau `::dire`: mỗi dòng của Table 2 là một detector **riêng**
# ---------------------------------------------------------------------------
def test_train_targets_resolve_all_and_reject_a_missing_generator():
    avail = ["adm", "dalle2", "pndm"]
    assert modal_app._train_targets("auto", avail) == [None]      # gộp: bài toán khác
    assert modal_app._train_targets("all", avail) == avail
    assert modal_app._train_targets("pndm", avail) == ["pndm"]
    assert modal_app._train_targets("pndm, adm", avail) == ["pndm", "adm"]
    with pytest.raises(ValueError) as exc:
        modal_app._train_targets("midjourney", avail)
    assert "midjourney" in str(exc.value) and "pndm" in str(exc.value)


def test_train_tags_are_unique_and_filesystem_safe():
    tags = [modal_app._train_tag(g) for g in ("adm", "diff-projectedgan", "ours_adm", None)]
    assert len(tags) == len(set(tags))
    for t in tags:
        assert all(c.isalnum() or c in "-._" for c in t), t
    # `auto` (gộp) không được trùng với một generator tên là "auto"
    assert modal_app._train_tag(None) == "auto"
    assert modal_app._train_tag("auto") != "auto"


def test_train_ledger_keeps_every_generator(tmp_path):
    """15 dòng của Table 2 phải cùng tồn tại — bản trước ghi đè, còn đúng 1 dòng.

    Lỗi thật: `train` ghi `runs/train.json` + `runs/detector.pth` cho **mọi**
    generator, nên chạy lần lượt 15 lệnh thì dòng cuối là dòng duy nhất còn lại,
    và `::report` đọc đúng một con số mà không nói 14 dòng kia đã mất.
    """
    def res(acc: float, **kw) -> dict:
        return {"best_epoch": 7, "best_val_ap": 0.9, "threshold_from_val": 0.5,
                "seconds": 12.0,
                "test": {"acc": acc, "ap": 0.95, "auc": 0.96, "n": 200,
                         "n_real": 100, "n_fake": 100, "acc_ci95": [acc - 0.02, acc + 0.02]}, **kw}

    out = str(tmp_path)
    modal_app._write_train_ledger(out, {"gen-adm": res(0.91), "gen-pndm": res(0.77)})
    led = modal_app._write_train_ledger(out, {"gen-sdv2": res(0.83)})

    index = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert sorted(index["by_generator"]) == ["gen-adm", "gen-pndm", "gen-sdv2"]
    assert index["by_generator"]["gen-pndm"]["acc"] == 0.77
    assert len(index["runs"]) == 3
    assert index["latest"] == "gen-sdv2"
    # khoá cũ ở cấp ngoài trỏ tới lượt gần nhất (chỗ đọc cũ không hỏng)
    assert index["threshold_from_val"] == 0.5
    assert led["n"] == 3


def test_train_writes_each_generator_to_its_own_directory():
    """Thư mục riêng, không chỉ file json riêng: `detector.pth` cũng phải tách."""
    src = Path(modal_app.__file__).read_text(encoding="utf-8")
    body = src[src.index("def train(epochs"):]
    body = body[: body.index("def _train_targets(")]
    assert 'f"{OUT}/train/{tag}"' in body
    assert '"runs/train.json"' not in body


# ---------------------------------------------------------------------------
# Sống sót sau khi mất mạng / bị kill
# ---------------------------------------------------------------------------
class _VolDem:
    """Volume giả: đếm số lần `commit()` và chụp lại đường dẫn đã ghi lúc đó."""

    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def test_compute_with_commits_persists_every_n_images(monkeypatch):
    """Volume v1 **không** tự ghi: mất mạng giữa lúc chạy mà chưa commit = mất hết.

    Đây là lỗi đắt nhất có thể có trong repo này: đã trả tiền cho từng giây GPU,
    rồi mất sạch vì `commit()` chỉ nằm ở cuối hàm.
    """
    vol = _VolDem()
    monkeypatch.setattr(modal_app, "vol", vol)

    def fake_compute(f, items, out_dir, meta, limit=None, skip_existing=False, progress=None):
        for i in range(1, 451):                 # 450 ảnh, commit mỗi 200
            if progress and i % 50 == 0:        # `compute_maps` gọi mỗi 50 ảnh
                progress(i)
        return {"n": 450, "n_new": 450, "n_reused": 0, "nfe_total": 0}

    out = modal_app._compute_with_commits(None, [], None, False, fake_compute, every=200)
    assert out["n_new"] == 450
    assert vol.commits == 2                     # tại 200 và 400; 450 chưa tới mốc
    assert vol.commits * 200 >= 400


def test_compute_with_commits_persists_even_when_it_crashes(monkeypatch):
    """Exception mềm (mất mạng, KeyboardInterrupt) ⇒ commit phần đã xong rồi mới ném."""
    vol = _VolDem()
    monkeypatch.setattr(modal_app, "vol", vol)

    def fake_compute(f, items, out_dir, meta, limit=None, skip_existing=False, progress=None):
        progress(50)
        raise ConnectionError("mất mạng")

    with pytest.raises(ConnectionError):
        modal_app._compute_with_commits(None, [], None, False, fake_compute, every=200)
    assert vol.commits == 1, "phải commit phần đã xong trước khi ném tiếp"


def test_dire_delegates_the_commit_cadence_instead_of_duplicating_it():
    src = Path(modal_app.__file__).read_text(encoding="utf-8")
    body = src[src.index("def dire("):]
    body = body[: body.index("def _compute_with_commits(")]
    assert "_compute_with_commits(" in body
    # đếm **dòng mã**, không đếm chữ trong docstring/thông báo lỗi
    code_commits = [ln.strip() for ln in body.splitlines() if ln.strip() == "vol.commit()"]
    assert code_commits == ["vol.commit()"], "commit chỉ nên nằm ở một chỗ trong `dire`"


def test_maps_inventory_counts_what_is_missing(tmp_path, monkeypatch):
    """`::maps` trả lời "đã đủ chưa" mà không cần GPU: đếm theo **ảnh đang có**."""
    data, maps = tmp_path / "data", tmp_path / "maps"
    for split, gen, n_img, n_map in (("train", "real", 4, 4), ("train", "adm", 4, 2),
                                     ("test", "real", 3, 1), ("test", "adm", 3, 0)):
        (data / split / gen).mkdir(parents=True, exist_ok=True)
        for i in range(n_img):
            (data / split / gen / f"{i}.png").write_bytes(b"png")
        (maps / split / gen).mkdir(parents=True, exist_ok=True)
        for i in range(n_map):
            (maps / split / gen / f"{i}.npy").write_bytes(b"npy" * 10)

    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))

    out = modal_app.maps.local(n=0, steps=20)
    cells = {c["cell"]: c for c in out["cells"]}
    assert set(cells) == {"train/real", "train/adm", "test/real", "test/adm"}
    assert cells["train/real"]["missing"] == 0
    assert cells["train/adm"]["maps"] == 2 and cells["train/adm"]["missing"] == 2
    assert cells["test/adm"]["missing"] == 3
    assert out["total_maps"] == 7                    # 4 + 2 + 1 + 0
    assert out["total_missing"] == 7                  # 0 + 2 + 2 + 3
    assert out["ready_for_train"] is False


def test_maps_honours_n_and_flags_a_leftover_fake_dir(tmp_path, monkeypatch):
    data, maps = tmp_path / "data", tmp_path / "maps"
    (data / "train" / "real").mkdir(parents=True)
    (data / "train" / "adm").mkdir(parents=True)
    for i in range(10):
        (data / "train" / "real" / f"{i}.png").write_bytes(b"png")
        (data / "train" / "adm" / f"{i}.png").write_bytes(b"png")
    (maps / "train" / "real").mkdir(parents=True)
    (maps / "train" / "fake").mkdir(parents=True)
    for i in range(5):
        (maps / "train" / "real" / f"{i}.npy").write_bytes(b"npy")
        (maps / "train" / "fake" / f"{i}.npy").write_bytes(b"npy")

    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    out = modal_app.maps.local(n=5, steps=20)
    cells = {c["cell"]: c for c in out["cells"]}
    assert cells["train/real"]["wanted"] == 5 and cells["train/real"]["missing"] == 0
    assert cells["train/real"]["extra"] == 0          # 5 map có, cần 5
    assert any("fake" in p for p in out["problems"]), out["problems"]


def test_maps_verify_catches_a_truncated_npy(tmp_path, monkeypatch):
    """File ghi dở vẫn tồn tại và khác rỗng ⇒ đếm file là chưa đủ, phải đọc thử."""
    data, maps = tmp_path / "data", tmp_path / "maps"
    (data / "train" / "real").mkdir(parents=True)
    (data / "train" / "adm").mkdir(parents=True)
    for i in range(3):
        (data / "train" / "real" / f"{i}.png").write_bytes(b"png")
        (data / "train" / "adm" / f"{i}.png").write_bytes(b"png")
    (maps / "train" / "real").mkdir(parents=True)
    (maps / "train" / "adm").mkdir(parents=True)
    import numpy as np
    for gen in ("real", "adm"):
        (maps / "train" / gen / "ok.npy").write_bytes(b"")
        np.save(maps / "train" / gen / "ok.npy", np.zeros((3, 8, 8), dtype="float16"))
    (maps / "train" / "adm" / "broken.npy").write_bytes(b"half a file")

    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    out = modal_app.maps.local(n=0, verify=5, steps=20)
    assert any("broken.npy" in b for b in out["bad_files"]), out["bad_files"]
    assert out["ready_for_train"] is False


# ---------------------------------------------------------------------------
# `::maps` sau khi chạy thật: ba lỗi lộ ra từ log
# ---------------------------------------------------------------------------
def _tree(tmp_path, spec: dict) -> tuple[Path, Path]:
    """``{(split, gen): (n_ảnh, n_map)}`` → cây data/ + maps/."""
    data, maps = tmp_path / "data", tmp_path / "maps"
    for (split, gen), (n_img, n_map) in spec.items():
        (data / split / gen).mkdir(parents=True, exist_ok=True)
        for i in range(n_img):
            (data / split / gen / f"{i}.png").write_bytes(b"png")
        (maps / split / gen).mkdir(parents=True, exist_ok=True)
        for i in range(n_map):
            (maps / split / gen / f"{i}.npy").write_bytes(b"x" * 300)
    return data, maps


def test_maps_never_suggests_generator_real(tmp_path, monkeypatch):
    """`--generator real` là lệnh **không chạy được**: `real` không nằm trong danh
    sách generator (nó có mặt trong mọi lệnh). Gợi ý sai tệ hơn không gợi ý."""
    data, maps = _tree(tmp_path, {("test", "real"): (100, 40), ("test", "adm"): (100, 100)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))

    out = modal_app.maps.local(n=100, steps=20)
    text = " ".join(out["problems"]) + " ".join(out["plan"]["commands"])
    assert "--generator real" not in text, "chuỗi trông như cờ sẽ bị copy nguyên văn"
    assert "--skip-existing" in text

    # và điều ngược lại: lệnh gợi ý cho `real` phải chạy được thật
    assert modal_app.select_generators(["adm", "pndm"], "adm") == ["adm"]
    with pytest.raises(ValueError):
        modal_app.select_generators(["adm", "pndm"], "real")


def test_maps_defaults_to_dires_n_and_says_so(tmp_path, monkeypatch):
    """`--n` mặc định phải là `n` của `::dire` (2 000), không phải cả cây.

    Log thật: chạy `::maps` không `--n` trên cây 40 000 ảnh/ô ⇒ "thiếu 212 583
    map". Đúng về số học, vô dụng về quyết định: không ai định tính map cho 212 583
    ảnh, và con số đó không nói phải chạy lệnh gì.
    """
    data, maps = _tree(tmp_path, {("train", "real"): (40000, 500), ("train", "adm"): (40000, 500)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))

    out = modal_app.maps.local(steps=20)          # không truyền n
    cells = {c["cell"]: c for c in out["cells"]}
    assert out["n"] == 2000
    assert cells["train/real"]["wanted"] == 2000 and cells["train/real"]["missing"] == 1500

    out_all = modal_app.maps.local(n=0, steps=20)
    assert {c["cell"]: c for c in out_all["cells"]}["train/real"]["wanted"] == 40000


def test_maps_prints_a_runnable_command_when_work_fits_one_run(tmp_path, monkeypatch):
    """Thiếu ít ⇒ **một** lệnh, đủ tham số, và `real` được nhắc là có sẵn trong đó."""
    data, maps = _tree(tmp_path, {("test", "real"): (100, 100), ("test", "adm"): (100, 20)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "bench.json").write_text(json.dumps({"sec_per_step": 0.0917}))

    out = modal_app.maps.local(n=100, steps=20)
    assert out["plan"]["gpu_h"] is not None
    cmds = out["plan"]["commands"]
    assert len(cmds) == 1, cmds
    assert "--generator adm" in cmds[0] and "--n 100" in cmds[0] and "--skip-existing" in cmds[0]
    assert "real" not in cmds[0].split("--generator")[1]
    # 80 ảnh × 40 NFE × 0,0917 s = 293 s = 0,081 h
    assert abs(out["plan"]["gpu_h"] - 0.081) < 0.01


def test_maps_splits_into_balanced_commands_when_over_the_time_budget(tmp_path, monkeypatch):
    data, maps = _tree(tmp_path, {("test", "real"): (100, 0),
                                  ("test", "adm"): (2500, 0),
                                  ("test", "pndm"): (2500, 0),
                                  ("test", "if"): (2500, 0)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "bench.json").write_text(json.dumps({"sec_per_step": 0.0917}))

    out = modal_app.maps.local(n=2500, steps=20)
    assert out["plan"]["gpu_h"] > 4.5                  # 7 600 ảnh ⇒ ~7,7 h
    cmds = out["plan"]["commands"]
    assert len(cmds) >= 3, cmds
    flat = [c.split("--generator")[1].split("#")[0].strip() for c in cmds]
    gens = [g for line in flat for g in line.split(",")]
    assert sorted(gens) == ["adm", "if", "pndm"]        # không bỏ sót, không lặp
    assert sum(1 for c in cmds if "real chưa có map" in c) == 1


def test_full_tree_inventory_does_not_try_to_schedule_n_zero(tmp_path, monkeypatch):
    """`maps --n 0` inventories the whole tree; it must never suggest `dire --n 0`.

    A real Modal run had 41,000 pending maps for one generator (41.77 GPU-h), so
    `_plan_remaining` raised `GateError` and destroyed an otherwise valid,
    read-only inventory command. Inventory is not a promise that one job fits.
    """
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))
    plan = modal_app._plan_remaining({"adm": 41_000, "real": 41_000},
                                     ["train", "val", "test"], n=0, steps=20)
    assert plan["commands"] == []
    assert plan["gpu_h"] is None and plan["usd"] is None
    assert "inventory toàn cây" in plan["note"]
    assert "::dire --n 0" in plan["note"]


def test_large_positive_maps_plan_reports_shard_needed_without_raising(tmp_path, monkeypatch):
    """Một generator vượt ngưỡng job cũng không được làm hỏng bước inventory."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "bench.json").write_text(json.dumps({"sec_per_step": SEC}))
    monkeypatch.setattr(modal_app, "OUT", str(runs))
    plan = modal_app._plan_remaining({"adm": 41_000, "real": 41_000},
                                     ["train", "val", "test"], n=41_000, steps=20)
    assert plan["commands"] == []
    assert plan["gpu_h"] > 4.5
    assert "adm" in plan["note"] and "giảm --n" in plan["note"]


def test_maps_without_bench_does_not_invent_a_price(tmp_path, monkeypatch):
    data, maps = _tree(tmp_path, {("test", "real"): (50, 0), ("test", "adm"): (50, 0)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "chưa-có"))

    out = modal_app.maps.local(n=50, steps=20)
    assert out["plan"]["gpu_h"] is None and out["plan"]["usd"] is None
    assert "::bench" in out["plan"]["note"]


def test_maps_warns_about_a_class_imbalance(tmp_path, monkeypatch):
    """`test/midjourney` 100 ảnh vs `test/real` 1 000 ⇒ 10:1: phải nói trước khi train."""
    data, maps = _tree(tmp_path, {("test", "real"): (1000, 1000), ("test", "midjourney"): (100, 100)})
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))

    out = modal_app.maps.local(n=0, steps=20)
    assert any("midjourney" in w and "10.0:1" in w for w in out["warnings"]), out["warnings"]
    assert out["total_missing"] == 0


# ---------------------------------------------------------------------------
# `::evaluate` — bước còn thiếu giữa `::train` và trang 2 của demo
# ---------------------------------------------------------------------------
def test_evaluate_refuses_when_nothing_was_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        modal_app.evaluate.local()
    assert "::train" in str(exc.value)


def test_evaluate_writes_scores_in_the_format_make_gallery_reads(tmp_path, monkeypatch):
    """Khoá phải đúng `f"{cls}_{stem}"` — nếu lệch, gallery im lặng "chưa có điểm".

    Đây là hợp đồng thật giữa `::evaluate` và `tools/make_gallery.py`: tool đọc
    `scores[name]` với `name = f"{prefix}{cls}_{npy.stem}"`. Lệch một ký tự thì
    không có lỗi nào, chỉ có một trang demo không có điểm.
    """
    import numpy as np

    out = tmp_path / "runs"
    (out / "train" / "gen-adm").mkdir(parents=True)
    (out / "train" / "gen-adm" / "train.json").write_text(json.dumps({
        "generators_used": ["adm"], "threshold_from_val": 0.3, "threshold_source": "val",
        "split": {"n_train": 80, "n_val": 20, "n_test": 20},
    }), encoding="utf-8")
    (out / "train" / "gen-adm" / "detector.pth").write_bytes(b"x")
    (out / "train.json").write_text(json.dumps({"by_generator": {"gen-adm": {"acc": 0.9}}}),
                                    encoding="utf-8")

    maps = tmp_path / "maps"
    for gen, label in (("real", 0), ("adm", 1)):
        d = maps / "test" / gen
        d.mkdir(parents=True)
        for i in range(3):
            np.save(d / f"{i:03d}.npy", np.zeros((3, 8, 8), dtype="float16"))

    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "vol", _VolDem())
    monkeypatch.setattr("dire_zero.pipeline._score",
                        lambda model, paths, labels, crop, device, bs=32: ([2.0, 2.0, 2.0] + [-2.0] * 3, labels))
    monkeypatch.setattr("dire_zero.detector.ResNetDetector", lambda *a, **k: _FakeDet())
    monkeypatch.setattr("dire_zero.detector.resnet50", lambda: None)
    import torch
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})

    res = modal_app.evaluate.local(generator="adm")
    assert res["evaluated"] == ["gen-adm"]

    scores = json.loads((out / "eval" / "gen-adm" / "scores.json").read_text(encoding="utf-8"))
    assert set(scores["samples"]) == {"real_000", "real_001", "real_002",
                                      "adm_000", "adm_001", "adm_002"}

    real = scores["samples"]["real_000"]
    fake = scores["samples"]["adm_000"]
    assert 0.0 <= real["p_fake"] <= 1.0 and real["p_fake"] < 0.5 < fake["p_fake"]
    assert real["gate"] == "passed" and real["threshold_split"] == "val"
    # ngưỡng logit 0,3 đổi sang xác suất — `make_gallery --threshold` cần [0,1]
    assert 0.0 <= real["threshold"] <= 1.0

    ev = json.loads((out / "eval" / "gen-adm" / "eval.json").read_text(encoding="utf-8"))
    assert ev["metrics"]["acc"] == 1.0 and ev["metrics"]["n"] == 6
    assert ev["split"] == "test"


def test_evaluate_marks_gate_failed_when_the_threshold_is_not_from_val(tmp_path, monkeypatch):
    """Ngưỡng không đến từ val ⇒ `gate: failed`; `provenance` sẽ từ chối hiện điểm.

    Đây là hành vi **đúng**, không phải lỗi cần lách: hiện điểm với ngưỡng chọn trên
    test là tự khen mình.
    """
    import numpy as np
    import torch

    out = tmp_path / "runs"
    (out / "train" / "gen-adm").mkdir(parents=True)
    (out / "train" / "gen-adm" / "train.json").write_text(json.dumps({
        "generators_used": ["adm"], "threshold_from_val": 0.0, "threshold_source": "default",
        "split": {"n_train": 100, "n_val": 0, "n_test": 20},
    }), encoding="utf-8")
    (out / "train" / "gen-adm" / "detector.pth").write_bytes(b"x")
    (out / "train.json").write_text(json.dumps({"by_generator": {"gen-adm": {}}}), encoding="utf-8")
    maps = tmp_path / "maps"
    for gen in ("real", "adm"):
        d = maps / "test" / gen
        d.mkdir(parents=True)
        np.save(d / "0.npy", np.zeros((3, 8, 8), dtype="float16"))

    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "vol", _VolDem())
    monkeypatch.setattr("dire_zero.pipeline._score",
                        lambda model, paths, labels, crop, device, bs=32: ([0.5] * len(paths), labels))
    monkeypatch.setattr("dire_zero.detector.ResNetDetector", lambda *a, **k: _FakeDet())
    monkeypatch.setattr("dire_zero.detector.resnet50", lambda: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})

    res = modal_app.evaluate.local(generator="adm")
    assert res["by_tag"]["gen-adm"]["gate"] == "failed"


class _FakeDet:
    def to(self, _d):
        return self

    def load_state_dict(self, _s):
        return self


def test_report_says_what_the_cumulative_counters_mean(tmp_path, monkeypatch):
    """`n`, `n_new`, `n_reused` **không** cộng lại thành một số.

    `n` là số map phân biệt; `n_new`/`n_reused` cộng dồn qua mọi lượt chạy, nên một
    map tính ở lượt trước rồi dùng lại ở lượt sau bị đếm hai lần. In ba số cạnh
    nhau mà không nói gì thì người đọc tưởng đó là phép cộng sai — đúng như log
    của một lượt chạy thật: 5 869 + 2 431 = 8 300 nhưng chỉ có 5 900 map.
    """
    data, maps = _tree(tmp_path, {("test", "real"): (10, 10), ("test", "adm"): (10, 10)})
    out = tmp_path / "runs"
    out.mkdir()
    (maps / "dire.json").write_text(json.dumps({
        "n": 20, "n_new": 5869, "n_reused": 2431, "steps": 20, "steps_mixed": [20],
        "nfe_total": 234760, "seconds": 25787.41,
        "runs": [{"created": "a"}, {"created": "b"}],
    }), encoding="utf-8")
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    monkeypatch.setattr(modal_app, "vol", _VolDem())

    rep = modal_app.report.local()
    assert rep["dire"]["n_distinct_maps"] == 20
    assert rep["dire"]["n_new_cumulative"] == 5869
    assert rep["dire"]["n_reused_cumulative"] == 2431
    assert rep["dire"]["n_runs"] == 2
    assert "cộng dồn" in rep["dire"]["n_note"]


def test_report_surfaces_a_leftover_fake_dir_and_a_ci_that_cannot_be_true(tmp_path, monkeypatch):
    """Hai thứ phải hiện thành cảnh báo, không phải để người đọc tự phát hiện:

    * `maps/*/fake/` còn sót ⇒ `fake` sẽ trộn với generator thật khi train;
    * `acc` nằm ngoài `acc_ci95` ⇒ con số không thể đúng (bản cũ tính CI tại 0,5).
    """
    data, maps = _tree(tmp_path, {("test", "real"): (10, 10), ("test", "adm"): (10, 10),
                                  ("test", "fake"): (0, 7)})
    out = tmp_path / "runs"
    out.mkdir()
    (out / "train.json").write_text(json.dumps({"by_generator": {"gen-adm": {
        "acc": 0.781, "acc_ci95": [0.788, 0.837], "acc_at_0.5": 0.813, "n": 1000,
        "ap": 0.911, "auc": 0.890}}}), encoding="utf-8")
    (out / "eval.json").write_text(json.dumps({"by_tag": {"gen-adm": {
        "acc": 0.781, "acc_ci95": [0.47, 0.529], "n": 1000, "gate": "passed"}}}),
        encoding="utf-8")
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    monkeypatch.setattr(modal_app, "vol", _VolDem())

    warnings = " | ".join(modal_app.report.local()["warnings"])
    assert "test/fake" in warnings and "relabel" in warnings
    assert "ngoài" in warnings and "gen-adm" in warnings
    assert "±" in warnings                      # cỡ mẫu thật, không phải "~200" chép tay


# ---------------------------------------------------------------------------
# `::train` — `fake/` không phải generator; một generator hỏng không giết cả lượt
# ---------------------------------------------------------------------------
def test_train_skips_a_leftover_fake_dir_and_keeps_going(tmp_path, monkeypatch):
    """Hai lỗi từ một log thật, cùng một chỗ.

    1. `maps/*/fake/` còn sót bị train thành một dòng `gen-fake` trong bảng —
       trông y như Table 2 nhưng nhãn của nó là "ảnh giả nói chung".
    2. Generator chỉ có map ở `test` (dalle2, ddpm, …) làm `train` chết giữa
       chừng, hoặc tệ hơn: âm thầm cho ra `acc = 0,500`.
    """
    from dire_zero.pipeline import SplitError

    data, maps = _tree(tmp_path, {
        ("train", "real"): (40, 40), ("test", "real"): (40, 40),
        ("train", "adm"): (40, 40), ("test", "adm"): (40, 40),
        ("test", "fake"): (0, 40),                  # thư mục cũ, chưa relabel
        ("test", "dalle2"): (0, 40),                # chỉ có ở test
    })
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(tmp_path / "runs"))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    monkeypatch.setattr(modal_app, "vol", _VolDem())

    seen: list[str] = []

    def fake_train(factory, subset, out_dir, **kw):
        seen.append(sorted({m["generator"] for m in subset}))
        if any(m["generator"] == "dalle2" for m in subset):
            raise SplitError("tập `train` chỉ có real — dùng --split-mode hash")
        return {"label": "R1+R2", "split_mode": "author", "best_epoch": 1,
                "best_val_ap": 0.9, "threshold_from_val": 0.0, "threshold_space": "logit",
                "seconds": 1.0,
                "test": {"acc": 0.9, "acc_at_0.5": 0.9, "acc_ci95": [0.85, 0.95],
                         "ap": 0.9, "auc": 0.9, "n": 80, "n_real": 40, "n_fake": 40}}

    monkeypatch.setattr("dire_zero.pipeline.train", fake_train)
    out = modal_app.train.local(generator="all", epochs=1)

    assert "gen-fake" not in seen                      # `fake/` bị loại
    assert out["trained"] == ["gen-adm"]
    assert any("dalle2" in s for s in out["skipped"])  # bỏ qua, không chết cả lượt
    ledger = json.loads((tmp_path / "runs" / "train.json").read_text(encoding="utf-8"))
    assert "gen-dalle2" in (ledger.get("skipped") or {})


def test_report_flags_a_row_that_is_just_a_constant_classifier(tmp_path, monkeypatch):
    """`acc` đúng 0,500 + AUC dưới 0,5 là chữ ký của "train trên một lớp".

    Dòng đó **không phải** kết quả, nhưng nếu bảng chỉ in `acc` và `AP` thì người
    đọc không có cách nào biết.
    """
    data, maps = _tree(tmp_path, {("train", "real"): (10, 10), ("test", "adm"): (0, 10)})
    out = tmp_path / "runs"
    out.mkdir()
    (out / "train.json").write_text(json.dumps({"by_generator": {
        "gen-ddpm": {"acc": 0.5, "acc_ci95": [0.471, 0.53], "ap": 0.346, "auc": 0.206,
                     "n": 1000, "n_real": 500, "n_fake": 500},
        "gen-adm": {"acc": 0.828, "acc_ci95": [0.8, 0.848], "ap": 0.903, "auc": 0.891,
                    "n": 1000, "n_real": 500, "n_fake": 500},
    }, "skipped": {"gen-ddpm": "gen-ddpm: tập `train` chỉ có real"}}), encoding="utf-8")
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    monkeypatch.setattr(modal_app, "vol", _VolDem())

    rep = modal_app.report.local()
    text = " | ".join(rep["warnings"])
    assert "gen-ddpm" in text and "một lớp" in text
    assert any("KHÔNG" in w and "kết quả" in w for w in rep["warnings"])
    # Dòng tốt không bị gắn cảnh báo "dự đoán một lớp"
    assert not any("gen-adm" in w and "một lớp" in w for w in rep["warnings"])
    assert "gen-ddpm" in (rep.get("skipped") or {})


def test_report_flags_an_eval_made_with_a_different_detector(tmp_path, monkeypatch):
    """`eval.json` cũ chấm bằng detector cũ ⇒ điểm đó không được ghép với ACC mới.

    Log thật cho thấy đúng cảnh này: `train_by_generator.gen-adm.acc = 0,828` (lượt
    mới) đứng cạnh `eval_by_generator.gen-adm.acc = 0,781` (lượt cũ). Hai con số
    của hai mô hình khác nhau, trong cùng một khối JSON.
    """
    data, maps = _tree(tmp_path, {("train", "real"): (10, 10), ("test", "adm"): (0, 10)})
    out = tmp_path / "runs"
    (out / "train" / "gen-adm").mkdir(parents=True)
    (out / "train" / "gen-adm" / "detector.pth").write_bytes(b"NEW-WEIGHTS")
    (out / "train.json").write_text(json.dumps({"by_generator": {"gen-adm": {
        "acc": 0.828, "acc_ci95": [0.8, 0.848], "ap": 0.903, "auc": 0.891, "n": 1000}}}),
        encoding="utf-8")
    (out / "eval.json").write_text(json.dumps({"by_tag": {"gen-adm": {
        "acc": 0.781, "acc_ci95": [0.75, 0.81], "n": 1000, "gate": "passed",
        "detector_sha256": "0" * 64}}}), encoding="utf-8")
    monkeypatch.setattr(modal_app, "DATA", str(data))
    monkeypatch.setattr(modal_app, "MAPS", str(maps))
    monkeypatch.setattr(modal_app, "OUT", str(out))
    monkeypatch.setattr(modal_app, "DATA_CARD", str(data / "data_card.json"))
    monkeypatch.setattr(modal_app, "vol", _VolDem())

    text = " | ".join(modal_app.report.local()["warnings"])
    assert "eval.json" in text and "detector" in text
    assert "CŨ" in text


def test_image_mounts_demolib_at_the_path_make_gallery_imports_from():
    """`::gallery` mount **một tệp** tool — và lần đầu, chỉ thế là đủ để chết.

    Container đã trả về::

        File "/root/tools/make_gallery.py", line 55, in <module>
          from demolib import provenance
        ModuleNotFoundError: No module named 'demolib'

    `make_gallery.py` tự thêm ``<repo>/platforms/hf-spaces`` vào `sys.path`, nên
    `demolib/` phải được mount ở **đúng** đường đó, không phải chỗ nào tiện.
    Ở máy không thấy gì vì repo ở ngay cạnh — chỉ container mới lộ.
    """
    import ast

    src_file = Path(modal_app.__file__).resolve()
    tree = ast.parse(src_file.read_text(encoding="utf-8"))
    mounts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_local_dir":
            local_expr, remote_expr = node.args[0], node.args[1]
            local = eval(  # noqa: S307 - biểu thức của chính repo, không phải input
                compile(ast.Expression(local_expr), str(src_file), "eval"),
                {"Path": Path, "__file__": str(src_file)},
            )
            mounts.append((Path(local), ast.literal_eval(remote_expr)))

    expected_local = src_file.parents[2] / "platforms" / "hf-spaces" / "demolib"
    assert (expected_local, "/root/platforms/hf-spaces/demolib") in mounts, (
        f"thiếu mount `demolib` cho `::gallery`; đang có: {mounts}"
    )
    # và thư mục đó phải thật sự tồn tại, nếu không Modal chỉ lỗi lúc build
    assert expected_local.is_dir()
    assert (expected_local / "provenance.py").is_file()
