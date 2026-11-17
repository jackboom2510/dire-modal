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
    report = _run()
    assert report["problems"] == []
    assert report["n_rows"] == 8 and report["n_png_on_disk"] == 8
    assert report["n_real"] == 4 and report["n_fake"] == 4
    assert report["sha_checked"] is True and report["manifest_sha_ok"] == 8
    assert report["warnings"] == []


def test_extra_pngs_outside_the_manifest_warn_but_do_not_block(volume_gia: Path):
    """`::gen` ghi `data/<split>/ours_adm` mà manifest không biết — bình thường."""
    _write_ok(volume_gia / "data", n_per_class=1)
    for i in range(3):
        (volume_gia / "data" / "train" / "ours_adm").mkdir(parents=True, exist_ok=True)
        (volume_gia / "data" / "train" / "ours_adm" / f"{i:016x}.png").write_bytes(b"x")
    report = _run()
    assert report["problems"] == []
    assert report["n_png_on_disk"] == report["n_rows"] + 3
    assert any("ours_adm" in w for w in report["warnings"])


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
    assert report["problems"] == [] and report["sha_checked"] is False
