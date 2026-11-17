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
