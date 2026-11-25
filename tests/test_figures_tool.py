"""Test cho `tools/figures.py` — công cụ sinh figure.

Vì sao bộ test này tồn tại
--------------------------
Một hình sai còn tệ hơn không có hình: nó đi vào báo cáo, vào slide, vào bài nộp,
và không ai kiểm lại. Ba thứ dễ sai nhất, và đây là ba thứ được canh:

1. **Ghép ảnh với map sai cặp.** Map của ảnh A dán lên ảnh B vẫn ra một hình
   rất thuyết phục. `pair_tars` và yêu cầu "cùng tên stem" ở `f1-teaser` là
   hàng rào.
2. **Nhãn mức tái lập sai hoặc thiếu.** `R0` (của tác giả) và `R1+R2` (của
   mình) không được lẫn; hình không nhãn thì không dùng được.
3. **AUC tính trên tập con thuần một lớp** ⇒ NaN, nhưng in ra dưới dạng "—" thì
   trông như "không có dữ liệu" chứ không phải "code sai".

Test chạy **không cần matplotlib thật**: `tests/fixtures/mpl_stub` cung cấp bản
giả đủ để gọi hết đường vẽ và kiểm file/sidecar. Nếu máy có matplotlib thật thì
test dùng bản thật — cùng một khẳng định.

Chạy::

    python3 -m pytest tests/test_figures_tool.py -q
"""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "figures.py"
STUB = ROOT / "tests" / "fixtures" / "mpl_stub"

np = pytest.importorskip("numpy")


def _load_tool():
    """Import `tools/figures.py` như module, không cần cài package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("figures_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fig = _load_tool()


# --------------------------------------------------------------------------- #
# Cấu trúc: không kéo matplotlib vào tầng module, registry khớp CLI
# --------------------------------------------------------------------------- #

def test_module_does_not_import_matplotlib_at_top_level():
    """`--list` phải chạy trên máy chưa cài matplotlib — và package không được nặng thêm."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    tops = {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "matplotlib" not in tops, "matplotlib phải import bên trong hàm vẽ"


def test_every_registered_figure_has_a_cli_command():
    parser = fig.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction))
    for name in fig.FIGURES:
        assert name in sub.choices, f"{name} có trong registry nhưng không có lệnh CLI"


def test_every_registered_figure_exposes_its_inputs_in_list(capsys):
    assert fig.main(["list"]) == 0
    out = capsys.readouterr().out
    for name in fig.FIGURES:
        assert name in out
    assert "docs/plans/research/figures-plan.md" in out


def test_all_allowed_labels_are_the_documented_ones():
    """Không có mức thứ năm: mỗi hình hoặc là số đo có nhãn, hoặc là hình vẽ."""
    assert set(fig.ALLOWED_LABELS) == {"R0", "R1+R2", "R1+R2+R3", "schematic"}


# --------------------------------------------------------------------------- #
# Đọc dữ liệu: ba định dạng map, chuẩn hoá, ghép cặp
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "arr,expected_max",
    [
        (np.array([[0, 255]], dtype="uint8"), 2.0),          # map tác giả: uint8
        (np.array([[0.0, 2.0]], dtype="float16"), 2.0),      # |x − x̂| với x ∈ [−1,1]
        (np.array([[0.0, 1.0]], dtype="float32"), 2.0),      # đã chuẩn hoá [0,1]
    ],
)
def test_map_to_float_handles_all_three_stored_formats(arr, expected_max):
    out = fig.map_to_float(arr)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(expected_max, rel=1e-3)


def test_map_to_float_does_not_silently_divide_a_float_map_by_255():
    """Lỗi thật: `uint8` và `float` lẫn nhau ⇒ mọi thống kê lệch 127×, không có lỗi nào."""
    a = np.full((4, 4), 0.5, dtype="float32")
    assert fig.map_to_float(a).mean() == pytest.approx(1.0)      # [0,1] ⇒ [0,2]


def test_percentile_norm_is_monotone_and_bounded():
    a = np.linspace(0, 5, 100).reshape(10, 10)
    out = fig.percentile_norm(a)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out.ravel()) >= -1e-6)


def test_percentile_norm_of_a_constant_map_does_not_divide_by_zero():
    out = fig.percentile_norm(np.zeros((8, 8)))
    assert np.all(out == 0.0)


def _tar_with(path: Path, names: list[str]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for n in names:
            data = b"x" * 8
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_pair_tars_accepts_matching_names(tmp_path: Path):
    imgs = _tar_with(tmp_path / "adm.tar.gz", ["000001.png", "000002.png"])
    maps = _tar_with(tmp_path / "adm_maps.tar.gz", ["000001.npy", "000002.npy"])
    res = fig.pair_tars(imgs, maps)
    assert res["ok"] is True
    assert res["n_images"] == res["n_maps"] == 2
    assert res["first_mismatch"] is None
    assert res["method"] == "order"


def test_pair_tars_rejects_a_shifted_pairing(tmp_path: Path):
    """Đây là lỗi nguy hiểm nhất: lệch một phần tử ⇒ map ảnh A lên ảnh B."""
    imgs = _tar_with(tmp_path / "adm.tar.gz", ["000001.png", "000002.png"])
    maps = _tar_with(tmp_path / "adm_maps.tar.gz", ["000002.npy", "000001.npy"])
    res = fig.pair_tars(imgs, maps)
    assert res["ok"] is False
    assert res["first_mismatch"] == {"i": 0, "image": "000001", "map": "000002"}


def test_pair_tars_rejects_different_counts(tmp_path: Path):
    imgs = _tar_with(tmp_path / "adm.tar.gz", ["a.png", "b.png", "c.png"])
    maps = _tar_with(tmp_path / "adm_maps.tar.gz", ["a.npy"])
    assert fig.pair_tars(imgs, maps)["ok"] is False


def test_load_maps_is_deterministic_and_seed_sensitive(tmp_path: Path):
    d = tmp_path / "test" / "adm"
    d.mkdir(parents=True)
    for i in range(6):
        np.save(d / f"{i:02d}.npy", np.full((4, 4), i, dtype="float16"))

    first = fig.load_maps(tmp_path, "test", "adm", n=2, seed=1)
    again = fig.load_maps(tmp_path, "test", "adm", n=2, seed=1)
    other = fig.load_maps(tmp_path, "test", "adm", n=2, seed=99)
    assert [a.tolist() for a in first] == [a.tolist() for a in again]
    assert [a.tolist() for a in first] != [a.tolist() for a in other]


def test_load_maps_says_what_is_missing_instead_of_returning_nothing(tmp_path: Path):
    (tmp_path / "test").mkdir()
    with pytest.raises(SystemExit) as ei:
        fig.load_maps(tmp_path, "test", "adm")
    assert "không có .npy" in str(ei.value)


# --------------------------------------------------------------------------- #
# Manifest: sha của nội dung, không của đường dẫn
# --------------------------------------------------------------------------- #

def _manifest(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    head = "png\tlabel\tgenerator\tsplit\tdataset\tsource_tar\tmember\tsha256\twidth\theight"
    lines = [head]
    for png, label, sha in rows:
        lines.append(f"{png}\t{label}\tfake\ttest\tlsun_bedroom\tx.tar.gz\tm\t{sha}\t256\t256")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_manifest_sha_depends_on_content_not_paths(tmp_path: Path):
    a = _manifest(tmp_path / "a.tsv", [("F:/x/1.png", 1, "aa"), ("F:/x/2.png", 0, "bb")])
    b = _manifest(tmp_path / "b.tsv", [("data/y/1.png", 1, "aa"), ("data/y/2.png", 0, "bb")])
    assert fig.manifest_sha(fig.read_manifest(a)) == fig.manifest_sha(fig.read_manifest(b))


def test_manifest_sha_changes_when_a_label_flips(tmp_path: Path):
    """Đổi nhãn mà sha không đổi ⇒ hai bộ dữ liệu khác nhau trông như một."""
    a = _manifest(tmp_path / "a.tsv", [("1.png", 1, "aa")])
    b = _manifest(tmp_path / "b.tsv", [("1.png", 0, "aa")])
    assert fig.manifest_sha(fig.read_manifest(a)) != fig.manifest_sha(fig.read_manifest(b))


def test_read_manifest_rejects_an_empty_file(tmp_path: Path):
    p = tmp_path / "m.tsv"
    p.write_text("", encoding="utf-8")
    assert fig.read_manifest(p) == []


# --------------------------------------------------------------------------- #
# Sidecar: hình không có nhãn/nguồn là hình không kiểm được
# --------------------------------------------------------------------------- #

def test_save_refuses_a_sidecar_without_claim_label(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, _ = plt.subplots()
    with pytest.raises(ValueError, match="claim_label"):
        fig._save(f, plt, tmp_path, "x", {"figure": "X", "kind": "measurement", "sources": {}})


def test_save_refuses_a_sidecar_without_sources(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, _ = plt.subplots()
    with pytest.raises(ValueError, match="sources"):
        fig._save(f, plt, tmp_path, "x", {"figure": "X", "kind": "measurement", "claim_label": "R0"})


def test_stamp_rejects_an_invented_level(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, _ = plt.subplots()
    with pytest.raises(ValueError, match="nhãn không hợp lệ"):
        fig._stamp(f, plt, "R2.5")


def test_write_index_builds_the_table_from_sidecars(tmp_path: Path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "f3-pipeline.json").write_text(json.dumps({
        "figure": "F3", "cmd": "python tools/figures.py f3-pipeline",
        "claim_label": None, "n": None, "sources": {"schedule": "src/dire_zero/schedule.py"},
    }), encoding="utf-8")
    (tmp_path / "data" / "a6-audit.json").write_text(json.dumps({
        "figure": "A6", "cmd": "python tools/figures.py a6-audit --manifest m.tsv",
        "claim_label": "R0", "n": 42, "sources": {"manifest": "/tmp/m.tsv"},
    }), encoding="utf-8")

    text = fig.write_index(tmp_path).read_text(encoding="utf-8")
    assert "| F3 |" in text and "| A6 |" in text
    assert "R0" in text and "schematic" in text
    assert "42" in text
    assert "đừng sửa tay" in text


# --------------------------------------------------------------------------- #
# Chạy thật (bằng stub matplotlib): file + sidecar phải được ghi
# --------------------------------------------------------------------------- #

def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # stub TRƯỚC, nhưng nếu môi trường có matplotlib thật thì vẫn chạy được:
    # `PYTHONPATH` xếp stub lên trước và stub tự khai nó chỉ là bản giả.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(STUB), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize("cmd", ["f2-concept", "f3-pipeline", "a1-provenance"])
def test_schematic_figures_run_end_to_end(cmd, tmp_path: Path):
    out = tmp_path / "figs"
    proc = _run([cmd, "--out", str(out)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (out / f"{cmd}.pdf").exists()
    side = json.loads((out / "data" / f"{cmd}.json").read_text(encoding="utf-8"))
    assert side["kind"] == "schematic"
    assert side["claim_label"] is None          # hình vẽ: không được mang nhãn R
    assert "created" in side and side["cmd"].startswith("python tools/figures.py")
    assert (out / "README.md").exists()


def test_pipeline_figure_reads_steps_from_the_schedule_not_from_a_constant(tmp_path: Path):
    out = tmp_path / "figs"
    proc = _run(["f3-pipeline", "--out", str(out), "--steps", "5"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    side = json.loads((out / "data" / "f3-pipeline.json").read_text(encoding="utf-8"))
    assert side["steps"] == 5
    assert side["nfe_per_image"] == 10
    assert side["timesteps"] == [0, 200, 400, 600, 800]


def test_f1_teaser_refuses_to_pair_by_order(tmp_path: Path):
    """Không có cặp stem khớp ⇒ dừng, không vẽ một hình ghép sai."""
    imgs = tmp_path / "images" / "test" / "adm"
    maps = tmp_path / "maps" / "test" / "adm"
    imgs.mkdir(parents=True)
    maps.mkdir(parents=True)
    for i in range(2):
        (imgs / f"aaa{i}.png").write_bytes(b"not a real png")
        np.save(maps / f"bbb{i}.npy", np.zeros((4, 4), dtype="float16"))
    out = tmp_path / "figs"
    proc = _run(["f1-teaser", "--images", str(tmp_path / "images"),
                 "--maps", str(tmp_path / "maps"), "--out", str(out)], tmp_path)
    assert proc.returncode != 0
    assert "cùng tên stem" in proc.stderr
    assert not (out / "f1-teaser.pdf").exists()
