"""`tools/make_gallery.py` sinh ra gallery mà **app đọc được** — kiểm bằng chính bộ đọc của app.

Bài kiểm này tồn tại vì một failure mode rất cụ thể: tool ghi `.json` với khoá hơi
khác (`steps` thay vì `S`, `hash` thay vì `sha256`), app từ chối toàn bộ, và trên
Space người xem thấy "Gallery đang rỗng" trong khi thư mục có 16 tệp. Không có lỗi
nào được ném ra ở đâu cả.

Nên chuỗi test ở đây là: dựng dữ liệu giả → chạy tool → `provenance.discover()` →
khẳng định **đúng số mẫu** đi qua. Đó là hợp đồng giữa tool và app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "platforms" / "hf-spaces"))
sys.path.insert(0, str(REPO / "tools"))

import make_gallery  # noqa: E402
from demolib import provenance  # noqa: E402


def _fake_tree(root: Path, n: int = 3, size: int = 256) -> tuple[Path, Path]:
    from PIL import Image

    maps, orig = root / "maps" / "test", root / "images" / "test"
    rng = np.random.default_rng(0)
    for cls in ("real", "adm"):
        (maps / cls).mkdir(parents=True)
        (orig / cls).mkdir(parents=True)
        for i in range(n):
            arr = rng.random((size, size)).astype("float16") * 2.0
            np.save(maps / cls / f"{i:04d}.npy", arr)
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
                orig / cls / f"{i:04d}.png")
    return maps, orig


def _args(maps: Path, orig: Path, out: Path, *extra: str) -> list[str]:
    return ["--maps", str(maps), "--orig", str(orig), "--out", str(out),
            "--n", "2", "--steps", "20", "--label", "R0", *extra]


def test_tool_writes_gallery_the_app_accepts(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src")
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0

    samples, problems = provenance.discover(out)
    assert problems == []
    assert len(samples) == 4  # 2 lớp × 2 mẫu
    assert {s.generator for s in samples} == {"real", "adm"}
    assert all(s.steps == 20 and s.label == "R0" for s in samples)
    assert all(s.recon is None for s in samples)  # ::dire không lưu recon

    # sha256 phải là hash của tệp .npy thật, không phải chuỗi bịa.
    import hashlib

    one = next(s for s in samples if s.generator == "adm")
    npy = maps / "adm" / f"{one.name.rsplit('_', 1)[-1]}.npy"
    assert one.sha256["map"] == hashlib.sha256(npy.read_bytes()).hexdigest()


def test_no_scores_means_no_score_shown(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    samples, _ = provenance.discover(out)
    assert samples
    for s in samples:
        score, reason = s.score()
        assert score is None and "detector" in reason


def test_scores_are_hidden_unless_gate_passed(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    out = tmp_path / "gallery"
    scores = tmp_path / "scores.json"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    names = [s.name for s in provenance.discover(out)[0]]
    scores.write_text(json.dumps({names[0]: {"p_fake": 0.97}}), encoding="utf-8")

    # Mặc định --gate failed: có p_fake trong .json nhưng app vẫn không hiện số.
    out2 = tmp_path / "g2"
    assert make_gallery.main(_args(maps, orig, out2, "--scores", str(scores))) == 0
    s = next(s for s in provenance.discover(out2)[0] if s.name == names[0])
    score, reason = s.score()
    assert score is None and "cổng 2" in reason

    out3 = tmp_path / "g3"
    assert make_gallery.main(_args(maps, orig, out3, "--scores", str(scores),
                                   "--gate", "passed")) == 0
    s = next(s for s in provenance.discover(out3)[0] if s.name == names[0])
    score, _ = s.score()
    assert score and score["p_fake"] == 0.97 and score["threshold_split"] == "val"


def test_missing_original_is_skipped_not_fatal(tmp_path: Path, capsys) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=2)
    (orig / "adm" / "0001.png").unlink()
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    samples, problems = provenance.discover(out)
    assert problems == []
    assert len(samples) == 3
    assert "không có ảnh gốc" in capsys.readouterr().err


def test_wrong_map_shape_fails_loudly(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    np.save(maps / "adm" / "0000.npy", np.zeros((64, 64), dtype="float16"))
    with pytest.raises(ValueError, match="shape"):
        make_gallery.main(_args(maps, orig, tmp_path / "gallery"))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    assert make_gallery.main(_args(maps, orig, out, "--dry-run")) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "adm_0000_diff.png", "adm_0000_orig.png", "real_0000_diff.png", "real_0000_orig.png"]


def test_existing_gallery_requires_force(tmp_path: Path) -> None:
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    with pytest.raises(SystemExit, match="--force"):
        make_gallery.main(_args(maps, orig, out))
    assert make_gallery.main(_args(maps, orig, out, "--force")) == 0


def test_missing_maps_dir_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--maps"):
        make_gallery.main(_args(tmp_path / "nope", tmp_path, tmp_path / "g"))


def test_display_scale_is_declared_not_implied(tmp_path: Path) -> None:
    """PNG chỉ để xem; `.npy` mới là dữ liệu. `.json` phải nói ra điều đó."""
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    meta = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert meta["display"]["authoritative"] == ".npy"
    assert meta["display"]["png_is_display_only"] is True


def test_missing_maps_dir_says_where_the_volume_is(tmp_path, capsys):
    """`--maps không phải thư mục: maps\test` là câu **không sửa được** gì.

    Ở máy Windows, map nằm trên Volume Modal chứ không nằm trong repo, nên lỗi
    thật là "chưa tải về" — thông báo phải nói ra đường dẫn đang được tra từ đâu
    và lệnh tải (hoặc lệnh dựng thẳng trên Modal).
    """
    with pytest.raises(SystemExit) as exc:
        make_gallery.main(["--maps", str(tmp_path / "maps" / "test"),
                           "--orig", str(tmp_path), "--out", str(tmp_path / "g")])
    msg = str(exc.value)
    assert "không phải thư mục" in msg
    assert "đang tra từ" in msg
    assert "modal volume get dire-zero maps" in msg
    assert "::gallery" in msg                   # đường không phải tải 800 MB


def test_scores_that_match_no_sample_is_an_error_not_a_silent_no_op(tmp_path):
    """Gallery dựng ra vẫn đẹp, mọi mẫu "chưa có điểm" — và không ai biết vì sao.

    Khoá phải đúng `f"{prefix}{cls}_{stem}"`; lệch (ví dụ dùng `gen-adm/000.npy`)
    là lỗi im lặng nguy hiểm nhất của tool này.
    """
    maps, orig = _fake_tree(tmp_path / "src")
    out = tmp_path / "g"
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"samples": {"gen-adm/000.npy": {"p_fake": 0.9}}}),
                      encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        make_gallery.main(["--maps", str(maps), "--orig", str(orig), "--out", str(out),
                           "--n", "1", "--scores", str(scores)])
    msg = str(exc.value)
    assert "KHÔNG khớp mẫu nào" in msg
    assert "adm_" in msg                        # nói luôn khoá đúng là gì


def test_it_recognises_a_placeholder_eaten_by_the_shell(tmp_path):
    """`--maps <volume>/maps/test` để lại `/maps/test` — không phải lỗi đường dẫn.

    Bash biến `<volume>` thành chuyển hướng stdin, bỏ token đó khỏi argv và để
    lại `/maps/test`. Trên Git Bash/MSYS đường dẫn còn bị dịch thành
    `C:/Program Files/Git/maps/test`. Thông báo phải chỉ đúng nguyên nhân, vì
    "không phải thư mục" khiến người dùng đi tìm lỗi ở chỗ khác.
    """
    with pytest.raises(SystemExit) as exc:
        make_gallery.main(["--maps", "/maps/test", "--orig", str(tmp_path),
                           "--out", str(tmp_path / "g")])
    msg = str(exc.value)
    assert "placeholder" in msg
    assert "<volume>" in msg
    assert "::gallery" in msg


def test_literal_angle_brackets_are_explained_too(tmp_path):
    """Trường hợp `--maps "<volume>/maps/test"` (có ngoặc kép ⇒ shell không ăn)."""
    with pytest.raises(SystemExit) as exc:
        make_gallery.main(["--maps", "<volume>/maps/test", "--orig", str(tmp_path),
                           "--out", str(tmp_path / "g")])
    msg = str(exc.value)
    assert "chuyển" in msg and "hướng stdin" in msg


# ---------------------------------------------------------------------------
# Chạy được **một mình** trong container, không có repo xung quanh.
#
# `::gallery` không tải cả repo lên Modal: nó mount đúng `tools/make_gallery.py`
# (và bây giờ là `demolib/`). Lần đầu chỉ mount tệp tool, và container chết ngay:
#
#     File "/root/tools/make_gallery.py", line 55, in <module>
#       from demolib import provenance
#     ModuleNotFoundError: No module named 'demolib'
#
# Ở máy thì không thấy gì, vì repo ở ngay đó. Hai test dưới dựng lại đúng bố cục
# container (chỉ `tools/` + `platforms/hf-spaces/demolib`) và chạy bằng subprocess
# với `-I`, nên `sys.path` chỉ còn đường mà chính tool tự thêm.
# ---------------------------------------------------------------------------

DEMOLIB = REPO / "platforms" / "hf-spaces" / "demolib"


def _container_root(tmp: Path, with_demolib: bool = True) -> Path:
    """Bố cục y như `modal_app.image`: `/root/tools` + `/root/platforms/hf-spaces`."""
    import shutil

    root = tmp / "root"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(REPO / "tools" / "make_gallery.py", root / "tools" / "make_gallery.py")
    if with_demolib:
        shutil.copytree(DEMOLIB, root / "platforms" / "hf-spaces" / "demolib")
    return root


def _run_isolated(root: Path, tmp: Path, *args: str):
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-I", str(root / "tools" / "make_gallery.py"), *args],
        cwd=tmp, env=env, capture_output=True, text=True, timeout=120,
    )


def test_tool_runs_with_only_the_two_mounted_paths(tmp_path):
    """Đúng những gì container có: tệp tool + `demolib/`. Không repo, không cwd."""
    root = _container_root(tmp_path)
    r = _run_isolated(root, tmp_path, "--help")
    assert r.returncode == 0, r.stderr
    assert "--maps" in r.stdout


def test_missing_demolib_names_the_mount_instead_of_a_bare_import_error(tmp_path):
    """Thiếu thư viện thì thông báo phải nói *thiếu cái gì, ở đâu* — lỗi cũ chỉ có
    `ModuleNotFoundError: No module named 'demolib'` kèm stack của importlib."""
    root = _container_root(tmp_path, with_demolib=False)
    r = _run_isolated(root, tmp_path, "--help")
    assert r.returncode != 0
    assert "demolib" in r.stderr
    assert "mount" in r.stderr
    assert "ModuleNotFoundError" not in r.stderr    # thay bằng thông báo có chỉ dẫn


def test_full_build_works_with_only_the_mounted_tree(tmp_path):
    """Không chỉ `--help`: dựng thật, trong bố cục container. Nếu `numpy`/`PIL` được
    import ở cấp module thì container (image chỉ có torch/numpy/pillow) sẽ chết ở
    đây chứ không phải ở `--help`."""
    import shutil

    root = _container_root(tmp_path)
    work = tmp_path / "vol"
    maps, orig = work / "maps" / "test", work / "data" / "test"
    rng = np.random.default_rng(1)
    from PIL import Image

    for cls in ("real", "adm"):
        (maps / cls).mkdir(parents=True)
        (orig / cls).mkdir(parents=True)
        for i in range(2):
            # (3, 256, 256): đúng shape `::dire` ghi cho ảnh RGB, không phải fixture xám
            np.save(maps / cls / f"{cls}_{i}.npy",
                    (rng.random((3, 256, 256)) * 2.0).astype("float16"))
            Image.fromarray((rng.random((256, 256)) * 255).astype("uint8")).save(
                orig / cls / f"{cls}_{i}.png")

    out = tmp_path / "gallery"
    r = _run_isolated(
        root, tmp_path,
        "--maps", str(maps), "--orig", str(orig), "--out", str(out),
        "--n", "2", "--steps", "20", "--label", "R1+R2",
        "--map-source", "test/*.npy · ::dire --steps 20",
    )
    assert r.returncode == 0, r.stderr + r.stdout
    made = sorted(p.name for p in out.glob("*.json"))
    assert len(made) == 4, made
    shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Map thật của `::dire` có **3 kênh**, không phải ảnh xám.
#
# Container đã trả về đúng lỗi này trên dữ liệu thật của repo:
#
#     ValueError: 004006349103115c.npy: shape (3, 256, 256), cần (256, 256)
#
# `::dire` lưu `dm.squeeze(0)` của một tensor RGB ⇒ (3, 256, 256). Tool chỉ nhận
# (256, 256) vì mọi fixture đều tự tạo ảnh xám — một lần nữa, fixture không đại
# diện cho dữ liệu thật.
# ---------------------------------------------------------------------------


def _rgb_map(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((3, 256, 256)) * 2.0).astype("float16")
    np.save(path, arr)


def test_three_channel_map_is_drawn_as_rgb_without_mixing_channels(tmp_path: Path) -> None:
    from PIL import Image

    npy = tmp_path / "x.npy"
    png = tmp_path / "x.png"
    rng = np.random.default_rng(7)
    arr = (rng.random((3, 256, 256)) * 2.0).astype("float16")
    np.save(npy, arr)

    mode = make_gallery.map_to_png(npy, png, 1.0)      # gamma 1 để so số chính xác
    assert mode == "RGB"
    with Image.open(png) as im:
        assert im.mode == "RGB"
        assert im.size == (256, 256)
        got = np.asarray(im).astype("float32") / 255.0
    want = np.clip(arr.astype("float32") / make_gallery.MAP_MAX, 0.0, 1.0)
    got = np.moveaxis(got, -1, 0)
    assert np.abs(got - want).max() <= 1 / 255 + 1e-6
    # kênh phải khác nhau — nếu bị trộn thành xám thì ba kênh bằng nhau
    assert np.abs(got[0] - got[1]).max() > 0.05


def test_single_channel_map_still_works(tmp_path: Path) -> None:
    from PIL import Image

    npy = tmp_path / "y.npy"
    png = tmp_path / "y.png"
    np.save(npy, np.zeros((1, 256, 256), dtype="float16"))
    assert make_gallery.map_to_png(npy, png, 0.5) == "L"
    with Image.open(png) as im:
        assert im.mode == "L"


def test_channels_last_map_is_read_as_rgb(tmp_path: Path) -> None:
    """Một số tool lưu `(H, W, 3)`; nhận cả hai thứ tự thay vì đoán sai trục."""
    from PIL import Image

    npy = tmp_path / "z.npy"
    png = tmp_path / "z.png"
    arr = np.zeros((256, 256, 3), dtype="float16")
    arr[..., 0] = 2.0
    np.save(npy, arr)
    assert make_gallery.map_to_png(npy, png, 1.0) == "RGB"
    with Image.open(png) as im:
        px = np.asarray(im)
    assert px[..., 0].min() == 255      # kênh 0 = 2.0/MAP_MAX → 255
    assert px[..., 1].max() == 0


def test_shape_error_lists_what_is_accepted(tmp_path: Path) -> None:
    """Thông báo cũ chỉ nói "cần (256, 256)" — đúng điều đã làm mất một vòng lặp
    deploy, vì nó không nói map thật có 3 kênh."""
    npy = tmp_path / "bad.npy"
    np.save(npy, np.zeros((2, 256, 256), dtype="float16"))
    with pytest.raises(ValueError) as err:
        make_gallery.map_to_png(npy, tmp_path / "bad.png", 0.5)
    msg = str(err.value)
    assert "(3, 256, 256)" in msg
    assert "3 kênh" in msg


def test_build_records_channel_count_in_metadata(tmp_path: Path) -> None:
    """`.json` phải nói PNG giữ 3 kênh hay 1 kênh — người đọc gallery cần biết
    ảnh mình đang xem là ba phép đo hay một."""
    maps, orig = _fake_tree(tmp_path / "src", n=1)
    (maps / "adm" / "0000.npy").unlink()
    _rgb_map(maps / "adm" / "0000.npy", seed=3)
    out = tmp_path / "gallery"
    assert make_gallery.main(_args(maps, orig, out)) == 0
    meta = json.loads((out / "adm_0000.json").read_text(encoding="utf-8"))
    assert meta["display"]["channels"] == 3
    assert meta["display"]["channel_order"] == "RGB"
