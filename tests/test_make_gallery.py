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
