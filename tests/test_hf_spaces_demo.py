"""Khoá những gì demo thuyết trình **không được** làm sai.

Hai nhóm test, hai lý do khác nhau:

**Số học** — `demolib.toy2d` phải trùng lịch và mốc của `dire_zero.schedule`, nếu
không thì demo nói một đằng còn pipeline tính một nẻo. Và nó phải tái hiện đúng
hiện tượng mà buổi nói dựa vào: ảnh sinh được tái tạo sát hơn ảnh thật.

**Liêm chính** — cổng chặn tái phát cho lỗi cũ: app không được có bất kỳ đường nào
nhận ảnh RGB rồi trả xác suất. Test đọc mã nguồn `app.py` chứ không chạy UI (chạy
`app.py` là dựng cả một Streamlit app), nên nó chặn được cả khi UI đổi mà đường
tắt vẫn còn.

Chạy: `pytest tests/test_hf_spaces_demo.py -q`
"""

from __future__ import annotations

import base64
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parent.parent / "platforms" / "hf-spaces"
sys.path.insert(0, str(SPACE))

from demolib import charts, cost, provenance, toy2d  # noqa: E402
from dire_zero import budget, schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Số học
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("steps", [1, 2, 5, 10, 20, 50, 200, 1000])
def test_marks_and_alpha_match_dire_zero(steps: int) -> None:
    assert toy2d.timesteps(1000, steps) == schedule.timesteps(1000, steps)
    ref = schedule.alphas_cumprod(schedule.linear_betas(1000))
    assert len(ref) == len(toy2d.A)
    assert max(abs(a - b) for a, b in zip(toy2d.A, ref)) < 1e-15


def test_alpha_starts_below_one() -> None:
    # ᾱ₀ = 1 − β₀ = 0,9999. Nếu là 1,0 thì phép đảo ngược ở bước đầu chia cho 0
    # và toàn bộ map thành NaN — không một exception nào.
    assert toy2d.A[0] == pytest.approx(0.9999, abs=1e-12)
    assert toy2d.A[-1] > 0.0


def test_nfe_is_two_per_step() -> None:
    for s in (1, 5, 20):
        assert toy2d.run(steps=s, n=10)["nfe"] == 2 * s


def test_generated_images_reconstruct_closer_than_real() -> None:
    """Luận điểm của cả buổi nói: ảnh nằm gần mô hình hơn thì tái tạo sát hơn.

    Ở đây mô hình hoàn hảo và `p_r` lệch khỏi `p_g`, nên hiệu ứng phải hiện ra
    rõ. Nếu test này đỏ thì phần hình học của lập luận sai, không phải chỉ số đẹp
    hay xấu.
    """
    r = toy2d.run(steps=20, n=100, seed=7)
    assert r["gen"]["mean"] < r["real"]["mean"]
    assert r["ratio"] > 1.4, r


def test_ratio_grows_with_steps_then_saturates() -> None:
    r5 = toy2d.run(steps=5, n=80, seed=7)["ratio"]
    r20 = toy2d.run(steps=20, n=80, seed=7)["ratio"]
    r100 = toy2d.run(steps=100, n=80, seed=7)["ratio"]
    assert r5 < r20 < r100
    # Bão hoà: gấp 5 lần số bước không còn gấp đôi tỉ lệ. Đây là điều trang 3
    # nói với người xem, nên nó phải đúng với tham số mặc định.
    assert r100 < 2 * r20 + 0.3


def test_result_is_json_serialisable_and_marked_as_toy() -> None:
    r = toy2d.run(steps=5, n=20)
    s = json.dumps(r, ensure_ascii=False)
    assert "không phải Table 2" in json.loads(s)["note"]


# ---------------------------------------------------------------------------
# Chi phí: hai nơi, một công thức
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,steps,gpu", [(1, 1, "A10G"), (800, 20, "A10G"),
                                         (1000, 50, "A100-40GB"), (0, 20, "T4")])
def test_estimate_matches_budget_module(n: int, steps: int, gpu: str) -> None:
    mine = cost.estimate(n, steps, 0.0917, gpu)
    ref = budget.Estimate(n_images=n, nfe_per_image=2 * steps, sec_per_step=0.0917, gpu=gpu)
    assert mine["nfe_total"] == ref.nfe_total
    assert mine["gpu_h"] == pytest.approx(ref.gpu_h, rel=1e-12)
    assert mine["usd"] == pytest.approx(ref.usd, rel=1e-12)
    assert mine["usd_band"] == pytest.approx(ref.band(), rel=1e-12)


def test_estimate_line_matches_budget_line() -> None:
    e = cost.estimate(800, 20, 0.0917, "A10G")
    ref = budget.Estimate(n_images=800, nfe_per_image=40, sec_per_step=0.0917)
    assert cost.estimate_line(e) == ref.line()


def test_prices_are_the_same_table() -> None:
    assert cost.PRICES == budget.PRICES
    assert cost.SPREAD == budget.SPREAD


# ---------------------------------------------------------------------------
# Liêm chính: app không được chấm ảnh RGB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def app_source() -> str:
    return (SPACE / "app.py").read_text(encoding="utf-8")


def test_app_has_no_upload_path(app_source: str) -> None:
    assert "file_uploader" not in app_source
    assert "P(fake)" in app_source  # vẫn có, nhưng chỉ trong trang gallery


def test_app_has_no_model_inference(app_source: str) -> None:
    for banned in ("sigmoid", "softmax", "torch", "torchvision", "hf_hub_download"):
        assert banned not in app_source, f"app.py không được chứa {banned!r}"


def test_app_never_claims_paper_accuracy(app_source: str) -> None:
    assert "số của paper (ICCV 2023)" in app_source
    for claim in ("độ chính xác 99", "accuracy 99", "99 % chính xác"):
        assert claim not in app_source.lower()


# ---------------------------------------------------------------------------
# Nguồn gốc mẫu
# ---------------------------------------------------------------------------
#: PNG 2×2 hợp lệ (79 byte). Phải là PNG **thật**: trang gallery đưa tệp cho Pillow
#: đọc, nên một chuỗi bắt đầu bằng magic header sẽ làm Streamlit ném
#: `UnidentifiedImageError` — đúng lỗi mà test này từng bắt được.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGM8ISfHwMDAxMDAwMDAAAANBAEI"
    "fXHKZgAAAABJRU5ErkJggg=="
)


def _write_sample(root: Path, name: str = "adm_0000", **meta) -> Path:
    for suffix in ("_orig.png", "_diff.png"):
        (root / f"{name}{suffix}").write_bytes(TINY_PNG)
    payload = {
        "name": name,
        "generator": "adm",
        "split": "test",
        "label": "R0",
        "map_source": "maps/test/adm/0000.npy",
        "sha256": {"orig": "a" * 64, "map": "b" * 64},
        "S": 20,
    }
    payload.update(meta)
    p = root / f"{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_gallery_discovers_a_valid_sample(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    samples, problems = provenance.discover(tmp_path)
    assert not problems
    assert samples[0].selector == "adm_0000 · adm/test"
    assert "nhãn: R0" in provenance.evidence_line(samples[0])
    assert "S=20" in provenance.evidence_line(samples[0])


@pytest.mark.parametrize("bad", [{"label": "R9"}, {"S": 0}, {"S": "hai mươi"},
                                 {"sha256": {"orig": "xyz"}}, {"map_source": "  "}])
def test_gallery_rejects_bad_metadata(tmp_path: Path, bad: dict) -> None:
    _write_sample(tmp_path, **bad)
    samples, problems = provenance.discover(tmp_path)
    assert not samples
    assert problems and "adm_0000.json" in problems[0]


def test_gallery_reports_missing_files_without_crashing(tmp_path: Path) -> None:
    p = _write_sample(tmp_path)
    (tmp_path / "adm_0000_diff.png").unlink()
    samples, problems = provenance.discover(tmp_path)
    assert not samples
    assert "thiếu tệp ảnh" in problems[0]
    assert p.exists()


def test_empty_gallery_is_an_honest_state(tmp_path: Path) -> None:
    samples, problems = provenance.discover(tmp_path)
    assert samples == [] and problems == []
    assert "make_gallery.py" in provenance.GENERATE_HELP


def test_score_hidden_without_passed_gate(tmp_path: Path) -> None:
    det = {"name": "lsun_adm.pth", "threshold": 0.5, "threshold_split": "val",
           "p_fake": 0.97, "source": "R0"}
    _write_sample(tmp_path, detector={**det, "gate": "failed"})
    samples, _ = provenance.discover(tmp_path)
    score, reason = samples[0].score()
    assert score is None and "cổng 2" in reason


def test_score_hidden_when_threshold_not_chosen_on_val(tmp_path: Path) -> None:
    _write_sample(tmp_path, detector={"name": "x", "gate": "passed", "threshold": 0.5,
                                      "threshold_split": "test", "p_fake": 0.97})
    samples, _ = provenance.discover(tmp_path)
    score, reason = samples[0].score()
    assert score is None and "val" in reason


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), "0.5", True, None])
def test_score_hidden_when_p_fake_invalid(tmp_path: Path, p) -> None:
    _write_sample(tmp_path, detector={"name": "x", "gate": "passed", "threshold": 0.5,
                                      "threshold_split": "val", "p_fake": p})
    samples, _ = provenance.discover(tmp_path)
    score, reason = samples[0].score()
    assert score is None and "p_fake" in reason


def test_score_shown_when_every_gate_passes(tmp_path: Path) -> None:
    _write_sample(tmp_path, detector={"name": "lsun_adm.pth", "gate": "passed",
                                      "threshold": 0.5, "threshold_split": "val",
                                      "p_fake": 0.97, "source": "R0"})
    samples, _ = provenance.discover(tmp_path)
    score, reason = samples[0].score()
    assert score and reason == ""
    assert "lsun_adm.pth" in provenance.evidence_line(samples[0])


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------
def _root(svg: str) -> ET.Element:
    return ET.fromstring(svg)  # ném nếu không phải XML hợp lệ


def test_histogram_is_valid_svg_with_one_rect_per_bin_and_class() -> None:
    svg = charts.histogram([0.0, 1.0, 2.0], [3, 5], [4, 1])
    root = _root(svg)
    assert root.tag.endswith("svg")
    bars = [e for e in root.iter() if e.tag.endswith("rect") and e.get("opacity") == "0.85"]
    assert len(bars) == 4


def test_histogram_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        charts.histogram([0.0, 1.0], [1, 2], [3])


def test_scatter_and_sweep_are_valid_svg() -> None:
    r = toy2d.run(steps=5, n=20, seed=1)
    assert _root(charts.scatter(r["pairs"])).tag.endswith("svg")
    assert _root(charts.sweep([{"steps": 5, "ratio": 1.3}, {"steps": 20, "ratio": 1.8}])).tag.endswith("svg")


def test_sweep_needs_points() -> None:
    with pytest.raises(ValueError):
        charts.sweep([])


# ---------------------------------------------------------------------------
# Giao diện: mọi trang phải dựng được, và trang gallery phải *ẩn* số khi chưa qua cổng
# ---------------------------------------------------------------------------
def _app_test(gallery: Path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DIRE_GALLERY_DIR", str(gallery))
    at = AppTest.from_file(str(SPACE / "app.py"), default_timeout=180)
    return at.run()


def test_every_page_renders_without_exception(tmp_path: Path, monkeypatch) -> None:
    """Chạy thật script Streamlit, không phải đọc mã nguồn.

    Trang 3 mất ~1,3 s vì nó tính 7 lượt quét `S`; đó là chi phí thật của lần mở
    đầu tiên, không phải cache hỏng.
    """
    at = _app_test(tmp_path, monkeypatch)
    assert not at.exception, at.exception
    options = list(at.radio[0].options)
    assert len(options) == 5
    for opt in options:
        at.radio[0].set_value(opt).run()
        assert not at.exception, (opt, at.exception)
        assert at.title[0].value


def test_gallery_page_hides_the_score_before_the_gate(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "gallery"
    store.mkdir()
    _write_sample(store, detector={"name": "lsun_adm.pth", "gate": "failed",
                                   "threshold": 0.5, "threshold_split": "val",
                                   "p_fake": 0.97})
    at = _app_test(store, monkeypatch)
    at.radio[0].set_value("2 · Gallery (chế độ A)").run()
    assert not at.exception, at.exception
    assert any(m.label == "P(fake)" and m.value == "—" for m in at.metric)
    assert any("cổng 2" in w.value for w in at.warning)


def test_gallery_page_shows_the_score_after_the_gate(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "gallery"
    store.mkdir()
    _write_sample(store, detector={"name": "lsun_adm.pth", "gate": "passed",
                                   "threshold": 0.5, "threshold_split": "val",
                                   "p_fake": 0.9734, "source": "R0"})
    at = _app_test(store, monkeypatch)
    at.radio[0].set_value("2 · Gallery (chế độ A)").run()
    assert not at.exception, at.exception
    assert any(m.label == "P(fake)" and m.value == "0.973" for m in at.metric)
    line = [m for m in at.markdown if "DIRE:" in m.value]
    assert line and "nhãn: R0" in line[0].value and "S=20" in line[0].value


def test_empty_gallery_shows_the_command_not_blank_panels(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "gallery"
    store.mkdir()
    at = _app_test(store, monkeypatch)
    at.radio[0].set_value("2 · Gallery (chế độ A)").run()
    assert not at.exception, at.exception
    assert any("Gallery đang rỗng" in w.value for w in at.warning)
    assert at.image == [] or len(at.image) == 0
    text = " ".join(m.value for m in at.markdown)
    assert "make_gallery.py" in text
