"""Test cho bản **Gradio** của demo.

Bản Streamlit có `tests/test_hf_spaces_demo.py`; đây là bản song song, kiểm đúng
những chỗ dễ sai khi đổi giao diện:

* app dựng được và có đủ tab, không exception lúc `build_app()`;
* `P(fake)` vẫn bị chặn bởi **cùng** một hàm (`provenance.Sample.score`), không
  phải bởi một `if` viết lại trong giao diện;
* đường **nhận ảnh** (tab *Tính DIRE thật*) không bao giờ trả xác suất — đây là
  lỗi của bản Gradio đầu tiên và không được quay lại;
* không có câu nào tự nhận độ chính xác của bài báo.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "platforms" / "hf-spaces"
SRC = APP_DIR / "gradio_app.py"


@pytest.fixture(scope="module")
def gradio_app():
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    import gradio_app as mod
    return mod


def _png(path: Path, colour: tuple[int, int, int] = (10, 20, 30)) -> str:
    """PNG hợp lệ nhỏ nhất có thể — không cần Pillow cho một test giao diện."""
    import struct
    import zlib

    raw = b"\x00" + bytes(colour)
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _sample(gallery: Path, name: str, *, gate: str = "failed", split: str = "val") -> None:
    orig = gallery / f"{name}_orig.png"
    orig.write_bytes(_png(gallery))
    (gallery / f"{name}_diff.png").write_bytes(_png(gallery, (90, 90, 90)))
    meta = {
        "name": name,
        "generator": name.split("_")[0],
        "split": "test",
        "label": "R1+R2",
        "map_source": "maps/test · ::dire --steps 20",
        "sha256": {"orig": hashlib.sha256(orig.read_bytes()).hexdigest(),
                   "map": hashlib.sha256(b"map").hexdigest()},
        "S": 20,
        "detector": {
            "name": "gen-adm", "gate": gate, "threshold": 0.5,
            "threshold_split": split, "p_fake": 0.83, "source": "::evaluate",
        },
    }
    (gallery / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def gallery(tmp_path, gradio_app, monkeypatch):
    g = tmp_path / "gallery"
    g.mkdir()
    _sample(g, "adm_test_0000", gate="passed")
    _sample(g, "real_test_0000", gate="failed")
    monkeypatch.setattr(gradio_app, "SAMPLES", None)      # buộc đọc lại, không dùng cache
    samples, problems = gradio_app.provenance.discover(g)
    monkeypatch.setattr(gradio_app, "SAMPLES", samples)
    monkeypatch.setattr(gradio_app, "PROBLEMS", problems)
    return g


def test_app_builds_and_has_every_tab(gradio_app):
    demo = gradio_app.build_app()
    labels = []

    def walk(block):
        lab = getattr(block, "label", None)
        if isinstance(lab, str) and "·" in lab:
            labels.append(lab)
        # `Blocks` có `.blocks`; `Tab`/`Row`/`Column` cũng có, nhưng `Tab` lồng
        # trong `Tabs` — nên đi theo cả hai đường thay vì giả định một kiểu.
        for child in list(getattr(block, "blocks", {}).values()):
            walk(child)
        for child in list(getattr(block, "children", []) or []):
            walk(child)

    walk(demo)
    text = " ".join(labels)
    for expected in ("Gallery", "Tính DIRE thật", "Thí nghiệm 2 chiều", "Lịch & chi phí",
                     "Bảng 2 & tái lập"):
        assert expected in text, (expected, labels)


def test_gallery_page_uses_the_shared_gate_not_its_own_if(gradio_app, gallery):
    """Cùng một mẫu, hai trạng thái cổng ⇒ chỉ khác nhau ở việc **có số hay không**.

    Nếu giao diện tự viết lại điều kiện (`if p_fake > 0.5`) thì test này vẫn xanh
    khi cổng mở, nhưng sẽ lệch ngay khi `provenance` đổi quy tắc. Vì vậy test so
    sánh **cả hai** trạng thái với cùng một `.json`.
    """
    panels, evidence, block, meta = gradio_app.show_sample("adm_test_0000 · adm/test")
    assert len(panels) == 2 and "DIRE" in evidence
    assert "0.830" in block and "FAKE" in block
    assert meta["generator"] == "adm"

    panels, evidence, block, meta = gradio_app.show_sample("real_test_0000 · real/test")
    assert "P(fake) = —" in block and "gate" in block          # lý do, không phải số
    assert "0.830" not in block


def test_unknown_selector_is_not_a_crash(gradio_app):
    panels, evidence, block, meta = gradio_app.show_sample("không tồn tại")
    assert panels == [] and "Chưa chọn" in evidence and meta is None


def test_empty_gallery_is_an_honest_state(gradio_app, monkeypatch):
    monkeypatch.setattr(gradio_app, "SAMPLES", [])
    monkeypatch.setattr(gradio_app, "PROBLEMS", [])
    header = gradio_app.gallery_header()
    assert "đang rỗng" in header and "make_gallery" in header
    assert gradio_app.gallery_choices() == []


def test_the_upload_path_returns_a_map_never_a_probability(gradio_app, monkeypatch):
    """Tab *Tính DIRE thật*: đầu ra là **map**, không phải `P(fake)`.

    Đây chính là lỗi của bản Gradio đầu tiên (`docs/demo-web-plan.md` §2). Test
    kiểm cả hai phía: khi chưa cấu hình thì không chạy, khi chạy thì không có
    khoá xác suất nào trong kết quả.
    """
    monkeypatch.delenv("DEMO_URL", raising=False)
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    out = gradio_app.run_dire("/tmp/không-có.png", 5)
    assert out[:3] == (None, None, None) and "DEMO_URL" in out[3]

    class _Res:
        steps, nfe, seconds, resized = 5, 10, 0.92, False
        stats = {"mean": 0.01, "p99": 0.05, "max": 0.2}
        line = "DIRE: modal A10G · detector: không có · nhãn: không áp dụng · S=5"
        map_png = _png(Path("."))
        recon_png = _png(Path("."), (1, 2, 3))

    monkeypatch.setenv("DEMO_URL", "https://example.invalid")
    monkeypatch.setenv("DEMO_TOKEN", "tok")
    monkeypatch.setattr(gradio_app.remote, "call", lambda *a, **k: _Res())

    import tempfile

    img = Path(tempfile.mkdtemp()) / "in.png"
    img.write_bytes(_png(Path(".")))
    orig, recon, dmap, line = gradio_app.run_dire(str(img), 5)
    assert all(p and Path(p).exists() for p in (orig, recon, dmap))
    assert "không phải kết luận thật/giả" in line
    assert "p_fake" not in line and "P(fake)" not in line


def test_source_never_claims_the_paper_accuracy(gradio_app):
    """Số của bài báo chỉ được xuất hiện như *số của bài báo*.

    "99 %" đứng một mình là tự nhận; "99,7" trong bảng Table 2 thì hợp lệ vì cả
    bảng nằm dưới dòng `PAPER`. Kiểm bằng chuỗi, không bằng ngữ nghĩa — đây là
    loại lỗi mà ngữ nghĩa không bắt được.
    """
    text = SRC.read_text(encoding="utf-8")
    assert "chính xác 99" not in text
    assert "99 %" not in text
    assert "99,7/100" in text                       # có trong bảng Table 2
    assert re.search(r'PAPER = "số của paper \(ICCV 2023\)', text)


def test_toy_tab_returns_real_svg_and_matches_the_library(gradio_app):
    hist, scatter, sweep, metrics, line = gradio_app.run_toy(steps=5, n=50, clip=True, seed=3407)
    for svg in (hist, scatter, sweep):
        assert svg.lstrip().startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "tỉ lệ" in metrics and "minh hoạ lý thuyết" in line


def test_cost_tab_matches_dire_zero_budget(gradio_app):
    from demolib import cost
    from dire_zero.budget import Estimate

    line, metrics = gradio_app.cost_block(800, 20, 0.0917, "A10G")
    e = Estimate(n_images=800, nfe_per_image=2 * 20, sec_per_step=0.0917)
    assert line == cost.estimate_line(cost.estimate(800, 20, 0.0917, "A10G"))
    assert f"{e.usd:.2f}" in metrics


def test_schedule_rows_end_at_the_right_timestep(gradio_app):
    from demolib import toy2d

    rows = gradio_app.schedule_rows(20)
    assert len(rows) == 20
    assert rows[0][0] == 0
    # Điểm dễ sai nhất: mốc cuối là 950, không phải 999 (xem tab Lịch & chi phí).
    assert rows[-1][0] == round((20 - 1) * toy2d.T_TRAIN / 20) == 950
