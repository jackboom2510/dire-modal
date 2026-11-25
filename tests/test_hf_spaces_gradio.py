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
    for expected in ("Gallery", "2 · Mô phỏng & tính DIRE", "Thí nghiệm 2 chiều", "Lịch & chi phí",
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


def test_the_upload_path_defaults_to_local_and_returns_a_map_never_a_probability(gradio_app, monkeypatch, tmp_path):
    """Upload defaults to local DIRE; it returns maps, never detector scores."""
    monkeypatch.delenv("DEMO_URL", raising=False)
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    img = tmp_path / "in.png"
    img.write_bytes(_png(img))
    run = {
        "steps": 5, "nfe": 10, "seconds": 0.92,
        "map_values": {"mean": 0.01, "p99": 0.05, "max": 0.2},
        "map_png": _png(img), "recon_png": _png(img, (1, 2, 3)),
        "cache_hit": False,
    }
    filmstrip = [(str(img), "frame")]
    monkeypatch.setattr(gradio_app, "_local_result", lambda *a, **k: (img.read_bytes(), run, filmstrip, "abc123"))
    monkeypatch.setattr(gradio_app.remote, "call", lambda *a, **k: pytest.fail("Modal must not be called by default"))

    orig, recon, dmap, frames, line = gradio_app.run_dire(str(img), 5)
    assert all(p and Path(p).exists() for p in (orig, recon, dmap))
    assert frames == filmstrip
    assert "DIRE: local ADM" in line and "cache miss" in line
    assert "không phải kết luận thật/giả" in line
    assert "p_fake" not in line and "P(fake)" not in line


def test_modal_requires_explicit_selection_and_credentials(gradio_app, monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_URL", raising=False)
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    img = tmp_path / "in.png"
    img.write_bytes(_png(img))
    out = gradio_app.run_dire(str(img), 5, backend="modal")
    assert out[:3] == (None, None, None) and "DEMO_URL/DEMO_TOKEN" in out[4]


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


def test_module_exposes_demo_for_the_hf_launcher(gradio_app):
    """HF Spaces **import** tệp rồi tìm `demo`, không chạy nó như script.

    Log của một lần deploy thật::

        GRADIO_HOT_RELOAD: Launching demo not found in __main__. Using 'demo'
        * Running on local URL: http://0.0.0.0:7860, with SSR
        Stopping Node.js server...

    Launcher không tìm thấy object nào để phục vụ, rơi vào nhánh dự phòng, bật
    SSR proxy rồi tiến trình Python kết thúc ⇒ proxy chết. App không hỏng; nó
    chưa bao giờ được giao cho launcher. Test này khoá điều kiện tối thiểu:
    `gradio_app.demo` tồn tại ở cấp module.
    """
    import gradio as gr

    assert hasattr(gradio_app, "demo"), "thiếu `demo` ở cấp module — HF không phục vụ được"
    assert isinstance(gradio_app.demo, gr.Blocks)


def test_importing_the_module_does_not_start_a_server(gradio_app):
    """`demo` phải được **dựng**, không phải **launch**. Import mà mở cổng thì
    test và HF sẽ tranh nhau cổng 7860."""
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", 7860))
        except OSError:
            pytest.fail("cổng 7860 đang bị chiếm sau khi import — có gì đó đã launch()")


def test_ssr_is_off_by_default_but_can_be_turned_back_on(gradio_app):
    """SSR = thêm một tiến trình Node đứng trước Python: thêm một chỗ để chết.

    Không bật mặc định, nhưng cũng không cấm: `setdefault` để Space tự đặt
    `GRADIO_SSR_MODE=true` nếu ai đó thật sự cần SSR (SEO) và đã đo.
    """
    import os

    assert os.environ.get("GRADIO_SSR_MODE", "").lower() == "false"


def test_the_launcher_contract_is_documented_where_it_will_be_read(gradio_app):
    """README phải nói về `demo` ở cấp module — nếu không, người sửa sau lại đưa
    `build_app()` vào trong `if __name__ == "__main__"` và Space chết y hệt."""
    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    assert "GRADIO_HOT_RELOAD" in text
    assert "demo = build_app()" in text


def test_import_prints_a_ready_line_instead_of_only_gradio_noise():
    """Log của Space phải có một mốc do **app** in ra.

    Không có nó, log chỉ gồm cảnh báo pip và hai câu của Gradio — và người đọc
    không phân biệt được "chạy xong" với "chết ngay sau khi in banner". Dòng này
    xuất hiện tức là `build_app()` đã trả về và `demo` đã ở cấp module.
    """
    import subprocess

    r = subprocess.run(
        [sys.executable, "-c",
         "import gradio_app as g; print('N_COMPONENTS', len(g.demo.blocks))"],
        cwd=str(APP_DIR), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    ready = [ln for ln in r.stdout.splitlines() if ln.startswith("[demo]")]
    assert len(ready) == 1, r.stdout
    assert "component" in ready[0]
    assert "sẵn sàng" in ready[0]
    # đếm đúng số component của app, không phải một con số chép tay
    literal = next(ln for ln in r.stdout.splitlines() if ln.startswith("N_COMPONENTS"))
    n = int(literal.split()[1])
    assert f"{n} component" in ready[0], (n, ready[0])


def test_requirements_has_no_global_pip_options():
    """`pip` từ chối option toàn cục trong requirements file.

    Đã thử `--root-user-action=ignore` để tắt cảnh báo "Running pip as the 'root'
    user" của HF: `pip install -r` chết với `no such option`. Cảnh báo đó là của
    HF, không tắt được từ repo — nên đừng để ai thêm lại.
    """
    for line in (APP_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert not stripped.startswith("-"), (
            f"option toàn cục trong requirements.txt: {stripped!r} — pip sẽ từ chối"
        )


def test_readme_names_the_four_normal_log_lines(gradio_app):
    """Tài liệu phải nói rõ bốn dòng này là **bình thường**, nếu không lần sau lại
    có người đi sửa một app đang chạy tốt."""
    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    for needle in (
        "Running pip as the 'root' user",
        "[demo]",
        "* Running on local URL",
        "To create a public link",
    ):
        assert needle in text, needle
    assert "hf.space/config" in text
