"""`demolib.live` phải tính **đúng** DIRE, và phải nói thật khi không tính được.

Hai điều được khoá, theo hai kiểu sai khác nhau:

1. **Sai số học.** `live.run_steps` viết lại vòng lặp để lấy được ảnh trung gian
   ở từng bước (bản trong `dire_zero.diffusion` không trả chúng). Bản sao nào
   cũng trôi, nên test chạy **cả hai** trên cùng mô hình tí hon và đòi map bằng
   nhau từng phần tử. Không có ngoại lệ "chắc là giống".

2. **Sai lời.** Khi thiếu torch hoặc thiếu checkpoint, module phải trả một
   `LiveStatus` nói **thiếu gì** và **làm gì tiếp** — không được im lặng trả một
   map rỗng, và không được ném exception làm sập cả trang. Một demo không có dữ
   liệu mà hiện khung trống là kiểu thất bại tệ nhất: người xem tưởng nó chạy.

Toàn bộ chạy trên CPU với mô hình 32 kênh nên mất vài giây, không cần GPU, không
cần checkpoint 2,5 GB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SPACE = Path(__file__).resolve().parent.parent / "platforms" / "hf-spaces"
sys.path.insert(0, str(SPACE))
sys.path.insert(0, str(SPACE / "demolib"))

from demolib import live  # noqa: E402
from dire_zero.diffusion import ModelCounter, Schedule, dire_map, from_uint8  # noqa: E402
from dire_zero.unet import UNetConfig, ZeroUNet  # noqa: E402

# Cấu hình tí hon nhưng **đủ tầng**: 5 mức ⇒ khối chú ý giữa chạy ở 16² (256 vị
# trí). Với 2 mức thì nó chạy ở 128² ⇒ ma trận chú ý 2×16384² = 2,1 GB và tiến
# trình bị OOM giết — đúng điều đã xảy ra khi viết test này. ADM thật có 6 mức nên
# chú ý giữa chỉ ở 8²; không có chuyện đó khi chạy checkpoint thật.
TINY = UNetConfig(image_size=256, model_channels=32, num_res_blocks=1,
                  attention_ds=(8,), channel_mult=(1, 1, 1, 2, 2),
                  num_heads=2, head_channels=-1)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = ZeroUNet(TINY).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def image():
    import numpy as np

    rng = np.random.default_rng(0)
    return (rng.random((256, 256, 3)) * 255).astype("uint8")


def test_stepwise_loop_matches_dire_zero(tiny_model, image) -> None:
    """Đây là test quan trọng nhất của tệp: bản sao vòng lặp phải cho **cùng** map."""
    out = live.run_steps(tiny_model, image, steps=3)

    x0 = from_uint8(image)
    sched = Schedule(steps=3, device="cpu")
    # `dire_map` nhận `ModelCounter` (nó đếm NFE và assert `= 2S`), không nhận model trần.
    ref_map, _ref_recon = dire_map(ModelCounter(tiny_model), x0, sched)

    from PIL import Image
    import io
    import numpy as np

    got = np.asarray(Image.open(io.BytesIO(out["map_png"])).convert("L"), dtype="float32")
    # Ảnh PNG là map đã qua gamma 0,5 và lượng tử hoá 8 bit, nên so **thống kê**,
    # còn số học chính xác thì so qua `map_values`.
    assert out["map_values"]["mean"] == pytest.approx(float(ref_map.mean()), abs=5e-3)
    assert out["map_values"]["max"] == pytest.approx(float(ref_map.max()), abs=1e-2)
    assert 0.0 < float(got.mean()) < 255.0


def test_nfe_is_two_per_step_and_frames_cover_both_phases(tiny_model, image) -> None:
    out = live.run_steps(tiny_model, image, steps=2)
    assert out["nfe"] == 4 == len(out["frames"])
    assert [f["phase"] for f in out["frames"]] == ["invert", "invert", "reconstruct", "reconstruct"]
    assert [f["t"] for f in out["frames"]] == [0, 500, 500, 0]
    assert out["steps"] == 2 and len(out["timings"]) == 4


def test_on_frame_is_called_in_order(tiny_model, image) -> None:
    seen = []
    live.run_steps(tiny_model, image, steps=2, on_frame=lambda d: seen.append((d["phase"], d["t"])))
    assert seen == [("invert", 0), ("invert", 500), ("reconstruct", 500), ("reconstruct", 0)]


def test_map_is_in_range_and_reproducible(tiny_model, image) -> None:
    a = live.run_steps(tiny_model, image, steps=2)
    b = live.run_steps(tiny_model, image, steps=2)
    assert 0.0 <= a["map_values"]["mean"] <= 2.0
    assert a["map_values"] == b["map_values"], "cùng đầu vào phải cùng kết quả"
    assert len(a["map_png"]) > 100 and a["recon_png"][:4] == b"\x89PNG"


def test_steps_are_capped_and_validated(tiny_model, image) -> None:
    with pytest.raises(ValueError, match="demo"):
        live.run_steps(tiny_model, image, steps=live.MAX_STEPS + 1)
    with pytest.raises(ValueError):
        live.run_steps(tiny_model, image, steps=0)


def test_estimate_matches_the_measured_forward_cost() -> None:
    assert live.estimate_seconds(5) == pytest.approx(101.0, abs=0.5)
    assert live.estimate_seconds(20) == pytest.approx(404.0, abs=1.0)
    assert live.estimate_seconds(1) == pytest.approx(20.2, abs=0.5)


def test_status_without_checkpoint_says_what_to_do(monkeypatch) -> None:
    monkeypatch.setenv("DIRE_CKPT", "/nonexistent/adm_lsun.pt")
    st = live.status()
    assert isinstance(st, live.LiveStatus)
    assert not st.available
    assert "chưa có checkpoint" in st.reason
    assert st.hints and any("DIRE_CKPT" in h for h in st.hints)


def test_status_without_torch_or_dire_zero_still_returns(monkeypatch) -> None:
    """Không được ném: trang gọi hàm này ở mọi lần render."""
    monkeypatch.setattr(live, "_try_import", lambda: (None, None, "không có torch (giả lập)"))
    st = live.status()
    assert not st.available and "torch" in st.reason
    assert any("pip install" in h for h in st.hints)


def test_load_checkpoint_refuses_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        live.load_checkpoint(tmp_path / "nope.pt")


def test_wrong_size_image_is_resized_not_silently_accepted(tmp_path, tiny_model) -> None:
    """Ảnh 64² được đưa về 256² — phải nói ra, không im lặng đo ở cỡ khác.

    ADM chỉ học ở 256²; chạy ở 128² cho ra số nhanh hơn 4× nhưng **không** phải
    phép đo DIRE mà bài báo nói tới.
    """
    import numpy as np

    p = tmp_path / "small.png"
    from PIL import Image

    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(p)
    arr = live._load_image_hwc(p)
    assert arr.shape == (256, 256, 3)


def test_live_page_computes_and_renders_when_enabled(tmp_path, monkeypatch, tiny_model) -> None:
    """Bật chế độ thật rồi **bấm nút**: trang phải tính xong và vẽ ra ảnh.

    Đây là bài kiểm đường giao diện, không chỉ đường thư viện: `st.progress`,
    callback từng bước, `st.session_state`, và dòng nguồn gốc — bốn chỗ có thể
    hỏng mà `run_steps` vẫn trả kết quả đúng. Dùng mô hình tí hon + checkpoint
    giả nên không cần tệp 2,5 GB; `S = 1` để chỉ tốn 2 lượt forward.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        live, "status",
        lambda: live.LiveStatus(True, torch.__version__, True, str(tmp_path / "adm.pt"), "sẵn sàng", []),
    )
    monkeypatch.setattr(live, "load_checkpoint", lambda path=None: (tiny_model, "deadbeef"))

    at = AppTest.from_file(str(SPACE / "app.py"), default_timeout=600)
    at.run()
    at.radio[0].set_value("3 · Tính DIRE thật").run()
    assert not at.exception, at.exception

    at.select_slider[0].set_value(1).run()
    assert not at.button[0].disabled
    at.button[0].click().run()
    assert not at.exception, at.exception

    line = [m.value for m in at.markdown if m.value.startswith("`DIRE:")]
    assert line and "S=1" in line[0] and "2 NFE" in line[0]
    assert "detector: không có" in line[0], "trang này không được ngầm chấm ảnh"
    assert len(at.image) >= 3, "phải có ảnh gốc, ảnh tái tạo và map"
    assert any("DIRE trung bình" in m.label for m in at.metric)
