"""Đường Space → Modal: đúng thì nhanh, sai thì phải **nói được lý do**.

Ba thứ dễ hỏng ở đây, và không thứ nào lộ ra khi chỉ đọc mã:

1. token rò ra thông báo lỗi (endpoint nhắc lại header trong `detail`);
2. endpoint lạ trả JSON "trông giống" kết quả nhưng thiếu dòng nguồn gốc — nếu
   chấp nhận, demo sẽ hiện một map không biết từ đâu;
3. mã lỗi HTTP bị nuốt thành "lỗi không rõ", khiến người vận hành không biết
   phải sửa token hay hạn mức.

Transport được tiêm vào, nên không có request mạng nào trong test.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parent.parent / "platforms" / "hf-spaces"
sys.path.insert(0, str(SPACE))

from demolib import remote  # noqa: E402

TOKEN = "tok-super-secret-123"
URL = "https://ws--dire-demo-api.modal.run"


def _png() -> bytes:
    from PIL import Image

    import numpy as np

    # 16×16 nhiễu: PNG thật, vài trăm byte — ảnh 4×4 đặc một màu chỉ ~80 byte và
    # sẽ trượt hàng rào `MIN_PNG_BYTES` (đúng như nó phải trượt).
    arr = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _ok_body(**over) -> bytes:
    body = {
        "steps": 5, "nfe": 10, "seconds": 0.91, "resized": False,
        "stats": {"mean": 0.11, "max": 0.9, "p99": 0.6},
        "line": "DIRE: ::dire --steps 5 (tính tại chỗ, ảnh gốc 256²) · detector: không có · nhãn: R1 · S=5",
        "map_png_b64": base64.b64encode(_png()).decode(),
        "recon_png_b64": base64.b64encode(_png()).decode(),
    }
    body.update(over)
    return json.dumps(body).encode()


def _transport(status: int, body: bytes, seen: dict | None = None):
    def t(url, payload, token, timeout):
        if seen is not None:
            seen.update(url=url, payload=payload, token=token, timeout=timeout)
        return status, body

    return t


def test_happy_path_decodes_everything() -> None:
    out = remote.call(URL, TOKEN, _png(), 5, transport=_transport(200, _ok_body()))
    assert out.steps == 5 and out.nfe == 10 and out.seconds == 0.91
    assert out.line.startswith("DIRE:") and "nhãn: R1" in out.line
    assert out.map_png.startswith(b"\x89PNG") and len(out.map_png) > 50


def test_request_carries_token_and_steps() -> None:
    seen: dict = {}
    remote.call(URL, TOKEN, b"abc", 7, transport=_transport(200, _ok_body(steps=7, nfe=14), seen))
    assert seen["url"] == URL and seen["token"] == TOKEN
    assert seen["payload"]["steps"] == 7
    assert base64.b64decode(seen["payload"]["image_b64"]) == b"abc"


def test_unconfigured_never_calls() -> None:
    called = []
    with pytest.raises(remote.RemoteError, match="DEMO_URL"):
        remote.call(URL, "", b"x", 5, transport=lambda *a: called.append(a) or (200, b""))
    assert called == []


@pytest.mark.parametrize("status,needle", [
    (401, "token"), (429, "Hạn mức"), (503, "đóng"), (413, "10 MB"), (422, "422"), (500, "500"),
])
def test_error_codes_become_readable_sentences(status: int, needle: str) -> None:
    with pytest.raises(remote.RemoteError) as ei:
        remote.call(URL, TOKEN, b"x", 5,
                    transport=_transport(status, json.dumps({"detail": "chi tiết"}).encode()))
    assert needle in str(ei.value)


def test_token_is_redacted_out_of_error_messages() -> None:
    """Nếu server nhắc lại token trong `detail`, nó không được vào log của Space."""
    body = json.dumps({"detail": f"bad token {TOKEN}"}).encode()
    with pytest.raises(remote.RemoteError) as ei:
        remote.call(URL, TOKEN, b"x", 5, transport=_transport(401, body))
    assert TOKEN not in str(ei.value)
    assert "***" in str(ei.value)


def test_result_without_provenance_line_is_rejected() -> None:
    with pytest.raises(remote.RemoteError, match="nguồn gốc"):
        remote.call(URL, TOKEN, b"x", 5,
                    transport=_transport(200, _ok_body(line="kết quả: 0.97")))


def test_result_with_empty_map_is_rejected() -> None:
    with pytest.raises(remote.RemoteError, match="PNG"):
        remote.call(URL, TOKEN, b"x", 5,
                    transport=_transport(200, _ok_body(map_png_b64=base64.b64encode(b"x").decode())))


def test_malformed_json_is_named_not_swallowed() -> None:
    with pytest.raises(remote.RemoteError, match="thiếu trường"):
        remote.call(URL, TOKEN, b"x", 5, transport=_transport(200, b'{"steps": 5}'))


def test_configured_requires_both_parts() -> None:
    assert remote.configured(URL, TOKEN)
    assert not remote.configured(URL, "")
    assert not remote.configured("", TOKEN)


def test_health_returns_none_instead_of_raising() -> None:
    assert remote.health("http://127.0.0.1:1", timeout=0.2) is None


# ---------------------------------------------------------------------------
# Trang 3 dùng endpoint khi có cấu hình — và không đụng tới torch
# ---------------------------------------------------------------------------
def test_live_page_uses_the_remote_endpoint_when_configured(monkeypatch) -> None:
    """Có `DEMO_URL`+`DEMO_TOKEN` ⇒ trang 3 đi qua endpoint, không nạp ADM tại chỗ.

    Đây là đường sẽ dùng trong buổi nói: 0,9 s thay vì 101 s. Test chặn hai kiểu
    hỏng im lặng: (a) trang vẫn gọi `live.status()` rồi tắt chế độ dù đã cấu hình
    endpoint; (b) kết quả hiện ra mà thiếu dòng nguồn gốc.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_URL", URL)
    monkeypatch.setenv("DEMO_TOKEN", TOKEN)

    def fake_health(url, timeout=10.0):
        return remote.Health(True, "adm (sha256 abc…)", (1, 20), 10 * 1024 * 1024)

    def fake_call(url, token, image, steps, *, timeout=90.0, transport=remote._default_transport):
        png = _png()
        return remote.Result(
            steps=steps, nfe=2 * steps, seconds=0.93, resized=False,
            stats={"mean": 0.111, "max": 0.8, "p99": 0.6},
            line=f"DIRE: ::dire --steps {steps} (tính tại chỗ) · detector: không có · "
                 f"nhãn: R1 (map tự tính) · S={steps} · nguồn: Modal A10G",
            map_png=png, recon_png=png,
        )

    import demolib.remote as remote_mod

    monkeypatch.setattr(remote_mod, "health", fake_health)
    monkeypatch.setattr(remote_mod, "call", fake_call)

    at = AppTest.from_file(str(SPACE / "app.py"), default_timeout=180)
    at.run()
    at.radio[0].set_value("3 · Tính DIRE thật").run()
    assert not at.exception, at.exception

    text = " ".join(
        [e.value for e in at.success] + [m.value for m in at.markdown]
        + [c.value for c in at.caption]
    )
    assert "Modal" in text and "A10G" in text
    assert not any("chờ" in w.value.lower() and "101" in w.value for w in at.warning)

    at.select_slider[0].set_value(5).run()
    at.button[0].click().run()
    assert not at.exception, at.exception

    line = [m.value for m in at.markdown if m.value.startswith("`DIRE:")]
    assert line and "S=5" in line[0] and "detector: không có" in line[0]
    assert len(at.image) >= 3
    labels = [m.label for m in at.metric]
    assert "DIRE trung bình" in labels and "giây (GPU)" in labels
