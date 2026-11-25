"""Cổng của endpoint demo — kiểm trên CPU, với engine giả, không tốn một xu GPU.

Một endpoint GPU công khai có ba cách chết, và cả ba đều **không** nằm ở phần tính
DIRE: mở không token, nhận body không giới hạn, và không có hạn mức. Nên test ở
đây tập trung vào đúng ba chỗ đó, cộng thêm phần định dạng trả về mà giao diện
dựa vào (`line`, `stats`, `resized`).

`TorchEngine` không được test ở đây: nó chỉ là lớp mỏng trên `dire_map`, mà
`dire_map` đã có test riêng (kể cả parity với bản tham chiếu). Test đường ảnh →
map trên GPU thật là việc của lần chạy `modal serve` đầu tiên, có ghi lại trong
`docs/plans/demos/demo-modal-plan.md` §6.
"""

from __future__ import annotations

import base64
import io
import json
import math

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from dire_zero.demo_service import (  # noqa: E402
    MAX_BYTES,
    S_RANGE,
    Budget,
    DireEngine,
    build_app,
    provenance_line,
)

TOKEN = "tok-demo-123"


def _png(n: int = 8) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (n, n), (30, 60, 90)).save(buf, format="PNG")
    return buf.getvalue()


class FakeEngine(DireEngine):
    """Engine giả: trả map tất định, ghi lại số lần gọi và `steps` nhận được."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def map(self, raw: bytes, steps: int) -> dict:
        self.calls.append((len(raw), steps))
        return {
            "steps": steps,
            "nfe": 2 * steps,
            "seconds": 0.01,
            "resized": True,
            "map_png": _png(4),
            "recon_png": _png(4),
            "stats": {"mean": 0.12345, "max": 0.9, "p99": 0.7},
        }


class BadEngine(DireEngine):
    name = "bad"

    def map(self, raw: bytes, steps: int) -> dict:
        return {"steps": steps, "nfe": 2 * steps, "seconds": 0.0, "resized": False,
                "map_png": b"", "recon_png": b"",
                "stats": {"mean": math.nan, "max": 0.0, "p99": 0.0}}


def _client(engine=None, token: str | None = TOKEN, **kw) -> TestClient:
    return TestClient(build_app(engine or FakeEngine(), token=token, **kw))


def _body(steps=5, raw=None) -> dict:
    return {"steps": steps,
            "image_b64": base64.b64encode(raw if raw is not None else _png()).decode()}


# ---------------------------------------------------------------------------
# Cổng 1 — token
# ---------------------------------------------------------------------------
def test_without_configured_token_the_endpoint_is_closed() -> None:
    """Không có `DEMO_TOKEN` phải là **503**, không phải "mở cho tiện"."""
    c = _client(token=None)
    r = c.post("/dire", json=_body())
    assert r.status_code == 503
    assert "DEMO_TOKEN" in r.json()["detail"]
    assert c.get("/health").json()["auth"].startswith("TẮT")


def test_missing_token_is_401() -> None:
    r = _client().post("/dire", json=_body())
    assert r.status_code == 401


def test_wrong_token_is_401() -> None:
    r = _client().post("/dire", json=_body(), headers={"X-Demo-Token": "sai"})
    assert r.status_code == 401


def test_correct_token_passes() -> None:
    c = _client()
    r = c.post("/dire", json=_body(), headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Cổng 2 — kích thước
# ---------------------------------------------------------------------------
def test_oversized_body_is_413_before_decoding() -> None:
    c = _client()
    big = json.dumps({"steps": 5, "image_b64": "A" * (MAX_BYTES + 10)})
    r = c.post("/dire", content=big, headers={"X-Demo-Token": TOKEN,
                                              "Content-Type": "application/json"})
    assert r.status_code == 413


def test_oversized_decoded_image_is_413() -> None:
    c = _client()
    raw = b"\x00" * (MAX_BYTES + 1)
    r = c.post("/dire", json=_body(raw=raw), headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Cổng 3 — hạn mức
# ---------------------------------------------------------------------------
def test_rate_limit_returns_429_with_a_wait() -> None:
    c = _client(budget=Budget(per_window=2, window_s=60, burst=0))
    h = {"X-Demo-Token": TOKEN}
    codes = [c.post("/dire", json=_body(), headers=h).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    assert "thử lại" in c.post("/dire", json=_body(), headers=h).json()["detail"]


def test_budget_refills_after_the_window() -> None:
    b = Budget(per_window=1, window_s=10, burst=0)
    assert b.allow("x", now=0.0)[0] is True
    assert b.allow("x", now=1.0)[0] is False
    ok, wait = b.allow("x", now=11.0)
    assert ok and wait == 0.0


def test_budget_is_per_client() -> None:
    b = Budget(per_window=1, window_s=60, burst=0)
    assert b.allow("a", now=0)[0] and b.allow("b", now=0)[0]


# ---------------------------------------------------------------------------
# Tham số và hình dạng trả về
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, 21, 500])
def test_steps_outside_range_is_400(bad: int) -> None:
    r = _client().post("/dire", json=_body(steps=bad), headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 400
    assert str(S_RANGE[0]) in r.json()["detail"] and str(S_RANGE[1]) in r.json()["detail"]


def test_steps_that_is_not_a_number_is_400() -> None:
    r = _client().post("/dire", json={"steps": "năm", "image_b64": "AA=="},
                       headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 400


def test_missing_image_is_400() -> None:
    r = _client().post("/dire", json={"steps": 5}, headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 400


def test_broken_base64_is_400() -> None:
    r = _client().post("/dire", json={"steps": 5, "image_b64": "!!!not-base64!!!"},
                       headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 400


def test_non_json_body_is_400() -> None:
    r = _client().post("/dire", content=b"khong phai json",
                       headers={"X-Demo-Token": TOKEN, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_response_has_provenance_line_and_pngs() -> None:
    c = _client()
    r = c.post("/dire", json=_body(steps=5), headers={"X-Demo-Token": TOKEN})
    body = r.json()
    assert body["nfe"] == 10 and body["steps"] == 5
    assert body["line"].startswith("DIRE: ::dire --steps 5")
    assert "detector: không có" in body["line"]
    assert "nhãn: R1" in body["line"]
    assert "S=5" in body["line"]
    assert base64.b64decode(body["map_png_b64"])[:4] == b"\x89PNG"
    assert base64.b64decode(body["recon_png_b64"])[:4] == b"\x89PNG"
    assert body["stats"]["mean"] == pytest.approx(0.12345)
    assert r.headers["X-DIRE-Engine"] == "fake"
    assert r.headers["Cache-Control"] == "no-store"


def test_engine_gets_the_requested_steps_and_raw_bytes() -> None:
    eng = FakeEngine()
    _client(eng).post("/dire", json=_body(steps=7), headers={"X-Demo-Token": TOKEN})
    assert eng.calls == [(len(_png()), 7)]


def test_nan_map_is_a_500_not_a_number() -> None:
    """Map toàn NaN là lỗi số học, không phải "kết quả thấp"."""
    r = _client(BadEngine()).post("/dire", json=_body(), headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 500
    assert "NaN" in r.json()["detail"]


def test_engine_exception_is_422_not_500() -> None:
    class Boom(DireEngine):
        name = "boom"

        def map(self, raw, steps):
            raise ValueError("ảnh hỏng")

    r = _client(Boom()).post("/dire", json=_body(), headers={"X-Demo-Token": TOKEN})
    assert r.status_code == 422
    assert "ValueError" in r.json()["detail"]


def test_health_lists_the_limits_so_clients_do_not_guess() -> None:
    h = _client().get("/health").json()
    assert h["ok"] is True and h["engine"] == "fake"
    assert h["s_range"] == list(S_RANGE)
    assert h["max_bytes"] == MAX_BYTES
    assert h["auth"] == "bật"


def test_provenance_line_flags_resizing() -> None:
    assert "ảnh gốc 256²" in provenance_line(5, resized=False, source="x")
    assert "resize" in provenance_line(5, resized=True, source="x")


def test_cors_only_when_origins_are_given() -> None:
    """Không có origin ⇒ không có header CORS. Đây là mặc định, và nó đúng."""
    c = _client()
    r = c.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    c2 = _client(origins=["https://dire-demo.hf.space"])
    r2 = c2.get("/health", headers={"Origin": "https://dire-demo.hf.space"})
    assert r2.headers.get("access-control-allow-origin") == "https://dire-demo.hf.space"
