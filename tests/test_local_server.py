"""HTTP API tại máy — kiểm bằng FastAPI `TestClient`, KHÔNG cần torch/model thật.

Vì sao có file này
------------------
`dire serve` và `dire_modal.api` dùng chung `dire_local.server.score_payload`.
Nếu chỉ test hàm thuần thì phần còn lại (mã lỗi HTTP, auth, schema response) vẫn
có thể sai — và `dire_modal.api` sẽ thừa hưởng nguyên lỗi đó.

Test dùng ``detector`` giả nên chạy trong ~0.1 s và không cần 269 MB checkpoint.
Khi máy không có ``fastapi``/``PIL`` thì **skip**, không đỏ — repo vẫn phải xanh
trên môi trường chỉ có ``modal + pytest``.

Chạy::

    python3 -m pytest tests/test_local_server.py -q
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402


class FakeDetector:
    """Điểm cố định để test không phụ thuộc model/ảnh thật."""

    def __init__(self, p: float = 0.9) -> None:
        self.p = p
        self.calls = 0

    def score_pils(self, pils, batch_size: int = 64):
        self.calls += 1
        return [self.p for _ in pils]

    def info(self):
        return {"preproc": "resize256_centercrop224_normalize", "n_out": 1,
                "prob": "sigmoid", "device": "cpu"}


def _png_b64(size=(8, 8), color=(120, 30, 200)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def client():
    from dire_local.server import build_app

    return TestClient(build_app(detector=FakeDetector(0.87)))


def test_healthz_reports_model_info(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"]["preproc"] == "resize256_centercrop224_normalize"


def test_detect_returns_score_and_verdict(client):
    r = client.post("/v1/detect", json={"image_b64": _png_b64()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["p_fake"] == pytest.approx(0.87)
    assert body["verdict"] == "fake"
    assert body["threshold"] == 0.5
    assert body["n_out"] == 1 and body["prob"] == "sigmoid"


def test_detect_real_below_threshold():
    from dire_local.server import build_app

    c = TestClient(build_app(detector=FakeDetector(0.1)))
    assert c.post("/v1/detect", json={"image_b64": _png_b64()}).json()["verdict"] == "real"


def test_bad_base64_is_400_not_500(client):
    r = client.post("/v1/detect", json={"image_b64": "!!!khong-phai-base64!!!"})
    assert r.status_code == 400
    assert "base64" in r.json()["detail"]


def test_missing_field_is_400(client):
    r = client.post("/v1/detect", json={})
    assert r.status_code == 400


def test_oversized_payload_is_rejected_before_decoding():
    """12 MB là trần; kiểm bằng chuỗi base64 lớn nhưng không phải ảnh."""
    from dire_local.server import MAX_IMAGE_BYTES, decode_image

    big = base64.b64encode(b"\0" * (MAX_IMAGE_BYTES + 1)).decode()
    with pytest.raises(ValueError, match="quá lớn"):
        decode_image(big)


def test_api_key_fails_closed_when_configured():
    from dire_local.server import build_app

    c = TestClient(build_app(detector=FakeDetector(), api_key="secret"))
    assert c.post("/v1/detect", json={"image_b64": _png_b64()}).status_code == 401
    ok = c.post("/v1/detect", json={"image_b64": _png_b64()},
                headers={"x-api-key": "secret"})
    assert ok.status_code == 200


def test_no_key_configured_means_open():
    """Không cấu hình key ⇒ mở. Đây là lựa chọn có ý thức, không phải lỗi."""
    from dire_local.server import build_app

    c = TestClient(build_app(detector=FakeDetector(), api_key=None))
    assert c.post("/v1/detect", json={"image_b64": _png_b64()}).status_code == 200


# ---------------------------------------------------------------------------
# CLI `dire detect` — cùng `dire_core.Detector`, thay model bằng detector giả
# ---------------------------------------------------------------------------
def test_cli_detect_uses_the_shared_detector(tmp_path, monkeypatch, capsys):
    from PIL import Image

    import dire_core
    from dire_local import cli

    img_path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img_path)

    class Fake:
        def score_pils(self, pils, batch_size=64):
            return [0.75 for _ in pils]

        def info(self):
            return {"preproc": "x", "n_out": 1, "prob": "sigmoid", "device": "cpu"}

    weights = tmp_path / "w.pth"
    weights.write_bytes(b"stub")  # CLI kiểm tồn tại trước khi gọi Detector.load
    monkeypatch.setattr(dire_core.Detector, "load", classmethod(lambda cls, w: Fake()))
    code = cli.main(["detect", str(img_path), "--weights", str(weights), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert '"p_fake": 0.75' in out
    assert '"verdict": "fake"' in out


def test_cli_batch_writes_csv(tmp_path, monkeypatch, capsys):
    from PIL import Image

    import dire_core
    from dire_local import cli

    for i in range(3):
        Image.new("RGB", (8, 8), (i, i, i)).save(tmp_path / f"{i}.png")

    class Fake:
        def score_pils(self, pils, batch_size=64):
            return [0.2, 0.8][: len(pils)] if len(pils) <= 2 else [0.2, 0.8, 0.9]

        def info(self):
            return {}

    weights = tmp_path / "w.pth"
    weights.write_bytes(b"stub")
    monkeypatch.setattr(dire_core.Detector, "load", classmethod(lambda cls, w: Fake()))
    out_csv = tmp_path / "scores.csv"
    code = cli.main(["batch", str(tmp_path), "--weights", str(weights),
                     "--out", str(out_csv)])
    assert code == 0
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "path,p_fake,verdict,error"
    assert len(lines) == 4
