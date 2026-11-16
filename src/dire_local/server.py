"""HTTP API tại máy — FastAPI thuần, **không** biết Modal tồn tại.

``dire serve`` dùng file này; ``dire_modal/api.py`` bọc **cùng** logic bằng
``@modal.fastapi_endpoint``. Nhờ vậy hai đường không thể lệch nhau về:

* cách đọc base64 → ảnh,
* cách trả lỗi (ảnh hỏng ⇒ 400, không phải 500),
* schema response (``p_fake``, ``verdict``, ``preproc``, ``n_out``).

Tách khỏi CLI để test được bằng ``fastapi.testclient`` mà không cần torch:
truyền ``detector`` giả vào :func:`build_app`.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

#: Trần kích thước ảnh nhận vào (bytes). Ảnh DIRE là 256²; 12 MB là rất rộng rãi
#: và chặn được việc dùng endpoint làm nơi tải file.
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def decode_image(data_b64: str):
    """base64 → PIL ảnh RGB. Raise ``ValueError`` với thông báo đọc được."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"base64 không hợp lệ: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"ảnh quá lớn: {len(raw)} B > {MAX_IMAGE_BYTES} B")
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except ImportError as exc:  # pragma: no cover
        raise ValueError("server cần Pillow để đọc ảnh") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"không đọc được ảnh: {exc}") from exc


def score_payload(detector, payload: dict[str, Any], threshold: float = 0.5) -> dict:
    """Bóc payload → chấm → dict response.

    Tách khỏi route để **cả** FastAPI local **và** endpoint Modal dùng chung, và
    để test gọi trực tiếp không cần HTTP.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload phải là object JSON")
    data = payload.get("image_b64") or payload.get("image")
    if not data or not isinstance(data, str):
        raise ValueError("thiếu `image_b64` (base64 của file ảnh)")
    img = decode_image(data)
    p_fake = float(detector.score_pils([img])[0])
    info = detector.info() if hasattr(detector, "info") else {}
    return {
        "p_fake": p_fake,
        "verdict": "fake" if p_fake > threshold else "real",
        "threshold": threshold,
        "preproc": info.get("preproc"),
        "n_out": info.get("n_out"),
        "prob": info.get("prob"),
    }


def build_app(detector=None, *, weights: str | None = None,
              threshold: float = 0.5, api_key: str | None = None):
    """Dựng FastAPI app. ``detector=None`` ⇒ nạp lười ở request đầu tiên.

    Nạp lười là cố ý: ``uvicorn --reload`` và test không phải trả giá 269 MB
    checkpoint cho mỗi lần khởi động.
    """
    from fastapi import FastAPI, Header, HTTPException

    state: dict[str, Any] = {"det": detector}

    def _det():
        if state["det"] is None:
            from dire_core import Detector

            if not weights:
                raise HTTPException(500, "server chưa cấu hình --weights")
            state["det"] = Detector.load(weights)
        return state["det"]

    app = FastAPI(title="DIRE local", version="0.1.0")

    def _auth(key: str | None) -> None:
        if api_key and key != api_key:
            raise HTTPException(401, "x-api-key không đúng")

    @app.get("/healthz")
    def healthz():
        info = _det().info() if state["det"] is not None else {"loaded": False}
        return {"ok": True, "model": info}

    # Nhận `dict` trần, KHÔNG dùng pydantic model khai báo trong hàm: file này có
    # `from __future__ import annotations` (PEP 563), nên annotation là CHUỖI và
    # FastAPI không resolve được class cục bộ ⇒ nó coi `req` là query param và
    # trả 422 "field required". Đây đúng họ lỗi `compute_dire.py` đã gặp với
    # Modal. Validate trong `score_payload` để local và cloud cùng một đường.
    @app.post("/v1/detect")
    def detect(payload: dict, x_api_key: str | None = Header(default=None)):
        _auth(x_api_key)
        try:
            return score_payload(_det(), payload, threshold)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
