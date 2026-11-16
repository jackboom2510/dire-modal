# src/dire_modal/api.py
"""Wrapper Modal cho SaaS: bọc **cùng** lõi ``dire_core`` bằng HTTP endpoint.

Quan hệ với ``dire_local``
--------------------------
::

    dire_local.server.build_app   ← FastAPI tại máy (uvicorn)
    dire_modal.api                ← @modal.fastapi_endpoint (scale-to-zero)

Cả hai đều gọi ``dire_local.server.score_payload`` — nghĩa là **một** đường
decode ảnh, **một** cách trả lỗi, **một** schema response. Nếu wrapper tự viết
lại phần decode thì hai đường sẽ trôi khỏi nhau, và đó đúng là loại lỗi đã làm
``demo.py`` của tác giả lệch ``datasets.py``.

Bốn quyết định vận hành (đều có lý do, không phải mặc định)
-----------------------------------------------------------
1. **Weight nhúng vào image** (``add_local_file``) nếu có file local. Cold start
   đọc 269 MB từ Volume mất 2–5 s; nhúng vào layer thì ~1 s. Không có file thì
   fallback Volume ``/raw/checkpoints``.
2. ``max_containers=2`` — endpoint public thì mỗi container song song là một
   đường đốt credit. Chặn cứng ở 2, và rate limit theo key ở phía gọi.
3. ``scaledown_window=120`` — scale-to-zero sau 2 phút (tên cũ
   ``container_idle_timeout`` đã bị Modal bỏ). ``keep_warm``
   24/7 tốn ~$8.6/tháng (0.25 core) = 29 % credit $30, nên **không** bật mặc định.
4. **Auth bằng header** ``x-api-key`` so với secret ``dire-api-key``. Không có
   secret ⇒ endpoint mở (chỉ dùng cho demo tạm) và có cảnh báo trong response
   ``/healthz`` chứ không im lặng.

Chạy::

    modal serve -m dire_modal.api          # dev, URL tạm
    modal deploy -m dire_modal.api         # deploy thật

    curl -s -X POST "$URL/v1/detect" \\
      -H "x-api-key: $DIRE_API_KEY" -H 'content-type: application/json' \\
      -d "{\\"image_b64\\": \\"$(base64 -w0 anh.jpg)\\"}"
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from .common import app, vol_raw

#: Ảnh mặc định nhúng vào image nếu có — 269.4 MB, đọc từ layer nhanh hơn Volume.
DEFAULT_WEIGHTS = Path("checkpoints/lsun_adm.pth")
BAKED_WEIGHTS = "/opt/dire_weights/lsun_adm.pth"
VOLUME_WEIGHTS = "/raw/checkpoints/lsun_adm.pth"

_MAX_CONTAINERS = 2
_SCALEDOWN_S = 120

# `modal.fastapi_endpoint` có từ Modal 0.64; guard để test (stub modal) import được.
_fastapi_endpoint = getattr(modal, "fastapi_endpoint", None)

# `Request` chỉ để đọc header. Không có fastapi (môi trường test) thì fallback
# `object`; endpoint thật luôn chạy trong image có fastapi.
try:  # pragma: no cover - phụ thuộc môi trường
    from fastapi import Request as _Request  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    _Request = object  # type: ignore[assignment, misc]


def _endpoint(*args, **kwargs):
    def deco(fn):
        if _fastapi_endpoint is None:  # pragma: no cover - chỉ xảy ra với stub
            return fn
        return _fastapi_endpoint(*args, **kwargs)(fn)

    return deco


def _api_secret() -> list:
    """Secret ``dire-api-key`` nếu đã tạo; chưa tạo thì chạy không auth.

    Không dùng list comprehension trong decorator để lỗi "secret chưa tạo" biến
    thành thông báo rõ ràng thay vì ``NotFoundError`` giữa lúc import.
    """
    secret_cls = getattr(modal, "Secret", None)
    if secret_cls is None:  # stub
        return []
    try:
        return [secret_cls.from_name("dire-api-key")]
    except Exception:  # noqa: BLE001
        return []


def api_image(weights: Path | str | None = None):
    """Image chứa torch + fastapi + (tuỳ chọn) weight nhúng sẵn."""
    from .common import gpu_image

    img = gpu_image.pip_install("fastapi==0.115.0", "uvicorn==0.30.6")
    p = Path(weights) if weights else DEFAULT_WEIGHTS
    add = getattr(img, "add_local_file", None)
    if add is not None and p.exists():
        img = add(str(p), remote_path=BAKED_WEIGHTS)
    return img


def weights_path() -> str:
    """Đường dẫn weight **bên trong container**: bản nhúng trước, Volume sau."""
    if Path(BAKED_WEIGHTS).exists():
        return BAKED_WEIGHTS
    return VOLUME_WEIGHTS


def check_key(given: str | None, expected: str | None) -> tuple[bool, int]:
    """Kiểm API key → ``(ok, http_status)``.

    Tách thành hàm thuần để test được không cần HTTP/Modal. Không có
    ``expected`` (chưa tạo secret) ⇒ cho qua; ``/healthz`` vẫn báo
    ``auth: "disabled"`` để không ai tưởng endpoint đang được bảo vệ.
    """
    if not expected:
        return True, 200
    if given == expected:
        return True, 200
    return False, 401


def _key_from(request, fallback: str | None = None) -> str | None:
    """Lấy ``x-api-key`` từ header; chấp nhận cả query ``?x_api_key=`` (tiện curl)."""
    headers = getattr(request, "headers", None)
    if headers is not None:
        try:
            got = headers.get("x-api-key")
        except Exception:  # noqa: BLE001
            got = None
        if got:
            return got
    return fallback


_DET_CACHE: dict[str, object] = {}


def _cached_detector():
    """Nạp model **một lần cho mỗi container** (269 MB, không nạp mỗi request)."""
    if "det" not in _DET_CACHE:
        from dire_core import Detector

        _DET_CACHE["det"] = Detector.load(weights_path())
    return _DET_CACHE["det"]


@app.function(
    image=api_image(),
    volumes={"/raw": vol_raw},
    timeout=120,
    max_containers=_MAX_CONTAINERS,
    scaledown_window=_SCALEDOWN_S,
    secrets=_api_secret(),
)
@_endpoint(method="POST", label="dire-detect")
def detect(payload: dict, request: _Request = None, x_api_key: str | None = None) -> dict:
    """``POST /v1/detect`` — chấm 1 ảnh base64.

    Trả ``{p_fake, verdict, threshold, preproc, n_out, prob}`` hoặc
    ``{error, status, detail}``. Mã lỗi do ``dire_local.server`` quyết định để
    local và cloud không lệch nhau.
    """
    from dire_local.server import score_payload

    ok, status = check_key(
        _key_from(request, x_api_key), os.environ.get("DIRE_API_KEY")
    )
    if not ok:
        return {"error": "unauthorized", "status": status}

    try:
        return score_payload(_cached_detector(), payload)
    except ValueError as exc:
        return {"error": "bad_request", "status": 400, "detail": str(exc)}


@app.function(
    image=api_image(),
    volumes={"/raw": vol_raw},
    timeout=60,
    max_containers=_MAX_CONTAINERS,
)
@_endpoint(method="GET", label="dire-healthz")
def healthz() -> dict:
    """``GET /healthz`` — model nạp được không + auth có bật không."""
    auth = "enabled" if os.environ.get("DIRE_API_KEY") else "disabled"
    try:
        return {"ok": True, "auth": auth, "model": _cached_detector().info()}
    except Exception as exc:  # noqa: BLE001 - health check trả JSON, không 500 HTML
        return {"ok": False, "auth": auth, "error": str(exc)}
