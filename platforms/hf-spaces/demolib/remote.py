"""Gọi endpoint DIRE trên Modal — **từ server của Space**, không từ trình duyệt.

Vì sao phải đi qua server
-------------------------
Token của endpoint là thứ duy nhất ngăn một URL công khai bị dùng để đốt credit
GPU. Nếu trang gọi Modal trực tiếp bằng JavaScript thì token nằm trong mã nguồn
của trang và **ai cũng đọc được** — nó không còn là bí mật, chỉ còn là một chuỗi.

Streamlit chạy trên server, nên:

* trình duyệt gọi **Space** (không token, cùng origin ⇒ không cần CORS);
* server của Space gọi **Modal** kèm token trong header;
* token nằm ở `st.secrets`, không bao giờ tới trình duyệt.

Đổi lại, mỗi lần bấm là hai chặng mạng thay vì một — vài chục ms, không đáng kể
so với 0,9 s tính DIRE.

Không thêm phụ thuộc
--------------------
Dùng `urllib.request` của thư viện chuẩn: Space chỉ có một dòng trong
`requirements.txt` và không có lý do gì để thêm `requests`/`httpx` cho một hàm.
Transport được **tiêm vào** để test không cần mạng.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ["RemoteError", "Result", "Health", "call", "health", "configured", "MIN_PNG_BYTES"]

#: PNG nhỏ hơn ngưỡng này chắc chắn không phải map 256² (map thật cỡ vài KB).
MIN_PNG_BYTES = 32


class RemoteError(RuntimeError):
    """Lỗi đã được dịch thành câu người đọc được — không lộ token."""


@dataclass(frozen=True)
class Result:
    steps: int
    nfe: int
    seconds: float
    resized: bool
    stats: dict
    line: str
    map_png: bytes
    recon_png: bytes


@dataclass(frozen=True)
class Health:
    ok: bool
    engine: str
    s_range: tuple[int, int]
    max_bytes: int


def configured(url: str | None, token: str | None) -> bool:
    """Cả hai đều phải có. Thiếu một ⇒ không gọi, và nói rõ thiếu cái nào."""
    return bool(url and token)


def _default_transport(url: str, payload: dict, token: str, timeout: float) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "X-Demo-Token": token,
                 "User-Agent": "dire-demo/1.0 (hf-space)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL do người vận hành đặt
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read() or b""


def _redact(text: str, token: str | None) -> str:
    """Không để token lọt vào log/exception nếu server nhắc lại header."""
    if token and token in text:
        return text.replace(token, "***")
    return text


def _friendly(status: int, raw: bytes, token: str | None) -> str:
    detail = ""
    try:
        payload = json.loads(raw or b"{}")
        detail = str(payload.get("detail", ""))[:300]
    except Exception:  # pragma: no cover - body không phải JSON
        pass
    detail = _redact(detail, token)
    tail = f" — {detail}" if detail else ""
    if status == 401:
        return ("Endpoint từ chối token (401). Kiểm `DEMO_TOKEN` ở Space secrets và ở Modal."
                + tail)
    if status == 429:
        return f"Hạn mức gọi (429){tail}"
    if status == 503:
        return f"Endpoint đang đóng (503){tail}"
    if status == 413:
        return f"Ảnh hoặc body quá lớn (413). Giới hạn 10 MB{tail}"
    if status == 422:
        return f"Endpoint không xử lý được ảnh này (422){tail}"
    return f"Endpoint trả {status}{tail}"




def call(
    url: str,
    token: str,
    image: bytes,
    steps: int,
    *,
    timeout: float = 90.0,
    transport=_default_transport,
) -> Result:
    """Một ảnh → `Result`. Ném `RemoteError` với câu đọc được khi hỏng."""
    if not configured(url, token):
        raise RemoteError("chưa cấu hình DEMO_URL/DEMO_TOKEN — không gọi endpoint nào")
    payload = {"image_b64": base64.b64encode(image).decode("ascii"), "steps": int(steps)}
    status, raw = transport(url, payload, token, timeout)
    if status != 200:
        raise RemoteError(_friendly(status, raw, token))
    try:
        body = json.loads(raw)
        out = Result(
            steps=int(body["steps"]),
            nfe=int(body["nfe"]),
            seconds=float(body["seconds"]),
            resized=bool(body.get("resized", False)),
            stats=dict(body["stats"]),
            line=str(body["line"]),
            map_png=base64.b64decode(body["map_png_b64"]),
            recon_png=base64.b64decode(body["recon_png_b64"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise RemoteError(f"endpoint trả JSON thiếu trường ({type(e).__name__}: {e})") from None

    # Hàng rào cuối: một endpoint lạ không được phép trả về "kết quả" mà thiếu
    # dòng nguồn gốc hoặc trả map rỗng.
    if not out.line.startswith("DIRE:"):
        raise RemoteError("endpoint trả kết quả không có dòng nguồn gốc `DIRE: …`")
    if len(out.map_png) < MIN_PNG_BYTES or not out.map_png.startswith(b"\x89PNG"):
        raise RemoteError("endpoint trả map không phải PNG")
    return out


def health(url: str, timeout: float = 10.0) -> Health | None:
    """/health không cần token; trả `None` nếu không gọi được (không ném)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:  # noqa: S310
            body = json.loads(r.read())
        lo, hi = body["s_range"]
        return Health(True, str(body["engine"]), (int(lo), int(hi)), int(body["max_bytes"]))
    except Exception:
        return None
