"""Dịch vụ DIRE chạy trong một container — phần **thuần**, không import `modal`.

Vì sao tách khỏi `demo_api.py`
------------------------------
`demo_api.py` chỉ có nhiệm vụ *nối* (image, GPU, volume, decorator). Còn toàn bộ
luật của endpoint — xác thực, giới hạn kích thước, giới hạn `S`, hạn mức gọi,
định dạng trả về — nằm ở đây. Nhờ vậy `tests/test_zero_demo_api.py` chạy được
`TestClient` trên CPU với một engine giả: không GPU, không Modal, không tốn tiền,
mà vẫn kiểm đúng những thứ có thể khiến một endpoint công khai bị lạm dụng.

Ba cổng, và **tất cả đều đóng** khi thiếu cấu hình
--------------------------------------------------
1. **Token.** Không đặt `DEMO_TOKEN` ⇒ endpoint trả `503`, không trả 401. Một
   endpoint không token mà mở công khai là chỗ đốt credit nhanh nhất có thể: mỗi
   ảnh `S=20` là 3,67 s A10G, và một vòng lặp `while true` là hoá đơn.
2. **Kích thước.** Ảnh > 10 MB bị từ chối trước khi giải mã base64.
3. **Hạn mức.** Token bucket theo client, mặc định 6 ảnh / 60 s, burst 2. Đủ cho
   một buổi nói, không đủ cho một bot.

Thêm một luật nữa: `S` chỉ nhận 1–20. Trần 20 là con số của bài báo (Table 4), và
mỗi bước là tiền thật — nhận `S=500` từ query string là cách tự phá hoá đơn.
"""

# Cố ý **không** có `from __future__ import annotations`: FastAPI đọc annotation của
# route bằng `get_type_hints`, và với annotation dạng chuỗi nó không thấy `Request`
# (được import trong `build_app`) ⇒ coi `request` là query param bắt buộc và trả 422
# cho mọi lời gọi. Đây là lỗi đã xảy ra thật, và nó chỉ hiện khi POST chứ không khi
# đọc mã.
import base64
import binascii
import hmac
import io
import math
import os
import time
from dataclasses import dataclass, field

__all__ = [
    "MAX_BYTES",
    "S_RANGE",
    "Budget",
    "DireEngine",
    "TorchEngine",
    "build_app",
    "provenance_line",
]

#: 10 MB — cùng giới hạn với `docs/plans/demos/demo-web-plan.md` §6.
MAX_BYTES = 10 * 1024 * 1024

#: `S` hợp lệ. 20 là cấu hình của bài (Table 4); trên 20 chỉ tốn tiền, không thêm
#: thông tin. 1 cho phép thử nhanh đường ống.
S_RANGE = (1, 20)

#: Ảnh vào phải đúng cỡ ADM học. Khác cỡ ⇒ resize **ở server** và ghi lại việc đó
#: trong response, thay vì im lặng đo ở một tỉ lệ khác.
NATIVE = 256


@dataclass
class Budget:
    """Token bucket theo client. Đủ cho người, chặn được bot."""

    per_window: int = 6
    window_s: float = 60.0
    burst: int = 2

    _calls: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, client: str, now: float | None = None) -> tuple[bool, float]:
        """Trả `(được gọi, số giây phải chờ)`. Không bao giờ ném."""
        t = time.monotonic() if now is None else now
        cap = self.per_window + self.burst
        calls = [x for x in self._calls.get(client, []) if t - x < self.window_s]
        if len(calls) >= cap:
            wait = self.window_s - (t - calls[0])
            self._calls[client] = calls
            return False, max(0.0, wait)
        calls.append(t)
        self._calls[client] = calls
        return True, 0.0


def provenance_line(steps: int, *, resized: bool, source: str) -> str:
    """Dòng bắt buộc dưới mỗi kết quả — cùng định dạng với demo và gallery."""
    note = "ảnh đã resize về 256²" if resized else "ảnh gốc 256²"
    return (
        f"DIRE: ::dire --steps {steps} (tính tại chỗ, {note}) · "
        f"detector: không có (endpoint này chỉ tính map) · "
        f"nhãn: R1 (map tự tính) · S={steps} · nguồn: {source}"
    )


class DireEngine:
    """Hợp đồng của engine: bytes ảnh → map + thống kê. Không biết gì về HTTP."""

    name = "engine"

    def map(self, raw: bytes, steps: int) -> dict:  # pragma: no cover - hợp đồng
        raise NotImplementedError


class TorchEngine(DireEngine):
    """Engine thật: ADM trên GPU (hoặc CPU), đúng hai vòng lặp của `dire_map`."""

    def __init__(self, counter, digest: str = "", device: str = "cuda") -> None:
        self.counter = counter
        self.digest = digest
        self.device = device
        self.name = f"adm (sha256 {digest[:12]}…)" if digest else "adm"

    def map(self, raw: bytes, steps: int) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        from .diffusion import Schedule, dire_map, from_uint8
        from .pipeline import to_hwc_uint8

        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            resized = im.size != (NATIVE, NATIVE)
            if resized:
                im = im.resize((NATIVE, NATIVE), Image.LANCZOS)
            arr = np.asarray(im)

        t0 = time.time()
        before = self.counter.nfe
        sched = Schedule(steps=steps, device=self.device)
        x0 = from_uint8(arr).to(self.device)
        with torch.inference_mode():
            dmap, recon = dire_map(self.counter, x0, sched)
        self.counter.nfe = before  # endpoint là dịch vụ: đừng tích luỹ NFE giữa request

        m = dmap.float().squeeze(0).mean(dim=0)
        stats = {
            "mean": round(float(m.mean()), 5),
            "max": round(float(m.max()), 5),
            "p99": round(float(m.flatten().kthvalue(max(1, int(0.99 * m.numel()))).values), 5),
        }
        # Map `[0, 2]` → ảnh xám, gamma 0,5 như `tools/make_gallery.py`.
        shade = ((m.clamp(0, 2) / 2.0) ** 0.5).mul(255).round().clamp(0, 255).to(torch.uint8)
        mp = Image.fromarray(shade.cpu().numpy(), mode="L")
        buf = io.BytesIO()
        mp.save(buf, format="PNG", optimize=True)

        rb = io.BytesIO()
        Image.fromarray(to_hwc_uint8(recon)).save(rb, format="PNG", optimize=True)

        return {
            "steps": steps,
            "nfe": 2 * steps,
            "seconds": round(time.time() - t0, 2),
            "resized": resized,
            "map_png": buf.getvalue(),
            "recon_png": rb.getvalue(),
            "stats": stats,
        }


def build_app(
    engine: DireEngine,
    *,
    token: str | None = None,
    origins: list[str] | None = None,
    budget: Budget | None = None,
    max_bytes: int = MAX_BYTES,
    source: str = "Modal A10G · container ấm",
):
    """Dựng FastAPI app. `token=None` ⇒ endpoint **đóng** (503), không phải mở."""
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    import json as _json

    app = FastAPI(title="DIRE demo endpoint", docs_url=None, redoc_url=None)
    bud = budget or Budget()

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Demo-Token"],
            max_age=600,
        )

    def _check(x_demo_token: str | None) -> None:
        if not token:
            raise HTTPException(
                status_code=503,
                detail="endpoint chưa cấu hình DEMO_TOKEN — đóng có chủ ý, "
                       "để một URL công khai không đốt credit GPU",
            )
        if not x_demo_token or not hmac.compare_digest(x_demo_token, token):
            raise HTTPException(status_code=401, detail="thiếu hoặc sai X-Demo-Token")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "engine": engine.name,
            "s_range": list(S_RANGE),
            "max_bytes": max_bytes,
            "auth": "bật" if token else "TẮT (endpoint đóng)",
            "rate": {"per_window": bud.per_window, "window_s": bud.window_s, "burst": bud.burst},
        }

    @app.post("/dire")
    async def dire(
        request: Request,
        x_demo_token: str | None = Header(default=None, alias="X-Demo-Token"),
    ):
        _check(x_demo_token)

        client = (x_demo_token or "anon")[:8] + "|" + (request.client.host if request.client else "?")
        ok, wait = bud.allow(client)
        if not ok:
            raise HTTPException(status_code=429, detail=f"quá hạn mức; thử lại sau {wait:.0f} s")

        raw_body = await request.body()
        if len(raw_body) > max_bytes:
            raise HTTPException(status_code=413, detail=f"body > {max_bytes} byte")
        try:
            payload = _json.loads(raw_body or b"{}")
        except _json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="body phải là JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body phải là object JSON")

        try:
            steps = int(payload.get("steps", 5))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="`steps` phải là số nguyên") from None
        lo, hi = S_RANGE
        if not lo <= steps <= hi:
            raise HTTPException(status_code=400, detail=f"`steps` phải trong [{lo}, {hi}]")

        b64 = payload.get("image_b64") or ""
        if not isinstance(b64, str) or not b64:
            raise HTTPException(status_code=400, detail="thiếu `image_b64` (ảnh PNG/JPEG, base64)")
        if len(b64) > max_bytes * 4 // 3 + 4:
            raise HTTPException(status_code=413, detail="ảnh base64 quá lớn")
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="`image_b64` không giải mã được") from None
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"ảnh > {max_bytes} byte")

        try:
            out = engine.map(raw, steps)
        except Exception as e:  # ảnh hỏng, mô hình lỗi — trả 422, không 500 mù
            raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}") from None

        if not math.isfinite(float(out["stats"]["mean"])):
            raise HTTPException(status_code=500, detail="map toàn NaN — dừng, đừng trả số")

        body = {
            "steps": out["steps"],
            "nfe": out["nfe"],
            "seconds": out["seconds"],
            "resized": out["resized"],
            "stats": out["stats"],
            "map_png_b64": base64.b64encode(out["map_png"]).decode("ascii"),
            "recon_png_b64": base64.b64encode(out["recon_png"]).decode("ascii"),
            "line": provenance_line(out["steps"], resized=bool(out["resized"]), source=source),
        }
        resp = JSONResponse(body)
        resp.headers["X-DIRE-Engine"] = engine.name
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


def token_from_env() -> str | None:
    """Token đọc từ env (Modal secret). Rỗng = không có = endpoint đóng."""
    return os.environ.get("DEMO_TOKEN") or None


def origins_from_env() -> list[str]:
    return [o.strip() for o in os.environ.get("DEMO_ORIGINS", "").split(",") if o.strip()]
