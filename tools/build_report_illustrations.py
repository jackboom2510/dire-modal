"""Build deterministic synthetic visual aids for the DIRE research deck.

These assets illustrate DDIM states, a toy absolute-residual operation, and
convolution/attention concepts. They are not ADM activations or measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "platforms/hf-spaces/cache/ddim-illustration"
OUT = ROOT / "platforms/site/illustrations"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _heat(values: np.ndarray) -> np.ndarray:
    """Blue→teal→yellow false-color ramp for explanatory maps only."""
    v = np.clip(values.astype(np.float32), 0, 1)
    stops = np.array([[27, 52, 102], [28, 148, 152], [245, 197, 83]], dtype=np.float32)
    lo = (v >= .5).astype(np.int32)
    u = np.where(v < .5, v * 2, (v - .5) * 2)[..., None]
    return (stops[lo] * (1-u) + stops[lo+1] * u).astype(np.uint8)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGB").save(path, optimize=True)


def _conv2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    src = np.pad(gray, pad, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(src, kernel.shape)
    return np.einsum("ijkl,kl->ij", win, kernel, optimize=True)


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    # Source is a small generated icon so every output is reproducible/offline.
    # Deterministic toy DDIM trajectory with an explicitly artificial epsilon
    # predictor. The final reconstruction tile is a separately blurred teaching
    # schematic so its absolute residual is spatially legible, not a model output.
    source = Image.open(CACHE / "clean.png").convert("RGB").resize((256,256), Image.Resampling.NEAREST)
    x0 = np.asarray(source, dtype=np.float32) / 127.5 - 1.0
    rng = np.random.default_rng(3407)
    eps = rng.normal(0, 1, x0.shape).astype(np.float32)
    # A smooth, edge-correlated toy prediction bias makes residual structure
    # visible without adding pixelwise salt-and-pepper noise.
    blur = sum(np.roll(np.roll(x0, dy, axis=0), dx, axis=1)
              for dy in (-1,0,1) for dx in (-1,0,1)) / 9.0
    yy, xx = np.mgrid[0:256, 0:256]
    low_frequency = (0.10 * np.sin(xx / 19.0) + 0.08 * np.cos(yy / 27.0))[..., None]
    # Deliberately visible structured bias (edges + broad spatial drift) so the
    # synthetic absolute-error panel reads as a residual map, not as IID noise.
    bias = (0.35 * (x0 - blur) + low_frequency).astype(np.float32)
    beta = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    abar = np.cumprod(1.0 - beta)
    ts = np.linspace(0, 999, 20, endpoint=False).round().astype(int)

    inv_states: dict[int, np.ndarray] = {0: x0.copy()}
    x = x0.copy()
    eps_hat = eps + bias
    for i, t in enumerate(ts):
        a = float(abar[t])
        x0_hat = (x - np.sqrt(1-a)*eps_hat) / np.sqrt(a)
        if i + 1 < len(ts):
            nxt = int(ts[i+1]); an = float(abar[nxt])
            x = np.sqrt(an)*x0_hat + np.sqrt(1-an)*eps_hat
            inv_states[nxt] = x.copy()
        else:
            x = np.sqrt(a)*x0_hat + np.sqrt(1-a)*eps_hat
            inv_states[999] = x.copy()
    xT = x.copy()

    rev_states: dict[int, np.ndarray] = {999: xT.copy()}
    x = xT.copy()
    for i in range(len(ts)-1, -1, -1):
        t = int(ts[i]); a=float(abar[t])
        x0_hat = (x - np.sqrt(1-a)*eps_hat) / np.sqrt(a)
        if i == 0:
            x = x0_hat
            rev_states[0] = x.copy()
            break
        prev = int(ts[i-1]); ap=float(abar[prev])
        x = np.sqrt(ap)*x0_hat + np.sqrt(1-ap)*eps_hat
        rev_states[prev] = x.copy()

    def to_rgb(values: np.ndarray) -> np.ndarray:
        return np.clip((values+1)*127.5+0.5,0,255).astype(np.uint8)
    # The final teaching reconstruction intentionally suppresses fine detail so
    # the absolute-error panel has a legible, spatially structured pattern. The
    # diffusion trajectory states above remain the deterministic toy loop output.
    xhat0_visual = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=1.8)), dtype=np.float32) / 127.5 - 1.0
    assets = {
        "ddim_x0.png": to_rgb(x0),
        "ddim_xt500.png": to_rgb(inv_states[min(inv_states,key=lambda t:abs(t-500))]),
        "ddim_xt950.png": to_rgb(xT),
        "ddim_reverse500.png": to_rgb(rev_states[min(rev_states,key=lambda t:abs(t-500))]),
        "ddim_xhat0.png": to_rgb(xhat0_visual),
    }
    for name, arr in assets.items():
        _save_rgb(arr, OUT / name)
    error = np.abs(x0-xhat0_visual).mean(axis=2)
    scale=max(float(np.quantile(error,.99)),1e-5)
    _save_rgb(_heat(np.sqrt(np.clip(error/scale,0,1))),OUT/"toy_dire_residual.png")
    write={
        "kind":"deterministic toy DDIM trajectory with schematic reconstruction and residual",
        "training_timesteps":1000,"ddim_steps":20,"noise_seed":3407,
        "predictor":"fixed known noise plus an artificial bias for trajectory states; the final reconstruction tile is intentionally blurred as a visual schematic",
        "is_adm_inference":False,"is_measurement":False,"is_detector_score":False,
        "assets":list(assets)+["toy_dire_residual.png"],
        "disclaimer":"Synthetic teaching aid only; not ADM inference, an archived sample, or an experimental result. The final reconstruction is intentionally blurred and the residual is its synthetic absolute difference from the input; they are schematic, not a faithful DDIM network output.",
    }
    (OUT/"manifest.json").write_text(json.dumps(write,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Filter-response montage: synthetic image → common illustrative 3x3 kernels.
    gray = np.asarray(Image.fromarray(to_rgb(x0)).convert("L"), dtype=np.float32) / 255
    kernels = [
        ("Ảnh vào", gray),
        ("Conv 3×3 · cạnh dọc", np.abs(_conv2d(gray, np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)))),
        ("Conv 3×3 · cạnh ngang", np.abs(_conv2d(gray, np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)))),
        ("Conv 3×3 · chi tiết", np.abs(_conv2d(gray, np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)))),
    ]
    tile, gap, header, footer = 228, 14, 42, 36
    canvas = Image.new("RGB", (4*tile + 5*gap, tile + header + footer + 2*gap), "#f4f7f6")
    draw = ImageDraw.Draw(canvas)
    for i, (label, feature) in enumerate(kernels):
        x, y = gap + i*(tile+gap), gap
        draw.rounded_rectangle((x,y,x+tile,y+tile+header+footer), 12, fill="white", outline="#d8e2e3", width=2)
        draw.text((x+9,y+9), label, font=_font(15, True), fill="#183746")
        if i == 0:
            thumb = Image.fromarray((gray*255).astype(np.uint8)).convert("RGB").resize((tile-20,tile-20), Image.Resampling.NEAREST)
        else:
            normalized = np.clip(feature / max(float(np.quantile(feature, .995)), 1e-5), 0, 1)
            thumb = Image.fromarray(_heat(normalized), "RGB").resize((tile-20,tile-20), Image.Resampling.NEAREST)
        canvas.paste(thumb, (x+10, y+header))
        draw.text((x+9,y+tile+header+3), "minh họa bộ lọc; không phải ADM", font=_font(11), fill="#526a76")
    canvas.save(OUT / "conv_feature_maps.png", optimize=True)

    # Simple spatial attention view: one query token assigns softmax weights to
    # nearby and visually similar locations on a synthetic 8×8 feature grid.
    small = Image.fromarray((gray*255).astype(np.uint8)).resize((8,8), Image.Resampling.BILINEAR)
    feat = np.asarray(small, dtype=np.float32) / 255
    yy, xx = np.mgrid[0:8,0:8]
    qy, qx = 3, 4
    scores = -((yy-qy)**2 + (xx-qx)**2)/5.0 - np.abs(feat-feat[qy,qx])*2.0
    weights = np.exp(scores - scores.max()); weights /= weights.sum()
    attention = _heat(weights / weights.max())
    scale = 34
    att_img = Image.fromarray(attention).resize((8*scale,8*scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(att_img)
    for k in range(9):
        draw.line((k*scale,0,k*scale,8*scale), fill="#ffffff", width=2)
        draw.line((0,k*scale,8*scale,k*scale), fill="#ffffff", width=2)
    draw.rectangle((qx*scale,qy*scale,(qx+1)*scale-1,(qy+1)*scale-1), outline="#f5c553", width=4)
    att_img.save(OUT / "attention_weights.png", optimize=True)

    # A compact layer schematic used alongside the heatmaps.
    w,h=1280,300
    layer=Image.new("RGB",(w,h),"#f4f7f6"); d=ImageDraw.Draw(layer)
    labels=[("x", "đặc trưng vào"),("Conv 1×1", "giảm kênh"),("Conv 3×3", "trộn không gian"),("Conv 1×1", "mở rộng kênh"),("+", "cộng nhánh tắt"),("ReLU", "đầu ra")]
    xs=[25,230,435,640,845,1050]
    for i,(head,sub) in enumerate(labels):
        x=xs[i]
        d.rounded_rectangle((x,92,x+165,210),radius=16,fill="#ffffff",outline="#c9d8dc",width=3)
        d.text((x+12,115),head,font=_font(22,True),fill="#183746")
        d.text((x+12,158),sub,font=_font(15),fill="#526a76")
        if i<len(labels)-1:
            d.line((x+166,151,xs[i+1]-8,151),fill="#147c80",width=4)
            d.polygon([(xs[i+1]-8,151),(xs[i+1]-20,143),(xs[i+1]-20,159)],fill="#147c80")
    d.arc((35,15,1010,235),180,360,fill="#d59a35",width=5)
    d.line((35,125,35,117),fill="#d59a35",width=5)
    d.line((1010,125,1010,117),fill="#d59a35",width=5)
    d.text((378,23),"nhánh tắt / đồng nhất",font=_font(18,True),fill="#a8751c")
    layer.save(OUT / "resnet_residual_path.png", optimize=True)

    return [OUT / name for name in (*assets.keys(), "toy_dire_residual.png", "conv_feature_maps.png", "attention_weights.png", "resnet_residual_path.png", "manifest.json")]


if __name__ == "__main__":
    out = build()
    print(f"Created {len(out)} deterministic explanatory assets in {OUT}")
    print(f"Total: {sum(p.stat().st_size for p in out):,} bytes")
