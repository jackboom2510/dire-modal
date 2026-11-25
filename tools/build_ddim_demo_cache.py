#!/usr/bin/env python3
"""Pre-render deterministic, synthetic DDIM teaching frames (not ADM results).

The demo uses a known noise realization as an oracle so the forward and reverse
updates are exactly reproducible without weights, network access, or GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "platforms/hf-spaces/cache/ddim-illustration"


def make_clean() -> np.ndarray:
    w = h = 256
    yy, xx = np.mgrid[0:h, 0:w]
    r = 0.18 + 0.52 * xx / (w - 1)
    g = 0.16 + 0.55 * yy / (h - 1)
    b = 0.32 + 0.25 * (1 - xx / (w - 1))
    rgb = np.stack([r, g, b], axis=-1)
    im = Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((42, 44, 214, 212), radius=23, fill=(28, 88, 110), outline=(255, 222, 145), width=5)
    d.ellipse((82, 80, 174, 172), fill=(231, 166, 67), outline=(255, 255, 255), width=4)
    d.polygon([(128, 96), (151, 143), (105, 143)], fill=(22, 89, 98))
    return np.asarray(im, dtype=np.float32) / 127.5 - 1.0


def save_state(arr: np.ndarray, path: Path) -> None:
    rgb = np.uint8(np.clip((arr + 1) * 127.5 + 0.5, 0, 255))
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (128, 128):
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
    image.save(path, format="PNG", optimize=True)


def build(out: Path, steps: tuple[int, ...] = (5, 10, 20), seed: int = 3407) -> None:
    # Import the exact 1000-step linear beta schedule used by the toy DDIM demo.
    import sys
    sys.path.insert(0, str(ROOT / "platforms/hf-spaces"))
    from demolib.toy2d import A, timesteps

    x0 = make_clean()
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(x0.shape, dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    save_state(x0, out / "clean.png")
    manifest = {
        "version": 1,
        "label": "MINH HỌA TOÁN HỌC · ẢNH TỔNG HỢP · NHIỄU ORACLE",
        "is_adm_inference": False,
        "is_measurement": False,
        "noise_seed": seed,
        "schedule": "linear beta, 1000 training steps; selected DDIM timesteps",
        "steps": {},
    }
    for s in steps:
        ts = timesteps(1000, s)
        folder = out / f"S{s:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, t in enumerate(ts):
            a = float(A[t])
            xt = math.sqrt(a) * x0 + math.sqrt(1.0 - a) * eps
            xt = np.clip(xt, -1, 1)
            xt_name = f"forward-{i:02d}-t{t:04d}-xt.png"
            x0h_name = f"forward-{i:02d}-t{t:04d}-x0hat.png"
            save_state(xt, folder / xt_name)
            # Oracle noise is known for this teaching-only example; x0-hat is exact.
            x0hat = np.clip((xt - math.sqrt(1.0 - a) * eps) / math.sqrt(a), -1, 1)
            save_state(x0hat, folder / x0h_name)
            frames.append({"phase": "Thêm nhiễu", "t": t, "x_t": f"S{s:02d}/{xt_name}", "x0_hat": f"S{s:02d}/{x0h_name}"})

        # Deterministic DDIM eta=0 reverse update, using the same known epsilon.
        for j, i in enumerate(range(len(ts) - 1, -1, -1)):
            t = ts[i]
            a = float(A[t])
            xt = np.clip(math.sqrt(a) * x0 + math.sqrt(1.0 - a) * eps, -1, 1)
            x0hat = np.clip((xt - math.sqrt(1.0 - a) * eps) / math.sqrt(a), -1, 1)
            t_prev = ts[i - 1] if i > 0 else -1
            a_prev = float(A[t_prev]) if t_prev >= 0 else 1.0
            x_prev = np.clip(math.sqrt(a_prev) * x0hat + math.sqrt(1.0 - a_prev) * eps, -1, 1)
            xt_name = f"reverse-{j:02d}-t{t:04d}-xt.png"
            x0h_name = f"reverse-{j:02d}-t{t:04d}-x0hat.png"
            save_state(xt, folder / xt_name)
            save_state(x0hat, folder / x0h_name)
            frames.append({"phase": "Khử nhiễu DDIM (η=0)", "t": t, "t_next": t_prev, "x_t": f"S{s:02d}/{xt_name}", "x0_hat": f"S{s:02d}/{x0h_name}"})
        manifest["steps"][str(s)] = frames
    (out / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--steps", default="5,10,20")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    steps = tuple(int(x) for x in args.steps.split(",") if x.strip())
    if not steps or any(s not in (5, 10, 20) for s in steps):
        raise SystemExit("only S=5, 10, 20 are permitted for this demonstration cache")
    build(args.out, steps, args.seed)
    print(f"Cached {sum(len(x) for x in json.loads((args.out/'index.json').read_text())['steps'].values())} frames under {args.out}")


if __name__ == "__main__":
    main()
