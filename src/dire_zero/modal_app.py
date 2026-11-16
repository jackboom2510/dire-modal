"""Chạy trên Modal — image **không** chứa repo của tác giả.

Điểm quan trọng nhất của file này không phải là nó chạy được, mà là nó chạy
được **mà không có `git clone` nào**. Image cũ (`dire_modal.common.gpu_image`)
clone `ZhendongWang6/DIRE` vào `/opt/dire` để lấy `guided_diffusion`. Image ở
đây không làm vậy, và `selfcheck` assert rằng `guided_diffusion` **không** import
được. Nếu ai đó sau này thêm `pip install guided-diffusion` cho tiện, cổng đó
đỏ ngay.

Lệnh::

    modal run -m dire_zero.modal_app::selfcheck
    modal run -m dire_zero.modal_app::bench
    modal run -m dire_zero.modal_app::gen --n 500
    modal run -m dire_zero.modal_app::dire --n 500
    modal run -m dire_zero.modal_app::train --epochs 40
    modal run -m dire_zero.modal_app::report

Đường dẫn trên Volume (tách hẳn khỏi `/raw/scratch/**` của đường cũ, để không
bao giờ trộn hai bộ số)::

    /raw/zero/ckpt/adm_lsun.pt      2,5 GB, tải tay
    /raw/zero/real/*.png            ảnh thật LSUN (đã xuất PNG)
    /raw/zero/gen/*.png             R3: ảnh mình sinh
    /raw/zero/maps/real/*.npy       R1
    /raw/zero/maps/fake/*.npy       R1
    /ckpt/zero/detector.pth         R2
    /ckpt/zero/{gen,dire,train,eval,bench}.json
"""

from __future__ import annotations

import os
import sys
import time

import modal

APP_NAME = "dire-zero"
CKPT = "/raw/zero/ckpt/adm_lsun.pt"
REAL = "/raw/zero/real"
GEN = "/raw/zero/gen"
MAPS_REAL = "/raw/zero/maps/real"
MAPS_FAKE = "/raw/zero/maps/fake"
OUT = "/ckpt/zero"

#: Image riêng: chỉ torch + numpy + pillow. **Không** clone repo tác giả.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "pillow==10.4.0")
    .add_local_python_source("dire_zero")
)

app = modal.App(APP_NAME, image=image)

vol_raw = modal.Volume.from_name("dire-raw", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("dire-ckpt", create_if_missing=True)
VOLUMES = {"/raw": vol_raw, "/ckpt": vol_ckpt}


def _counter(ckpt: str = CKPT, device: str = "cuda", dtype: str = "auto"):
    """Nạp ADM và bọc bộ đếm NFE. Dùng chung cho mọi bước."""
    from .checkpoint import load_adm, sha256_file
    from .diffusion import ModelCounter

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"thiếu {ckpt}. Đẩy lên bằng:\n"
            "  modal volume put dire-raw <256x256_diffusion_uncond.pt> zero/ckpt/adm_lsun.pt"
        )
    dt = "float16" if (dtype == "auto" and device == "cuda") else ("float32" if dtype == "auto" else dtype)
    t0 = time.time()
    model = load_adm(ckpt, device=device, dtype=dt, strict=True)
    print(f"[nạp] {ckpt} · {dt} · {time.time() - t0:.1f}s · sha256 {sha256_file(ckpt)[:12]}…", flush=True)
    return ModelCounter(model), sha256_file(ckpt)


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def selfcheck() -> dict:
    """Cổng nạp trọng số: `strict=True` + một ảnh thật + kiểm NFE và NaN."""
    import torch  # noqa: PLC0415

    from .diffusion import Schedule, dire_map, sigma_floor

    # CLEAN-ROOM-CHECK
    import importlib.util  # noqa: PLC0415

    assert importlib.util.find_spec("guided_diffusion") is None, (
        "image có guided_diffusion — vi phạm điều kiện clean-room"
    )
    assert not os.path.isdir("/opt/dire"), "image có /opt/dire — vi phạm điều kiện clean-room"
    # END-CLEAN-ROOM-CHECK

    f, digest = _counter()
    sched = Schedule(steps=20, device="cuda")

    # (2) một ảnh thật; nếu chưa có thì nhiễu, để cổng vẫn chạy được sớm
    from .data import list_images

    pngs = list_images(REAL) if os.path.isdir(REAL) else []
    if pngs:
        from .pipeline import load_image

        img = load_image(pngs[0]).cuda()
        src = f"ảnh thật {pngs[0].name}"
    else:
        img = torch.rand(1, 3, 256, 256, device="cuda") * 2 - 1
        src = "nhiễu (chưa có ảnh thật trên Volume)"

    t0 = time.time()
    dm, recon = dire_map(f, img, sched)
    dt = time.time() - t0

    report = {
        "ckpt_sha256": digest,
        "source": src,
        "nfe": f.nfe,
        "nfe_expected": 2 * sched.steps,
        "finite": bool(torch.isfinite(dm).all() and torch.isfinite(recon).all()),
        "dire_min": float(dm.min()),
        "dire_max": float(dm.max()),
        "dire_mean": float(dm.mean()),
        "recon_abs_mean": float(recon.abs().mean()),
        "sigma_floor": sigma_floor(sched),
        "seconds_1_image": round(dt, 3),
        "sec_per_step": round(dt / (2 * sched.steps), 4),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "clean_room": "gói của tác giả không tồn tại trong image (xem khối CLEAN-ROOM-CHECK)",
    }
    assert report["nfe"] == report["nfe_expected"], report
    assert report["finite"], "map có NaN/Inf — thường là ᾱ[0] = 1,0 (chia cho 0)"
    assert 0.0 <= report["dire_min"] and report["dire_max"] <= 2.0 + 1e-5, report
    print(report, flush=True)
    return report


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def bench(n_warmup: int = 1, n_measure: int = 5, steps: int = 20) -> dict:
    """Đo `sec/step` — cổng G8 cũ, nay là điều kiện của mọi ước lượng tiền."""
    import torch  # noqa: PLC0415

    from .diffusion import Schedule
    from .budget import save_sec_per_step

    f, digest = _counter(dtype="float16")
    sched = Schedule(steps=steps, device="cuda")
    x = torch.randn(1, 3, 256, 256, device="cuda")

    for _ in range(n_warmup):
        f(x, sched.timesteps[0])
    torch.cuda.synchronize()

    t0 = time.time()
    for i in range(n_measure):
        f(x, sched.timesteps[i % len(sched.timesteps)])
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_measure

    payload = {
        "sec_per_step": round(dt, 4),
        "n_measure": n_measure,
        "gpu": torch.cuda.get_device_name(0),
        "batch": 1,
        "ckpt_sha256": digest,
        "note": "một ảnh 256×256, batch 1, fp16; đây là số dùng cho mọi ước lượng tiền",
    }
    save_sec_per_step(f"{OUT}/bench.json", payload["sec_per_step"], source="modal bench", gpu=payload["gpu"], batch=1)
    vol_ckpt.commit()
    print(payload, flush=True)
    return payload


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def gen(n: int = 500, steps: int = 20, seed_base: int = 1000, size: int = 256, max_usd: float = 5.0) -> dict:
    """R3 — sinh ảnh. Từ chối chạy nếu vượt trần tiền."""
    from .budget import Estimate, load_sec_per_step, require_estimate
    from .pipeline import RunMeta, generate

    sec = load_sec_per_step(f"{OUT}/bench.json")
    est = Estimate(n_images=n, nfe_per_image=steps, sec_per_step=sec)
    print("[tiền]", est.line(), flush=True)
    require_estimate(est, max_usd)

    f, digest = _counter(dtype="float16")
    out = generate(f, n=n, out_dir=GEN, meta=RunMeta(digest, steps), seed_base=seed_base, size=size)
    vol_raw.commit()
    return {"n": n, "out": GEN, "gen_json": {k: out[k] for k in ("n", "steps", "nfe_total", "seconds")}}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def dire(n: int = 500, steps: int = 20, max_usd: float = 5.0) -> dict:
    """R1 — DIRE map cho cả hai phía, cùng một lịch."""
    from .budget import Estimate, load_sec_per_step, require_estimate
    from .pipeline import RunMeta, compute_maps

    sec = load_sec_per_step(f"{OUT}/bench.json")
    est = Estimate(n_images=2 * n, nfe_per_image=2 * steps, sec_per_step=sec)
    print("[tiền]", est.line(), flush=True)
    require_estimate(est, max_usd)

    f, digest = _counter(dtype="float16")
    meta = RunMeta(digest, steps)

    def listing(root: str, label: int, limit: int) -> list[tuple[str, int]]:
        from .data import list_images

        if not os.path.isdir(root):
            raise FileNotFoundError(f"thiếu {root} — chạy bước trước (real: tải tay; fake: `gen`)")
        return [(str(p), label) for p in list_images(root)[:limit]]

    real = compute_maps(f, listing(REAL, 0, n), MAPS_REAL, meta, limit=n)
    fake = compute_maps(f, listing(GEN, 1, n), MAPS_FAKE, meta, limit=n)
    vol_raw.commit()
    return {"real": real["n"], "fake": fake["n"], "nfe_total": real["nfe_total"] + fake["nfe_total"], "est": est.as_dict()}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def train(epochs: int = 40, bs: int = 32, lr: float = 1e-4, crop: int = 224, seed: int = 3407) -> dict:
    """R2 — train detector trên map **của mình**."""
    from .detector import ResNetDetector, resnet50
    from .pipeline import collect_maps, train as train_fn

    maps = collect_maps([MAPS_REAL, MAPS_FAKE], [0, 1])
    if not maps:
        raise FileNotFoundError(f"không có .npy trong {MAPS_REAL} / {MAPS_FAKE} — chạy `dire` trước")
    payload = train_fn(
        lambda: ResNetDetector(resnet50()),
        maps,
        OUT,
        epochs=epochs,
        lr=lr,
        bs=bs,
        crop=crop,
        device="cuda",
        seed=seed,
        init="scratch",
    )
    vol_ckpt.commit()
    print({k: payload[k] for k in ("best_epoch", "best_val_ap", "threshold_from_val")}, flush=True)
    print("[test]", payload["test"], flush=True)
    return payload


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def report() -> dict:
    """Gom `gen/dire/train` thành một báo cáo có nhãn mức tái lập và cảnh báo."""
    import json  # noqa: PLC0415

    from .pipeline import write_json

    def read(name: str) -> dict:
        p = f"{OUT}/{name}.json"
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    payload = {
        "label": "R1+R2+R3 (clean-room, không trọng số ImageNet)",
        "what_is_ours": ["DIRE map", "detector", "ảnh sinh"],
        "what_is_not": ["trọng số ADM (OpenAI)", "LSUN-Bedroom (dữ liệu thật)", "kiến trúc U-Net/ResNet"],
        "gen": {k: v for k, v in read("gen").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "dire": {k: v for k, v in read("dire").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "train": {k: v for k, v in read("train").items() if k in ("test", "best_epoch", "threshold_from_val", "init")},
        "bench": read("bench"),
        "warnings": [
            "ACC trên ~200 ảnh test có CI ±5 điểm — không dùng để so với Table 2",
            "detector khởi tạo ngẫu nhiên, không ImageNet như bài báo",
            "một checkpoint ADM duy nhất ⇒ chỉ đo in-distribution (ADM), không đo cross-model",
        ],
    }
    write_json(f"{OUT}/report.json", payload)
    vol_ckpt.commit()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - tiện cho CLI
    argv = argv or sys.argv[1:]
    print(__doc__)
    return 0
