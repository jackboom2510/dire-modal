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

Đường dẫn trên Volume — **một volume, một gốc**, xem `dire_zero/volume.py`::

    dire-zero  →  /zero
    /zero/ckpt/adm_lsun.pt          2,5 GB, đẩy tay (hoặc bê từ volume cũ sang)
    /zero/ckpt/detector.pth         R2
    /zero/data/<split>/<gen>/*.png  kết quả ingest (real + ảnh tác giả/mình sinh)
    /zero/data/manifest.tsv
    /zero/data/data_card.json
    /zero/maps/<split>/<gen>/*.npy  DIRE map
    /zero/gen/<split>/<gen>/*.png   R3: ảnh mình sinh
    /zero/runs/*.json               gen/dire/train/eval/bench/report

Volume cũ (`dire-raw` + `dire-ckpt`) **không bị sửa**. Chuyển dữ liệu sang bố cục
này bằng `dire_zero.migrate` (chỉ đọc cũ, chỉ ghi mới, không xoá gì).
"""

from __future__ import annotations

import os
import sys
import time

import modal

from . import volume as V

APP_NAME = "dire-zero"
#: Đường dẫn lấy từ **một** nguồn duy nhất (`volume.py`) — trước đây cùng một
#: checkpoint được nhắc bằng sáu đường khác nhau trong repo, và không lỗi nào báo.
CKPT = V.CKPT
#: Cây dữ liệu sau `python -m dire_zero ingest`: <split>/<generator>/*.png
#: `real` là ảnh thật; mọi tên khác là ảnh sinh (adm, pndm, ldm, ours_adm, …).
DATA = V.DATA
MANIFEST = V.MANIFEST
DATA_CARD = V.DATA_CARD
GEN = V.GEN                   # chỉ dùng khi chưa có ảnh giả nào
MAPS = V.MAPS                 # <split>/<generator>/*.npy
OUT = V.RUNS

#: Image riêng: chỉ torch + numpy + pillow. **Không** clone repo tác giả.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "pillow==10.4.0")
    .add_local_python_source("dire_zero")
)

app = modal.App(APP_NAME, image=image)

#: Một volume duy nhất. Hai volume cũ nghĩa là hai lần commit, hai chỗ để quên
#: commit, và lớp lỗi "file có trên Volume nhưng container khác không thấy".
vol = modal.Volume.from_name(V.VOLUME_NAME, create_if_missing=True)
VOLUMES = {V.MOUNT: vol}


def _counter(ckpt: str = CKPT, device: str = "cuda", dtype: str = "auto"):
    """Nạp ADM và bọc bộ đếm NFE. Dùng chung cho mọi bước."""
    from .checkpoint import load_adm, sha256_file
    from .diffusion import ModelCounter

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"thiếu {ckpt}. Đẩy lên bằng:\n"
            "  modal volume put dire-zero <256x256_diffusion_uncond.pt> ckpt/adm_lsun.pt"
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

    real_dir = f"{DATA}/test/real"
    pngs = list_images(real_dir) if os.path.isdir(real_dir) else []
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


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def checkdata() -> dict:
    """Cổng dữ liệu: kiểm manifest + data_card **trước** khi tiêu GPU.

    Kiểm bốn thứ mà một lần chạy GPU 3 tiếng sẽ không báo:

    1. manifest có trên Volume và sha256 từng PNG khớp (file bị sửa/xoá);
    2. có đủ hai lớp, và tỉ lệ hai lớp không lệch quá 5:1 (ACC sẽ vô nghĩa);
    3. `data_card.label` phù hợp với nguồn ảnh giả (`author` ⇒ không được `R3`);
    4. rò rỉ pixel giữa hai lớp (nếu ingest đã chạy, việc này đã chặn; ở đây
       kiểm lại trên bản đã đẩy lên vì Volume có thể bị ghi đè từ ngoài).
    """
    import collections
    import json  # noqa: PLC0415

    from .dataset import read_manifest_tsv, verify_manifest

    card = json.load(open(DATA_CARD, encoding="utf-8")) if os.path.exists(DATA_CARD) else {}
    label = card.get("label", "?")
    ok, bad = verify_manifest(MANIFEST) if os.path.exists(MANIFEST) else (0, ["thiếu manifest.tsv"])

    rows = read_manifest_tsv(MANIFEST) if os.path.exists(MANIFEST) else []
    counts = collections.Counter((r["split"], r["generator"]) for r in rows)
    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)

    problems = []
    if not rows:
        problems.append("manifest rỗng — chạy `python -m dire_zero ingest` rồi đẩy lên")
    if bad:
        problems.append(f"{len(bad)} ảnh lệch sha256/xoá: {bad[:2]}")
    if n_real and n_fake and max(n_real, n_fake) / max(min(n_real, n_fake), 1) > 5:
        problems.append(f"hai lớp lệch {n_real}:{n_fake} — ACC sẽ bị lớp đông lấn át")
    if not n_real or not n_fake:
        problems.append(f"thiếu một lớp (thật {n_real}, giả {n_fake})")
    if card.get("fake_source") == "author" and "R3" in label:
        problems.append("data_card nói ảnh giả là của tác giả nhưng nhãn lại có R3")

    report = {
        "label": label,
        "fake_source": card.get("fake_source", "?"),
        "dataset": card.get("dataset", "?"),
        "license": card.get("license", "?"),
        "comparable_to_paper": card.get("comparable_to_paper"),
        "n_rows": len(rows),
        "n_real": n_real,
        "n_fake": n_fake,
        "by_split_generator": {f"{s}/{g}": n for (s, g), n in sorted(counts.items())},
        "manifest_sha_ok": ok,
        "problems": problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if problems:
        raise RuntimeError("dữ liệu chưa dùng được:\n  " + "\n  ".join(problems))
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
    vol.commit()
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
    out_root = f"{DATA}/gen"          # chưa biết split ở bước này; `ingest` không đụng vào
    out = generate(f, n=n, out_dir=out_root, meta=RunMeta(digest, steps), seed_base=seed_base, size=size)

    # đặt vào cây dữ liệu theo split đã ghi trong gen.json, dưới tên `ours_adm`
    import shutil  # noqa: PLC0415

    for split, names in out["split"].items():
        target = f"{DATA}/{split}/ours_adm"
        os.makedirs(target, exist_ok=True)
        for nm in names:
            shutil.move(f"{out_root}/{nm}", f"{target}/{nm}")
    vol.commit()
    return {"n": n, "out": f"{DATA}/<split>/ours_adm", "fake_source": "gen",
            "gen_json": {k: out[k] for k in ("n", "steps", "nfe_total", "seconds")}}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def dire(splits: str = "train,test", generator: str = "auto", n: int = 2000,
         steps: int = 20, max_usd: float = 5.0) -> dict:
    """R1 — DIRE map cho cả hai phía, cùng một lịch, **giữ split của người đóng gói**.

    ``generator="auto"``: nếu đã có ảnh giả trong cây dữ liệu (tác giả hoặc tự
    sinh) thì dùng; nếu chưa, đọc ảnh từ `gen/` (kết quả `::gen`).
    """
    from .budget import Estimate, load_sec_per_step, require_estimate
    from .data import list_images
    from .pipeline import RunMeta, compute_maps

    want = tuple(x.strip() for x in splits.split(",") if x.strip())
    gens = _generators_in(DATA)
    if generator != "auto":
        gens = [g for g in gens if g == generator]
    if not gens:
        raise FileNotFoundError(
            f"không có thư mục generator nào trong {DATA} (có: {_generators_in(DATA)}).\n"
            "Chạy `python -m dire_zero ingest` rồi đẩy lên, hoặc `::gen` để tự sinh ảnh giả."
        )

    n_pairs = sum(len(list_images(f"{DATA}/{s}/real")) for s in want if os.path.isdir(f"{DATA}/{s}/real"))
    n_fake = sum(len(list_images(f"{DATA}/{s}/{g}")) for s in want for g in gens if os.path.isdir(f"{DATA}/{s}/{g}"))
    sec = load_sec_per_step(f"{OUT}/bench.json")
    est = Estimate(n_images=n_pairs + n_fake, nfe_per_image=2 * steps, sec_per_step=sec)
    print("[tiền]", est.line(), flush=True)
    require_estimate(est, max_usd)

    f, digest = _counter(dtype="float16")
    meta = RunMeta(digest, steps)

    items: list[tuple[str, int, str]] = []
    for split in want:
        real_dir = f"{DATA}/{split}/real"
        if not os.path.isdir(real_dir):
            raise FileNotFoundError(f"thiếu {real_dir} — ingest chưa đẩy split này?")
        for p in list_images(real_dir)[:n]:
            items.append((str(p), 0, split))
        for g in gens:
            d = f"{DATA}/{split}/{g}"
            if os.path.isdir(d):
                for p in list_images(d)[:n]:
                    items.append((str(p), 1, split))

    out = compute_maps(f, items, MAPS, meta, limit=None)
    vol.commit()
    return {"n": out["n"], "generators": gens, "splits": list(want), "nfe_total": out["nfe_total"], "est": est.as_dict()}


def _generators_in(root: str) -> list[str]:
    """Các generator có mặt trong cây dữ liệu (bỏ `real`)."""
    found = set()
    if not os.path.isdir(root):
        return []
    for split in os.listdir(root):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            found.update(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and x != "real")
    return sorted(found)


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def train(epochs: int = 40, bs: int = 32, lr: float = 1e-4, crop: int = 224, seed: int = 3407) -> dict:
    """R2 — train detector trên map **của mình**."""
    from .pipeline import collect_maps, train as train_fn

    from .detector import ResNetDetector, resnet50

    maps = collect_maps(MAPS)
    if not maps:
        raise FileNotFoundError(f"không có .npy trong {MAPS} — chạy `dire` trước")
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
    vol.commit()
    print({k: payload[k] for k in ("best_epoch", "best_val_ap", "threshold_from_val")}, flush=True)
    print("[test]", payload["test"], flush=True)
    return payload


def _read_json(path: str) -> dict:
    import json  # noqa: PLC0415

    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def report() -> dict:
    """Gom `gen/dire/train` thành một báo cáo có nhãn mức tái lập và cảnh báo."""
    import json  # noqa: PLC0415

    from .pipeline import write_json

    def read(name: str) -> dict:
        p = f"{MAPS}/{name}.json" if name == "dire" else f"{OUT}/{name}.json"
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    payload = {
        "label": _read_json(DATA_CARD).get("label", "R1+R2+R3 (clean-room)"),
        "what_is_ours": ["DIRE map", "detector", "ảnh sinh"],
        "what_is_not": ["trọng số ADM (OpenAI)", "LSUN-Bedroom (dữ liệu thật)", "kiến trúc U-Net/ResNet"],
        "gen": {k: v for k, v in read("gen").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "dire": {k: v for k, v in read("dire").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "data": _read_json(DATA_CARD),
        "train": {k: v for k, v in read("train").items() if k in ("test", "best_epoch", "threshold_from_val", "init")},
        "bench": read("bench"),
        "warnings": [
            "ACC trên ~200 ảnh test có CI ±5 điểm — không dùng để so với Table 2",
            "detector khởi tạo ngẫu nhiên, không ImageNet như bài báo",
            "một checkpoint ADM duy nhất ⇒ chỉ đo in-distribution (ADM), không đo cross-model",
        ],
    }
    write_json(f"{OUT}/report.json", payload)
    vol.commit()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - tiện cho CLI
    argv = argv or sys.argv[1:]
    print(__doc__)
    return 0
