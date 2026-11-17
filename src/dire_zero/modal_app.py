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
from pathlib import Path
from typing import Sequence

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
        "dtype": str(next(f.model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "clean_room": "gói của tác giả không tồn tại trong image (xem khối CLEAN-ROOM-CHECK)",
    }
    assert report["nfe"] == report["nfe_expected"], report
    assert report["finite"], "map có NaN/Inf — thường là ᾱ[0] = 1,0 (chia cho 0)"
    assert 0.0 <= report["dire_min"] and report["dire_max"] <= 2.0 + 1e-5, report
    print(report, flush=True)
    return report


#: Trần mất cân bằng **trong một bài toán nhị phân** (real vs một generator).
#: Không áp cho tổng gộp: 1000 thật + 14×1000 giả là cấu trúc chuẩn của
#: DiffusionForensics cho Table 2, không phải dữ liệu hỏng.
MAX_IMBALANCE = 5.0


def _check_data(
    skip_sha: bool = False,
    workers: int = 64,
    max_ratio: float = MAX_IMBALANCE,
    splits: str = "",
    sample: int = 0,
    count_png: bool = False,
    progress_every: int = 20000,
) -> dict:
    """Thân của `checkdata`, tách riêng để test được mà không cần Modal.

    `modal.Function` không phải callable và không lộ hàm gốc, nên một entrypoint
    gộp cả logic thì chỉ kiểm được bằng cách chạy thật trên Volume — đúng thứ
    đang hỏng. Đây là toàn bộ lý do hàm này tồn tại.

    Cổng dữ liệu: kiểm manifest + data_card **trước** khi tiêu GPU.

    Kiểm bốn thứ mà một lần chạy GPU 3 tiếng sẽ không báo:

    1. manifest có trên Volume và sha256 từng PNG khớp (file bị sửa/xoá);
    2. có đủ hai lớp, và tỉ lệ hai lớp không lệch quá 5:1 (ACC sẽ vô nghĩa);
    3. `data_card.label` phù hợp với nguồn ảnh giả (`author` ⇒ không được `R3`);
    4. rò rỉ pixel giữa hai lớp (nếu ingest đã chạy, việc này đã chặn; ở đây
       kiểm lại trên bản đã đẩy lên vì Volume có thể bị ghi đè từ ngoài).

    `skip_sha=True` bỏ bước băm (nhanh, dùng khi chỉ muốn kiểm cấu trúc). Mặc
    định **không** bỏ: đẩy 200 nghìn file là chỗ hỏng thật đã xảy ra, và chỉ có
    băm mới phát hiện được file đẩy dở.
    """
    import collections
    import json  # noqa: PLC0415

    from .dataset import read_manifest_tsv, verify_rows

    card = json.load(open(DATA_CARD, encoding="utf-8")) if os.path.exists(DATA_CARD) else {}
    label = card.get("label", "?")
    has_manifest = os.path.exists(MANIFEST)
    rows = read_manifest_tsv(MANIFEST) if has_manifest else []

    # Lọc split trước khi băm: kiểm `test` (vài nghìn ảnh) mất vài giây, còn kiểm
    # cả 217 585 ảnh là việc của một lần chạy dài, không phải mỗi lần sửa code.
    want = tuple(x.strip() for x in splits.split(",") if x.strip())
    if want:
        rows = [r for r in rows if r["split"] in want]

    # `sample` băm một tập con **cách đều** (tất định, không ngẫu nhiên theo giờ):
    # nhanh, nhưng báo cáo phải nói rõ là mẫu — không được để ai đọc thành "đã
    # kiểm toàn bộ".
    sampled = bool(sample) and len(rows) > sample
    rows_to_hash = rows
    if sampled:
        step = len(rows) / float(sample)
        rows_to_hash = [rows[int(i * step)] for i in range(sample)]

    t0 = time.time()
    if has_manifest and not skip_sha and rows_to_hash:
        def _progress(done: int, total: int) -> None:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[băm] {done}/{total} · {rate:.0f} file/s · "
                  f"còn ~{(total - done) / max(rate, 1e-9) / 60:.1f} phút", flush=True)

        ok, bad = verify_rows(rows_to_hash, Path(MANIFEST).parent, workers=workers,
                              progress=_progress, progress_every=progress_every)
    else:
        ok, bad = 0, []
    sha_seconds = round(time.time() - t0, 1)

    # Đếm PNG thật trên đĩa — mặc định TẮT: một vòng `os.walk` qua 217 585 entry
    # trên FUSE cũng tốn hàng phút, và nó không phải thứ chặn bước nào.
    n_png = None
    if count_png:
        n_png = 0
        for split in os.listdir(DATA) if os.path.isdir(DATA) else []:
            for _dirpath, _dirnames, filenames in os.walk(os.path.join(DATA, split)):
                n_png += sum(1 for f in filenames if f.endswith(".png"))

    counts = collections.Counter((r["split"], r["generator"]) for r in rows)
    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)

    problems = []
    # Ba nguyên nhân khác nhau của "không có dòng nào", ba cách sửa khác nhau.
    # Gộp chúng thành "manifest rỗng" là đã dẫn người dùng đi sai một lần.
    if not has_manifest:
        problems.append(
            f"thiếu {MANIFEST} — file này nằm ở **gốc** cây local, nên đẩy từng thư mục "
            "con sẽ bỏ sót. Đẩy riêng:\n"
            f"  modal volume put {V.VOLUME_NAME} <local>/manifest.tsv data/manifest.tsv"
        )
    elif not rows:
        problems.append(
            f"{MANIFEST} chỉ có dòng tiêu đề — tệp rỗng. Chạy `python -m dire_zero ingest` "
            "trên máy rồi đẩy lại (manifest + data_card), không cần đẩy lại ảnh."
        )
    if not os.path.exists(DATA_CARD):
        problems.append(
            f"thiếu {DATA_CARD} — cùng gốc với manifest; thiếu nó thì không biết nhãn "
            "tối đa được nói (R0/R1+R2/R1+R2+R3) và không kiểm được nguồn ảnh giả."
        )
    if bad:
        problems.append(f"{len(bad)} ảnh lệch sha256/xoá: {bad[:2]}")
    # Cân bằng lớp phải xét **từng bài toán**, không xét tổng.
    # `::dire --generator adm` dựng bài toán `real vs adm`; 14 generator × 1000
    # ảnh cộng lại là 1:12, nhưng **mỗi** generator vẫn 1000:1000. Kiểm trên tổng
    # là chặn oan đúng cấu trúc mà Table 2 cần để so từng dòng.
    reals = collections.Counter(r["split"] for r in rows if r["label"] == 0)
    gen_counts = collections.Counter((r["split"], r["generator"]) for r in rows if r["label"] == 1)
    tasks, unbalanced, no_real = [], [], []
    for (split, gen), n_fake_gen in sorted(gen_counts.items()):
        n_real_gen = reals.get(split, 0)
        ratio = max(n_real_gen, n_fake_gen) / max(min(n_real_gen, n_fake_gen), 1)
        entry = {"split": split, "generator": gen, "real": n_real_gen,
                 "fake": n_fake_gen, "ratio": round(ratio, 2),
                 "ok": n_real_gen > 0 and ratio <= max_ratio}
        tasks.append(entry)
        if n_real_gen == 0:
            no_real.append(f"{split}/{gen}")
        elif ratio > max_ratio:
            unbalanced.append(f"{split}/{gen}: thật {n_real_gen} vs giả {n_fake_gen} "
                              f"({ratio:.1f}:1)")

    if not n_real or not n_fake:
        problems.append(f"thiếu một lớp (thật {n_real}, giả {n_fake})")
    if no_real:
        problems.append(
            f"{len(no_real)} generator không có ảnh thật cùng split ⇒ không dựng được "
            f"bài toán nhị phân: {', '.join(no_real[:5])}"
        )
    if unbalanced:
        problems.append(
            f"{len(unbalanced)} bài toán lệch quá {max_ratio:g}:1 — ACC sẽ bị lớp đông "
            f"lấn át: {', '.join(unbalanced[:5])}"
        )
    if card.get("fake_source") == "author" and "R3" in label:
        problems.append("data_card nói ảnh giả là của tác giả nhưng nhãn lại có R3")

    # Không phải lỗi: `::gen` ghi thêm `data/<split>/ours_adm` mà manifest (do
    # ingest tạo) không biết. Nhưng chênh lệch lớn thì phải nói ra.
    warnings = []
    n_gen = len({g for (_sp, g) in gen_counts})
    if n_gen > 1:
        pooled = max(n_real, n_fake) / max(min(n_real, n_fake), 1)
        warnings.append(
            f"{n_gen} generator trong manifest ⇒ gộp tất cả thành một lớp giả cho "
            f"{n_real}:{n_fake} ({pooled:.1f}:1), **không** so được với Table 2. "
            "Table 2 là từng dòng = từng generator: chạy `::dire --generator <tên>` "
            "rồi `::train` trên `maps/` chỉ có generator đó."
        )
    if n_png is not None and rows and n_png != len(rows):
        warnings.append(
            f"{n_png} PNG trên đĩa vs {len(rows)} dòng manifest — chênh "
            f"{n_png - len(rows):+d}. Bình thường nếu đã chạy `::gen` (ảnh `ours_adm` "
            "không nằm trong manifest); không bình thường nếu vừa ingest xong."
        )

    report = {
        "label": label,
        "fake_source": card.get("fake_source", "?"),
        "dataset": card.get("dataset", "?"),
        "license": card.get("license", "?"),
        "comparable_to_paper": card.get("comparable_to_paper"),
        "n_rows": len(rows),
        "n_real": n_real,
        "n_fake": n_fake,
        "by_split_generator": {f"{sp}/{g}": n for (sp, g), n in sorted(counts.items())},
        "tasks": tasks,
        "max_ratio": max_ratio,
        "n_png_on_disk": n_png,
        "manifest_sha_ok": ok,
        "sha_scope": ("sample" if sampled else "full") if (has_manifest and not skip_sha)
        else "none",
        "sha_checked_rows": len(rows_to_hash),
        "sha_seconds": sha_seconds,
        "sha_workers": workers,
        "filtered_splits": list(want),
        "warnings": warnings,
        "problems": problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if problems:
        raise RuntimeError("dữ liệu chưa dùng được:\n  " + "\n  ".join(problems))
    return report


#: `cpu=8` và timeout 4 giờ, không phải 2 core/30 phút: 217 585 PNG nhỏ là bài
#: toán **độ trễ mở file**, không phải băng thông. Lần chạy thật với `cpu=2`,
#: một luồng, timeout 1800 s đã bị cắt trước khi băm xong.
@app.function(volumes=VOLUMES, cpu=8, timeout=4 * 3600)
def checkdata(skip_sha: bool = False, workers: int = 64, splits: str = "",
              sample: int = 0, count_png: bool = False,
              max_ratio: float = MAX_IMBALANCE) -> dict:
    """Cổng dữ liệu — xem `_check_data`.

    Ví dụ::

        modal run -m dire_zero.modal_app::checkdata --splits test   # vài giây
        modal run -m dire_zero.modal_app::checkdata --sample 2000   # mẫu, có ghi rõ
        modal run -m dire_zero.modal_app::checkdata                 # toàn bộ, 64 luồng
    """
    return _check_data(skip_sha, workers=workers, splits=splits, sample=sample,
                       count_png=count_png, max_ratio=max_ratio)


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
        "dtype": "float16",
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
         steps: int = 20, max_usd: float = 5.0, skip_existing: bool = False) -> dict:
    """R1 — DIRE map cho cả hai phía, cùng một lịch, **giữ split của người đóng gói**.

    ``generator="auto"``: dùng **mọi** thư mục generator đang có trong cây dữ liệu
    (tác giả hoặc tự sinh). Ảnh `gen` sinh sau lần chạy trước thì chỉ cần chạy lại
    với ``--skip-existing``: map đã có sẽ được **dùng lại**, không tính lại.

    Vì sao phải có ``--skip-existing``: ảnh `real` luôn nằm trong danh sách, nên
    mỗi lần thêm một generator là mỗi lần trả tiền cho toàn bộ `real` lần nữa —
    rồi ghi đè đúng những file vừa trả tiền để tạo. Ước lượng tiền cũng tính theo
    số ảnh **sẽ tính thật**, không theo số ảnh trong danh sách.
    """
    from .budget import Estimate, load_sec_per_step, require_estimate
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

    items = _items_to_map(DATA, want, gens, n)
    if not items:
        raise FileNotFoundError(
            f"không có ảnh nào để chạy: splits={list(want)} generators={gens}. "
            "Kiểm `::checkdata` trước."
        )
    existing = _count_existing(items, MAPS) if skip_existing else 0
    to_compute = len(items) - existing
    sec = load_sec_per_step(f"{OUT}/bench.json")
    print(f"[kế hoạch] {len(items)} ảnh (n ≤ {n} mỗi ô, {len(want)} split, "
          f"{len(gens)} generator: {', '.join(gens)})", flush=True)
    print(f"[kế hoạch] sẽ tính {to_compute}, dùng lại {existing} map đã có"
          + ("" if skip_existing else "  (thiếu --skip-existing nên không bỏ qua map nào)"), flush=True)
    if to_compute == 0:
        print("[kế hoạch] không có gì mới — không gọi GPU", flush=True)
        return {"n": 0, "n_reused": existing, "generators": gens, "splits": list(want),
                "nfe_total": 0, "est": Estimate(0, 2 * steps, sec).as_dict(),
                "note": "mọi map đã có; bỏ --skip-existing nếu muốn tính lại từ đầu"}

    est = Estimate(n_images=to_compute, nfe_per_image=2 * steps, sec_per_step=sec)
    print("[tiền]", est.line(), flush=True)
    require_estimate(est, max_usd)

    f, digest = _counter(dtype="float16")
    meta = RunMeta(digest, steps)
    out = compute_maps(f, items, MAPS, meta, limit=None, skip_existing=skip_existing)
    vol.commit()
    return {"n": out["n"], "n_new": out["n_new"], "n_reused": out["n_reused"],
            "generators": gens, "splits": list(want), "nfe_total": out["nfe_total"],
            "est": est.as_dict()}


def _items_to_map(root: str, want: Sequence[str], gens: Sequence[str], n: int) -> list[tuple]:
    """Danh sách ``(path, label, split, generator)`` mà `::dire` sẽ chạy.

    Generator **phải** đi cùng item. Bản trước bỏ nó đi, nên `_as_item` gán mọi
    ảnh giả thành generator `"fake"`: map của `ours_adm` và của `adm` rơi vào
    cùng một thư mục, tên file trùng nhau (tên PNG sinh là `seed.png`) thì ghi
    đè lên nhau, và `::train --generator ours_adm` không bao giờ tìm thấy gì.
    """
    from .data import list_images  # noqa: PLC0415

    items: list[tuple] = []
    for split in want:
        real_dir = f"{root}/{split}/real"
        if not os.path.isdir(real_dir):
            raise FileNotFoundError(f"thiếu {real_dir} — ingest chưa đẩy split này?")
        for p in list_images(real_dir)[:n]:
            items.append((str(p), 0, split, "real"))
        for g in gens:
            d = f"{root}/{split}/{g}"
            if os.path.isdir(d):
                for p in list_images(d)[:n]:
                    items.append((str(p), 1, split, g))
    return items


def _count_existing(items: Sequence[tuple], maps_root: str) -> int:
    """Bao nhiêu item đã có map **khác rỗng** — con số dùng để ước lượng tiền."""
    from .pipeline import map_rel_path  # noqa: PLC0415

    n = 0
    for it in items:
        p = os.path.join(maps_root, map_rel_path(it).as_posix())
        if os.path.exists(p) and os.path.getsize(p) > 0:
            n += 1
    return n


def _plan_images(root: str, want: Sequence[str], gens: Sequence[str], n: int) -> tuple[int, int]:
    """Đếm **đúng** số ảnh `::dire` sẽ chạy — nay là một cách đọc của `_items_to_map`.

    Giữ tên cũ vì `docs/from-zero-clean-room.md` §3 dẫn tới nó, nhưng **không**
    còn phép đếm thứ hai: hai bản sao của quy tắc "ảnh nào được chọn" là hai cơ
    hội để ước lượng tiền và vòng lặp thật lệch nhau. Lỗi cũ: đếm cả cây — 82 000
    ảnh cho `train,test` — nên `--n 200` bị ước lượng 91,90 USD thay vì ~0,90 và
    `--max-usd` chặn oan một lượt chạy rẻ.
    """
    splits = [s for s in want if os.path.isdir(f"{root}/{s}/real")]
    if not splits:
        return 0, 0
    items = _items_to_map(root, splits, gens, n)
    n_real = sum(1 for it in items if it[1] == 0)
    return n_real, len(items) - n_real


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


@app.function(volumes=VOLUMES, timeout=1800)
def relabel(apply: bool = False) -> dict:
    """Sửa nhãn generator cho map cũ nằm ở `maps/<split>/fake/` (xem `maps_index`).

    Chạy **không cần GPU**: đây là đổi tên file theo `source` đã ghi trong
    `dire.json`, không phải tính lại map. Mặc định chỉ in kế hoạch.

    ::

        modal run -m dire_zero.modal_app::relabel              # xem trước
        modal run -m dire_zero.modal_app::relabel --apply       # đổi tên thật
    """
    import json  # noqa: PLC0415

    from .maps_index import apply as apply_fn, plan as plan_fn  # noqa: PLC0415

    p = plan_fn(MAPS)
    out = {"plan": {k: v for k, v in p.items() if k != "moves"}, "n_moves": p["n_moves"],
           "apply": apply}
    if apply and p["moves"]:
        out["applied"] = apply_fn(p)["n_moves"]
        vol.commit()
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.function(volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def train(epochs: int = 40, bs: int = 32, lr: float = 1e-4, crop: int = 224,
          seed: int = 3407, generator: str = "auto") -> dict:
    """R2 — train detector trên map **của mình**.

    `generator` chọn **một** generator để dựng bài toán `real vs generator đó` —
    đúng một dòng của Table 2. `auto` giữ mọi generator, nghĩa là bài toán
    `real vs bất kỳ fake`; đó là một phép đo khác, không so được với Table 2, và
    báo cáo phải nói rõ (xem `generators_used`).
    """
    from .pipeline import collect_maps, train as train_fn

    from .detector import ResNetDetector, resnet50

    maps = collect_maps(MAPS)
    if not maps:
        raise FileNotFoundError(f"không có .npy trong {MAPS} — chạy `dire` trước")

    available = sorted({m["generator"] for m in maps if m["generator"] != "real"})
    if generator != "auto":
        if generator not in available:
            raise ValueError(
                f"không có map của generator {generator!r} trong {MAPS}. "
                f"Có: {available or '(chưa có generator nào)'}. "
                f"Chạy `::dire --generator {generator}` trước."
            )
        maps = [m for m in maps if m["generator"] in ("real", generator)]
        used = [generator]
    else:
        used = available
    if not used:
        raise FileNotFoundError(f"chỉ có map `real` trong {MAPS} — chưa có ảnh giả nào")
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
        extra={
            "generators_used": used,
            "generator_filter": generator,
            "note_scope": (
                "một generator ⇒ so được với đúng một dòng của Table 2"
                if len(used) == 1 else
                f"{len(used)} generator gộp lại ⇒ bài toán `real vs bất kỳ fake`, "
                "KHÔNG so được với Table 2 (mỗi dòng của bảng là một generator)"
            ),
        },
    )
    vol.commit()
    print({k: payload[k] for k in ("best_epoch", "best_val_ap", "threshold_from_val")}, flush=True)
    print("[phạm vi]", payload["note_scope"], flush=True)
    print("[test]", payload["test"], flush=True)
    return payload


def _map_inventory(maps_root: str) -> dict:
    """Đếm `.npy` theo ``<split>/<generator>`` — nguồn sự thật về map đã có."""
    if not os.path.isdir(maps_root):
        return {"total": 0, "by_split_generator": {}}
    counts: dict[str, int] = {}
    for dirpath, _dirnames, filenames in os.walk(maps_root):
        rel = os.path.relpath(dirpath, maps_root).replace(os.sep, "/")
        if rel == "." or rel.startswith("dire-runs"):
            continue
        n = sum(1 for f in filenames if f.endswith(".npy"))
        if n:
            counts[rel] = counts.get(rel, 0) + n
    return {"total": sum(counts.values()), "by_split_generator": dict(sorted(counts.items()))}


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

    dire_json = read("dire")
    payload = {
        "label": _read_json(DATA_CARD).get("label", "R1+R2+R3 (clean-room)"),
        "what_is_ours": ["DIRE map", "detector", "ảnh sinh"],
        "what_is_not": ["trọng số ADM (OpenAI)", "LSUN-Bedroom (dữ liệu thật)", "kiến trúc U-Net/ResNet"],
        "gen": {k: v for k, v in read("gen").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "dire": {k: v for k, v in dire_json.items()
                 if k in ("n", "steps", "steps_mixed", "nfe_total", "seconds", "n_new", "n_reused")},
        # Tồn kho map **thực tế trên đĩa**, không phải điều `dire.json` tự khai:
        # sau khi thêm generator, câu hỏi đúng là "map nào đã có", và một thư mục
        # `fake/` còn sót từ bản cũ sẽ lộ ra ở đây chứ không lộ ở `dire.json`.
        "maps": _map_inventory(MAPS),
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
