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

import math
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
    # `::gallery` chạy đúng `tools/make_gallery.py` của repo (một nguồn sự thật cho
    # định dạng `.json` mà `demolib/provenance.py` đọc). Máy Windows không phải tải
    # 800 MB map về để dựng 64 mẫu.
    #
    # `demolib/` phải đi kèm, và phải **đúng chỗ**: `make_gallery.py` tự thêm
    # `<repo>/platforms/hf-spaces` vào `sys.path` rồi `from demolib import provenance`.
    # Mount mỗi tệp tool mà quên thư viện ⇒ container chết ngay ở dòng import:
    #
    #     File "/root/tools/make_gallery.py", line 55, in <module>
    #         from demolib import provenance
    #     ModuleNotFoundError: No module named 'demolib'
    #
    # Đây là toàn bộ `demolib` (6 tệp, thuần stdlib + numpy), không phải cả Space.
    .add_local_file(Path(__file__).resolve().parents[2] / "tools" / "make_gallery.py",
                    "/root/tools/make_gallery.py")
    .add_local_dir(Path(__file__).resolve().parents[2] / "platforms" / "hf-spaces" / "demolib",
                   "/root/platforms/hf-spaces/demolib")
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


#: Timeout của `dire`/`gen`/`train`. Dùng làm **cổng thời gian** trong `dire`, không
#: chỉ để khai báo: một lượt ước tính 9,9 GPU-h trong container 6 giờ thì bị kill
#: giữa lúc đang chạy, và `vol.commit()` ở cuối không bao giờ chạy.
WORK_TIMEOUT_S = 6 * 3600

#: Bao nhiêu ảnh thì `commit()` Volume một lần trong `::dire`.
#: Volume v1 **không** tự ghi: thay đổi chỉ bền sau `commit()`. Trước đây chỉ có
#: `commit()` ở cuối, nên một lượt bị kill (mất mạng, hết timeout, `--detach` bị
#: dừng) làm mất **toàn bộ** map đã tính — đã trả tiền cho từng giây GPU.
COMMIT_EVERY = 200

#: Chỉ dùng tối đa 75 % timeout cho phép tính: nạp checkpoint (10,7 s), ghi
#: `.npy`, và commit Volume đều nằm ngoài phần đã ước lượng.
SAFE_TIME_FRACTION = 0.75

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


@app.function(volumes=VOLUMES, gpu="A10G", timeout=WORK_TIMEOUT_S)
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


@app.function(volumes=VOLUMES, gpu="A10G", timeout=WORK_TIMEOUT_S)
def dire(splits: str = "train,test", generator: str = "auto", n: int = 2000,
         steps: int = 20, max_usd: float = 5.0, skip_existing: bool = False) -> dict:
    """R1 — DIRE map cho cả hai phía, cùng một lịch, **giữ split của người đóng gói**.

    ``generator="auto"``: dùng **mọi** thư mục generator đang có trong cây dữ liệu
    (tác giả hoặc tự sinh). Nhận danh sách: ``--generator adm,pndm``. Ảnh `gen`
    sinh sau lần chạy trước thì chỉ cần chạy lại với ``--skip-existing``: map đã có
    sẽ được **dùng lại**, không tính lại.

    Hai cổng **trước** khi xin GPU, cả hai đều dừng chương trình:

    * **tiền** (`--max-usd`): 10 100 ảnh ở `S=20` là 10,87 USD — vượt trần mặc định
      5 USD thì dừng, chứ không tự "thương lượng" rồi chạy;
    * **thời gian**: ước tính 9,88 GPU-h trong container có timeout 6 h thì dù có
      nâng trần tiền, lượt chạy vẫn bị kill ở giờ thứ 6 và `vol.commit()` cuối
      không bao giờ chạy. Cổng này in luôn **nhóm generator** gợi ý để chia nhỏ.

    ``--skip-existing`` cần cho cả hai lý do trên: nó làm mỗi lượt chia nhỏ **cộng
    dồn** thay vì trả tiền lại cho ảnh `real`, và làm lượt bị kill không mất phần
    đã xong (map đã ghi thì lần sau dùng lại).
    """
    from .budget import Estimate, GateError, load_sec_per_step, require_estimate
    from .pipeline import RunMeta, compute_maps

    want = tuple(x.strip() for x in splits.split(",") if x.strip())
    gens = select_generators(_generators_in(DATA), generator)
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
    pending = _pending_by_generator(items, MAPS, skip_existing)
    existing = len(items) - sum(pending.values())
    to_compute = sum(pending.values())
    sec = load_sec_per_step(f"{OUT}/bench.json")
    print(f"[kế hoạch] {len(items)} ảnh (n ≤ {n} mỗi ô, {len(want)} split, "
          f"{len(gens)} generator: {', '.join(gens)})", flush=True)
    print(f"[kế hoạch] sẽ tính {to_compute}, dùng lại {existing} map đã có"
          + ("" if skip_existing else "  (thiếu --skip-existing nên không bỏ qua map nào)"), flush=True)
    for g, k in sorted(pending.items(), key=lambda kv: -kv[1]):
        if k:
            one = Estimate(k, 2 * steps, sec)
            print(f"  · {g:<18} {k:>5} ảnh  {one.gpu_h:6.2f} GPU-h  {one.usd:5.2f} USD", flush=True)
    if to_compute == 0:
        print("[kế hoạch] không có gì mới — không gọi GPU", flush=True)
        return {"n": 0, "n_reused": existing, "generators": gens, "splits": list(want),
                "nfe_total": 0, "est": Estimate(0, 2 * steps, sec).as_dict(),
                "note": "mọi map đã có; bỏ --skip-existing nếu muốn tính lại từ đầu"}

    est = Estimate(n_images=to_compute, nfe_per_image=2 * steps, sec_per_step=sec)
    print("[tiền]", est.line(), flush=True)

    # Cổng **thời gian** đứng trước cổng tiền: khi cả hai cùng vượt, lời khuyên của
    # cổng tiền ("nâng --max-usd") là lời khuyên **sai** — nâng trần tiền không làm
    # container sống lâu hơn, nó chỉ làm hoá đơn của một lượt chạy chắc chắn bị kill
    # trở nên đắt hơn.
    budget_h = WORK_TIMEOUT_S / 3600 * SAFE_TIME_FRACTION
    if est.gpu_h > budget_h:
        groups = group_generators(pending, sec, steps, budget_h)
        lines = plan_commands(groups, n=n, steps=steps, splits=splits)
        raise GateError(
            f"ước tính {est.gpu_h:.2f} GPU-h > trần thời gian {budget_h:.2f} h "
            f"(timeout {WORK_TIMEOUT_S / 3600:.0f} h × {SAFE_TIME_FRACTION:.0%}).\n"
            f"  {est.line()}\n"
            "Nâng `--max-usd` cũng **không** cứu được lượt này: container bị kill ở giờ thứ "
            f"{WORK_TIMEOUT_S / 3600:.0f} và `vol.commit()` cuối không chạy.\n"
            f"Chạy {len(groups)} lệnh dưới đây, đúng thứ tự (mỗi lệnh đã tính cả `real`; "
            "lệnh sau dùng lại map đã có nhờ --skip-existing):\n"
            + "\n".join(lines)
        )
    require_estimate(est, max_usd)

    f, digest = _counter(dtype="float16")
    meta = RunMeta(digest, steps)
    out = _compute_with_commits(f, items, meta, skip_existing, compute_maps)
    vol.commit()
    return {"n": out["n"], "n_new": out["n_new"], "n_reused": out["n_reused"],
            "generators": gens, "splits": list(want), "nfe_total": out["nfe_total"],
            "est": est.as_dict()}


def _compute_with_commits(f, items, meta, skip_existing: bool, compute_maps,
                          every: int | None = None) -> dict:
    """Chạy `compute_maps` và **ghi bền** Volume trên đường đi.

    Volume v1 không tự ghi: thay đổi chỉ bền sau `commit()`. Bản trước chỉ commit ở
    cuối, nên một lượt bị dừng giữa đường (mất mạng, `--detach` bị kill, hết
    timeout) mất **toàn bộ** map đã tính — đã trả tiền cho từng giây GPU rồi mất.

    Hai lớp:

    1. **định kỳ** mỗi ``every`` ảnh (mặc định `COMMIT_EVERY`): kill cứng (OOM,
       timeout) mất tối đa chừng ấy ảnh;
    2. **trong `except`**: exception mềm (lỗi mạng, `KeyboardInterrupt`) commit
       phần đã xong trước khi ném tiếp.

    ``every`` tách ra để test được mà không phải chạy 200 ảnh.
    """
    every = every or COMMIT_EVERY

    def checkpoint_progress(i: int) -> None:
        if i % every:
            return
        vol.commit()
        print(f"[bền] {i} ảnh đã commit lên Volume "
              f"(mất mạng từ đây chỉ mất ≤ {every} ảnh)", flush=True)

    try:
        out = compute_maps(f, items, MAPS, meta, limit=None, skip_existing=skip_existing,
                           progress=checkpoint_progress)
    except BaseException:
        vol.commit()
        print("[bền] commit phần đã xong trước khi thoát do lỗi", flush=True)
        raise
    return out


def select_generators(available: Sequence[str], spec: str) -> list[str]:
    """Lọc danh sách generator có thật theo ``--generator`` (``auto`` hoặc ``a,b``).

    Tên không tồn tại là **lỗi**, không phải "bỏ qua": gõ `sdv1` trong khi cây chỉ
    có `sdv2` mà vẫn chạy thì kết quả là một lượt GPU trả tiền cho `real` rồi báo
    "xong" với 0 map mới.
    """
    spec = (spec or "auto").strip()
    if spec == "auto":
        return list(available)
    picked = [x.strip() for x in spec.split(",") if x.strip()]
    missing = [g for g in picked if g not in available]
    if missing:
        raise ValueError(f"không có generator {missing}. Có: {list(available)}")
    return [g for g in available if g in picked]


def _pending_by_generator(items: Sequence[tuple], maps_root: str, skip_existing: bool) -> dict:
    """Số ảnh **sẽ tính** theo từng generator (0 nếu đã có map hoặc không bỏ qua).

    Dùng cho hai việc: in kế hoạch để thấy generator nào tốn tiền, và chia nhóm
    theo thời gian. `real` xuất hiện ở đây như mọi generator khác — nó cũng là ảnh
    phải tính, chỉ là thường đã có map từ lần chạy trước.
    """
    from .pipeline import map_rel_path  # noqa: PLC0415

    counts: dict[str, int] = {}
    for it in items:
        gen = it[3]
        counts.setdefault(gen, 0)
        if skip_existing:
            p = os.path.join(maps_root, map_rel_path(it).as_posix())
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
        counts[gen] += 1
    return counts


def _group_hours(n_images: int, sec_per_step: float, steps: int) -> float:
    return n_images * 2 * steps * sec_per_step / 3600


def group_generators(pending: dict, sec_per_step: float, steps: int,
                     budget_h: float) -> list[dict]:
    """Chia generator thành các nhóm **cân bằng**, mỗi nhóm ≤ ``budget_h`` GPU-h.

    Hai lỗi của bản trước, cả hai đều làm người dùng bị chặn **lần thứ hai** với
    đúng bản gợi ý vừa in ra:

    1. `real` bị loại khỏi phép chia nhưng **có mặt trong mọi lệnh**. Nhóm đầu in
       ra 4 000 ảnh (4,08 h), cộng 600 ảnh `real` chưa có map ⇒ lệnh thật là
       4,69 h và bị chặn y như cũ. Nay `real` được tính vào **nhóm chạy đầu tiên**
       (chỉ nhóm đó trả tiền cho nó; lệnh sau dùng lại map nhờ `--skip-existing`).
    2. Chia kiểu "nhét đầy" (greedy) cho 4,08 h + 0,51 h — nhóm đầu vẫn sát trần
       nên một lần chậm 10 % là bị kill. Nay chia **cân bằng** (LPT: việc lớn nhất
       vào nhóm đang nhẹ nhất), số nhóm = `ceil(tổng / trần)`.

    Trả về ``[{"generators": [...], "n_images": int, "includes_real": int, "gpu_h": float}, …]``
    theo đúng thứ tự nên chạy.
    """
    from .budget import GateError  # noqa: PLC0415

    items = [(g, k, _group_hours(k, sec_per_step, steps)) for g, k in pending.items() if k]
    for g, k, h in items:
        if h > budget_h:
            raise GateError(
                f"riêng generator {g!r} đã ước tính {h:.2f} GPU-h > {budget_h:.2f} h — "
                f"giảm `--n` (hiện {k} ảnh) hoặc `--steps`"
            )

    total_h = sum(h for _g, _k, h in items)
    n_real = pending.get("real", 0)
    rest = [x for x in items if x[0] != "real"]

    def pack(k: int) -> list[dict]:
        groups = [{"generators": [], "n_images": 0, "includes_real": 0, "gpu_h": 0.0} for _ in range(k)]
        groups[0]["includes_real"] = n_real
        groups[0]["gpu_h"] = _group_hours(n_real, sec_per_step, steps)
        # LPT: việc lớn nhất vào nhóm đang nhẹ nhất (nhóm 0 đã có `real` trong gpu_h)
        for g, n_img, h in sorted(rest, key=lambda x: -x[2]):
            target = min(groups, key=lambda grp: grp["gpu_h"])
            target["generators"].append(g)
            target["n_images"] += n_img
            target["gpu_h"] += h
        return groups

    # LPT bảo đảm ≤ 4/3 OPT, **không** bảo đảm ≤ trần với `ceil(tổng/trần)` nhóm:
    # 10 000 ảnh / trần 2 940 ảnh cho 4 nhóm vẫn ra 3 000 ảnh/nhóm. Nên tăng dần
    # số nhóm cho tới khi **mọi** nhóm nằm dưới trần — dừng muộn nhất ở `k = số
    # generator`, nơi mỗi nhóm một generator (đã kiểm từng cái ≤ trần ở trên).
    groups = pack(max(1, math.ceil(round(total_h, 6) / budget_h)))
    while any(grp["gpu_h"] > budget_h + 1e-9 for grp in groups) and len(groups) < len(items):
        groups = pack(len(groups) + 1)

    for grp in groups:
        grp["generators"].sort()
        grp["gpu_h"] = round(grp["gpu_h"], 2)
    return groups


def plan_commands(groups: list[dict], n: int, steps: int, splits: str) -> list[str]:
    """Lệnh copy–paste được, mỗi nhóm một dòng, kèm ước tính của **chính nhóm đó**.

    In `--generator …` trơ trọi là chưa đủ: người dùng còn phải nhớ `--n`,
    `--steps`, `--splits` và `--skip-existing`, và thiếu `--skip-existing` thì
    nhóm thứ hai trả tiền lại cho `real` — đúng thứ vừa được chia nhỏ để tránh.
    """
    lines = []
    for grp in groups:
        cmd = (f"modal run -m dire_zero.modal_app::dire --n {n} --steps {steps} "
               f"--splits {splits} --skip-existing --generator {','.join(grp['generators'])}")
        n_images = grp["n_images"] + grp["includes_real"]
        note = f"  # {n_images} ảnh"
        if grp["includes_real"]:
            note += f" (gồm {grp['includes_real']} real chưa có map)"
        note += f" = {grp['gpu_h']:.2f} GPU-h ≈ {grp['gpu_h'] * 1.10:.2f} USD"
        lines.append(f"  {cmd}{note}")
    return lines


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


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def maps(n: int = 2000, verify: int = 0, generator: str = "auto",
         steps: int = 20) -> dict:
    """Kiểm **tồn kho** DIRE map và in **lệnh chạy tiếp** — không cần GPU.

    Dùng đúng vào lúc cần: một lượt `::dire` bị dừng giữa đường (mất mạng, hết
    timeout, `--detach` bị kill). Câu hỏi khi đó không phải "app còn sống không"
    (nó chết rồi, không nối lại được) mà "đã kịp ghi bao nhiêu, chạy tiếp từ đâu".

    ::

        modal run -m dire_zero.modal_app::maps --n 500          # cùng --n với lệnh định chạy
        modal run -m dire_zero.modal_app::maps --n 500 --verify 20
        modal run -m dire_zero.modal_app::maps --n 0            # inventory cả cây, không sinh lệnh chạy

    `--n 0` là **chế độ inventory toàn cây**, không phải giá trị chạy được của
    `::dire`: ở `::dire`, `n=0` chọn 0 ảnh. Vì vậy `::maps --n 0` chỉ đếm/kiểm kê;
    nó không lập lệnh chạy cho workload vượt ngân sách. Dùng `--n` dương để lập
    kế hoạch cho một đợt chạy.

    ``--n`` PHẢI là ``n`` của `::dire` mà bạn định chạy — mặc định 2 000 chỉ là mặc
    định của `::dire`. Đếm cả cây rồi báo "thiếu 212 583 map" là con số đúng nhưng
    vô dụng: không ai định tính map cho 212 583 ảnh, và nó không cho biết phải chạy
    lệnh gì tiếp.

    ``verify`` đọc thử ngẫu nhiên từng ấy file bằng ``np.load``: file ghi dở do bị
    kill **vẫn tồn tại với kích thước khác 0**, nên đếm file là chưa đủ.
    """
    import numpy as np  # noqa: PLC0415

    from .data import list_images  # noqa: PLC0415

    data_gens = _cell_generators(DATA)
    gens = select_generators(data_gens, generator) if data_gens else []
    cells: list[dict] = []
    for split in sorted(os.listdir(DATA)) if os.path.isdir(DATA) else []:
        sdir = os.path.join(DATA, split)
        if not os.path.isdir(sdir):
            continue
        for gen in sorted(os.listdir(sdir)):
            gdir = os.path.join(sdir, gen)
            if not os.path.isdir(gdir) or (gens and gen != "real" and gen not in gens):
                continue
            imgs = list_images(gdir)
            want = min(len(imgs), n) if n else len(imgs)
            mdir = os.path.join(MAPS, split, gen)
            have = sorted(p for p in os.listdir(mdir) if p.endswith(".npy")) if os.path.isdir(mdir) else []
            nonempty = [p for p in have if os.path.getsize(os.path.join(mdir, p)) > 0]
            cells.append({
                "cell": f"{split}/{gen}",
                "images": len(imgs),
                "wanted": want,
                "maps": len(nonempty),
                "missing": max(0, want - len(nonempty)),
                "extra": max(0, len(nonempty) - want),
                "pct": round(100 * min(len(nonempty), want) / want, 1) if want else 0.0,
            })

    import json  # noqa: PLC0415

    index_path = os.path.join(MAPS, "dire.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else {}
    runs = index.get("runs") or []

    # Thiếu map, gộp theo generator trên mọi split. `real` phải nằm trong đây: nó
    # có trong **mọi** lệnh `::dire`, nên nếu bỏ ra khỏi phép tính thì lệnh gợi ý
    # vẫn bị chặn — đúng cái bẫy đã xảy ra ở phần chia nhóm (§3.9).
    pending: dict[str, int] = {}
    for c in cells:
        if c["missing"]:
            pending[c["cell"].split("/")[-1]] = pending.get(c["cell"].split("/")[-1], 0) + c["missing"]

    problems: list[str] = []
    for c in cells:
        if c["missing"] and c["cell"].split("/")[-1] == "real":
            # Không in `--generator real`: `real` không phải generator hợp lệ, và một
            # chuỗi trông như cờ để copy là đủ để có người copy nó.
            problems.append(f"{c['cell']}: thiếu {c['missing']} map — chạy bất kỳ lệnh "
                            "`::dire` nào với `--skip-existing` (ảnh `real` luôn được đưa vào)")
    # `fake/` chỉ nằm trong `maps/`, không có ảnh tương ứng trong `data/` ⇒ không
    # xuất hiện như một ô. Nó vẫn phải bị báo: map ở đó bị `collect_maps` đọc thành
    # một generator tên `fake`, tức là một lớp giả **không có thật** trong dữ liệu.
    if os.path.isdir(MAPS):
        for split in sorted(os.listdir(MAPS)):
            if os.path.isdir(os.path.join(MAPS, split, "fake")):
                problems.append(f"{split}/fake: còn thư mục `fake/` từ bản cũ — "
                                "chạy `::relabel --apply` trước khi train")

    # Mất cân bằng lớp: `test/midjourney` chỉ có 100 ảnh trong khi `test/real` có
    # 1 000 ⇒ bài toán `real vs midjourney` là 10:1. Recall của nó sẽ thấp vì **ít
    # mẫu**, không phải vì DIRE kém; báo ở đây để quyết trước khi train.
    warnings: list[str] = []
    for split in sorted({c["cell"].split("/")[0] for c in cells}):
        by = {c["cell"].split("/")[-1]: c["wanted"] for c in cells if c["cell"].startswith(split + "/")}
        real = by.get("real", 0)
        for gen, k in sorted(by.items()):
            if gen == "real" or not k or not real:
                continue
            ratio = max(real, k) / max(min(real, k), 1)
            if ratio > MAX_IMBALANCE:
                warnings.append(f"{split}: `real` {real} vs `{gen}` {k} = {ratio:.1f}:1 > "
                                f"{MAX_IMBALANCE:.0f}:1 — recall của dòng này phản ánh số mẫu")

    bad_files: list[str] = []
    if verify:
        rng = np.random.default_rng(0)
        for c in cells:
            mdir = os.path.join(MAPS, *c["cell"].split("/"))
            files = [p for p in os.listdir(mdir) if p.endswith(".npy")] if os.path.isdir(mdir) else []
            for name in rng.choice(files, size=min(verify, len(files)), replace=False) if files else []:
                path = os.path.join(mdir, str(name))
                try:
                    arr = np.load(path)
                    if arr.ndim != 3 or arr.shape[0] != 3:
                        bad_files.append(f"{c['cell']}/{name}: shape {arr.shape}")
                except Exception as e:  # noqa: BLE001 - chẩn đoán, in ra nguyên nhân
                    bad_files.append(f"{c['cell']}/{name}: {type(e).__name__}: {e}")

    # Ước lượng phần còn thiếu + lệnh chạy tiếp, cùng một công thức với cổng của
    # `::dire`. Nếu chỉ in "thiếu 3 500 map" thì người đọc vẫn phải tự chia nhóm.
    splits = sorted({c["cell"].split("/")[0] for c in cells})
    plan = _plan_remaining(pending, splits, n, steps)
    total_missing = sum(c["missing"] for c in cells)
    has_fake = any(not c["cell"].endswith("/real") and c["maps"] for c in cells)
    ready = bool(cells) and not problems and not bad_files and has_fake and total_missing == 0
    payload = {
        "n": n,
        "cells": cells,
        "total_maps": sum(c["maps"] for c in cells),
        "total_missing": total_missing,
        "ready_for_train": ready,
        "problems": problems,
        "warnings": warnings,
        "bad_files": bad_files,
        "plan": plan,
        "dire_runs": [
            {k: r.get(k) for k in ("created", "steps", "n", "n_new", "seconds", "skip_existing")}
            for r in runs[-5:]
        ],
        "steps_mixed": index.get("steps_mixed"),
    }

    print(f"{'ô':<28}{'ảnh':>7}{'cần':>7}{'map':>7}{'thiếu':>7}  %", flush=True)
    for c in cells:
        print(f"{c['cell']:<28}{c['images']:>7}{c['wanted']:>7}{c['maps']:>7}{c['missing']:>7}"
              f"  {c['pct']:5.1f}", flush=True)
    line = f"\ntổng: {payload['total_maps']} map · thiếu {total_missing}"
    if plan.get("gpu_h") is not None:
        line += (f" · ~{plan['gpu_h']:.2f} GPU-h ≈ {plan['usd']:.2f} USD "
                 f"(S={steps}, dùng lại {payload['total_maps']} map đã có)")
    print(line + f" · train được: {'CÓ' if ready else 'CHƯA'}", flush=True)
    for w in warnings:
        print(f"  ~ {w}", flush=True)
    for p in problems:
        print(f"  ! {p}", flush=True)
    for b in bad_files:
        print(f"  ! file hỏng: {b}", flush=True)
    if plan.get("commands"):
        print(f"\nchạy tiếp ({len(plan['commands'])} lệnh, mỗi lệnh đã có --skip-existing):", flush=True)
        for cmd in plan["commands"]:
            print(cmd, flush=True)
    if plan.get("note"):
        print(f"\n{plan['note']}", flush=True)
    if runs:
        print(f"\nlần chạy gần nhất: {runs[-1].get('created')} · {runs[-1].get('n_new')} map mới", flush=True)
    return payload


def _plan_remaining(pending: dict, splits: Sequence[str], n: int, steps: int) -> dict:
    """Ước lượng phần còn thiếu + lệnh chạy tiếp, dùng **cùng** công thức với `::dire`.

    Trả ``{"gpu_h", "usd", "commands", "groups", "note"}``. Không có `bench.json`
    thì không ước lượng (``gpu_h = None``) — đoán `sec_per_step` là cách chắc chắn
    nhất để mọi con số tiền sau đó sai.
    """
    from .budget import Estimate, GateError, load_sec_per_step  # noqa: PLC0415

    todo = {g: k for g, k in pending.items() if k}
    if not todo:
        return {"gpu_h": None, "usd": None, "commands": [],
                "note": "không thiếu map nào — không có gì để chạy"}
    if n == 0:
        # `::maps --n 0` is explicitly the whole-tree inventory mode. Never turn
        # that into `::dire --n 0`: `_items_to_map(..., n=0)` selects [:0], and
        # so the suggested command would process no images. It must also remain
        # usable for inventories whose one-generator total exceeds one job's
        # time budget; planning limits are not inventory limits.
        return {
            "gpu_h": None,
            "usd": None,
            "commands": [],
            "groups": [],
            "note": (f"inventory toàn cây: còn {sum(todo.values())} map; chưa sinh lệnh chạy vì "
                     "`::dire --n 0` không chọn ảnh và một generator có thể vượt thời gian/job. "
                     "Sau khi xem inventory, dùng `::maps --n <số dương>` để lập một đợt có giới hạn; "
                     "chạy `::bench` trước nếu chưa có bench.json."),
        }
    try:
        sec = load_sec_per_step(f"{OUT}/bench.json")
    except Exception:  # noqa: BLE001 - thiếu bench.json là trạng thái hợp lệ
        return {"gpu_h": None, "usd": None, "commands": [],
                "groups": [],
                "note": (f"cần {sum(todo.values())} map nữa, nhưng chưa có `{OUT}/bench.json` "
                         "nên không ước lượng được tiền/thời gian — chạy `::bench` trước")}

    est = Estimate(sum(todo.values()), 2 * steps, sec)
    budget_h = WORK_TIMEOUT_S / 3600 * SAFE_TIME_FRACTION
    split_arg = ",".join(splits) if splits else "train,test"
    if est.gpu_h <= budget_h:
        groups = [{"generators": sorted(g for g in todo if g != "real"),
                   "n_images": sum(k for g, k in todo.items() if g != "real"),
                   "includes_real": todo.get("real", 0), "gpu_h": round(est.gpu_h, 2)}]
        note = (f"một lệnh là đủ ({est.gpu_h:.2f} h ≤ trần {budget_h:.2f} h). "
                "Nhớ `--skip-existing` để không tính lại map đã có.")
        if not groups[0]["generators"]:
            note = ("chỉ còn thiếu map của `real` — mọi lệnh `::dire` đều mang nó, "
                    "chạy bất kỳ generator nào với `--skip-existing` là đủ")
            groups = []
    else:
        try:
            groups = group_generators(todo, sec, steps, budget_h)
        except GateError as exc:
            # Inventory and plan generation are separate concerns: a single large
            # generator can exceed one job even though `::maps` has successfully
            # counted every cell. Return the inventory plus an actionable note,
            # rather than turning a read-only check into an uncaught exception.
            return {
                "gpu_h": round(est.gpu_h, 2),
                "usd": round(est.usd, 2),
                "commands": [],
                "groups": [],
                "note": (f"chưa sinh lệnh chạy an toàn: {exc}. `::maps` vẫn kiểm kê đầy đủ; "
                         "giảm --n cho một đợt nhỏ hơn rồi kiểm tra lại. Không dùng `--n 0` "
                         "cho `::dire` và không giảm `--steps` khỏi protocol chỉ để qua cổng."),
            }
        note = (f"ước tính {est.gpu_h:.2f} h > trần {budget_h:.2f} h ⇒ chia {len(groups)} lệnh; "
                "mỗi lệnh dùng lại map đã có nhờ `--skip-existing`.")
    commands = plan_commands(groups, n=n, steps=steps, splits=split_arg) if groups else []
    return {"gpu_h": round(est.gpu_h, 2), "usd": round(est.usd, 2),
            "commands": commands, "groups": groups, "note": note}


def _cell_generators(data_root: str) -> list[str]:
    """Generator có trong **dữ liệu ảnh** (`data/`), không phải trong `maps/`.

    Khác `_generators_in(MAPS)`: câu hỏi "đã đủ map chưa" phải đếm theo ảnh đang
    có, nếu không thì một generator chưa có map nào sẽ **không hiện** trong bảng —
    đúng thứ cần thấy.
    """
    return _generators_in(data_root)


@app.function(volumes=VOLUMES, gpu="A10G", timeout=WORK_TIMEOUT_S)
def train(epochs: int = 40, bs: int = 32, lr: float = 1e-4, crop: int = 224,
          seed: int = 3407, generator: str = "auto", split_mode: str = "author") -> dict:
    """R2 — train detector trên map **của mình**.

    ``generator``:

    * một tên (`adm`) ⇒ bài toán `real vs adm`, **đúng một dòng của Table 2**;
    * `a,b,c` ⇒ lần lượt từng generator, mỗi cái một detector riêng;
    * ``all`` ⇒ **mọi** generator đang có map (đường đi thường dùng sau `::dire`);
    * ``auto`` ⇒ gộp tất cả vào một bài toán `real vs bất kỳ fake` — **không** so
      được với Table 2, và `note_scope` trong `train.json` sẽ nói đúng như vậy.

    Mỗi generator ghi vào **thư mục riêng** (`runs/train/<generator>/`), nên chạy
    lần lượt 15 dòng của Table 2 không ghi đè nhau. Bản trước ghi tất cả vào
    `runs/train.json` + `runs/detector.pth`: dòng thứ 15 là dòng duy nhất còn lại,
    và `::report` đọc đúng một dòng mà không nói rằng 14 dòng kia đã mất.

    Không có ước lượng tiền ở đây: mỗi dòng của Table 2 là một lượt train ResNet-50
    trên ~2 000 map, **giây** chứ không phải GPU-giờ như DIRE. Thay vào đó, mỗi
    lượt in số giây thật và `--generator all` dừng trước khi chạm timeout, liệt kê
    generator còn lại để chạy tiếp.
    """
    from .detector import ResNetDetector, resnet50
    from .pipeline import SplitError, collect_maps, train as train_fn

    maps = collect_maps(MAPS)
    if not maps:
        raise FileNotFoundError(f"không có .npy trong {MAPS} — chạy `dire` trước")

    # `fake/` là thư mục map cũ (chưa relabel), không phải một generator. Train nó
    # cho ra một dòng `gen-fake` trong bảng — một dòng trông như Table 2 nhưng nhãn
    # của nó là "ảnh giả nói chung", và nó trùng lặp với generator thật.
    leftover = sorted({m["generator"] for m in maps if m["generator"] == "fake"})
    available = sorted({m["generator"] for m in maps if m["generator"] not in ("real", "fake")})
    if leftover:
        print(f"[bỏ qua] {MAPS}/*/fake: {sum(1 for m in maps if m['generator'] == 'fake')} map "
              "chưa có nhãn generator — chạy `::relabel --apply` trước; "
              "`fake` không phải một dòng của Table 2", flush=True)
    if not available:
        raise FileNotFoundError(f"chỉ có map `real` trong {MAPS} — chưa có ảnh giả nào")
    targets = _train_targets(generator, available)

    deadline = time.time() + WORK_TIMEOUT_S * SAFE_TIME_FRACTION
    results: dict[str, dict] = {}
    stopped: list[str] = []
    skipped: list[str] = []
    for gen in targets:
        left = deadline - time.time()
        if left <= 60:
            stopped = targets[targets.index(gen):]
            print(f"[dừng] hết ngân sách thời gian; còn {len(stopped)} generator: "
                  f"{', '.join(stopped)}", flush=True)
            break
        subset = maps if gen is None else [m for m in maps if m["generator"] in ("real", gen)]
        real_n = sum(1 for m in subset if m["label"] == 0)
        fake_n = len(subset) - real_n
        ratio = max(real_n, fake_n) / max(min(real_n, fake_n), 1)
        if ratio > MAX_IMBALANCE:
            print(f"[cảnh báo] {gen or 'auto'}: {real_n} real vs {fake_n} fake "
                  f"(tỉ lệ {ratio:.1f}:1 > {MAX_IMBALANCE:.0f}:1) — recall là số yếu, "
                  "và nó phản ánh số mẫu chứ không phải chất lượng DIRE", flush=True)

        tag = _train_tag(gen)
        print(f"\n=== train {tag}: {real_n} real + {fake_n} fake = {len(subset)} map ===", flush=True)
        try:
            payload = train_fn(
                lambda: ResNetDetector(resnet50()),
                subset,
                f"{OUT}/train/{tag}",
                label=_read_json(DATA_CARD).get("label") or "R1+R2",
                epochs=epochs,
                lr=lr,
                bs=bs,
                crop=crop,
                device="cuda",
                seed=seed,
                init="scratch",
                split_mode=split_mode,
                extra={
                "generators_used": [gen] if gen else available,
                "generator_filter": gen or "auto",
                "n_real": real_n,
                "n_fake": fake_n,
                "imbalance_ratio": round(ratio, 2),
                    "note_scope": (
                        f"một generator ({gen}) ⇒ so được với đúng một dòng của Table 2"
                        if gen else
                        f"{len(available)} generator gộp lại ⇒ bài toán `real vs bất kỳ fake`, "
                        "KHÔNG so được với Table 2 (mỗi dòng của bảng là một generator)"
                    ),
                },
            )
        except SplitError as e:
            # Một generator hỏng không được làm hỏng cả lượt `--generator all`:
            # ghi lý do, bỏ qua, chạy tiếp. Lượt trước im lặng train nó và in ra
            # một dòng `acc = 0,500`, trông y như kết quả.
            print(f"[bỏ qua] {tag}: {e}", flush=True)
            skipped.append(f"{tag}: {e}")
            continue
        vol.commit()
        results[tag] = {k: payload.get(k) for k in
                        ("label", "split_mode", "best_epoch", "best_val_ap",
                         "threshold_from_val", "threshold_space", "test", "seconds")}
        print(f"[{tag}] {payload['test']['acc']:.3f} acc · AP {payload['test']['ap']:.3f} · "
              f"{payload['seconds']:.0f} s", flush=True)

    ledger = _write_train_ledger(OUT, results, skipped)
    return {"trained": sorted(results), "skipped": skipped, "stopped": stopped,
            "ledger": ledger}


def _train_targets(spec: str, available: Sequence[str]) -> list[str | None]:
    """``--generator`` → danh sách việc; ``None`` nghĩa là gộp mọi generator."""
    spec = (spec or "auto").strip()
    if spec == "auto":
        return [None]
    if spec == "all":
        return list(available)
    picked = [x.strip() for x in spec.split(",") if x.strip()]
    missing = [g for g in picked if g not in available]
    if missing:
        raise ValueError(
            f"không có map của generator {missing} trong {MAPS}. Có: {available}. "
            f"Chạy `::dire --generator {','.join(missing)}` trước."
        )
    return picked


def _train_tag(gen: str | None) -> str:
    """Tên thư mục cho một lượt train: an toàn cho mọi hệ tệp, không trùng nhau."""
    if gen is None:
        return "auto"
    keep = "".join(c if (c.isalnum() or c in "-._") else "_" for c in gen)
    return f"gen-{keep}" if keep != "auto" else "gen-auto"


def _sha256(path: str) -> str:
    """sha256 một tệp, rỗng nếu không có — dùng để biết `eval.json` còn khớp detector."""
    import hashlib  # noqa: PLC0415

    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_train_ledger(out_root: str, results: dict[str, dict],
                        skipped: Sequence[str] = ()) -> dict:
    """Gộp các lượt train vào `runs/train.json` — bảng nhiều dòng, không ghi đè.

    Cùng lý do với `dire.json`: 15 dòng của Table 2 là **15 kết quả**, không phải
    một kết quả được ghi 15 lần. `runs` giữ mọi lượt (kể cả lượt cũ cùng tag, để
    so lại sau khi đổi seed), và `by_generator` là bảng đọc nhanh.
    """
    path = os.path.join(out_root, "train.json")
    old: dict = {}
    if os.path.exists(path):
        try:
            import json  # noqa: PLC0415

            old = json.load(open(path, encoding="utf-8"))
        except ValueError:
            old = {}

    import json  # noqa: PLC0415

    runs = [dict(r) for r in (old.get("runs") or []) if isinstance(r, dict)]
    if not runs and old.get("test"):
        runs.append({k: old.get(k) for k in ("best_epoch", "best_val_ap", "threshold_from_val",
                                             "seconds", "generators_used")})
    for tag, res in results.items():
        runs.append({"tag": tag, **res})
    by_generator = dict(old.get("by_generator") or {})
    # Một generator bị bỏ qua ở lượt này thì dòng cũ của nó **không còn là kết quả
    # hiện tại** — nhưng nó vẫn nằm trong `by_generator` từ lượt trước. Đánh dấu,
    # để `::report` cảnh báo thay vì trình bày một con số đã bị thay thế.
    if skipped:
        skipped_tags = [ln.split(":", 1)[0].strip() for ln in skipped]
        old_skipped = dict(old.get("skipped") or {})
        for t in skipped_tags:
            old_skipped[t] = next((ln for ln in skipped if ln.startswith(t + ":")), t)
        body_extra = {"skipped": old_skipped}
    else:
        body_extra = {"skipped": dict(old.get("skipped") or {})}

    for tag, res in results.items():
        t = res.get("test") or {}
        by_generator[tag] = {
            # `acc_at_0.5` đi cùng `acc` để đọc được ngay mức lệch chuẩn hoá: lệch
            # lớn nghĩa là ACC đang phụ thuộc vào một ngưỡng chọn trên val ít mẫu.
            "acc": t.get("acc"), "acc_at_0.5": t.get("acc_at_0.5"),
            "acc_ci95": t.get("acc_ci95"), "ap": t.get("ap"),
            "auc": t.get("auc"), "n": t.get("n"), "n_real": t.get("n_real"),
            "n_fake": t.get("n_fake"), "best_epoch": res.get("best_epoch"),
        }
    latest = runs[-1] if runs else {}
    payload = {
        # Nhãn lấy từ chính lượt chạy, không phải hằng số: bản trước ghi cứng
        # "R1+R2+R3" ở đây trong khi `train.json` của từng detector đã ghi R1+R2.
        "label": latest.get("label") or "R1+R2",
        "what": "mỗi mục của `by_generator` là một detector riêng (một dòng Table 2)",
        "by_generator": by_generator,
        "runs": runs,
        **body_extra,
        # giữ khoá cũ ở cấp ngoài cho chỗ đọc cũ (`::report`), trỏ tới lượt gần nhất
        **{k: latest.get(k) for k in ("best_epoch", "best_val_ap", "threshold_from_val", "seconds")},
        "latest": latest.get("tag"),
    }
    from .pipeline import write_json  # noqa: PLC0415

    write_json(path, payload)
    return {"n": len(by_generator), "by_generator": by_generator, "latest": payload["latest"]}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def evaluate(generator: str = "all", split: str = "test", crop: int = 224,
             limit: int = 0) -> dict:
    """Chấm detector **đã train** trên map của split chấm, và xuất `scores.json`.

    Đây là bước còn thiếu giữa `::train` và trang 2 của demo: `tools/make_gallery.py
    --scores` cần `{tên mẫu: {p_fake: …}}`, mà trước hàm này repo chỉ có `train.json`
    (số tổng hợp), không có điểm từng ảnh. Không có nó, gallery vẫn dựng được nhưng
    trang 2 hiện "chưa có điểm" — mất đúng phần demo muốn khoe.

    ::

        modal run -m dire_zero.modal_app::evaluate                       # mọi detector đã train
        modal run -m dire_zero.modal_app::evaluate --generator adm       # một cái
        modal run -m dire_zero.modal_app::evaluate --split val           # chấm trên val

    Ghi ra ``runs/eval/<tag>/eval.json`` (metric + CI) và ``.../scores.json``
    (điểm từng ảnh, đúng định dạng `make_gallery` đọc).

    **Ngưỡng**: dùng ngưỡng đã chốt trên val lúc train, đổi sang thang xác suất bằng
    sigmoid cho khớp `p_fake ∈ [0,1]`. Đổi thang không đổi ACC/AP/AUC (hàm đơn điệu
    tăng), nhưng đổi *cách đọc*: `make_gallery --threshold` yêu cầu `[0,1]`, còn
    ngưỡng trong `train.json` là logit (có thể âm) — trộn hai thang là hiện điểm
    sai mà không có lỗi nào báo.

    **Cổng**: `gate: passed` chỉ khi ngưỡng thật sự đến từ val với ≥ 20 mẫu. Không
    đạt thì `gate: failed`, và `demolib/provenance.py` sẽ **từ chối** hiện điểm —
    đó là hành vi đúng, không phải lỗi cần lách.
    """
    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from . import metrics as M  # noqa: PLC0415
    from .detector import ResNetDetector, resnet50  # noqa: PLC0415
    from .pipeline import _score, collect_maps, write_json  # noqa: PLC0415

    ledger_path = f"{OUT}/train.json"
    ledger = json.load(open(ledger_path, encoding="utf-8")) if os.path.exists(ledger_path) else {}
    tags = sorted(ledger.get("by_generator") or {})
    if not tags:
        raise FileNotFoundError(f"chưa có detector nào trong {OUT}/train.json — chạy `::train` trước")
    want = _train_targets(generator, [t[4:] if t.startswith("gen-") else t for t in tags])
    if want and want[0] is None:                      # `auto` ⇒ mọi detector đã train
        chosen = tags
    else:
        chosen = [_train_tag(g) for g in want]
        missing = [t for t in chosen if t not in tags]
        if missing:
            raise FileNotFoundError(f"chưa train {missing}. Có: {tags}")

    all_maps = collect_maps(MAPS)
    results: dict[str, dict] = {}
    for tag in chosen:
        tdir = f"{OUT}/train/{tag}"
        tpath = f"{tdir}/train.json"
        ckpt = f"{tdir}/detector.pth"
        if not (os.path.exists(tpath) and os.path.exists(ckpt)):
            print(f"[bỏ qua] {tag}: thiếu train.json hoặc detector.pth", flush=True)
            continue
        train_meta = json.load(open(tpath, encoding="utf-8"))
        gens = train_meta.get("generators_used") or []
        rows = [m for m in all_maps if m["split"] == split and m["generator"] in ("real", *gens)]
        if limit:
            keep_real = [m for m in rows if m["label"] == 0][:limit]
            keep_fake = [m for m in rows if m["label"] == 1][:limit]
            rows = keep_real + keep_fake
        if not rows or len({m["label"] for m in rows}) < 2:
            print(f"[bỏ qua] {tag}: split {split!r} không có đủ hai lớp", flush=True)
            continue

        model = ResNetDetector(resnet50()).to("cuda")
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        detector_sha = _sha256(ckpt)
        logits, labels = _score(model, [m["path"] for m in rows], [m["label"] for m in rows],
                                crop, "cuda")
        probs = [float(torch.sigmoid(torch.tensor(z))) for z in logits]

        thr_raw = train_meta.get("threshold_from_val", 0.0)
        space = train_meta.get("threshold_space", "logit")
        if space == "logit":
            thr = float(torch.sigmoid(torch.tensor(float(thr_raw))))
        else:                              # detector cũ đã ghi sẵn xác suất
            thr = float(thr_raw)
        src = train_meta.get("threshold_source")
        n_val = int((train_meta.get("split") or {}).get("n_val") or 0)
        gate = "passed" if src == "val" and n_val >= 20 else "failed"
        ev = M.evaluate(probs, labels, threshold=thr, seed=3407)

        samples = {}
        for m, p in zip(rows, probs):
            # đúng khoá `make_gallery` sinh ra: f"{prefix}{cls}_{stem}"
            samples[f"{m['generator']}_{os.path.basename(m['path'])[:-4]}"] = {
                "p_fake": round(p, 6),
                "gate": gate,
                "threshold": round(thr, 6),
                "threshold_split": "val",
            }

        payload = {
            "tag": tag,
            "split": split,
            # Dấu vân tay của detector **đã chấm**. `::report` so với `detector.pth`
            # hiện tại; lệch nghĩa là `train` đã chạy lại sau `evaluate`, và điểm
            # trong `scores.json` là của một mô hình khác — hiện nó cạnh ACC mới là
            # ghép hai kết quả khác nhau thành một dòng bảng.
            "detector_sha256": detector_sha,
            "generators_used": gens,
            "threshold_raw": thr_raw,
            "threshold_space": space,
            "threshold_prob": round(thr, 6),
            "threshold_source": src,
            "n_val": n_val,
            "gate": gate,
            "metrics": ev,
            "crop": crop,
            "honesty": (
                "chấm trên map **tự tính** (R1) bằng detector train from scratch; "
                "không so được giá trị tuyệt đối với Table 2 của bài báo "
                "(khác init, khác số ảnh train, khác generator)."
            ),
        }
        write_json(f"{OUT}/eval/{tag}/eval.json", payload)
        write_json(f"{OUT}/eval/{tag}/scores.json", {"samples": samples})
        results[tag] = {"acc": ev["acc"], "ap": ev["ap"], "auc": ev["auc"],
                        "acc_ci95": ev["acc_ci95"], "n": ev["n"], "gate": gate,
                        "detector_sha256": detector_sha, "split": split,
                        "threshold": round(thr, 6)}
        print(f"[{tag}] {split}: acc {ev['acc']:.3f} "
              f"[{ev['acc_ci95'][0]:.3f}, {ev['acc_ci95'][1]:.3f}] · "
              f"AP {ev['ap']:.3f} · n {ev['n']} · gate {gate}", flush=True)
    vol.commit()
    _write_eval_ledger(results)
    return {"evaluated": sorted(results), "split": split, "by_tag": results}


def _write_eval_ledger(results: dict[str, dict]) -> None:
    """Gộp điểm đã chấm vào `runs/eval.json` — dùng cho `::report` và cho bảng báo cáo."""
    import json  # noqa: PLC0415

    from .pipeline import write_json  # noqa: PLC0415

    path = f"{OUT}/eval.json"
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    by_tag = dict(old.get("by_tag") or {})
    by_tag.update(results)
    write_json(path, {"by_tag": by_tag,
                      "note": "metric chấm trên map tự tính (R1); xem runs/eval/<tag>/eval.json"})


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


def _report_warnings(inventory: dict, dire_json: dict, train_by_gen: dict,
                     eval_by_gen: dict, skipped: dict | None = None) -> list[str]:
    """Cảnh báo **suy ra từ số liệu**, không phải câu chép tay.

    Bản trước hardcode "ACC trên ~200 ảnh test" trong khi lượt chạy thật có
    `n = 1000`: một câu cảnh báo sai số liệu còn tệ hơn không có cảnh báo, vì nó
    dạy người đọc bỏ qua cả mục.
    """
    out: list[str] = []

    # 1. `fake/` còn sót ⇒ nhãn `fake` sẽ lẫn với generator thật khi train.
    leftover = sorted(k for k in (inventory.get("by_split_generator") or {}) if k.endswith("/fake"))
    if leftover:
        out.append(
            f"còn thư mục cũ {leftover} ({sum((inventory['by_split_generator'] or {})[k] for k in leftover)} map) "
            "— chạy `::relabel --apply` trước khi train, nếu không `fake` trộn với generator thật"
        )

    # 2. CI không phủ điểm ước lượng. Sau khi sửa `metrics.evaluate` điều này không
    #    còn xảy ra; nếu vẫn thấy thì đó là số cũ (ghi trước bản sửa) — đừng trích dẫn.
    for tag, row in sorted({**train_by_gen, **eval_by_gen}.items()):
        acc, ci = row.get("acc"), row.get("acc_ci95")
        if isinstance(acc, (int, float)) and isinstance(ci, (list, tuple)) and len(ci) == 2:
            if not (ci[0] <= acc <= ci[1]):
                out.append(
                    f"{tag}: acc {acc:.3f} nằm **ngoài** CI {ci} — số ghi bởi bản cũ "
                    "(CI tính tại 0,5, ACC tại ngưỡng val); chạy lại `::train`/`::evaluate`"
                )

    # 3. Ngưỡng quá sát 0,5 ⇒ detector gần như không phân biệt được; ACC khi đó là
    #    hệ quả của một ngưỡng may mắn, không phải của chất lượng map.
    for tag, row in sorted(train_by_gen.items()):
        acc, acc05 = row.get("acc"), row.get("acc_at_0.5")
        if isinstance(acc, (int, float)) and isinstance(acc05, (int, float)) and abs(acc - acc05) > 0.10:
            out.append(
                f"{tag}: acc tại ngưỡng {acc:.3f} lệch acc@0,5 {acc05:.3f} > 10 điểm "
                "— detector chưa hiệu chuẩn; đọc ACC kèm AP/AUC, đừng đọc riêng ACC"
            )

    # 3b. Dòng "kết quả" thực chất là bộ phân loại hằng số. `acc = 0,500` tuyệt đối
    #     + AUC dưới 0,5 là chữ ký của việc train trên tập một lớp (hoặc mô hình
    #     sụp về một phía); in nó như một dòng Table 2 là trình bày rác thành kết quả.
    for tag, row in sorted(train_by_gen.items()):
        acc, auc = row.get("acc"), row.get("auc")
        if not isinstance(acc, (int, float)) or not isinstance(auc, (int, float)):
            continue
        if abs(acc - 0.5) < 1e-9 and auc < 0.55:
            out.append(
                f"{tag}: acc đúng 0,500 và AUC {auc:.3f} — mô hình dự đoán **một lớp** "
                "(train trên tập chỉ có một lớp, hoặc sụp về hằng số). Dòng này KHÔNG "
                "phải kết quả; chạy lại với `::train --generator "
                f"{tag[4:] if tag.startswith('gen-') else tag} --split-mode hash` "
                "nếu dữ liệu không có map ở split `train`"
            )
        elif auc < 0.5:
            out.append(f"{tag}: AUC {auc:.3f} < 0,5 — xếp hạng **ngược**; "
                       "đọc ACC của dòng này là vô nghĩa")

    # 3c. Generator bị bỏ qua ở lượt train gần nhất nhưng vẫn còn dòng cũ trong bảng.
    for tag, why in sorted((skipped or {}).items()):
        if tag in train_by_gen:
            out.append(f"{tag}: bị bỏ qua ở lượt train gần nhất ({str(why).split(':', 1)[-1].strip()[:120]}…) "
                       "— dòng trong bảng là của lượt CŨ")

    # 4. Cỡ mẫu thật, thay cho con số 200 chép tay.
    ns = [row.get("n") for row in eval_by_gen.values() if isinstance(row.get("n"), int)]
    if ns:
        n = min(ns)
        # nửa độ rộng CI 95 % cho tỉ lệ quanh 0,8: 1,96·√(p(1−p)/n) ≈ 0,78/√n
        out.append(f"n nhỏ nhất trong bảng = {n} ⇒ nửa độ rộng CI 95 % ≈ ±{0.78 / max(n, 1) ** 0.5 * 100:.1f} điểm ACC")

    # 4b. `eval.json` chấm bằng một detector khác với `detector.pth` hiện tại.
    for tag, row in sorted(eval_by_gen.items()):
        sha = row.get("detector_sha256")
        cur = _sha256(f"{OUT}/train/{tag}/detector.pth")
        if sha and cur and sha != cur:
            out.append(
                f"{tag}: `eval.json` chấm bằng detector {sha[:12]}… nhưng `detector.pth` "
                f"hiện tại là {cur[:12]}… — điểm đó là của mô hình CŨ; chạy lại `::evaluate`"
            )

    if train_by_gen:
        out.append("detector khởi tạo ngẫu nhiên (`init=scratch`), không ImageNet như bài báo")
    gens = sorted({str(r.get("n_fake")) for r in train_by_gen.values()})
    if len(gens) == 1 and list(train_by_gen) == ["gen-adm"]:
        out.append("một checkpoint ADM duy nhất ⇒ chỉ đo in-distribution (ADM), không đo cross-model")
    return out


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def gallery(n: int = 8, steps: int = 20, split: str = "test", label: str = "R1+R2",
            classes: str = "", force: bool = False, with_scores: bool = False,
            dry_run: bool = False) -> dict:
    """Dựng `platforms/hf-spaces/gallery/` **ngay trên Volume**, từ map + ảnh đã có.

    Vì sao không chạy ở máy local: map nằm trên Volume (5 900 `.npy` ≈ 800 MB), còn
    gallery chỉ là ~64 mẫu (~13 MB). Tải lên để dựng rồi tải xuống là hai lần di
    chuyển 800 MB qua mạng nhà, và trên Windows còn thêm lớp lỗi đường dẫn
    (`maps\test` không tồn tại ở máy). Ở đây mọi thứ đã nằm cạnh nhau.

    ::

        modal run -m dire_zero.modal_app::gallery --n 8 --steps 20
        modal volume get dire-zero gallery ./platforms/hf-spaces/gallery

    ``--with-scores`` chỉ dùng được khi `::evaluate` đã chạy **và** ngưỡng của nó
    khớp detector hiện tại. Điều kiện thứ hai kiểm ngay tại đây bằng sha256: điểm
    của một mô hình cũ đi kèm ảnh của mô hình mới là một dòng bảng ghép từ hai kết
    quả khác nhau.
    """
    import importlib.util  # noqa: PLC0415
    import types  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("make_gallery", "/root/tools/make_gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]

    # Lớp lấy từ **map đã có**, không phải từ cây dữ liệu: gallery cần `.npy`, và
    # một generator có ảnh nhưng chưa có map thì không dựng được mẫu nào.
    have = sorted(d for d in os.listdir(f"{MAPS}/{split}") if os.path.isdir(f"{MAPS}/{split}/{d}"))
    wanted = [c.strip() for c in classes.split(",") if c.strip()] or have
    maps_root = f"{MAPS}/{split}"
    orig_root = f"{DATA}/{split}"

    scores_path = ""
    thr = 0.5
    if with_scores:
        tags = sorted((_read_json(f"{OUT}/train.json").get("by_generator") or {}))
        if not tags:
            raise FileNotFoundError("chưa train detector nào — chạy `::train` rồi `::evaluate`")
        # Gom điểm của mọi tag đã chấm, mỗi tag một tệp; gộp theo tên mẫu.
        merged: dict[str, dict] = {}
        used: list[str] = []
        for tag in tags:
            ev_path = f"{OUT}/eval/{tag}/eval.json"
            sc_path = f"{OUT}/eval/{tag}/scores.json"
            if not (os.path.exists(ev_path) and os.path.exists(sc_path)):
                continue
            ev = _read_json(ev_path)
            cur = _sha256(f"{OUT}/train/{tag}/detector.pth")
            if ev.get("detector_sha256") and cur and ev["detector_sha256"] != cur:
                print(f"[bỏ qua] {tag}: eval.json chấm bằng detector cũ — chạy lại `::evaluate`",
                      flush=True)
                continue
            if ev.get("gate") != "passed":
                print(f"[bỏ qua] {tag}: gate = {ev.get('gate')!r} (ngưỡng không đến từ val)",
                      flush=True)
                continue
            merged.update(_read_json(sc_path).get("samples") or {})
            used.append(tag)
            thr = float(ev.get("threshold_prob", 0.5))
        if not merged:
            raise FileNotFoundError(
                "không có `scores.json` nào dùng được — chạy `::evaluate` trước, "
                "hoặc bỏ `--with-scores` để dựng gallery không kèm điểm"
            )
        scores_path = f"{OUT}/gallery_scores.json"
        from .pipeline import write_json  # noqa: PLC0415

        write_json(scores_path, {"samples": merged, "tags": used})
        print(f"[điểm] {len(merged)} khoá từ {used}, ngưỡng {thr:.4f}", flush=True)

    args = types.SimpleNamespace(
        maps=maps_root, orig=orig_root, out=f"{V.MOUNT}/gallery", n=n, steps=steps,
        split=split, label=label, map_source=f"{split}/*.npy · ::dire --steps {steps}",
        scores=scores_path, gate="passed" if with_scores else "failed",
        threshold=thr, gamma=0.5, prefix="", dry_run=dry_run, force=force,
    )
    existing = len(glob_many(f"{V.MOUNT}/gallery/*.json"))
    if existing and not force and not dry_run:
        raise FileExistsError(
            f"{V.MOUNT}/gallery đã có {existing} mẫu — dùng `--force` để ghi đè"
        )
    rows = mod.build(args)
    made = [r for r in rows if "_skipped" not in r]
    skipped = [s for r in rows if "_skipped" in r for s in r["_skipped"]]
    vol.commit()
    for s in skipped[:10]:
        print(f"[bỏ qua] {s}", flush=True)
    print(f"[gallery] {len(made)} mẫu → {V.MOUNT}/gallery"
          + (" (dry-run)" if dry_run else ""), flush=True)
    print("tải về: modal volume get dire-zero gallery ./platforms/hf-spaces/gallery", flush=True)
    return {"n": len(made), "skipped": skipped, "classes": wanted, "with_scores": bool(scores_path),
            "out": f"{V.MOUNT}/gallery", "fetch": "modal volume get dire-zero gallery ./platforms/hf-spaces/gallery"}


def glob_many(pattern: str) -> list[str]:
    import glob  # noqa: PLC0415

    return glob.glob(pattern)


@app.function(volumes=VOLUMES, gpu="A10G", timeout=1800)
def report() -> dict:
    """Gom `gen/dire/train` thành một báo cáo có nhãn mức tái lập và cảnh báo."""
    import json  # noqa: PLC0415

    from .pipeline import write_json

    def read(name: str) -> dict:
        p = f"{MAPS}/{name}.json" if name == "dire" else f"{OUT}/{name}.json"
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    dire_json = read("dire")
    train_json = read("train")
    eval_json = _read_json(f"{OUT}/eval.json") or {}
    inventory = _map_inventory(MAPS)
    train_by_gen = train_json.get("by_generator", {})
    eval_by_gen = eval_json.get("by_tag", {})
    payload = {
        "label": _read_json(DATA_CARD).get("label", "R1+R2"),
        "what_is_ours": ["DIRE map", "detector", "ảnh sinh"],
        "what_is_not": ["trọng số ADM (OpenAI)", "LSUN-Bedroom (dữ liệu thật)", "kiến trúc U-Net/ResNet"],
        "gen": {k: v for k, v in read("gen").items() if k in ("n", "steps", "nfe_total", "seconds")},
        "dire": {
            **{k: v for k, v in dire_json.items()
               if k in ("n", "steps", "steps_mixed", "nfe_total", "seconds")},
            # `n` là số **map phân biệt** (`len(by_path)`), còn `n_new`/`n_reused` là
            # **tổng cộng dồn qua mọi lượt chạy**. Một map tính ở lượt trước rồi dùng
            # lại ở lượt sau được đếm ở cả hai phía, nên `n_new + n_reused > n` là
            # bình thường. Bản trước in ba số cạnh nhau mà không nói gì, đọc ra như
            # một phép cộng sai.
            "n_distinct_maps": dire_json.get("n"),
            "n_new_cumulative": dire_json.get("n_new"),
            "n_reused_cumulative": dire_json.get("n_reused"),
            "n_runs": len(dire_json.get("runs") or []),
            "n_note": ("n_distinct_maps = số `.npy` phân biệt; n_new/n_reused cộng dồn mọi "
                       "lượt chạy nên tổng của chúng lớn hơn n_distinct_maps là bình thường"),
        },
        # Tồn kho map **thực tế trên đĩa**, không phải điều `dire.json` tự khai:
        # sau khi thêm generator, câu hỏi đúng là "map nào đã có", và một thư mục
        # `fake/` còn sót từ bản cũ sẽ lộ ra ở đây chứ không lộ ở `dire.json`.
        "maps": inventory,
        "data": _read_json(DATA_CARD),
        "train": {k: v for k, v in train_json.items()
                  if k in ("test", "best_epoch", "threshold_from_val", "threshold_space",
                           "threshold_source", "init", "latest", "split")},
        # **Bảng** của Table 2: mỗi generator một detector riêng, không phải một
        # con số gộp. Thiếu mục nào nghĩa là generator đó chưa train.
        "train_by_generator": train_by_gen,
        "skipped": train_json.get("skipped") or {},
        # Điểm đã chấm trên split chấm (mỗi tag một detector) — có `acc_ci95` để
        # không ai đọc chênh lệch 2 điểm thành thứ hạng.
        "eval_by_generator": eval_by_gen,
        "bench": read("bench"),
        "warnings": _report_warnings(inventory, dire_json, train_by_gen, eval_by_gen,
                                     skipped=train_json.get("skipped") or {}),
    }
    write_json(f"{OUT}/report.json", payload)
    vol.commit()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - tiện cho CLI
    argv = argv or sys.argv[1:]
    print(__doc__)
    return 0
