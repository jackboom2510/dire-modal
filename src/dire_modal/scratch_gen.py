"""R3 + R1: sinh ảnh bằng ADM, rồi tự tính DIRE map trên ảnh đó và trên LSUN thật.

Hai bước, cố ý tách
-------------------
* **Bước sinh (R3)** — ảnh "giả" do mình tạo, seed mình chọn. Ghi `gen.json`
  kèm sha256 checkpoint, sampler, số bước, danh sách seed. Thiếu một trường là
  người khác không dựng lại được, và cổng G3 đỏ.
* **Bước DIRE (R1)** — tính map cho **cả hai** phía (ảnh thật LSUN và ảnh mình
  sinh) bằng đúng một mô hình, một lịch, một `S`. Nếu hai phía dùng hai cấu
  hình khác nhau thì DIRE đang đo cấu hình, không đo ảnh.

Đây là điểm khác bản chất so với `01_repro.sh`: bản đó nạp map **đã phát hành**
(`/raw/dire/**`) và vì thế chỉ chấm lại. Ở đây `assert_no_published` chặn ngay
từ danh sách đầu vào.

Chạy::

    modal run -m dire_modal.scratch_gen::generate --n 500 --seed-base 1000
    modal run -m dire_modal.scratch_gen::dire --n 500 --side both
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from dire_core.scratch import (
    assert_no_published,
    check_gen_meta,
    sha256_file,
    split_seeds,
    write_json,
)

from .common import VOLUMES, app, gpu_image
from .ddim import Schedule, dire_map, load_adm, sample, to_uint8

CKPT = "/raw/ckpt/adm_lsun.pt"
GEN_ROOT = "/raw/scratch/gen"
MAP_ROOT = "/raw/scratch/maps"
REAL_ROOT = "/raw/scratch/real"


def _save_png(arr, path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(arr).save(path, format="PNG", optimize=False)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A10G", timeout=3600)
def generate(
    n: int = 200,
    seed_base: int = 1000,
    steps: int = 20,
    ckpt: str = CKPT,
    out_root: str = GEN_ROOT,
    generator: str = "adm",
) -> dict:
    """Sinh `n` ảnh 256×256 từ ADM, lưu PNG + `gen.json`."""
    if generator != "adm":
        raise NotImplementedError(
            f"generator={generator!r}: đường scratch mới cài ADM. Muốn thêm "
            "iDDPM/PNDM cần checkpoint tương ứng + loader riêng; đừng trộn "
            "ảnh của generator khác vào cùng một lần chạy mà không ghi metadata."
        )
    if not Path(ckpt).exists():
        raise SystemExit(f"[!] thiếu checkpoint {ckpt} — chạy tools/scratch_repro.sh --stage fetch")
    assert_no_published([ckpt])

    out = Path(out_root) / generator
    out.mkdir(parents=True, exist_ok=True)
    seeds = split_seeds(n, max(1, n // 5), base=seed_base)
    all_seeds = seeds["train"] + seeds["test"]

    model, device = load_adm(ckpt, "cuda")
    schedule = Schedule(1000, steps)
    written, nfe_total = 0, 0
    for i, seed in enumerate(all_seeds):
        x0, nfe = sample(model, (1, 3, 256, 256), schedule, device, seed)
        nfe_total += nfe
        arr = to_uint8(x0).squeeze(0).permute(1, 2, 0).numpy()
        _save_png(arr, out / f"{i:06d}_s{seed}.png")
        written += 1
        if written % 25 == 0:
            print(f"[sinh] {written}/{len(all_seeds)}")

    meta = {
        "generator": generator,
        "ckpt": ckpt,
        "ckpt_sha256": sha256_file(ckpt),
        "sampler": "ddim",
        "steps": steps,
        "seeds": all_seeds,
        "split": {"train": seeds["train"], "test": seeds["test"]},
        "n_images": written,
        "resolution": 256,
        "prompt": None,  # ADM LSUN là unconditional
        "nfe": nfe_total,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    missing = check_gen_meta(meta)
    if missing:
        raise AssertionError(f"gen.json thiếu trường (G3): {missing}")
    write_json(out / "gen.json", meta)
    print(f"[xong] {written} ảnh → {out}  ({nfe_total} NFE)")
    return {"n": written, "out": str(out), "nfe": nfe_total}


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A10G", timeout=3600)
def dire(
    limit: int = 200,
    steps: int = 20,
    ckpt: str = CKPT,
    gen_root: str = GEN_ROOT,
    real_root: str = REAL_ROOT,
    out_root: str = MAP_ROOT,
) -> dict:
    """Tính DIRE map cho ảnh thật + ảnh sinh, lưu `.npy` float16.

    Map lưu ở thang **gốc** `[0, 2]` (không nhân 4, không clip) để phép so sánh
    sau này không bị khoá vào một quy ước hiển thị.
    """
    assert_no_published([ckpt, gen_root, real_root])

    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model, device = load_adm(ckpt, "cuda")
    schedule = Schedule(1000, steps)

    def _one(path: Path) -> tuple:
        img = Image.open(path).convert("RGB")
        if img.size != (256, 256):
            # Ảnh thật LSUN có sẵn 256²; ảnh khác phải bicubic về 256 trước khi
            # vào mô hình — đúng `reshape_image` của tác giả.
            img = img.resize((256, 256), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        d, recon, nfe = dire_map(model, x, schedule, device)
        return d[0].to(torch.float16).cpu().numpy(), recon, nfe

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    done, nfe_total, n_images, t0 = {}, 0, 0, time.perf_counter()
    for side, root in (("real", Path(real_root)), ("fake", Path(gen_root) / "adm")):
        files = sorted(root.rglob("*.png"))[:limit]
        if not files:
            print(f"[bỏ qua] {side}: không có ảnh trong {root}")
            continue
        (out / side).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            d, _recon, nfe = _one(f)
            nfe_total += nfe
            # giữ tên gốc để seed trong gen.json ↔ file map không bị lạc
            np.save(out / side / f"{f.stem}.npy", d)
            n_images += 1
        done[side] = len(files)
        print(f"[DIRE] {side}: {len(files)} map")

    elapsed = max(1e-6, time.perf_counter() - t0)
    s_per_image = elapsed / max(1, n_images)
    # Con số DUY NHẤT biến mọi ước lượng tiền thành số đo (cổng G8).
    print(f"[DIRE] s_per_image={s_per_image:.3f} "
          f"(sec/step={s_per_image / (2 * steps):.4f}, {n_images} ảnh)")
    meta = {
        "mode": "scratch",
        "s_per_image": round(s_per_image, 4),
        "sec_per_step": round(s_per_image / (2 * steps), 6),
        "ddim_steps": steps,
        "nfe_per_image": 2 * steps,
        "nfe_total": nfe_total,
        "counts": done,
        "scale": "|x0 - recon| in [0, 2], float16, không khuếch đại",
        "ckpt_sha256": sha256_file(ckpt),
    }
    write_json(out / "dire.json", meta)
    return meta
