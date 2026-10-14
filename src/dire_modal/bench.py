"""Benchmark thật cho dự án DIRE: STORAGE (Volume vs R2) và COMPUTE (DIRE proxy).

Trước đây file này trùng hoàn toàn `pack.py` và `scripts/03_pack_shards.sh`
gọi `bench::throughput` — một hàm KHÔNG tồn tại. Đã sửa: mỗi benchmark là một
function riêng, in kết quả JSON, và `main()` chạy những cái nào môi trường cho phép.

Chạy
----
    modal run -m dire_modal.bench::main                    # storage + compute nếu có GPU
    modal run -m dire_modal.bench::volume_seq_read --path /shards/train/lsun_bedroom/adm/shard-000000.tar
    modal run -m dire_modal.bench::dire_throughput --n 100 --steps 20

Ghi chú thiết kế
----------------
* Số đo được ghi vào `manifests/bench_<tên>_<YYYYmmdd-HHMM>.json` (không commit .tsv;
  JSON kết quả nên commit khi đã chốt — xem `.gitignore`).
* `dire_throughput` là **PROXY**: dựng một UNet cỡ ~552M tham số (bằng ADM 256×256 của
  guided-diffusion) và đo ảnh/giây với `steps` lần forward/ảnh. Đây KHÔNG phải số chính
  thức của ADM — sai số ước tính ±30%. Khi `compute_dire.py` được port xong, dùng
  `--impl real` để đo trên đúng model.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from .common import (
    app,
    cpu_image,
    gpu_image,
    r2_secret,
    vol_raw,
    vol_shards,
)

# ---------------------------------------------------------------------------
# R2 mount — cấu hình đọc từ ENV Ở MÁY LOCAL lúc `modal run`.
# `scripts/bench_storage.sh` export R2_ACCOUNT_ID/R2_BUCKET từ .env trước khi chạy.
# Credentials vẫn lấy từ Modal Secret `r2-cred` lúc runtime (không nhúng vào code).
# ---------------------------------------------------------------------------
_R2_ACCOUNT = os.environ.get("R2_ACCOUNT_ID", "")
_R2_BUCKET = os.environ.get("R2_BUCKET", "dire-archive")
R2_MOUNT_PATH = "/r2"
R2_MOUNT_ENABLED = bool(_R2_ACCOUNT)

if R2_MOUNT_ENABLED:
    r2_mount = modal.CloudBucketMount(
        bucket_name=_R2_BUCKET,
        secret=r2_secret,
        bucket_endpoint_url=f"https://{_R2_ACCOUNT}.r2.cloudflarestorage.com",
        read_only=True,
    )
else:  # không có R2_ACCOUNT_ID → bỏ qua benchmark mount, pipeline vẫn chạy bình thường
    r2_mount = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")


def _emit(name: str, payload: dict) -> dict:
    """In JSON + ghi vào Volume /shards/_bench/ và trả về payload cho caller."""
    payload = {"bench": name, "ts_utc": datetime.now(timezone.utc).isoformat(), **payload}
    print("BENCH_RESULT " + json.dumps(payload, ensure_ascii=False))
    try:
        out_dir = Path("/shards/_bench")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"bench_{name}_{_now_tag()}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        vol_shards.commit()
    except Exception as exc:  # /shards có thể chưa mount ở một số function
        print(f"[warn] không ghi được kết quả vào /shards: {exc}")
    return payload


def _human(bps: float) -> str:
    for unit in ("B/s", "KiB/s", "MiB/s", "GiB/s"):
        if bps < 1024:
            return f"{bps:.1f} {unit}"
        bps /= 1024
    return f"{bps:.1f} TiB/s"


def _read_chunked(path: Path, chunk_bytes: int, max_bytes: int | None = None) -> tuple[float, int]:
    """Đọc tuần tự theo chunk, trả (giây, byte đã đọc)."""
    total = 0
    t0 = time.perf_counter()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            total += len(buf)
            if max_bytes and total >= max_bytes:
                break
    return time.perf_counter() - t0, total


# ===========================================================================
# STORAGE BENCHMARKS
# ===========================================================================
@app.function(
    image=cpu_image,
    volumes={"/shards": vol_shards},
    timeout=1800,
    cpu=2.0,
)
def volume_seq_read(path: str, chunk_mb: int = 16, max_mb: int = 2048, repeats: int = 3) -> dict:
    """Đọc tuần tự 1 file lớn trên Volume (đại diện: shard WebDataset / .tar.gz)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} không tồn tại trên /shards")
    size_mb = p.stat().st_size / 1e6
    rates = []
    for i in range(repeats):
        dt, nbytes = _read_chunked(p, chunk_mb * 1024 * 1024, max_mb * 1024 * 1024)
        rates.append(nbytes / dt)
        print(f"  lần {i + 1}: {nbytes / 1e6:.0f} MB trong {dt:.2f}s → {_human(nbytes / dt)}")
    return _emit("volume_seq_read", {
        "path": path, "file_mb": round(size_mb, 1), "repeats": repeats,
        "median_mb_s": round(statistics.median(rates) / 1e6, 1),
        "max_mb_s": round(max(rates) / 1e6, 1),
        "gb_per_s": round(statistics.median(rates) / 1e9, 3),
    })


@app.function(
    image=cpu_image,
    volumes={"/shards": vol_shards},
    timeout=1800,
    cpu=2.0,
)
def volume_small_read(dir_path: str = "/shards/test", n_files: int = 200, repeats: int = 1) -> dict:
    """Latency đọc file nhỏ ngẫu nhiên trên Volume (mô phỏng DataLoader đọc nhiều file)."""
    files = [f for f in Path(dir_path).rglob("*") if f.is_file()][:20000]
    if not files:
        raise FileNotFoundError(f"không có file nào dưới {dir_path}")
    sample = random.sample(files, min(n_files, len(files)))
    lat_ms, bytes_total = [], 0
    for r in range(repeats):
        for f in sample:
            t0 = time.perf_counter()
            bytes_total += len(f.read_bytes())
            lat_ms.append((time.perf_counter() - t0) * 1000)
    lat_ms.sort()
    p50 = lat_ms[len(lat_ms) // 2]
    p95 = lat_ms[int(len(lat_ms) * 0.95)]
    return _emit("volume_small_read", {
        "dir": dir_path, "n_files": len(sample), "total_requests": len(lat_ms),
        "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
        "files_per_s": round(1000 / p50, 1),
        "avg_kb": round(bytes_total / len(lat_ms) / 1024, 1),
    })


@app.function(
    image=cpu_image,
    volumes={"/shards": vol_shards},
    timeout=1800,
    cpu=2.0,
)
def volume_write(size_mb: int = 512, chunk_mb: int = 16) -> dict:
    """Throughput ghi lên Volume + thời gian commit (commit là chi phí riêng của Modal)."""
    out = Path("/shards/_bench") / f"write_test_{size_mb}mb.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\0" * (chunk_mb * 1024 * 1024)
    t0 = time.perf_counter()
    with out.open("wb") as fh:
        for _ in range(size_mb // chunk_mb):
            fh.write(payload)
    write_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    vol_shards.commit()
    commit_s = time.perf_counter() - t1
    out.unlink(missing_ok=True)
    return _emit("volume_write", {
        "size_mb": size_mb,
        "write_mb_s": round(size_mb / write_s, 1),
        "commit_s": round(commit_s, 2),
        "note": "commit là chi phí riêng của Modal Volume — đừng commit mỗi file nhỏ",
    })


@app.function(
    image=cpu_image,
    secrets=[r2_secret],
    volumes={R2_MOUNT_PATH: r2_mount} if r2_mount is not None else {},
    timeout=1800,
    cpu=2.0,
)
def r2_mount_seq_read(key: str = "diffusion-forensics/v1/dire/train/lsun_bedroom/adm.tar.gz",
                      chunk_mb: int = 16, max_mb: int = 2048, repeats: int = 3) -> dict:
    """Đọc tuần tự qua CloudBucketMount (mountpoint-s3) — so sánh với Volume."""
    if r2_mount is None:
        raise RuntimeError(
            "Chưa cấu hình R2: export R2_ACCOUNT_ID=<account id> trước khi `modal run` "
            "(scripts/bench_storage.sh tự đọc từ .env)."
        )
    p = Path(R2_MOUNT_PATH) / key
    if not p.exists():
        raise FileNotFoundError(f"{p} không tồn tại trong bucket {_R2_BUCKET}")
    rates = []
    for i in range(repeats):
        dt, nbytes = _read_chunked(p, chunk_mb * 1024 * 1024, max_mb * 1024 * 1024)
        rates.append(nbytes / dt)
        print(f"  lần {i + 1}: {nbytes / 1e6:.0f} MB trong {dt:.2f}s → {_human(nbytes / dt)}")
    return _emit("r2_mount_seq_read", {
        "key": key, "repeats": repeats,
        "median_mb_s": round(statistics.median(rates) / 1e6, 1),
        "max_mb_s": round(max(rates) / 1e6, 1),
        "gb_per_s": round(statistics.median(rates) / 1e9, 3),
        "caveat": "mountpoint-s3 tối ưu cho đọc file lớn tuần tự; không suy ra được cho small-file",
    })


@app.function(
    image=cpu_image,
    secrets=[r2_secret],
    volumes={R2_MOUNT_PATH: r2_mount} if r2_mount is not None else {},
    timeout=1800,
    cpu=2.0,
)
def r2_mount_small_read(prefix: str = "diffusion-forensics/v1", n_files: int = 100) -> dict:
    """Latency đọc file nhỏ ngẫu nhiên qua R2 mount — mặt yếu của mountpoint-s3."""
    if r2_mount is None:
        raise RuntimeError("Chưa cấu hình R2: export R2_ACCOUNT_ID trước khi `modal run`.")
    base = Path(R2_MOUNT_PATH) / prefix
    files = [f for f in base.rglob("*") if f.is_file()][:5000]
    if not files:
        raise FileNotFoundError(f"không có file dưới {base}")
    sample = random.sample(files, min(n_files, len(files)))
    lat_ms, bytes_total = [], 0
    for f in sample:
        t0 = time.perf_counter()
        bytes_total += len(f.read_bytes())
        lat_ms.append((time.perf_counter() - t0) * 1000)
    lat_ms.sort()
    p50 = lat_ms[len(lat_ms) // 2]
    p95 = lat_ms[int(len(lat_ms) * 0.95)]
    return _emit("r2_mount_small_read", {
        "prefix": prefix, "n_files": len(sample), "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3), "files_per_s": round(1000 / p50, 1),
        "avg_kb": round(bytes_total / len(lat_ms) / 1024, 1),
    })


@app.function(image=cpu_image, secrets=[r2_secret], timeout=3600, cpu=1.0)
def r2_api_io(size_mb: int = 256, prefix: str = "_bench") -> dict:
    """Đo PUT/GET thẳng qua S3 API (boto3) — đường đi thật của `archive.py`."""
    import io
    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ.get("R2_BUCKET", _R2_BUCKET)
    key = f"{prefix}/bench_{size_mb}mb.bin"
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024, max_concurrency=8, use_threads=True)

    buf = io.BytesIO(b"\1" * (size_mb * 1024 * 1024))
    t0 = time.perf_counter()
    s3.upload_fileobj(buf, bucket, key, Config=cfg)
    put_s = time.perf_counter() - t0

    buf2 = io.BytesIO()
    t1 = time.perf_counter()
    s3.download_fileobj(bucket, key, buf2)
    get_s = time.perf_counter() - t1

    s3.delete_object(Bucket=bucket, Key=key)
    return _emit("r2_api_io", {
        "size_mb": size_mb, "bucket": bucket,
        "put_mb_s": round(size_mb / put_s, 1), "get_mb_s": round(size_mb / get_s, 1),
        "note": "multipart 64 MB, 8 luồng — cấu hình giống archive.py",
    })


# ===========================================================================
# COMPUTE BENCHMARK — chi phí DIRE (DDIM inversion)
# ===========================================================================
def _build_proxy_unet(params_m: int):
    """UNet nhỏ-cỡ-lớn, cùng bậc tham số với ADM 256×256 (~552M) của guided-diffusion.

    Mục đích: đo throughput forward ở 256×256 để ước lượng ảnh/giây của DDIM inversion.
    KHÔNG phải ADM thật — xem docstring đầu file.
    """
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))

    class ProxyUNet(nn.Module):
        def __init__(self, widths: tuple[int, ...]):
            super().__init__()
            self.enc = nn.ModuleList()
            cin = 3
            for w in widths:
                self.enc.append(block(cin, w))
                cin = w
            self.mid = block(widths[-1], widths[-1])
            self.dec = nn.ModuleList()
            for w in reversed(widths[:-1]):
                self.dec.append(block(cin, w))
                cin = w
            self.out = nn.Conv2d(cin, 3, 3, padding=1)

        def forward(self, x, t):  # t chỉ để giống chữ ký ADM
            h = x
            for m in self.enc:
                h = m(h)
            h = self.mid(h)
            for m in self.dec:
                h = m(h)
            return self.out(h)

    # tăng dần bề rộng cho đủ gần 552M tham số
    widths = (64, 96, 128, 192, 256, 320)
    model = ProxyUNet(widths)
    n_params = sum(p.numel() for p in model.parameters())
    # scale bề rộng nếu lệch xa mục tiêu (1 lần tinh chỉnh)
    if n_params * 3 < params_m * 1e6:
        model = ProxyUNet(tuple(int(w * 1.6) for w in widths))
        n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


@app.function(image=gpu_image, timeout=3600, gpu="A100")
def dire_throughput(n: int = 100, steps: int = 20, batch: int = 4, params_m: int = 552,
                    impl: str = "proxy") -> dict:
    """Ảnh/giây của pipeline DIRE (DDIM inversion `steps` bước) trên GPU.

    impl="proxy" (mặc định): UNet cỡ ~ADM 256×256 → số ĐO ĐƯỢC nhưng là proxy (±30%).
    impl="real": đòi hỏi compute_dire đã port; sẽ báo lỗi rõ nếu chưa.
    """
    import torch

    if impl == "real":
        raise RuntimeError(
            "impl='real' cần compute_dire.py được port (đang TODO). "
            "Dùng impl='proxy' để có số đo tạm, và chạy lại lệnh này khi đã port xong."
        )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, n_params = _build_proxy_unet(params_m)
    model = model.to(dev).eval()
    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"

    # warmup
    with torch.inference_mode():
        x = torch.randn(batch, 3, 256, 256, device=dev)
        t = torch.full((batch,), 0.5, device=dev)
        for _ in range(2):
            model(x, t)
        if dev == "cuda":
            torch.cuda.synchronize()

    done, t0 = 0, time.perf_counter()
    with torch.inference_mode():
        while done < n:
            b = min(batch, n - done)
            x = torch.randn(b, 3, 256, 256, device=dev)
            t = torch.full((b,), 0.5, device=dev)
            for _ in range(steps):  # `steps` lần forward/ảnh ≈ DDIM-20
                model(x, t)
            done += b
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    img_s = n / elapsed
    per_img_s = 1 / img_s
    est = {
        "200k ảnh": 200_000 * per_img_s / 3600,
        "50k ảnh": 50_000 * per_img_s / 3600,
        "5k ảnh": 5_000 * per_img_s / 3600,
    }
    return _emit("dire_throughput", {
        "impl": "proxy", "gpu": gpu_name, "n_images": n, "steps": steps, "batch": batch,
        "model_params_m": round(n_params / 1e6, 1),
        "images_per_s": round(img_s, 2),
        "sec_per_image": round(per_img_s, 4),
        "est_gpu_hours": {k: round(v, 2) for k, v in est.items()},
        "est_cost_usd_a100_80gb": {k: round(v * 2.50, 2) for k, v in est.items()},
        "caveat": ("PROXY: UNet ~%dM tham số cùng bậc với ADM 256×256 (~552M). Sai số ±30%%. "
                   "Chạy lại với impl='real' sau khi port compute_dire.py." % round(n_params / 1e6)),
    })


# ===========================================================================
# ENTRYPOINT
# ===========================================================================
@app.function(image=cpu_image, timeout=300)
def _plan() -> dict:
    has_shards = Path("/shards").exists()
    return {"shards_mounted": has_shards}


@modal.local_entrypoint()
def main(shard_path: str = "", skip_storage: bool = False, with_gpu: bool = True,
         n_images: int = 100, steps: int = 20) -> None:
    """Chạy các benchmark khả dụng. `modal run -m dire_modal.bench::main`"""
    print("=" * 72)
    print("BENCH DIRE — storage (Volume/R2) + compute (DIRE proxy)")
    print("=" * 72)

    if not skip_storage:
        print("\n[1/5] Volume: ghi + commit")
        print(json.dumps(volume_write.remote(size_mb=256), ensure_ascii=False))

        print("\n[2/5] Volume: đọc file nhỏ ngẫu nhiên")
        print(json.dumps(volume_small_read.remote(n_files=200), ensure_ascii=False))

        if shard_path:
            print(f"\n[3/5] Volume: đọc tuần tự {shard_path}")
            print(json.dumps(volume_seq_read.remote(path=shard_path), ensure_ascii=False))
        else:
            print("\n[3/5] Bỏ qua đọc tuần tự Volume (truyền --shard-path để đo)")

        if R2_MOUNT_ENABLED:
            print("\n[4/5] R2: PUT/GET qua S3 API")
            try:
                print(json.dumps(r2_api_io.remote(size_mb=256), ensure_ascii=False))
            except Exception as exc:
                print(f"  [warn] benchmark R2 API lỗi (secret/bucket?): {exc}")
        else:
            print("\n[4/5] Bỏ qua R2 (chưa export R2_ACCOUNT_ID)")
    else:
        print("\n[storage] bỏ qua theo yêu cầu")

    if with_gpu:
        print("\n[5/5] GPU: throughput DIRE (proxy)")
        try:
            print(json.dumps(dire_throughput.remote(n=n_images, steps=steps), ensure_ascii=False))
        except Exception as exc:
            print(f"  [warn] benchmark GPU lỗi (quota/GPU?): {exc}")
    else:
        print("\n[gpu] bỏ qua theo yêu cầu (--no-gpu)")

    print("\nXong. Kết quả JSON nằm trong /shards/_bench/ — tải về bằng:")
    print("  modal volume ls dire-shards /_bench")
    print("  modal volume get dire-shards /_bench/<file>.json .")
