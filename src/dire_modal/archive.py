"""Upload .tar.gz từ Volume /raw → R2 cold archive.
Chạy 1 lần duy nhất sau khi download xong. Không dùng cho hot path.
"""

import os
from pathlib import Path

import modal

from .common import app, cpu_image, vol_raw, r2_secret


def _r2_client():
    import boto3

    account = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


@app.function(
    image=cpu_image,
    secrets=[r2_secret],
    volumes={"/raw": vol_raw},
    timeout=12 * 3600,
    cpu=1.0,
)
def push_all(prefix: str = "diffusion-forensics/v1") -> None:
    """modal run -m dire_modal.archive::push_all"""
    from boto3.s3.transfer import TransferConfig

    s3 = _r2_client()
    bucket = os.environ["R2_BUCKET"]
    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    files = sorted(Path("/raw").rglob("*.tar.gz")) + sorted(Path("/raw").rglob("*.pth"))
    print(f"[plan] push {len(files)} files → s3://{bucket}/{prefix}/")

    for p in files:
        key = f"{prefix}/{p.relative_to('/raw').as_posix()}"
        # Idempotent: skip nếu đã tồn tại cùng size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] == p.stat().st_size:
                print(f"[SKIP] {key}")
                continue
        except s3.exceptions.ClientError:
            pass
        print(f"[PUT ] {key}  ({p.stat().st_size/1e9:.2f} GB)")
        s3.upload_file(str(p), bucket, key, Config=cfg)
    print("[done]")


@app.function(
    image=cpu_image,
    secrets=[r2_secret],
    volumes={"/raw": vol_raw},
    timeout=12 * 3600,
)
def pull_all(prefix: str = "diffusion-forensics/v1") -> None:
    """Khôi phục /raw từ R2 (dùng khi tạo Volume mới).
    modal run -m dire_modal.archive::pull_all"""
    s3 = _r2_client()
    bucket = os.environ["R2_BUCKET"]
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            local = Path("/raw") / rel
            if local.exists() and local.stat().st_size == obj["Size"]:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"[GET ] {key}")
            s3.download_file(bucket, key, str(local))
    vol_raw.commit()
