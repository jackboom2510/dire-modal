"""Đẩy / kéo dữ liệu giữa Volume `/raw` và các đích ngoài: R2 (S3 API), HF Hub, Kaggle Datasets.

Vai trò từng đích (xem `docs/storage-comparison.md` §5):
  * R2        — kho NHỎ: checkpoint đã chốt + manifest sha256 (10 GB free, egress $0)
  * HF Hub    — mirror CÔNG KHAI chính: dataset (~300 GB/repo, file ≤50 GB), free
  * Kaggle    — mirror CÔNG KHAI phụ: 200 GB/dataset; **tối đa 50 file top-level**

Chạy
----
    modal run -m dire_modal.archive::push_r2        --prefix diffusion-forensics/v1
    modal run -m dire_modal.archive::push_hf        --repo-id <user>/diffusionforensics-dire
    modal run -m dire_modal.archive::push_kaggle    --dataset-id <user>/dire-tier-a --src-subdir dire/train/lsun_bedroom
    modal run -m dire_modal.archive::verify_hf      --repo-id <user>/diffusionforensics-dire

Secrets cần có (xem scripts/setup.sh):
    r2-cred      : R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
    hf-cred      : HF_TOKEN
    kaggle-cred  : KAGGLE_USERNAME, KAGGLE_KEY
"""

import hashlib
import os
from pathlib import Path

import modal

from .common import app, cpu_image, vol_raw, vol_ckpt, r2_secret


def _optional_secret(name: str):
    """Secret không bắt buộc: thiếu thì hàm tương ứng báo lỗi rõ ràng, không làm chết cả app."""
    try:
        return modal.Secret.from_name(name)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        print(f"[warn] secret {name!r} chưa tạo: {exc}")
        return None


hf_secret = _optional_secret("hf-cred")
kaggle_secret = _optional_secret("kaggle-cred")


def _sha256_file(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


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


def _files_under(root: Path, include: tuple[str, ...]) -> list[Path]:
    """Lọc theo ĐUÔI TÊN (endswith) — vì '.tar.gz' không phải là suffix của Path."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith(include))


def _list_hub_files(api, repo_id: str) -> dict[str, dict]:
    """Trả {path: {"size": int, "sha256": str|None}} cho mọi file trong repo HF.

    Chịu được nhiều phiên bản huggingface_hub: ưu tiên `list_repo_tree(expand=True)`
    (có LFS sha256), fallback sang liệt kê tên file nếu API khác.
    """
    out: dict[str, dict] = {}
    try:
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True):
            path = getattr(entry, "path", None)
            if path is None or getattr(entry, "size", None) is None:
                continue  # bỏ qua thư mục
            lfs = getattr(entry, "lfs", None)
            sha = None
            if isinstance(lfs, dict):
                sha = lfs.get("sha256")
            elif lfs is not None:
                sha = getattr(lfs, "sha256", None)
            out[path] = {"size": entry.size, "sha256": sha}
    except TypeError:
        # phiên bản cũ không có `expand`
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            path = getattr(entry, "path", None)
            if path is None or getattr(entry, "size", None) is None:
                continue
            out[path] = {"size": entry.size, "sha256": None}
    except Exception as exc:
        print(f"[i] không liệt kê được repo ({exc}) — coi như repo trống")
    return out


# ===========================================================================
# R2 — kho nhỏ (checkpoint + manifest)
# ===========================================================================
@app.function(image=cpu_image, secrets=[r2_secret], volumes={"/raw": vol_raw, "/ckpt": vol_ckpt},
              timeout=12 * 3600, cpu=1.0)
def push_r2(prefix: str = "diffusion-forensics/v1", include_ckpt: bool = True,
            include_tar: bool = False) -> dict:
    """Đẩy lên R2. Mặc định CHỈ checkpoint + manifest (vì R2 free chỉ 10 GB).

    include_tar=True nếu bạn thật sự muốn đẩy cả .tar.gz (tốn tiền khi >10 GB).
    """
    from boto3.s3.transfer import TransferConfig

    s3 = _r2_client()
    bucket = os.environ["R2_BUCKET"]
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                         multipart_chunksize=64 * 1024 * 1024,
                         max_concurrency=8, use_threads=True)

    targets: list[tuple[Path, str]] = []
    if include_tar:
        for p in sorted(Path("/raw").rglob("*.tar.gz")):
            targets.append((p, f"{prefix}/raw/{p.relative_to('/raw').as_posix()}"))
    for p in sorted(Path("/raw").rglob("*.tsv")):
        targets.append((p, f"{prefix}/manifests/{p.name}"))
    if include_ckpt and Path("/ckpt").exists():
        for p in _files_under(Path("/ckpt"), (".pth", ".pt")):
            targets.append((p, f"{prefix}/ckpt/{p.relative_to('/ckpt').as_posix()}"))

    if not targets:
        raise SystemExit("[!] không có file nào để đẩy — kiểm tra --include-tar/--include-ckpt")

    total = sum(p.stat().st_size for p, _ in targets)
    print(f"[plan] {len(targets)} file · {total / 1e9:.2f} GB → s3://{bucket}/{prefix}/")
    if total > 9e9:
        print(f"[!] CẢNH BÁO: {total / 1e9:.2f} GB > 9 GB — vượt free tier R2 (10 GB), sẽ phát sinh phí $0.015/GB/tháng")

    pushed, skipped = [], []
    for p, key in targets:
        size = p.stat().st_size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] == size:
                skipped.append(key)
                print(f"[SKIP] {key} (đã có, cùng size)")
                continue
        except s3.exceptions.ClientError:
            pass
        print(f"[PUT ] {key}  ({size / 1e9:.2f} GB)")
        s3.upload_file(str(p), bucket, key, Config=cfg)
        pushed.append(key)
    return {"pushed": len(pushed), "skipped": len(skipped), "bytes": total, "prefix": prefix}


@app.function(image=cpu_image, secrets=[r2_secret], volumes={"/raw": vol_raw}, timeout=12 * 3600)
def pull_r2(prefix: str = "diffusion-forensics/v1") -> dict:
    """Khôi phục /raw từ R2 (dùng khi tạo Volume mới)."""
    s3 = _r2_client()
    bucket = os.environ["R2_BUCKET"]
    n = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel.startswith("raw/"):
                continue
            local = Path("/raw") / rel[len("raw/"):]
            if local.exists() and local.stat().st_size == obj["Size"]:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"[GET ] {key}")
            s3.download_file(bucket, key, str(local))
            n += 1
    vol_raw.commit()
    return {"downloaded": n}


# ===========================================================================
# Hugging Face Hub — mirror công khai chính
# ===========================================================================
@app.function(image=cpu_image, secrets=[hf_secret] if hf_secret else [],
              volumes={"/raw": vol_raw}, timeout=12 * 3600, cpu=2.0)
def push_hf(repo_id: str = "", src: str = "/raw", prefix_in_repo: str = "",
            private: bool = False, include: str = ".tar.gz,.tsv,.pth,.pt,.json",
            verify_sha: bool = True) -> dict:
    """Đẩy thư mục lên HF dataset repo (idempotent: skip file đã có cùng SHA256).

    repo_id ví dụ: "jackboom2510/diffusionforensics-dire"
    """
    if not repo_id or "/" not in repo_id:
        raise SystemExit("[!] cần --repo-id dạng <user>/<repo>")
    if hf_secret is None:
        raise SystemExit("[!] chưa có secret 'hf-cred' — chạy: modal secret create hf-cred HF_TOKEN=hf_xxx")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    root = Path(src)
    exts = tuple(e.strip() for e in include.split(",") if e.strip())
    files = _files_under(root, exts)

    existing = _list_hub_files(api, repo_id) if verify_sha else {}
    print(f"[i] repo hiện có {len(existing)} file")

    pushed, skipped, total = 0, 0, 0
    for p in files:
        rel = (Path(prefix_in_repo) / p.relative_to(root)).as_posix() if prefix_in_repo else p.relative_to(root).as_posix()
        size = p.stat().st_size
        if size > 50 * 1024**3:
            print(f"[SKIP] {rel}: {size / 1e9:.1f} GB > giới hạn 50 GB/file của HF — hãy chia nhỏ")
            continue
        have = existing.get(rel)
        if have and have["sha256"] and verify_sha:
            if _sha256_file(p) == have["sha256"]:
                skipped += 1
                print(f"[SKIP] {rel} (sha256 khớp)")
                continue
        elif have and have["size"] == size:
            skipped += 1
            print(f"[SKIP] {rel} (cùng size; chưa có sha256 để đối chiếu)")
            continue
        print(f"[PUT ] {rel}  ({size / 1e9:.2f} GB)")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=repo_id, repo_type="dataset")
        pushed += 1
        total += size
    return {"repo_id": repo_id, "pushed": pushed, "skipped": skipped, "bytes": total,
            "url": f"https://huggingface.co/datasets/{repo_id}"}


@app.function(image=cpu_image, secrets=[hf_secret] if hf_secret else [],
              volumes={"/raw": vol_raw}, timeout=12 * 3600, cpu=2.0)
def verify_hf(repo_id: str = "", src: str = "/raw", prefix_in_repo: str = "") -> dict:
    """Đối chiếu sha256 giữa file trên Volume và file đã đẩy lên HF. Đây là cổng kiểm tra mirror."""
    if not repo_id:
        raise SystemExit("[!] cần --repo-id")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    remote = _list_hub_files(api, repo_id)

    root = Path(src)
    ok = mismatch = missing = no_lfs = 0
    detail = []
    for local_path, rel in ((p, p.relative_to(root).as_posix()) for p in _files_under(root, (".tar.gz", ".tsv", ".pth", ".pt", ".json"))):
        key = (Path(prefix_in_repo) / rel).as_posix() if prefix_in_repo else rel
        if key not in remote:
            missing += 1
            detail.append({"path": key, "status": "missing_on_hub"})
            continue
        remote_sha = remote[key]["sha256"]
        if not remote_sha:
            if remote[key]["size"] == local_path.stat().st_size:
                ok += 1  # xác nhận bằng size khi HF không trả sha256
                continue
            no_lfs += 1
            detail.append({"path": key, "status": "no_lfs_sha & size khác nhau"})
            continue
        local_sha = _sha256_file(local_path)
        if local_sha == remote_sha:
            ok += 1
        else:
            mismatch += 1
            detail.append({"path": key, "status": "MISMATCH", "local": local_sha[:12], "remote": remote_sha[:12]})
    result = {"repo_id": repo_id, "ok": ok, "mismatch": mismatch, "missing": missing,
              "no_lfs_sha": no_lfs, "detail": detail[:50]}
    print("VERIFY_RESULT " + str(result))
    if mismatch or missing:
        raise SystemExit(f"[!] mirror KHÔNG khớp: {mismatch} lệch, {missing} thiếu")
    return result


# ===========================================================================
# Kaggle Datasets — mirror công khai phụ (200 GB/dataset, ≤50 file top-level)
# ===========================================================================
@app.function(image=cpu_image, secrets=[kaggle_secret] if kaggle_secret else [],
              volumes={"/raw": vol_raw}, timeout=12 * 3600, cpu=2.0)
def push_kaggle(dataset_id: str = "", src_subdir: str = "", title: str = "",
                licence: str = "CC-BY-NC-SA-4.0", create: bool = True, dir_mode: str = "zip") -> dict:
    """Tạo/ cập nhật Kaggle Dataset từ một thư mục con của /raw.

    Lưu ý giới hạn Kaggle: **tối đa 50 file top-level** và 200 GB/dataset.
    Vì vậy hãy đẩy theo từng thư mục nhỏ, ví dụ:
        --src-subdir dire/train/lsun_bedroom --dataset-id <user>/dire-tier-a-train-lsun
    """
    import json
    import shutil
    import subprocess

    if not dataset_id or "/" not in dataset_id:
        raise SystemExit("[!] cần --dataset-id dạng <user>/<slug>")
    if kaggle_secret is None:
        raise SystemExit("[!] chưa có secret 'kaggle-cred' — chạy: modal secret create kaggle-cred KAGGLE_USERNAME=... KAGGLE_KEY=...")

    src = Path("/raw") / src_subdir
    if not src.exists():
        raise SystemExit(f"[!] không thấy {src} trên Volume")

    # Kaggle CLI đọc ~/.kaggle/kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text(json.dumps({
        "username": os.environ["KAGGLE_USERNAME"],
        "key": os.environ["KAGGLE_KEY"],
    }))
    (kaggle_dir / "kaggle.json").chmod(0o600)

    # staging: copy (hoặc symlink) để có chỗ đặt dataset-metadata.json
    staging = Path("/tmp/kaggle_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    top = sorted(p for p in src.iterdir())
    if len(top) > 50:
        raise SystemExit(f"[!] {len(top)} file top-level > giới hạn 50 của Kaggle — hãy đẩy theo thư mục nhỏ hơn")
    for p in top:
        dst = staging / p.name
        try:
            os.symlink(p, dst)  # tiết kiệm dung lượng
        except OSError:
            shutil.copy2(p, dst)

    slug = dataset_id.split("/", 1)[1]
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": title or f"DIRE DiffusionForensics {slug}",
        "id": dataset_id,
        "licenses": [{"name": licence}],
        "description": (
            "Mirror của DiffusionForensics (DIRE, ICCV 2023) cho mục đích nghiên cứu. "
            "Nguồn gốc: https://github.com/ZhendongWang6/DIRE — ghi nguồn khi sử dụng. "
            "Bản quyền thuộc nhóm tác giả; mirror này để chống link rot."
        ),
    }, ensure_ascii=False, indent=2))

    cmd = ["kaggle", "datasets", "create" if create else "version",
           "-p", str(staging), "--dir-mode", dir_mode]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise SystemExit(proc.returncode)
    return {"dataset_id": dataset_id, "files": len(top), "dir_mode": dir_mode,
            "cmd": " ".join(cmd), "url": f"https://www.kaggle.com/datasets/{dataset_id}"}


# ===========================================================================
# Manifest sha256 — nguồn sự thật để kiểm tra mọi mirror
# ===========================================================================
@app.function(image=cpu_image, volumes={"/raw": vol_raw, "/ckpt": vol_ckpt}, timeout=6 * 3600, cpu=2.0)
def make_manifest(out: str = "/raw/_manifest_sha256.tsv") -> dict:
    """Sinh manifest sha256 cho mọi file .tar.gz/.pth trong /raw và /ckpt."""
    import csv

    rows = []
    for base in (Path("/raw"), Path("/ckpt")):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix not in (".gz", ".pth", ".pt"):
                continue
            rows.append({
                "source": base.name,
                "path": p.relative_to(base).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            })
    dest = Path(out)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "path", "size_bytes", "sha256"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    vol_raw.commit()
    print(f"[ok] {len(rows)} file → {dest} ({sum(r['size_bytes'] for r in rows) / 1e9:.2f} GB)")
    return {"files": len(rows), "manifest": str(dest)}
