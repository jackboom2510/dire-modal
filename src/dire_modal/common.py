"""Shared Modal objects. Import mọi thứ từ đây."""
import modal

APP_NAME = "dire"

# ---------- Image ----------
# Base image cho mọi job không cần GPU
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "aria2", "pigz", "curl", "ca-certificates")
    .pip_install(
        "pyyaml==6.0.2",
        "boto3==1.35.0",
        "tqdm==4.66.5",
        "webdataset==0.2.100",
        "pillow==10.4.0",
        # cho mirror công khai (archive.py): HF Hub + Kaggle Datasets
        "huggingface_hub>=0.34",
        "kaggle>=1.6.17",
    )
    .run_commands(
        # BaiduPCS-Go cho Baidu Netdisk
        "wget -q https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/v4.0.2/"
        "BaiduPCS-Go-v4.0.2-linux-amd64.zip",
        "unzip -q BaiduPCS-Go-v4.0.2-linux-amd64.zip",
        "mv BaiduPCS-Go-v4.0.2-linux-amd64/BaiduPCS-Go /usr/local/bin/",
        "chmod +x /usr/local/bin/BaiduPCS-Go",
        "rm -rf BaiduPCS-Go-v4.0.2-linux-amd64*",
    )
)

# Image cho GPU jobs (compute_dire, train, eval)
gpu_image = (
    cpu_image
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy==1.26.4",
        "scikit-learn==1.5.1",
        "timm==1.0.9",
    )
    .run_commands(
        # guided-diffusion của repo DIRE gốc (cần cho DDIM inversion ở compute_dire.py).
        # Repo gốc đã archive 2025-07-08 nhưng vẫn clone được.
        "git clone --depth 1 https://github.com/ZhendongWang6/DIRE.git /opt/dire",
        "ls /opt/dire/guided-diffusion | head",
    )
)

# ---------- Volumes ----------
# Tách 3 vai trò rõ ràng (đã bàn ở lượt trước)
vol_raw    = modal.Volume.from_name("dire-raw",    create_if_missing=True)
vol_shards = modal.Volume.from_name("dire-shards", create_if_missing=True)
vol_ckpt   = modal.Volume.from_name("dire-ckpt",   create_if_missing=True)

VOLUMES = {
    "/raw":    vol_raw,     # .tar.gz gốc từ Baidu/Rec
    "/shards": vol_shards,  # WebDataset shards (hot path train)
    "/ckpt":   vol_ckpt,    # model checkpoint + log
}

# ---------- Secrets ----------
# Tạo bằng: modal secret create baidu-cred BDUSS=... STOKEN=...
#           modal secret create r2-cred R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
baidu_secret = modal.Secret.from_name("baidu-cred")
r2_secret    = modal.Secret.from_name("r2-cred")

# ---------- App ----------
app = modal.App(APP_NAME, image=cpu_image)