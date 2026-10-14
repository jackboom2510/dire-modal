#!/usr/bin/env bash
set -euo pipefail
# Chạy DIRE trên RunPod Pod (A100 80GB ~$1.79/h — rẻ hơn Modal $2.50/h).
#
# Dùng khi: job dài > 4 GPU-h và đã ổn định (xem docs/adr-0002 §ngưỡng rời Modal).
# Dữ liệu lấy từ R2/HF (KHÔNG copy dữ liệu vào pod: pod có thể bị thu hồi).
#
# Cách dùng:
#   1. Tạo RunPod Pod: template "RunPod PyTorch 2.4", GPU A100 80GB, Network Volume (tuỳ chọn)
#   2. SSH vào pod rồi chạy script này (hoặc dán vào "Start command")
#
# Biến môi trường cần có (KHÔNG commit):
#   R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET
#   HF_TOKEN (nếu tải mirror qua HF thay vì R2)

: "${R2_BUCKET:=dire-archive}"
WORKDIR="${WORKDIR:-/workspace/dire-modal}"

echo "[1/5] clone code"
if [[ ! -d "$WORKDIR" ]]; then
  git clone https://github.com/jackboom2510/dire-modal "$WORKDIR"
fi
cd "$WORKDIR"

echo "[2/5] cài phụ thuộc"
pip install -q -e . webdataset boto3 pillow
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 || true

echo "[3/5] lấy dữ liệu về ổ local của pod (KHÔNG làm việc trực tiếp trên network volume)"
mkdir -p /workspace/data
if [[ -n "${HF_REPO_ID:-}" ]]; then
  # Đường khuyến nghị: mirror công khai trên HF (nhanh, không cần credential đặc biệt)
  pip install -q "huggingface_hub[cli]"
  huggingface-cli download "$HF_REPO_ID" --repo-type dataset \
    --local-dir /workspace/data --max-workers 8
elif [[ -n "${R2_ACCOUNT_ID:-}" ]]; then
  # Đường dự phòng: kéo shards từ R2 bằng boto3 (cùng logic với archive.py::pull_r2)
  python - <<'PY'
import os, sys
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig

bucket = os.environ.get("R2_BUCKET", "dire-archive")
prefix = os.environ.get("R2_PREFIX", "diffusion-forensics/v1")
dest = Path("/workspace/data")
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024, max_concurrency=8, use_threads=True)
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel = key[len(prefix):].lstrip("/")
        if not rel.startswith("raw/"):
            continue
        local = dest / rel[len("raw/"):]
        if local.exists() and local.stat().st_size == obj["Size"]:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        print("[GET]", key)
        s3.download_file(bucket, key, str(local), Config=cfg)
        n += 1
print(f"[ok] tải {n} file về {dest}")
PY
else
  echo "[!] cần HF_REPO_ID hoặc R2_* — xem docs/runbook-free.md bước 5"
  exit 1
fi

echo "[4/5] chạy job"
cat <<'EOF'
  # Code train hiện là TODO trong repo (xem README §Trạng thái):
  #   src/dire_modal/train.py — port từ src/dire_modal/vendor/ + repo DIRE gốc
  #
  # Khi đã port, chạy trực tiếp bằng python trên pod, ví dụ:
  #   python training_entry.py --shards /workspace/data/shards --out /workspace/ckpt
  # (KHÁC với `modal run` — trên VM bạn chạy script thường)
EOF

echo "[5/5] đẩy checkpoint lên R2 (BẮT BUỘC trước khi pod bị thu hồi)"
cat <<'EOF'
  Chạy từ máy local:
    modal run -m dire_modal.archive::push_r2 --prefix diffusion-forensics/v1
  hoặc copy thủ công từ pod:
    modal volume put ... (nếu có Modal) / hoặc aws s3 cp --endpoint-url ...
EOF

echo "[ok] xong"
