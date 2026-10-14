#!/usr/bin/env bash
set -euo pipefail
# Bootstrap trên Vast.ai instance (giá marketplace, thường rẻ nhất; có thể bị preempt).
#
# Vast phù hợp cho: bulk GPU dài hơi, khi đã ổn định code. BẮT BUỘC checkpoint thường xuyên
# vì instance có thể bị thu hồi bất kỳ lúc nào (đặc biệt với host "interruptible").
#
# Cách dùng:
#   vastai create instance <offer_id> --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime --disk 80 --onstart bootstrap.sh
# hoặc SSH vào rồi: bash bootstrap.sh

WORKDIR="${WORKDIR:-/workspace/dire-modal}"
DATA_DIR="${DATA_DIR:-/workspace/data}"

echo "[1/6] kiểm tra GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "[2/6] clone code"
[[ -d "$WORKDIR" ]] || git clone https://github.com/jackboom2510/dire-modal "$WORKDIR"
cd "$WORKDIR"

echo "[3/6] cài phụ thuộc"
pip install -q -e . webdataset boto3 pillow

echo "[4/6] kiểm tra kết nối tới kho dữ liệu (R2/HF) — nhanh, để phát hiện sớm lỗi mạng/credential"
python - <<'PY'
import os
print("  R2_ACCOUNT_ID:", "có" if os.environ.get("R2_ACCOUNT_ID") else "THIẾU → sẽ phải tải qua HF")
print("  HF_TOKEN     :", "có" if os.environ.get("HF_TOKEN") else "THIẾU")
PY

echo "[5/6] tải dữ liệu về ổ của instance (Vast thường kèm dung lượng lớn)"
mkdir -p "$DATA_DIR"
if [[ -n "${HF_TOKEN:-}" && -n "${HF_REPO_ID:-}" ]]; then
  pip install -q "huggingface_hub[cli]"
  huggingface-cli download "$HF_REPO_ID" --repo-type dataset --local-dir "$DATA_DIR" --max-workers 8
else
  echo "[!] chưa có HF_REPO_ID → tải thủ công rồi chạy lại"
fi

echo "[6/6] chạy job (ví dụ: đo throughput DIRE trước khi cam kết giờ GPU)"
cat <<EOF
  # đo trước:
  #   python -c "import torch; print(torch.cuda.get_device_name(0))"
  # train:
  #   python -m dire_modal.train --exp lsun_adm_repro
  # ĐỪNG QUÊN: đẩy checkpoint ra ngoài (R2/HF) mỗi epoch — script train đã làm nếu volumes được map.
EOF

echo "[ok] bootstrap xong. Nhắc lại: instance Vast CÓ THỂ bị thu hồi — luôn có checkpoint ngoài máy."
