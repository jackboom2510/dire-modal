#!/usr/bin/env bash
set -euo pipefail
# Đo throughput DIRE (ảnh/giây) trên A100 để quy ra tổng giờ GPU cho stage-2.
# Đây là số ĐO ĐƯỢC nhưng là PROXY (~UNet 552M ~ ADM 256×256), không phải ADM thật.
#
# Dùng:
#   ./scripts/06_measure_dire.sh                # 100 ảnh, 20 bước DDIM
#   ./scripts/06_measure_dire.sh 300 20         # 300 ảnh, 20 bước
#   ./scripts/06_measure_dire.sh 100 20 --no-storage

N="${1:-100}"
STEPS="${2:-20}"
shift 2 2>/dev/null || true

echo "[i] đo DIRE proxy: n=$N ảnh · steps=$STEPS"
modal run -m dire_modal.bench::dire_throughput --n "$N" --steps "$STEPS"

cat <<'EOF'

--- Cách đọc kết quả ---
JSON có trường:
  images_per_s          → ảnh/giây đo được
  sec_per_image         → giây/ảnh
  est_gpu_hours         → số giờ GPU ước tính cho 5k / 50k / 200k ảnh
  est_cost_usd_a100_80gb→ quy ra tiền ở giá Modal $2.50/h
  caveat                → nhắc rằng đây là proxy ±30%

Cập nhật docs/iaas-options.md §4.3 bằng số thật thay cho ước lượng "30–60 A100-h".
EOF
