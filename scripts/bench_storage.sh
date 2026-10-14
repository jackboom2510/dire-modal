#!/usr/bin/env bash
set -euo pipefail
# Đo storage: Modal Volume vs R2 (CloudBucketMount + S3 API), và throughput DIRE proxy.
#
# R2_ACCOUNT_ID/R2_BUCKET phải có Ở MÁY LOCAL trước khi `modal run` vì
# modal.CloudBucketMount được dựng trong tiến trình local (credentials thì lấy từ Secret).
#
# Dùng:
#   ./scripts/bench_storage.sh
#   ./scripts/bench_storage.sh --shard-path /shards/train/lsun_bedroom/adm/shard-000000.tar
#   ./scripts/bench_storage.sh --no-gpu

if [[ -f .env ]]; then
  set -a; source .env; set +a
  echo "[i] đã nạp .env"
else
  echo "[!] không thấy .env — R2 sẽ bị bỏ qua, chỉ đo Volume + GPU"
fi

if [[ -z "${R2_ACCOUNT_ID:-}" ]]; then
  echo "[!] R2_ACCOUNT_ID trống → bỏ qua benchmark R2 (Volume + GPU vẫn chạy)"
fi

echo "[i] R2 bucket: ${R2_BUCKET:-dire-archive}"

modal run -m dire_modal.bench::main -- "$@"

cat <<'EOF'

--- Xem kết quả ---
  modal volume ls dire-shards /_bench
  modal volume get dire-shards /_bench/<file>.json manifests/
  # rồi mở manifests/bench_*.json và đối chiếu với docs/storage-comparison.md §3
EOF
