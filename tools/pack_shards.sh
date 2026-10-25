#!/usr/bin/env bash
# TIỆN ÍCH — không phải quy trình thực nghiệm (pack shards chỉ cần khi TRAIN,
# mà dự án không train lại — spec.md N1). Giữ để đo/đóng gói kho.
set -euo pipefail
# Pack /raw → /shards (WebDataset) và ĐO benchmark storage + compute.
# (Bản cũ gọi `dire_modal.bench::throughput` — hàm không tồn tại. Đã sửa.)

# 1. Pack — tier `train` (2 tar train + 2 tar val LSUN).
#    Bản cũ dùng `--tiers A`: alias đó giờ = e2,train nên sẽ pack thêm 16 tar
#    test không cần cho huấn luyện. Muốn pack cả test: --tiers e2,train
PACK_TIERS="${PACK_TIERS:-train}"
modal run -m dire_modal.pack::main --tiers "$PACK_TIERS"
modal run -m dire_modal.pack::stats

# 2. Benchmark: Volume vs R2 + throughput DIRE proxy
#    Truyền --shard-path để đo đọc tuần tự trên 1 shard thật.
SHARD="${1:-}"
if [[ -n "$SHARD" ]]; then
  modal run -m dire_modal.bench::main -- --shard-path "$SHARD"
else
  echo "[i] không truyền shard path → bỏ qua đo đọc tuần tự (dùng: bash tools/pack_shards.sh /shards/train/lsun_bedroom/adm/shard-000000.tar)"
  modal run -m dire_modal.bench::main
fi

echo
echo "[ok] Kết quả JSON: modal volume ls dire-shards /_bench"
