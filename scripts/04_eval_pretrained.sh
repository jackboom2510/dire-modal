#!/usr/bin/env bash
# Eval bằng checkpoint PRE-TRAINED của tác giả (không train gì).
#
#   bash scripts/04_eval_pretrained.sh            # tier e2 (17 file, đầy đủ 16 gen)
#   TIERS=e2min bash scripts/04_eval_pretrained.sh  # 3 file, chỉ cổng S2
#
# Bản cũ copy ckpt sang Volume khác bằng `modal volume cp /raw/...` — lệnh đó
# không chạy được: `modal volume cp` đòi dạng `<volume>:<path>`, còn `/raw/...`
# chỉ tồn tại BÊN TRONG container (do mount), không phải trên máy local.
# Không cần copy nữa: eval::main tự tìm /raw/checkpoints/lsun_adm.pth trước,
# rồi /ckpt/pretrained/lsun_adm.pth.
set -euo pipefail

TIERS="${TIERS:-e2}"
CKPT="${CKPT:-}"          # để trống = để eval::main tự tìm
LIMIT="${LIMIT:-0}"

if [ -n "$CKPT" ]; then
  modal run -m dire_modal.eval::main --tiers "$TIERS" --ckpt "$CKPT" --limit "$LIMIT"
else
  modal run -m dire_modal.eval::main --tiers "$TIERS" --limit "$LIMIT"
fi

cat <<'EOF'

✅ CỔNG S2 (in trong log): ACC trên generator `adm` phải >= 0.99.
   KHÔNG ĐẠT thì đừng đọc bảng AP — kiểm theo thứ tự:
     1. ls -l /raw/checkpoints/lsun_adm.pth   (đúng file chưa, 269.4 MB?)
     2. n_real / n_fake trong log             (bằng nhau và > 0?)
     3. run.json → "preproc" phải là resize256_centercrop224_normalize

Kết quả: /ckpt/results/<exp_id>/dire.csv + run.json
  modal volume get dire-ckpt results ./results
EOF
