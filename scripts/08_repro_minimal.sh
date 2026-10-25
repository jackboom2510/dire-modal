#!/usr/bin/env bash
# Tái lập thử nghiệm DIRE với checkpoint PRE-TRAINED — KHÔNG train lại ResNet.
#
#   bash scripts/08_repro_minimal.sh
#   TIERS=e2 bash scripts/08_repro_minimal.sh        # đủ 16 generator (7.4 giờ)
#   SKIP_FETCH=1 bash scripts/08_repro_minimal.sh    # đã có dữ liệu, chỉ eval
#   SOURCE=hf HF_REPO=<user>/<repo> bash scripts/08_repro_minimal.sh
#
# Vì sao có script này: mọi thứ dưới đây đã đo được (2026-10-25).
#   * Baidu cho **76.6 KiB/s cho CẢ tài khoản** — thêm container không nhanh hơn
#     (k=1: 76.6, k=2: 75.0) mà chỉ làm mỗi file chậm đi một nửa.
#   * `e2` = 1.94 GB ⇒ 7.4 giờ. `e2min` = 3 file ≈ 400 MB ⇒ ~1.5 giờ.
#   * `e2,e4` = 3.39 GB ⇒ 12.9 giờ.
# Nên đường ngắn nhất tới CON SỐ ĐẦU TIÊN là `e2min`.
set -euo pipefail

TIERS="${TIERS:-e2min}"
SOURCE="${SOURCE:-baidu}"          # baidu | hf
HF_REPO="${HF_REPO:-}"
SKIP_FETCH="${SKIP_FETCH:-0}"
LIMIT="${LIMIT:-0}"                # >0: chỉ N ảnh mỗi archive (smoke test)

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "0. Kiểm cấu hình (miễn phí, không bật container tải)"
modal run -m dire_modal.download::local_status

say "1. Kế hoạch tải — chưa tải gì"
if [ "$SOURCE" = "hf" ]; then
  echo "[i] nguồn = HF Hub ($HF_REPO) — bỏ qua bước plan của Baidu"
else
  modal run -m dire_modal.download::plan --tiers "$TIERS"
fi

if [ "$SKIP_FETCH" != "1" ]; then
  say "2. Tải ($TIERS) — tự chạy nhiều vòng, resume được, ghi manifest từng file"
  if [ "$SOURCE" = "hf" ]; then
    [ -n "$HF_REPO" ] || { echo "[!] SOURCE=hf cần HF_REPO=<user>/<repo>"; exit 1; }
    modal run -m dire_modal.download::from_hf --repo-id "$HF_REPO" --tiers "$TIERS"
  else
    modal run -m dire_modal.download::main --tiers "$TIERS" --mode fetch
  fi
fi

say "3. Kiểm toàn vẹn dữ liệu đã tải"
modal run -m dire_modal.download::verify

say "4. Eval bằng checkpoint PRE-TRAINED (không train gì)"
echo "[i] ckpt: /raw/checkpoints/lsun_adm.pth — do eval::main tự tìm"
if [ "$LIMIT" != "0" ]; then
  echo "[i] giới hạn $LIMIT ảnh/archive (smoke test)"
fi
modal run -m dire_modal.eval::main --tiers "$TIERS" --limit "$LIMIT"

say "XONG"
cat <<'EOF'
Kết quả ở Volume dire-ckpt: results/<exp_id>/dire.csv + run.json
Tải về:
  modal volume get dire-ckpt results ./results

Cổng S2 của SPEC-001 (đã in trong log eval):
  ACC trên generator `adm` phải >= 0.99.
  Nếu KHÔNG ĐẠT, đừng đọc tiếp bảng AP:
     1. checkpoint có đúng lsun_adm.pth không?  (ls -l /raw/checkpoints)
     2. n_real và n_fake có bằng nhau và > 0 không?  (eval in ra cảnh báo)
     3. tiền xử lý phải là resize256 → centercrop224 → Normalize ImageNet
        (xem run.json: "preproc": "resize256_centercrop224_normalize")
EOF
