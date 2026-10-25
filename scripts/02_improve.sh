#!/usr/bin/env bash
# ============================================================================
# QUY TRÌNH 2 — CẢI THIỆN (training-free, trên nền checkpoint tác giả)
# ============================================================================
# Ràng buộc C3/N1 của spec.md: **không train lại** detector. Nên mọi cải thiện
# phải nằm NGOÀI trọng số: hiệu chỉnh, kết hợp điểm, định tuyến, phân tích lỗi.
#
# Hai nguồn ý tưởng, ghi rõ nguồn để không lẫn "của mình" với "của paper":
#
#   (A) AEROBLADE — Ricker et al., CVPR 2024 ("AEROBLADE: Training-Free
#       Detection of Latent Diffusion Images Using Autoencoder Reconstruction
#       Error"). Training-free: nạp VAE của LDM, đo lỗi tái tạo ảnh, lấy ĐIỂM ÂM
#       của khoảng cách (best AE = max). Mạnh trên ảnh LDM/SD, yếu trên ADM vì
#       ADM không dùng VAE cùng họ. ⇒ dùng làm tín hiệu THỨ HAI, không thay DIRE.
#
#   (B) Ý tưởng của dự án (chưa có trong paper nào): kết hợp DIRE ⊕ AEROBLADE
#       bằng rank-normalise + trọng số fit trên **val** (FR-5); định lượng
#       format bias JPEG/PNG vì DIRE nhạy với codec (FR-4); đường cong robustness
#       theo biến đổi của AEROBLADE (FR-6); định tuyến theo họ generator.
#
#   bash scripts/02_improve.sh                  # baseline trên val + in lộ trình
#   bash scripts/02_improve.sh --stage baseline # điểm DIRE trên e5 (val)
#   bash scripts/02_improve.sh --stage equivalence   # FR-2 (chưa có code)
#   bash scripts/02_improve.sh --stage aero          # FR-3 (chưa có code)
#   bash scripts/02_improve.sh --stage fusion        # FR-5 (chưa có code)
#   bash scripts/02_improve.sh --stage plan     # chỉ in lộ trình, không chạy gì
#
# ⚠ Stage CHƯA CÓ CODE sẽ DỪNG và nói rõ thiếu gì, trỏ về mốc trong spec.md.
#   Không có stage nào giả vờ đã chạy: một báo cáo "xong" không đo gì là đúng
#   loại lỗi tài liệu repo này đã gặp.
#
# Dữ liệu: baseline trên val cần `e5` (11 file / 2.11 GB / ~8 h ở 76.6 KiB/s).
# Nếu đã có `e2` thì generator test đã sẵn; `e5` chỉ thêm phần val.
# ============================================================================
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGES=""
VAL_TIERS="${VAL_TIERS:-e5}"
TEST_TIERS="${TEST_TIERS:-e2}"
CK="${CK:-lsun_adm}"

usage() {
  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGES="${2// /}"; shift 2 ;;
    --stages) STAGES="${2// /}"; shift 2 ;;
    --ck) CK="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) die "tham số lạ: $1 (xem --help)" ;;
  esac
done

need_cmd modal

# Mặc định chỉ chạy stage CÓ CODE. Chạy hết mọi stage sẽ dừng ngay ở FR-2 (chưa
# có code) và người dùng tưởng quy trình hỏng — trong khi nó chỉ chưa tới mốc M2.
[ -z "$STAGES" ] && STAGES="baseline"

roadmap() {
  cat <<'EOF'

LỘ TRÌNH CẢI THIỆN (thứ tự là RÀNG BUỘC, không phải đề xuất — spec.md §12)

  FR-2  equivalence   chứng minh DIRE tự tính khớp map tác giả (Pearson ≥ 0.90)
                      └─ CHẶN mọi kết luận dùng map tự tính (FR-4, FR-6)
  FR-3  aero          AEROBLADE làm tín hiệu thứ hai, cùng ảnh nguồn với FR-1
  FR-4  bias          AP ở clean / jpeg90 / png → tách dấu vết sinh ảnh khỏi codec
  FR-5  fusion        DIRE ⊕ AEROBLADE: rank-normalise + trọng số fit trên VAL
                      └─ chỉ gọi là "cải thiện" nếu AP(test) > cả hai thành phần
                         ở ≥ 12/16 generator (AC-5.3). Nếu không → báo cáo là
                         "không cải thiện", đó là kết quả hợp lệ.
  FR-6  robustness    20 điều kiện × 2 detector (jpeg/blur/crop/noise)

  Tổng chi phí trần: ≤ $20 (NFR-1), mỗi lần chạy có preflight $5.
  Hiện có code: FR-1 (eval DIRE). Còn lại là việc của spec.md §12 M2–M5.
EOF
}

# ---------------------------------------------------------------------------
if want baseline; then
  banner "QUY TRÌNH 2 — BASELINE DIRE TRÊN VAL (FR-1, có code)"

  say "kiểm dữ liệu val ($VAL_TIERS)"
  run_modal -m dire_modal.download::plan --tiers "$VAL_TIERS"
  if [ "${FETCH:-0}" = "1" ]; then
    run_modal -m dire_modal.download::main --tiers "$VAL_TIERS" --mode fetch
    run_modal -m dire_modal.download::verify
  else
    note "chưa tải; thêm FETCH=1 để tải ngay ($VAL_TIERS ≈ 2.11 GB ≈ 8 h)"
  fi

  say "eval trên VAL — ngưỡng/trọng số của FR-5 sẽ chỉ được fit ở đây"
  run_modal -m dire_modal.eval::main \
    --tiers "$VAL_TIERS" --ckpt "/raw/checkpoints/${CK}.pth" \
    --split val --exp-id "improve_val_dire_${CK}"

  say "eval trên TEST — để so 'trước/sau' trên cùng tập"
  run_modal -m dire_modal.eval::main \
    --tiers "$TEST_TIERS" --ckpt "/raw/checkpoints/${CK}.pth" \
    --exp-id "improve_test_dire_${CK}"

  note "test KHÔNG được chạm khi chọn trọng số — AC-5.1. Chỉ dùng để báo cáo."
  roadmap
fi

# ---------------------------------------------------------------------------
if want equivalence; then
  say "FR-2 — DIRE tự tính khớp map tác giả"
  missing_stage equivalence "spec.md FR-2 · mốc M2" \
    "compute_dire::equivalence (so 50 real + 50 adm, Pearson ≥ 0.90)" \
    "modal run -m dire_modal.compute_dire::equivalence --n 100 --exp-id fr2"
fi

if want aero; then
  say "FR-3 — AEROBLADE"
  missing_stage aero "spec.md FR-3 · mốc M3" \
    "module src/dire_modal/aero.py (nạp VAE SD1/SD2/KD2.1, LPIPS layer 2, điểm ÂM)" \
    "modal run -m dire_modal.aero::main --tiers e4 --exp-id fr3"
fi

if want bias; then
  say "FR-4 — format bias"
  missing_stage bias "spec.md FR-4 · điều kiện AC-4.2: FR-2 phải ĐẠT trước" \
    "recompute DIRE ở clean/jpeg90/png trên 200 real + 200 fake" \
    "modal run -m dire_modal.compute_dire::dire_batch --n 400 --recode jpeg90 --dry-run"
fi

if want fusion; then
  say "FR-5 — kết hợp DIRE ⊕ AEROBLADE"
  missing_stage fusion "spec.md FR-5 · mốc M4 (cần FR-3 xong)" \
    "module src/dire_modal/fusion.py (rank-normalise + trọng số fit trên val)" \
    "modal run -m dire_modal.fusion::fit --val-tier e5 --exp-id fr5"
fi

if want robustness; then
  say "FR-6 — đường cong robustness"
  missing_stage robustness "spec.md FR-6 (cần FR-2 + FR-3)" \
    "20 điều kiện × 2 detector, chi phí ≤ \$8" \
    "modal run -m dire_modal.aero::main --transforms jpeg_90,blur_2 --exp-id fr6"
fi

if want plan; then
  roadmap
fi

banner "XONG các stage đã yêu cầu"
cat <<'EOF'
Kết quả: /ckpt/results/<exp_id>/dire.csv + run.json
Tải về:  modal volume get dire-ckpt results ./results

Cổng chặn cần nhớ (spec.md §11):
  FR-2 KHÔNG ĐẠT ⇒ dừng, mọi kết luận sau chỉ dựa trên map TÁC GIẢ phát hành.
  FR-5 phải báo cáo cả 3 dòng (DIRE / AEROBLADE / kết hợp) trên cùng tập.
  AC-5.3: không đạt ≥ 12/16 generator thì ghi "không cải thiện" — đừng uốn số.
EOF
