#!/usr/bin/env bash
# ============================================================================
# QUY TRÌNH 1 — TÁI LẬP THỰC NGHIỆM CỦA TÁC GIẢ (không train lại)
# ============================================================================
# Mục tiêu: tái tạo khối "DIRE (ours)" của **Table 2** trong paper
# (Wang et al., ICCV 2023 — arXiv:2303.09295v1):
#
#     4 hàng  = 4 detector của tác giả (train trên ADM / PNDM / iDDPM / StyleGAN)
#     8 cột   = 8 generator (ADM, DDPM, iDDPM, PNDM, SD-v1, SD-v2, LDM, VQ-Diffusion)
#     32 ô    = ACC/AP, headline là Avg 99.9% ACC / 100% AP
#
# Không train gì: nhánh `dire/` của dataset đã chứa DIRE map tác giả tính sẵn
# (ADM-LSUN, DDIM S=20), nên chỉ cần ResNet-50 forward trên CPU.
#
#   bash scripts/01_repro.sh                 # data → eval → report
#   bash scripts/01_repro.sh --stage data    # chỉ tải + verify
#   bash scripts/01_repro.sh --stage eval    # chỉ eval (dữ liệu đã có)
#   bash scripts/01_repro.sh --stage report  # chỉ in bảng so với paper
#   bash scripts/01_repro.sh --stage cross   # Table 3 (cross-dataset ImageNet)
#   bash scripts/01_repro.sh --quick         # e2min: 3 file, 1 ô (cổng S2)
#   DETACH=1 bash scripts/01_repro.sh        # app sống độc lập với máy local
#
# Dữ liệu (số đo 2026-10-25, trần tải Baidu ~76.6 KiB/s cho CẢ tài khoản):
#   e2min      3 file    0.40 GB   ~1.5 h   → 1 ô (cổng S2, ACC(adm) ≥ 0.99)
#   e2         17 file   1.94 GB   ~7.4 h   → hàng ADM: 8 cột
#   e2,ckpts   22 file   3.25 GB   ~12.4 h  → TOÀN BỘ 32 ô  ← mức của quy trình này
# ============================================================================
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGES=""
QUICK=0
CKOPS_DEFAULT="e2,ckpts"

usage() {
  sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGES="${2// /}"; shift 2 ;;
    --stages) STAGES="${2// /}"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --tiers) TIERS_ARG="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) die "tham số lạ: $1 (xem --help)" ;;
  esac
done

need_cmd modal
if [ "$QUICK" = "1" ]; then
  TIERS="${TIERS_ARG:-e2min}"
  CKOPS="lsun_adm"
  EXP_PREFIX="repro_quick"
else
  TIERS="${TIERS_ARG:-$CKOPS_DEFAULT}"
  CKOPS="lsun_adm lsun_pndm lsun_iddpm lsun_stylegan"
  EXP_PREFIX="repro"
fi

banner "QUY TRÌNH 1 — TÁI LẬP TABLE 2 (DIRE, ICCV 2023) · tiers=$TIERS · det=${CKOPS}"
note "Số tham chiếu nằm ở src/dire_modal/paper_table2.tsv (chép từ paper, có ghi nguồn)."
[ "$QUICK" = "1" ] && note "--quick: chỉ 1 detector × 1 generator, để chạy cổng S2 nhanh."

# ---------------------------------------------------------------------------
if want data; then
  say "1/3 DỮ LIỆU — plan (0 credit) → fetch → verify"
  run_modal -m dire_modal.download::local_status

  # `plan` in size thật + tổng GB, KHÔNG tải. Luôn chạy trước: nó là bước duy
  # nhất phát hiện được tier trỏ vào đường dẫn sai mà không tốn một giờ chờ.
  run_modal -m dire_modal.download::plan --tiers "$TIERS"

  # `main` tự lặp nhiều vòng tới 20 h (ORCHESTRATOR_BUDGET_S) và resume từ
  # /raw/.staging, nên không phải gõ lại lệnh nhiều lần.
  say "tải thật — tự resume, tự chạy nhiều vòng"
  run_modal -m dire_modal.download::main --tiers "$TIERS" --mode fetch

  run_modal -m dire_modal.download::verify
fi

# ---------------------------------------------------------------------------
if want eval; then
  say "2/3 EVAL — mỗi detector một lần (eval.py nhận 1 ckpt/lần)"

  if [ -n "${LIMIT:-}" ]; then note "LIMIT=$LIMIT ảnh/archive (smoke test)"; fi
  for CK in $CKOPS; do
    CKPT="/raw/checkpoints/${CK}.pth"
    # Không kiểm tồn tại file ở đây: chỉ container mới thấy /raw. eval::main tự
    # resolve và in đường dẫn đã thử nếu thiếu — thông báo đó tốt hơn của script.
    say "eval ${CK}"
    # `--exp-id` cố định theo tên detector: `eval::report` nhóm theo stem của
    # ckpt đọc từ run.json, nhưng tên thư mục dễ đọc vẫn có ích khi `ls` Volume.
    run_modal -m dire_modal.eval::main \
      --tiers "$TIERS" \
      --ckpt "$CKPT" \
      --exp-id "${EXP_PREFIX}_${CK}" \
      ${LIMIT:+--limit "$LIMIT"}
  done
fi

# ---------------------------------------------------------------------------
if want report; then
  say "3/3 BÁO CÁO — gộp mọi lần eval, in Δ so với paper"
  run_modal -m dire_modal.eval::report
  cat <<'EOF'

Cách đọc:
  * Δ ≈ 0 ở cả 32 ô  → tái lập ĐẠT. Ghi số vào báo cáo/PR.
  * Δ lớn ở một cột    → nghi tiền xử lý (phải là resize256 → centercrop224 →
                         Normalize ImageNet; xem trường "preproc" trong run.json).
  * Cột nào "thiếu dữ liệu" → chưa tải đủ. `--tiers e2,ckpts` cho generator +
                         detector; `--tiers e2` nếu chỉ thiếu generator.
  * 8 cột của Table 2 KHÔNG gồm stylegan/projectedgan/dalle2/if/midjouney… —
    nhánh `e2` có 15 generator: 8 cột của paper + 7 generator mở rộng của dự án
   (stylegan, projectedgan, diff-*, dalle2, if, midjouney).

Tải kết quả về máy:
  modal volume get dire-ckpt results ./results
EOF
fi

# ---------------------------------------------------------------------------
if want cross; then
  # Table 3 của paper = cross-dataset. Chỉ cần 3 file, KHÔNG cần cả tier `e3`
  # (4.85 GB): nhánh `dire/test/imagenet/imagenet/` có đúng bộ của paper.
  say "TÙY CHỌN — Table 3 (cross-dataset ImageNet)"
  CROSS_PATHS="dire/test/imagenet/imagenet/real.tar.gz,\
dire/test/imagenet/imagenet/adm.tar.gz,\
dire/test/imagenet/imagenet/sdv1.tar.gz"
  run_modal -m dire_modal.download::main --paths "$CROSS_PATHS" --mode fetch
  for CK in lsun_adm; do
    run_modal -m dire_modal.eval::main \
      --ckpt "/raw/checkpoints/${CK}.pth" --exp-id "cross_${CK}_imagenet"
  done
  note "Đối chiếu với Table 3 chỉ khi đã tải đúng ckpt tương ứng"
  note "(paper: train trên LSUN-Bedroom → test trên ImageNet)."
fi

banner "XONG — tái lập"
cat <<'EOF'
Số của chúng ta ở:  /ckpt/results/<exp_id>/dire.csv + run.json
Tải về:             modal volume get dire-ckpt results ./results

Nếu đang chạy DETACH=1:  modal app list · modal app logs <app-id>
Việc còn treo của quy trình này: xem plan.md (Table 4/5/6 cần TRAIN LẠI nên
ngoài phạm vi — spec.md N1).
EOF
