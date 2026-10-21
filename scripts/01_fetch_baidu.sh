#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Tải dữ liệu từ Baidu/Rec về Volume /raw.
#
# Bước 0–2 KHÔNG tải gì và gần như không tốn tiền — chạy chúng trước để thấy
# size thật. Chỉ bước 3 mới tải.
#
# Mặc định tải tier e2+e4 = 21/154 file trong cây Baidu (~86% số file là thừa
# với mục tiêu eval + demo). Đổi bằng --tiers hoặc sửa default_tiers trong
# src/dire_modal/tiers.yaml.
#
# Chạy không có TTY (CI, nohup): đặt ASSUME_YES=1 để bỏ qua lời nhắc ở bước 3.
# ============================================================================

TIERS="${TIERS:-}"          # rỗng ⇒ dùng default_tiers trong tiers.yaml
ARGS=()
if [[ -n "$TIERS" ]]; then ARGS+=(--tiers "$TIERS"); fi

echo "================ 0. POLICY (chạy local, 0 credit) ================"
modal run -m dire_modal.download::local_status

echo
echo "================ 1. DÒ CÂY REMOTE + ĐỐI CHIẾU MANIFEST ================"
echo "-- liệt kê /apps để xác nhận BAIDU_ROOT --"
modal run -m dire_modal.download::ls_remote --path /apps
echo
echo "-- audit: file nào có thật, file nào manifest kê sai, file nào bị NEVER --"
modal run -m dire_modal.download::audit

echo
echo "================ 2. KẾ HOẠCH + SIZE THẬT (không tải) ================"
modal run -m dire_modal.download::plan "${ARGS[@]}"

echo
echo "================ 3. TẢI THẬT ================"
if [[ "${ASSUME_YES:-0}" != "1" ]]; then
  # Nếu không có TTY, `read` thất bại và set -e dừng script TRƯỚC khi tải.
  # Đó là hành vi mong muốn: không bao giờ tải khi chưa có người duyệt bảng.
  read -r -p ">>> Kiểm tra bảng trên. Gõ Enter để tải, Ctrl-C để dừng: " _
fi
modal run -m dire_modal.download::main "${ARGS[@]}" --mode fetch

echo
echo "================ 4. VERIFY ================"
modal run -m dire_modal.download::verify

echo
echo "Xong. Nếu phát hiện đã tải thừa, thu hồi bằng:"
echo "  modal run -m dire_modal.download::prune ${ARGS[*]}            # chỉ in"
echo "  modal run -m dire_modal.download::prune ${ARGS[*]} --mode apply  # xoá thật"
