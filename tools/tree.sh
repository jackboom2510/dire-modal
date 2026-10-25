#!/usr/bin/env bash
# TIỆN ÍCH — xem Volume, không phải quy trình thực nghiệm.
# Liệt kê TOÀN BỘ nội dung một Volume dưới dạng cây (hoặc danh sách phẳng).
#
#   bash tools/tree.sh                          # /raw của dire-raw
#   PATH_IN_VOL=/raw/checkpoints bash tools/tree.sh
#   FLAT=1 SAVE=1 bash tools/tree.sh            # <size>\t<path>, lưu ra file
#   bash tools/tree.sh --depth 2                # giới hạn độ sâu
#
# Vì sao cần script này: `modal volume ls` CHỈ đọc một tầng (không có --recursive)
# và khi stdout không phải TTY thì chỉ in đường dẫn, mất cột size. Nên không thể
# ra cây bằng CLI thuần.
set -euo pipefail

PATH_IN_VOL="${PATH_IN_VOL:-/raw}"
VOL_NAME="${VOL_NAME:-dire-raw}"
ARGS=()

if [ "${FLAT:-0}" = "1" ]; then
  ARGS+=(--flat)
fi
if [ "${SAVE:-0}" = "1" ]; then
  REMOTE_OUT="${REMOTE_OUT:-/raw/_tree.txt}"
  ARGS+=(--out "$REMOTE_OUT")
fi

modal run -m dire_modal.download::tree --path "$PATH_IN_VOL" "${ARGS[@]}" "$@"

if [ "${SAVE:-0}" = "1" ]; then
  cat <<TIP

[i] Muốn kéo bản lưu về máy (log của Modal có thể bị cắt):
      modal volume get ${VOL_NAME} ${REMOTE_OUT#/raw} ./_tree.txt
TIP
fi
