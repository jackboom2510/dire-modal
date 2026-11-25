#!/usr/bin/env bash
# Tải artefact Table 4 đã có từ Modal Volume về máy.
# Không chạy job, không tính DIRE, không sinh ảnh và không xoá dữ liệu trên Volume.
#
# Mặc định:
#   remote: dire-zero/experiments/table4/table4-lsun-pilot
#   local : ./dire-zero/experiments/table4/table4-lsun-pilot
#
# Dùng:
#   bash scripts/fetch_table4_artifacts.sh
#   bash scripts/fetch_table4_artifacts.sh --remote-path experiments/table4/table4-lsun-pilot \
#       --out ./dire-zero/experiments/table4/table4-lsun-pilot
#   bash scripts/fetch_table4_artifacts.sh --dry-run
set -euo pipefail

VOLUME="${DIRE_VOLUME:-dire-zero}"
REMOTE="experiments/table4/table4-lsun-pilot"
OUT="./dire-zero/experiments/table4/table4-lsun-pilot"
DRY_RUN=0

usage() {
  sed -n '2,13p' "$0" | sed 's/^# //'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --volume) VOLUME="$2"; shift 2 ;;
    --remote-path) REMOTE="${2#/}"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Tham số không hợp lệ: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$VOLUME$REMOTE$OUT" in
  *".."*) echo "Không cho phép đường dẫn chứa '..'." >&2; exit 2 ;;
esac

# Modal volume get nhận thư mục remote và tải recursive toàn bộ nội dung bên dưới.
# Dùng thư mục tạm để tránh làm hỏng bản tải dở khi máy bị ngắt giữa chừng.
TMP="${OUT}.partial"
CMD=(modal volume get "$VOLUME" "$REMOTE" "$TMP")

echo "Volume : $VOLUME"
echo "Remote : /$REMOTE"
echo "Local  : $OUT"
echo "Mode   : $([ "$DRY_RUN" = 1 ] && echo dry-run || echo download)"

echo
echo "Chỉ tải các artefact hiện có; không chạy Modal function."
if [ "$DRY_RUN" = 1 ]; then
  printf 'Lệnh: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

command -v modal >/dev/null 2>&1 || {
  echo "Thiếu lệnh modal. Cài/đăng nhập Modal trước khi chạy." >&2
  exit 127
}

mkdir -p "$(dirname "$OUT")"
rm -rf "$TMP"
"${CMD[@]}"

# Chỉ đổi tên sau khi modal volume get kết thúc thành công.
rm -rf "$OUT"
mv "$TMP" "$OUT"

count=$(find "$OUT" -type f | wc -l | tr -d ' ')
json_count=$(find "$OUT" -type f -name '*.json' | wc -l | tr -d ' ')
echo "[OK] Đã tải $count file ($json_count JSON) vào $OUT"
find "$OUT" -type f -name '*.json' -print | sort
