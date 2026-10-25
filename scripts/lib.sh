#!/usr/bin/env bash
# lib.sh — hàm dùng chung cho 2 quy trình thực nghiệm.
# Không chạy trực tiếp file này: `source scripts/lib.sh`.
#
# Vì sao có file này: trước đây `scripts/` là một dãy 11 script đánh số rời rạc
# (00…10) — mỗi người đọc phải tự ghép lại thứ tự, và mỗi lần thêm một bước
# lại thêm một file. Nay chỉ còn ĐÚNG HAI quy trình, phần lặp lại nằm ở đây.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; YEL=$'\033[33m'; OFF=$'\033[0m'

say() { printf '\n%s== %s ==%s\n' "$BOLD" "$*" "$OFF"; }
note() { printf '%s   %s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '%s[!] %s%s\n' "$YEL" "$*" "$OFF"; }
die() { printf '%s[STOP] %s%s\n' "$RED" "$*" "$OFF" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "thiếu lệnh '$1'"; }

# `modal run` có/không --detach.
#
# Vì sao quan trọng: `modal run` ở chế độ thường là app EPHEMERAL — app sống theo
# client. Mất DNS ở máy local một lúc là heartbeat chết, CLI thoát, app bị dọn
# (đã xảy ra thật: `ConnectionError: [Errno 11001] getaddrinfo failed`). Với
# `--detach`, app không phụ thuộc client; xem log bằng `modal app logs`.
DETACH="${DETACH:-0}"
run_modal() {
  if [ "$DETACH" = "1" ]; then
    modal run --detach "$@"
  else
    modal run "$@"
  fi
}

# Chạy một stage nếu nó nằm trong danh sách `--stage`.
# STAGES rỗng nghĩa là "tất cả" (mỗi quy trình tự định nghĩa thứ tự).
STAGES="${STAGES:-}"
want() {
  [ -z "$STAGES" ] && return 0
  case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# Chốt chặn cho phần CHƯA CÓ CODE.
#
# Vì sao không viết script giả chạy được: một quy trình in ra "xong" trong khi
# chưa hề đo gì là đúng loại lỗi tài liệu mà repo này đã gặp. Ở đây nó phải
# DỪNG, nói rõ thiếu gì, và trỏ về mốc trong spec.md.
missing_stage() {
  local stage="$1" ref="$2" what="$3" cmd="$4"
  warn "stage '$stage' chưa có code — dừng (không giả vờ đã chạy)."
  printf '   spec:  %s\n' "$ref"
  printf '   cần:   %s\n' "$what"
  printf '   lệnh sẽ là: %s\n' "$cmd"
  printf '   Theo dõi ở plan.md §"Việc tiếp theo".\n'
  return 3
}

# Tổng dung lượng cần tải, để người chạy biết mình đang hứa bao nhiêu giờ.
# 76.6 KiB/s là SỐ ĐO (bench_concurrency 2026-10-25), không phải ước lượng.
eta_hours() {
  python3 - "$1" <<'PY'
import sys
gb = float(sys.argv[1])
rate = 76.6 * 1024            # B/s
print(f"{gb * 1024**3 / rate / 3600:.1f}")
PY
}

confirm() {
  if [ "${ASSUME_YES:-0}" = "1" ]; then return 0; fi
  read -r -p ">>> $1 [Enter = tiếp, Ctrl-C = dừng] " _
}

banner() {
  printf '\n%s%s%s\n%s\n%s%s%s\n' \
    "$BOLD" "==============================================================================" "$OFF" \
    "$1" \
    "$BOLD" "==============================================================================" "$OFF"
}
