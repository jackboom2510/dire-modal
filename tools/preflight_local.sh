#!/usr/bin/env bash
# ============================================================================
# KIỂM MÁY CÁ NHÂN trước khi chạy bất cứ thứ gì tốn tiền
# ============================================================================
# Chín cổng G0–G8 kiểm *dữ liệu và quy trình*. Script này kiểm *cái máy*:
# Python, Git Bash, `modal`, dung lượng trống, ổ FAT32, và DIRE_LOCAL_ROOT.
# Chạy trước, mất 5 giây, tránh mất 40 phút.
#
#   bash tools/preflight_local.sh
# ============================================================================
set -uo pipefail

export MSYS_NO_PATHCONV=1      # Git Bash: "/raw" → "C:/Program Files/Git/raw"
export MSYS2_ARG_CONV_EXCL='*'

FAIL=0
say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[xanh]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[ cẩn ]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[ ĐỎ ]\033[0m %s\n' "$*"; FAIL=1; }

say "KIỂM MÁY — $(uname -s) / $(uname -m)"

# ---- 1. Python --------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else bad "không thấy python3/python trong PATH"; PY=""
fi
if [[ -n "$PY" ]]; then
  ver="$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  case "$ver" in
    3.1[0-9]|3.[2-9][0-9]) ok "Python $ver" ;;
    *) bad "Python $ver — cần ≥ 3.10 (pyproject: requires-python)" ;;
  esac
  $PY -c 'import yaml' 2>/dev/null && ok "pyyaml (dùng để đọc tiers.yaml)" \
    || warn "thiếu pyyaml → policy dùng fallback (không chết, nhưng nên cài: pip install -e .)"
fi

# ---- 2. Git Bash trên Windows ----------------------------------------------
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  ok "Git Bash (MSYS2) — đã bật MSYS_NO_PATHCONV cho tiến trình này"
  if grep -q 'MSYS_NO_PATHCONV' "$(dirname "${BASH_SOURCE[0]}")/scratch_repro.sh" 2>/dev/null; then
    ok "scratch_repro.sh cũng tự bật"
  else
    warn "scratch_repro.sh không thấy MSYS_NO_PATHCONV — kiểm lại"
  fi
fi

# ---- 3. DIRE_LOCAL_ROOT -----------------------------------------------------
if [[ -z "${DIRE_LOCAL_ROOT:-}" ]]; then
  bad "DIRE_LOCAL_ROOT chưa đặt (ví dụ: export DIRE_LOCAL_ROOT=D:/dire-data)"
else
  mkdir -p "$DIRE_LOCAL_ROOT" 2>/dev/null
  if [[ -w "$DIRE_LOCAL_ROOT" ]]; then ok "DIRE_LOCAL_ROOT=$DIRE_LOCAL_ROOT (ghi được)"
  else bad "DIRE_LOCAL_ROOT=$DIRE_LOCAL_ROOT không ghi được"; fi
fi

# ---- 4. dung lượng trống ----------------------------------------------------
ROOT="${DIRE_LOCAL_ROOT:-data/raw}"
if [[ -d "$ROOT" ]]; then
  # df -Pk: POSIX, có trên MSYS và Linux
  free_mb="$(df -Pk "$ROOT" 2>/dev/null | awk 'NR==2{print int($4/1024)}')"
  fstype="$(stat -f -c %T "$ROOT" 2>/dev/null || echo '?')"
  if [[ -n "${free_mb:-}" ]]; then
    if   (( free_mb >= 15000 )); then ok "trống ${free_mb} MB (đủ cho A+B+C ≈ 9 GB + đệm)"
    elif (( free_mb >= 9000  )); then warn "trống ${free_mb} MB — đủ A+B+C nhưng không đệm được"
    else bad "trống ${free_mb} MB — A+B+C cần ~9 GB"; fi
  fi
  case "$fstype" in
    vfat|msdos|fat) bad "ổ FAT32/exFAT ($fstype): giới hạn 4 GB/file — ADM 2,5 GB lọt nhưng .part cộng dồn dễ tràn; dùng NTFS" ;;
    *) ok "filesystem: $fstype" ;;
  esac
fi

# ---- 5. công cụ tuỳ chọn ----------------------------------------------------
for c in curl unzip sha256sum; do
  command -v "$c" >/dev/null 2>&1 && ok "$c" || warn "thiếu $c"
done
if command -v modal >/dev/null 2>&1; then
  ok "modal $(${PY:-python3} -c 'import modal;print(modal.__version__)' 2>/dev/null || echo '?') CLI"
  if modal volume ls >/dev/null 2>&1; then ok "đã đăng nhập Modal (volume ls chạy được)"
  else warn "chưa đăng nhập Modal → chạy: modal token new"; fi
else
  warn "chưa có CLI modal → pip install 'modal>=0.64'; chỉ cần khi chạy GPU"
fi
$PY -c 'import torch' 2>/dev/null && ok "torch (chỉ cần nếu chạy local, không cần cho Modal)" \
  || warn "không có torch ở local — bình thường, GPU chạy trên Modal"

# ---- 6. repo ---------------------------------------------------------------
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/../src/dire_modal/manifest.py" ]]; then
  ok "đang ở trong repo dire-modal"
else
  bad "không thấy src/dire_modal/manifest.py — chạy từ repo root"
fi

echo
if [[ $FAIL -ne 0 ]]; then
  say "CÒN CỔNG ĐỎ — sửa trước khi tải dữ liệu (tải 9 GB rồi mới phát hiện thiếu chỗ là việc tệ nhất)"
  exit 3
fi
say "MÁY ĐỦ ĐIỀU KIỆN. Bước tiếp:"
echo "  bash tools/scratch_repro.sh --stage list --need a,gallery,e2min"
