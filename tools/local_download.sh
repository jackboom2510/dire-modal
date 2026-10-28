#!/usr/bin/env bash
# TIỆN ÍCH — tải dữ liệu DIRE về MÁY LOCAL, không dùng Modal, không dùng Volume.
#
#   bash tools/local_download.sh                             # kế hoạch e2min (không tải)
#   DEST=./data/raw bash tools/local_download.sh --mode fetch
#   TIERS=e2,ckpts bash tools/local_download.sh --mode fetch  # 3.25 GB, ~12 h
#   bash tools/local_download.sh --mode verify                # đã tải đúng chưa?
#   bash tools/local_download.sh --mode manifest              # sha256 TSV (để mirror)
#   MSYS_NO_PATHCONV=1 ... (script tự bật)
#
# Vì sao có script này: đường Modal bị ba trần của nền tảng (attempt 1800 s, lần
# gọi 7200 s, app 20 h). Ở 76.6 KiB/s — trần của TÀI KHOẢN Baidu — thì e2 1.94 GB
# = 7.4 h, nên một file 269 MB luôn bị cắt giữa chừng. Máy local không có ba trần
# đó: bật terminal, để chạy, Ctrl+C lúc nào cũng được.
#
# Yêu cầu (một lần):
#   1. BaiduPCS-Go trong PATH   https://github.com/qjfoidnh/BaiduPCS-Go/releases
#      ⚠ user free PHẢI đặt  max_parallel=1, max_download_load=1  trong config,
#        nếu không Baidu throttle 0 B/s hàng giờ.
#   2. Đăng nhập:  BaiduPCS-Go login -bduss=<BDUSS>
#
# Script KHÔNG tự tải gì nếu thiếu --mode fetch (mặc định là kế hoạch).
set -euo pipefail

# Git Bash (MSYS2) đổi mọi tham số bắt đầu bằng "/" thành đường dẫn Windows:
#   --path /raw  →  C:/Program Files/Git/raw  →  container Linux không thấy.
# Hai biến này tắt việc đó; trên Linux/macOS chúng bị bỏ qua, vô hại.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=tools/lib/native_path.sh
. "$REPO_ROOT/tools/lib/native_path.sh"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "[STOP] thiếu python3/python" >&2; exit 1; }

TIERS="${TIERS:-e2min}"
DEST="${DEST:-data/raw}"

echo "== tải LOCAL (không Modal, không Volume) =="
echo "   nguồn : Baidu — cần BaiduPCS-Go + đã login"
echo "   đích  : $DEST   (thư mục này KHÔNG bị commit: xem .gitignore)"
echo "   tier  : $TIERS"
echo "   Ctrl+C bất cứ lúc nào: phần đã tải nằm trong $DEST/.staging và sẽ resume."
echo

# PYTHONPATH để chạy được cả khi chưa `pip install -e .`
# PYTHONPATH là BIẾN MÔI TRƯỜNG nên MSYS không đổi hộ: trên Git Bash
# `$REPO_ROOT` là `/d/...`, python.exe của Windows sẽ tìm `D:\d\...` và báo
# "No module named dire_modal" — trông y như chưa cài package.
PYTHONPATH="$(native_path "$REPO_ROOT/src")${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m dire_modal.local_fetch --tiers "$TIERS" --dest "$DEST" "$@"
