#!/usr/bin/env bash

set -euo pipefail

# Git Bash (MSYS2) đổi mọi tham số bắt đầu bằng "/" thành đường dẫn Windows:
#   --path /raw  →  C:/Program Files/Git/raw  →  container Linux không thấy.
# Hai biến này tắt việc đó; trên Linux/macOS chúng bị bỏ qua, vô hại.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
# CỔNG 0 — kiểm tra repo import được BẰNG MODAL THẬT trước khi tốn tiền.
#
# Vì sao cần: `compute_dire.py` từng có `from __future__ import annotations`
# (PEP 563). Modal thật đọc `__annotations__` thô nên thấy 'str' thay vì str và
# chết ngay lúc IMPORT module:
#
#     KeyError: 'str'
#     → InvalidError("No class parameter encoder implemented for type `...`")
#     → AttributeError: 'str' object has no attribute '__name__'
#
# AttributeError cuối che mất nguyên nhân, và lỗi ở import nên MỌI lệnh của
# module chết — kể cả `dire_batch --dry-run`. Bộ test dùng stub không bắt được
# (stub cũ trả thẳng default, không kiểm annotation), nên phải có bước này.
#
# Dùng:
#   bash scripts/check.sh              # tạo .venv lần đầu, rồi chạy
#   bash scripts/check.sh --recreate   # xoá .venv và cài lại
#
# Không gọi cloud, không cần token, không tốn credit: chỉ import module.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"

if [ "${1:-}" = "--recreate" ]; then
  rm -rf "$VENV"
fi

# venv trên Windows đặt binary ở `Scripts/`, KHÔNG phải `bin/` — bản cũ hardcode
# `.venv/bin/python` nên CỔNG 0 không bao giờ chạy được trên Git Bash, dù README
# ghi "chạy trước mọi lệnh khác". Dò cả hai để một script chạy ở cả hai hệ.
if [ -x "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"; PIP="$VENV/Scripts/pip.exe"
elif [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"; PIP="$VENV/bin/pip"
else
  echo "[i] tạo venv tại $VENV"
  if command -v py >/dev/null 2>&1; then py -3 -m venv "$VENV"; else python3 -m venv "$VENV"; fi
  if [ -x "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"; PIP="$VENV/Scripts/pip.exe"
  else
    PY="$VENV/bin/python"; PIP="$VENV/bin/pip"
  fi
fi

# In tiếng Việt trên console cp1252 (Windows mặc định) sẽ nổ `UnicodeEncodeError`
# giữa lúc chạy test. Ép UTF-8 cho mọi tiến trình con.
export PYTHONIOENCODING=utf-8

echo "[i] cài phụ thuộc dev (modal THẬT, không stub)"
"$PIP" -q install -r requirements-dev.txt
"$PIP" -q install -e . 2>/dev/null || echo "[i] bỏ qua 'pip install -e .' (không bắt buộc để test)"

echo
echo "[1/3] import bằng modal thật"
"$PY" -m pytest tests/test_real_modal_import.py -q

echo
echo "[2/3] toàn bộ test (stub + thật)"
"$PY" -m pytest tests/ -q

echo
echo "[3/3] pyflakes trên code đã sửa"
"$PY" -m pyflakes src/dire_core src/dire_local src/dire_modal tests/ || {
  echo "[!] pyflakes báo lỗi — xem ở trên"; exit 1; }

echo
echo "✅ CỔNG 0 đạt — repo import được bằng Modal thật, test xanh."
echo "   Chạy trên máy nhà (không Modal):  dire info && dire detect anh.jpg --weights <ckpt>"
echo "   Quy trình 1 (tái lập paper):  bash scripts/01_repro.sh --quick"
echo "   Quy trình 2 (cải thiện):      bash scripts/02_improve.sh --stage plan"
