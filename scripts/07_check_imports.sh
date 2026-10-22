#!/usr/bin/env bash
set -euo pipefail
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
#   ./scripts/07_check_imports.sh              # tạo .venv lần đầu, rồi chạy
#   ./scripts/07_check_imports.sh --recreate   # xoá .venv và cài lại
#
# Không gọi cloud, không cần token, không tốn credit: chỉ import module.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"

if [ "${1:-}" = "--recreate" ]; then
  rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[i] tạo venv tại $VENV"
  python3 -m venv "$VENV"
fi

echo "[i] cài phụ thuộc dev (modal THẬT, không stub)"
"$VENV/bin/pip" -q install -r requirements-dev.txt

echo
echo "[1/3] import bằng modal thật"
"$VENV/bin/python" -m pytest tests/test_real_modal_import.py -q

echo
echo "[2/3] toàn bộ test (stub + thật)"
"$VENV/bin/python" -m pytest tests/ -q

echo
echo "[3/3] pyflakes trên code đã sửa"
"$VENV/bin/python" -m pyflakes src/dire_modal/compute_dire.py tests/ || {
  echo "[!] pyflakes báo lỗi — xem ở trên"; exit 1; }

echo
echo "✅ CỔNG 0 đạt — repo import được bằng Modal thật, test xanh."
echo "   Bước tiếp: modal run -m dire_modal.compute_dire::dire_batch --n 200 --dry-run"
