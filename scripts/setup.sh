#!/usr/bin/env bash
# ĐIỀU KIỆN TIÊN QUYẾT (một lần) — KHÔNG phải quy trình thực nghiệm.
#
#   bash scripts/setup.sh
#
# Tạo secret trên Modal từ .env (KHÔNG commit .env). Idempotent: dùng --force.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ -f .env ] || die "không thấy .env — chép .env.example thành .env rồi điền token"
need_cmd modal

# shellcheck disable=SC1091
set -a; . ./.env; set +a

say "tạo secret baidu-cred (bắt buộc để tải dữ liệu)"
[ -n "${BAIDU_BDUSS:-}" ] || die "BAIDU_BDUSS trống trong .env"
modal secret create baidu-cred \
  BDUSS="$BAIDU_BDUSS" STOKEN="${BAIDU_STOKEN:-}" --force

say "r2-cred (tùy chọn — kho checkpoint + manifest sha256)"
if [ -n "${R2_ACCOUNT_ID:-}" ]; then
  modal secret create r2-cred \
    R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
    R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    R2_BUCKET="${R2_BUCKET:-dire-archive}" --force
else
  note "bỏ qua (R2_ACCOUNT_ID trống)"
fi

say "hf-cred (mirror công khai — đường thoát khỏi 76.6 KiB/s của Baidu)"
if [ -n "${HF_TOKEN:-}" ]; then
  modal secret create hf-cred HF_TOKEN="$HF_TOKEN" --force
else
  note "bỏ qua (HF_TOKEN trống)"
fi

say "kaggle-cred (mirror phụ)"
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  modal secret create kaggle-cred \
    KAGGLE_USERNAME="$KAGGLE_USERNAME" KAGGLE_KEY="$KAGGLE_KEY" --force
else
  note "bỏ qua (KAGGLE_USERNAME/KAGGLE_KEY trống)"
fi

say "xong"
note "kiểm: modal secret list"
note "venv cho test: python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt"
note "cổng 0: bash scripts/check.sh"
