#!/usr/bin/env bash
set -euo pipefail
# Nạp .env (KHÔNG commit)
set -a; source .env; set +a

# --- Bắt buộc cho pipeline tải dữ liệu ---
modal secret create baidu-cred \
  BDUSS="$BAIDU_BDUSS" \
  STOKEN="$BAIDU_STOKEN" \
  --force

modal secret create r2-cred \
  R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
  R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  R2_BUCKET="$R2_BUCKET" \
  --force

# --- Cần cho mirror công khai (archive.py) ---
# HF token: https://huggingface.co/settings/tokens (quyền "write")
if [[ -n "${HF_TOKEN:-}" ]]; then
  modal secret create hf-cred HF_TOKEN="$HF_TOKEN" --force
  echo "[ok] hf-cred"
else
  echo "[!] bỏ qua hf-cred (HF_TOKEN trống trong .env)"
fi

# Kaggle API key: https://www.kaggle.com/settings → "Create New Token" (tải kaggle.json)
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  modal secret create kaggle-cred \
    KAGGLE_USERNAME="$KAGGLE_USERNAME" \
    KAGGLE_KEY="$KAGGLE_KEY" \
    --force
  echo "[ok] kaggle-cred"
else
  echo "[!] bỏ qua kaggle-cred (KAGGLE_USERNAME/KAGGLE_KEY trống trong .env)"
fi

echo "[ok] secrets đã tạo"
echo
echo "Kiểm tra:  modal secret list"
