#!/usr/bin/env bash
set -euo pipefail
# Nạp .env (KHÔNG commit)
set -a; source .env; set +a

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

echo "[ok] secrets đã tạo"