#!/usr/bin/env bash
set -euo pipefail
# Đưa checkpoint của tác giả vào vị trí quy ước
# (chạy 1 lần, Volume CLI)
modal volume cp /raw/checkpoints/lsun_adm.pth \
  dire-ckpt:/pretrained/lsun_adm.pth

modal run -m dire_modal.eval::main --ckpt /ckpt/pretrained/lsun_adm.pth
# ✅ Cổng chặn: ACC trên ADM phải ≈ 99%