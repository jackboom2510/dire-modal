#!/usr/bin/env bash
set -euo pipefail
# 1. Dò cây remote (chạy 1 lần, chép path vào manifest.py::BAIDU_ROOT)
# modal run -m dire_modal.download::ls_remote --path /apps

# 2. Tải checkpoints + Tier A (phương án tối thiểu reproduce)
modal run -m dire_modal.download::main --tiers "ckpt,A" --concurrency 4