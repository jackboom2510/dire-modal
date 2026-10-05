#!/usr/bin/env bash
set -euo pipefail
modal run --detach -m dire_modal.train::main --exp lsun_adm_repro
# --detach: job vẫn chạy khi đóng terminal
# Theo dõi: modal app logs dire