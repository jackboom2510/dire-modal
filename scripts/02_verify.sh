#!/usr/bin/env bash
set -euo pipefail
modal run -m dire_modal.download::verify
# Tùy chọn: backup cold lên R2 sau khi verify OK
# modal run -m dire_modal.archive::push_all