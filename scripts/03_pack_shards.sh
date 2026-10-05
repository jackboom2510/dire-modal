#!/usr/bin/env bash
set -euo pipefail
modal run -m dire_modal.pack::main --tiers A
modal run -m dire_modal.pack::stats
modal run -m dire_modal.bench::throughput