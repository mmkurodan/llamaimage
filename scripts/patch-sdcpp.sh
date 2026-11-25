#!/usr/bin/env bash
# Out-of-tree modifications to the fetched stable-diffusion.cpp sources (run by fetch-sdcpp.sh).
#
# None are needed at the current pin: the option-based C API provides what earlier pins had to
# be patched for — cancellation (sd_cancel_generation), staged progress (generation progress
# callback with stage "sampling" vs. decode) and latent previews (sd_set_preview_callback).
# Keep this script as the place for future patches; each must be idempotent (sentinel-guarded)
# and its anchor must match exactly once.
set -euo pipefail
DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/app/src/main/cpp/sd}"
echo "[patch-sdcpp] no patches for $DEST"
