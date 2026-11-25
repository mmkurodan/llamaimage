#!/usr/bin/env bash
# Fetch the pinned native sources used by the image-generation layer. They are NOT committed to
# this repository (see .gitignore); CMake invokes this script at configure time, and CI runs it
# before the build.
#
#   stable-diffusion.cpp @ 79dc7ca9 (2026-11-24) — option-based C API (sd_ctx_opt_t /
#     sd_img_gen_opt_t) with built-in cancellation, staged progress and latent previews, so the
#     engine is used unpatched; runtime backend selection (`backend` option) incl. the OpenCL
#     backend for Adreno GPUs.
#   ggml — leejet/ggml at the exact gitlink sd.cpp @ 79dc7ca9 references (its patched GGML
#     extensions), so the ggml API the engine compiles against always matches.
#   OpenCL-Headers + OpenCL-ICD-Loader (Khronos, Apache-2.0) — only to COMPILE and LINK the GPU
#     variant (libsd_jni_gpu.so). The loader's libOpenCL.so is never packaged: at runtime the
#     device's own vendor libOpenCL.so is used (declared via <uses-native-library>).
#
# A .pin file records what was fetched; a checkout from another pin is replaced.
set -euo pipefail

SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp.git"
SDCPP_COMMIT="79dc7ca9074d79419a46380cdb95a4524db9b74c"

GGML_REPO="https://github.com/leejet/ggml.git"
# gitlink recorded by stable-diffusion.cpp @ SDCPP_COMMIT (git ls-tree HEAD ggml)
GGML_COMMIT="eeda9ec48fc84a3df0bb9dcc19ae9053075a2539"

CL_HEADERS_REPO="https://github.com/KhronosGroup/OpenCL-Headers.git"
CL_HEADERS_COMMIT="82f60316cc142e7b21dc1d8983b26de18858fbd9"
CL_LOADER_REPO="https://github.com/KhronosGroup/OpenCL-ICD-Loader.git"
CL_LOADER_COMMIT="d0be526adb156de7f701024acc811182c984d4c5"

# Resolve repo root from this script's location (repo_root/scripts/fetch-sdcpp.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${1:-$REPO_ROOT/app/src/main/cpp/sd}"
CL_DEST="$(dirname "$DEST")/opencl"
PIN="sdcpp=$SDCPP_COMMIT ggml=$GGML_COMMIT clh=$CL_HEADERS_COMMIT icd=$CL_LOADER_COMMIT"

# Fetch a single commit by SHA (GitHub allows reachable-SHA fetches). This keeps the
# download tiny compared to a full clone.
fetch_commit() {
  local repo="$1" commit="$2" dir="$3"
  rm -rf "$dir"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$repo"
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
  rm -rf "$dir/.git"
}

if [ -f "$DEST/.pin" ] && [ "$(cat "$DEST/.pin")" = "$PIN" ] \
   && [ -f "$DEST/include/stable-diffusion.h" ] && [ -f "$DEST/ggml/CMakeLists.txt" ] \
   && [ -f "$CL_DEST/headers/CL/cl.h" ] && [ -f "$CL_DEST/icd-loader/CMakeLists.txt" ]; then
  echo "[fetch-sdcpp] pinned sources already present at $DEST — skipping fetch"
else
  echo "[fetch-sdcpp] fetching stable-diffusion.cpp @ ${SDCPP_COMMIT:0:12} -> $DEST"
  fetch_commit "$SDCPP_REPO" "$SDCPP_COMMIT" "$DEST"
  echo "[fetch-sdcpp] fetching ggml @ ${GGML_COMMIT:0:12} -> $DEST/ggml"
  # sd.cpp leaves an empty 'ggml' gitlink dir on a non-submodule checkout; replace it.
  fetch_commit "$GGML_REPO" "$GGML_COMMIT" "$DEST/ggml"
  echo "[fetch-sdcpp] fetching OpenCL headers / ICD loader -> $CL_DEST"
  fetch_commit "$CL_HEADERS_REPO" "$CL_HEADERS_COMMIT" "$CL_DEST/headers"
  fetch_commit "$CL_LOADER_REPO" "$CL_LOADER_COMMIT" "$CL_DEST/icd-loader"
  echo "$PIN" > "$DEST/.pin"
  echo "[fetch-sdcpp] done."
fi

# Local patches (none at this pin; kept as the hook for future ones).
bash "$SCRIPT_DIR/patch-sdcpp.sh" "$DEST"
