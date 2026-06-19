#!/usr/bin/env bash
# Fetch the pinned stable-diffusion.cpp + ggml sources used by the image-generation
# native layer. The sources are NOT committed to this repository (see .gitignore);
# CMake invokes this script at configure time, and CI runs it before the build.
#
# Why these exact commits:
#   - stable-diffusion.cpp @ 9578fdcc (2024-11-30) has the flat txt2img() API that is
#     trivial to bind from JNI, loads an all-in-one SD1.5 GGUF directly via model_path,
#     and carries only a single CLIP vocab (no multi-GB tokenizer headers that later
#     master revisions added for FLUX/SD3/Qwen). Minimal == right for this PoC.
#   - ggml is pinned to the exact gitlink commit that sd.cpp @ 9578fdcc references,
#     so the ggml API the engine compiles against always matches.
set -euo pipefail

SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp.git"
SDCPP_COMMIT="9578fdcc4632dc3de5565f28e2fb16b7c18f8d48"

GGML_REPO="https://github.com/ggerganov/ggml.git"
# gitlink recorded by stable-diffusion.cpp @ SDCPP_COMMIT (git ls-tree HEAD ggml)
GGML_COMMIT="6fcbd60bc72ac3f7ad43f78c87e535f2e6206f58"

# Resolve repo root from this script's location (repo_root/scripts/fetch-sdcpp.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${1:-$REPO_ROOT/app/src/main/cpp/sd}"

# Idempotent: skip the network fetch if both trees are already materialized,
# but always run the patch step below so local modifications are applied.
SKIP_FETCH=0
if [ -f "$DEST/stable-diffusion.h" ] && [ -f "$DEST/ggml/CMakeLists.txt" ]; then
  echo "[fetch-sdcpp] already present at $DEST — skipping fetch"
  SKIP_FETCH=1
fi

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

if [ "$SKIP_FETCH" -eq 0 ]; then
  echo "[fetch-sdcpp] fetching stable-diffusion.cpp @ ${SDCPP_COMMIT:0:12} -> $DEST"
  fetch_commit "$SDCPP_REPO" "$SDCPP_COMMIT" "$DEST"

  echo "[fetch-sdcpp] fetching ggml @ ${GGML_COMMIT:0:12} -> $DEST/ggml"
  # sd.cpp leaves an empty 'ggml' gitlink dir on a non-submodule checkout; replace it.
  fetch_commit "$GGML_REPO" "$GGML_COMMIT" "$DEST/ggml"

  echo "[fetch-sdcpp] done."
fi

# Always apply local patches (idempotent — already-patched files are skipped).
bash "$SCRIPT_DIR/patch-sdcpp.sh" "$DEST"
