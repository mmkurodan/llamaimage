#!/usr/bin/env bash
# Applies out-of-tree modifications to the freshly-fetched stable-diffusion.cpp +
# ggml sources. Designed to run on a PRISTINE checkout (as CI does after fetch);
# each section is guarded by a sentinel so re-runs are idempotent.
#
# Patches:
#   1. util.{h,cpp}            — globals g_sd_step_latent / g_sd_use_gpu / g_sd_active_gpu
#   2. stable-diffusion.cpp    — expose denoised latent for preview; gate the Vulkan
#                                backend on g_sd_use_gpu and report if it initialised
#   3. ggml-vulkan/CMakeLists  — use a host-built vulkan-shaders-gen (cross-compile) and
#                                don't require glslc as a find_package COMPONENT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$(cd "$SCRIPT_DIR/.." && pwd)/app/src/main/cpp/sd}"

echo "[patch-sdcpp] applying patches to $DEST"

py_replace() {
    # py_replace <file> <sentinel> <old> <new>  — replace once; skip if sentinel present.
    python3 - "$@" <<'PYEOF'
import sys
path, sentinel, old, new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(path).read()
if sentinel in text:
    print(f"[patch-sdcpp] {path} already patched ({sentinel}) — skipping")
    sys.exit(0)
if old not in text:
    raise RuntimeError(f"Expected pattern not found in {path} for sentinel {sentinel}")
open(path, "w").write(text.replace(old, new, 1))
print(f"[patch-sdcpp] patched {path} ({sentinel})")
PYEOF
}

# ---------------------------------------------------------------------------
# 1. util.h / util.cpp — shared globals
# ---------------------------------------------------------------------------
py_replace "$DEST/util.h" "g_sd_active_gpu" \
'void pretty_progress(int step, int steps, float time);' \
'void pretty_progress(int step, int steps, float time);

// --- llamaimage hooks (out-of-tree, see scripts/patch-sdcpp.sh) -----------
struct ggml_tensor;
// Current denoised latent, set just before each step progress callback (valid only
// during the callback). The JNI layer reads it to build a lightweight preview.
extern struct ggml_tensor* g_sd_step_latent;
// Requested at context creation: try a GPU (Vulkan) backend when true.
extern bool g_sd_use_gpu;
// Set to 1 by the backend selector when a GPU backend actually initialised.
extern int g_sd_active_gpu;'

py_replace "$DEST/util.cpp" "g_sd_active_gpu" \
'void* sd_progress_cb_data              = NULL;' \
'void* sd_progress_cb_data              = NULL;

struct ggml_tensor* g_sd_step_latent = nullptr;
bool g_sd_use_gpu    = false;
int  g_sd_active_gpu = 0;'

# ---------------------------------------------------------------------------
# 2a. stable-diffusion.cpp — expose denoised latent around the progress callback
# ---------------------------------------------------------------------------
py_replace "$DEST/stable-diffusion.cpp" "g_sd_step_latent" \
'                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);' \
'                g_sd_step_latent = denoised;  // expose for JNI preview
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                g_sd_step_latent = nullptr;'

# ---------------------------------------------------------------------------
# 2b. stable-diffusion.cpp — gate Vulkan on g_sd_use_gpu; record success
# ---------------------------------------------------------------------------
py_replace "$DEST/stable-diffusion.cpp" "g_sd_active_gpu = 1" \
'#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif' \
'#ifdef SD_USE_VULKAN
        if (g_sd_use_gpu) {
            LOG_DEBUG("Using Vulkan backend");
            for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
                backend = ggml_backend_vk_init(device);
            }
            if (!backend) {
                LOG_WARN("Failed to initialize Vulkan backend");
            } else {
                g_sd_active_gpu = 1;
            }
        }
#endif'

# ---------------------------------------------------------------------------
# 3. ggml-vulkan/CMakeLists.txt — host shader generator + relaxed glslc find
#    (only consulted when GGML_VULKAN is ON, i.e. a GPU build)
# ---------------------------------------------------------------------------
VK_CMAKE="$DEST/ggml/src/ggml-vulkan/CMakeLists.txt"
if [ -f "$VK_CMAKE" ]; then
    py_replace "$VK_CMAKE" "find_package(Vulkan REQUIRED)" \
'find_package(Vulkan COMPONENTS glslc REQUIRED)' \
'find_package(Vulkan REQUIRED)'

    py_replace "$VK_CMAKE" "GGML_VK_HOST_SHADERS_GEN" \
'    add_subdirectory(vulkan-shaders)

    set (_ggml_vk_genshaders_cmd vulkan-shaders-gen)' \
'    # llamaimage: cross-compiling to Android cannot run a target-built shader gen,
    # so accept a host-built one via -DGGML_VK_HOST_SHADERS_GEN=<path>.
    if (DEFINED GGML_VK_HOST_SHADERS_GEN)
        set (_ggml_vk_genshaders_cmd ${GGML_VK_HOST_SHADERS_GEN})
    else()
        add_subdirectory(vulkan-shaders)
        set (_ggml_vk_genshaders_cmd vulkan-shaders-gen)
    endif()'
else
    echo "[patch-sdcpp] $VK_CMAKE not present — skipping Vulkan CMake patch"
fi

echo "[patch-sdcpp] done."
