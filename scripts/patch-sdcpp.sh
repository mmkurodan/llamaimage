#!/usr/bin/env bash
# Applies out-of-tree modifications to the freshly-fetched stable-diffusion.cpp sources.
# Designed to run on a PRISTINE checkout (as CI does after fetch); each section is guarded
# by a sentinel so re-runs are idempotent.
#
# Patch: expose the current denoised latent (g_sd_step_latent) just before each step
# progress callback, so the JNI layer can build a lightweight CPU preview of the image
# as it forms. CPU-only build; no GPU/Vulkan patches.
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

# util.h — declare the global
py_replace "$DEST/util.h" "g_sd_step_latent" \
'void pretty_progress(int step, int steps, float time);' \
'void pretty_progress(int step, int steps, float time);

// llamaimage hook (out-of-tree, see scripts/patch-sdcpp.sh): current denoised latent,
// set just before each step progress callback (valid only during the callback). The JNI
// layer reads it to build a lightweight preview of the image as it forms.
struct ggml_tensor;
extern struct ggml_tensor* g_sd_step_latent;'

# util.cpp — define the global
py_replace "$DEST/util.cpp" "g_sd_step_latent" \
'void* sd_progress_cb_data              = NULL;' \
'void* sd_progress_cb_data              = NULL;

struct ggml_tensor* g_sd_step_latent = nullptr;'

# stable-diffusion.cpp — publish the denoised latent around the progress callback
py_replace "$DEST/stable-diffusion.cpp" "g_sd_step_latent" \
'                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);' \
'                g_sd_step_latent = denoised;  // expose for JNI preview
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                g_sd_step_latent = nullptr;'

echo "[patch-sdcpp] done."
