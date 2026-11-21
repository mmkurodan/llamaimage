#!/usr/bin/env bash
# Applies out-of-tree modifications to the freshly-fetched stable-diffusion.cpp sources.
# Designed to run on a PRISTINE checkout (as CI does after fetch); each section is guarded
# by a sentinel so re-runs are idempotent, and every anchor must match exactly once.
#
# Patches (all consumed by app/src/main/cpp/jni/jni_sd.cpp):
#   1. g_sd_step_latent — the current denoised latent, published around each sampling-step
#      progress callback, so the JNI layer can build a lightweight preview as the image forms.
#   2. g_sd_phase — which stage is reporting progress. sd.cpp routes BOTH the sampling steps and
#      the VAE-decode tiles (sd_tiling) through the same progress callback; without this the
#      tile progress ("1/16 … 16/16") looks like a second generation run after "6/6".
#   3. sd_unwind_cleanup — frees work_ctx and the UNet compute buffer when an exception unwinds
#      through generate_image(). The JNI layer cancels a generation by throwing from the
#      progress callback, so this keeps a cancel from leaking memory.
# CPU-only build; no GPU/Vulkan patches.
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
count = text.count(old)
if count != 1:
    raise RuntimeError(f"Expected exactly one match in {path} for sentinel {sentinel}, found {count}")
open(path, "w").write(text.replace(old, new, 1))
print(f"[patch-sdcpp] patched {path} ({sentinel})")
PYEOF
}

# --- 1. step latent (preview) ------------------------------------------------------------

py_replace "$DEST/util.h" "g_sd_step_latent" \
'void pretty_progress(int step, int steps, float time);' \
'void pretty_progress(int step, int steps, float time);

// llamaimage hook (out-of-tree, see scripts/patch-sdcpp.sh): current denoised latent,
// set just before each step progress callback (valid only during the callback). The JNI
// layer reads it to build a lightweight preview of the image as it forms.
struct ggml_tensor;
extern struct ggml_tensor* g_sd_step_latent;'

py_replace "$DEST/util.cpp" "g_sd_step_latent" \
'void* sd_progress_cb_data              = NULL;' \
'void* sd_progress_cb_data              = NULL;

struct ggml_tensor* g_sd_step_latent = nullptr;'

py_replace "$DEST/stable-diffusion.cpp" "g_sd_step_latent" \
'                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);' \
'                g_sd_step_latent = denoised;  // expose for JNI preview
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                g_sd_step_latent = nullptr;'

# --- 2. progress phase (sampling vs. VAE decode tiles) -------------------------------------

py_replace "$DEST/util.h" "g_sd_phase" \
'extern struct ggml_tensor* g_sd_step_latent;' \
'extern struct ggml_tensor* g_sd_step_latent;

// llamaimage hook: which stage the progress callback is reporting. 1 = sampling steps,
// 2 = after sampling (VAE decode; with vae_tiling its tiles also call the progress
// callback). The JNI layer resets it to 0 before each generation.
extern int g_sd_phase;'

py_replace "$DEST/util.cpp" "g_sd_phase" \
'struct ggml_tensor* g_sd_step_latent = nullptr;' \
'struct ggml_tensor* g_sd_step_latent = nullptr;
int g_sd_phase = 0;'

py_replace "$DEST/stable-diffusion.cpp" "g_sd_phase" \
'        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng);' \
'        g_sd_phase = 1;  // llamaimage: progress callbacks below are sampling steps
        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng);
        g_sd_phase = 2;  // llamaimage: sampling done; later callbacks are VAE-decode tiles'

# --- 3. cleanup when a cancel exception unwinds through generate_image() -------------------

py_replace "$DEST/stable-diffusion.cpp" "sd_unwind_cleanup" \
'sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,' \
'// llamaimage hook (out-of-tree, see scripts/patch-sdcpp.sh): if an exception unwinds through
// generate_image() (the JNI layer cancels by throwing from the progress callback), release what
// the normal path would have: work_ctx (freed at the end of generate_image) and the UNet compute
// buffer (freed at the end of sample()). Does nothing on a normal return.
#include <exception>
struct sd_unwind_cleanup {
    sd_ctx_t* sd_ctx;
    struct ggml_context* work_ctx;
#if __cplusplus >= 201703L
    int uncaught_at_entry = std::uncaught_exceptions();
    bool unwinding() const { return std::uncaught_exceptions() > uncaught_at_entry; }
#else
    bool unwinding() const { return std::uncaught_exception(); }
#endif
    ~sd_unwind_cleanup() {
        if (!unwinding()) {
            return;
        }
        g_sd_step_latent = nullptr;
        if (sd_ctx != NULL && sd_ctx->sd != NULL && sd_ctx->sd->diffusion_model) {
            sd_ctx->sd->diffusion_model->free_compute_buffer();
        }
        if (work_ctx != NULL) {
            ggml_free(work_ctx);
        }
    }
};

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,'

py_replace "$DEST/stable-diffusion.cpp" "sd_unwind_guard" \
'    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.' \
'    sd_unwind_cleanup sd_unwind_guard{sd_ctx, work_ctx};  // llamaimage: cancel cleanup
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.'

echo "[patch-sdcpp] done."
