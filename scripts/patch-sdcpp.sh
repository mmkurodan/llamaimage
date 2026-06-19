#!/usr/bin/env bash
# Applies out-of-tree modifications to the fetched stable-diffusion.cpp sources.
# These add a g_sd_step_latent hook so that the JNI progress callback can read
# the current denoised latent tensor and produce a lightweight preview image.
#
# The script is idempotent: files that are already patched are silently skipped.
# Called by fetch-sdcpp.sh after each fetch, and safe to run standalone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$(cd "$SCRIPT_DIR/.." && pwd)/app/src/main/cpp/sd}"

echo "[patch-sdcpp] applying patches to $DEST"

# ---------------------------------------------------------------------------
# util.h — declare g_sd_step_latent
# ---------------------------------------------------------------------------
if grep -q "g_sd_step_latent" "$DEST/util.h" 2>/dev/null; then
    echo "[patch-sdcpp] util.h already patched — skipping"
else
    python3 - "$DEST/util.h" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
OLD = 'void pretty_progress(int step, int steps, float time);'
NEW = (
    OLD
    + '\n\n'
    + '// Set by the denoising loop to the current denoised latent tensor just before\n'
    + '// each pretty_progress() call (step > 0 only). Valid only during the callback.\n'
    + '// The JNI layer reads this to produce a lightweight latent preview image.\n'
    + 'struct ggml_tensor;\n'
    + 'extern struct ggml_tensor* g_sd_step_latent;'
)
if OLD not in text:
    raise RuntimeError(f'Expected pattern not found in {path}')
open(path, 'w').write(text.replace(OLD, NEW, 1))
PYEOF
    echo "[patch-sdcpp] patched util.h"
fi

# ---------------------------------------------------------------------------
# util.cpp — define g_sd_step_latent
# ---------------------------------------------------------------------------
if grep -q "g_sd_step_latent" "$DEST/util.cpp" 2>/dev/null; then
    echo "[patch-sdcpp] util.cpp already patched — skipping"
else
    python3 - "$DEST/util.cpp" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
OLD = 'void* sd_progress_cb_data              = NULL;'
NEW = OLD + '\n\nstruct ggml_tensor* g_sd_step_latent = nullptr;'
if OLD not in text:
    raise RuntimeError(f'Expected pattern not found in {path}')
open(path, 'w').write(text.replace(OLD, NEW, 1))
PYEOF
    echo "[patch-sdcpp] patched util.cpp"
fi

# ---------------------------------------------------------------------------
# stable-diffusion.cpp — expose denoised latent around the progress callback
# ---------------------------------------------------------------------------
if grep -q "g_sd_step_latent" "$DEST/stable-diffusion.cpp" 2>/dev/null; then
    echo "[patch-sdcpp] stable-diffusion.cpp already patched — skipping"
else
    python3 - "$DEST/stable-diffusion.cpp" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
OLD = '                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);'
NEW = (
    '                g_sd_step_latent = denoised;  // expose for JNI preview\n'
    + OLD + '\n'
    + '                g_sd_step_latent = nullptr;'
)
if OLD not in text:
    raise RuntimeError(f'Expected pattern not found in {path}')
open(path, 'w').write(text.replace(OLD, NEW, 1))
PYEOF
    echo "[patch-sdcpp] patched stable-diffusion.cpp"
fi

echo "[patch-sdcpp] done."
