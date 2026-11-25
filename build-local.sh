#!/usr/bin/env bash
# ============================================================================
# build-local.sh — llamaimage local APK build (mirrors /root/llama/build-local.sh).
#
# The NDK's own linker (x86_64 ld.lld) cannot run in this proot + qemu-on-aarch64 environment,
# so Gradle cannot do the native build here. The native libraries
# (libsd_jni.so = armv8-a baseline, libsd_jni_fp16.so = armv8.2-a+fp16+dotprod) come from:
#   --fetch (default)  the latest successful CI run's APK artifact (x86_64 runner, NDK lld)
#   --local-native     a local build: the NDK clang compiles (it runs fine under qemu) and a
#                      NATIVE aarch64 linker links (clang --ld-path; GNU ld unless a native
#                      ld.lld is on PATH). Same CMakeLists and arguments as the Gradle modules.
#   --no-fetch         whatever is already in app/src/main/jniLibs/arm64-v8a/
# and the APK is then assembled with -PsdPrebuiltNative=true: Gradle skips the NDK build (and the
# :sdjni-fp16 module) and packages the .so files from app/src/main/jniLibs/arm64-v8a/.
#
# The .so must match the Java JNI declarations. --fetch refuses a CI run whose commit differs
# from the local tree in native code (app/src/main/cpp, scripts/*sdcpp.sh): push first and wait
# for CI, build with --local-native, or pass --allow-stale.
#
# Environment note: aapt2 (x86_64) SIGILLs under proot's default `-q` qemu even with
# QEMU_CPU=max, but runs fine via an EXPLICIT `qemu-x86_64-static` call. AGP runs d8 /
# zip-align / apksigner in-process (JVM), so aapt2 is the only native tool we must wrap —
# done here via -Pandroid.aapt2FromMavenOverride pointing at a qemu wrapper.
#
# Usage: ./build-local.sh [--debug|--release] [--fetch|--local-native|--no-fetch] [--run-id ID]
#                         [--allow-stale]
#   --release  signed release APK (default)   --debug  debug APK
#   keystore:  KEYSTORE_PATH or ./keystore.jks or ~/keystore.jks or /root/keystore.jks
#              creds: KEYSTORE_PASSWORD/KEY_ALIAS/KEY_PASSWORD (default micklab/mykey/micklab)
#   output:    /sdcard/Download/llamaimage-app-<variant>.apk (or APK_OUTPUT_DIR)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO_ROOT"
MODULE="app"
JNILIBS="$REPO_ROOT/$MODULE/src/main/jniLibs/arm64-v8a"
NATIVE_PATHS=(app/src/main/cpp scripts/fetch-sdcpp.sh scripts/patch-sdcpp.sh)

VARIANT="release"; DO_FETCH=1; LOCAL_NATIVE=0; RUN_ID=""; ARTIFACT="app-release-apk"; ALLOW_STALE=0
KS_INSTALLED=""; TMP=""; WRAP_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)   VARIANT="debug";   ARTIFACT="app-debug-apk" ;;
    --release) VARIANT="release"; ARTIFACT="app-release-apk" ;;
    --fetch)   DO_FETCH=1; LOCAL_NATIVE=0 ;;
    --local-native) LOCAL_NATIVE=1; DO_FETCH=0 ;;
    --no-fetch) DO_FETCH=0; LOCAL_NATIVE=0 ;;
    --run-id)  RUN_ID="$2"; shift ;;
    --allow-stale) ALLOW_STALE=1 ;;
    -h|--help) sed -n '3,/^# ====/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac; shift
done

export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk}"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export QEMU_CPU="${QEMU_CPU:-max}"
GRADLE_RUNNER="$(command -v gradle || echo ./gradlew)"
REPO_SLUG="$(git remote get-url origin 2>/dev/null | sed -E 's|.*github\.com[:/]||; s|\.git$||' || true)"

cleanup(){
  [ -n "$KS_INSTALLED" ] && rm -f "$REPO_ROOT/$MODULE/keystore.jks" 2>/dev/null || true
  [ -n "$TMP" ] && rm -rf "$TMP" 2>/dev/null || true
  [ -n "$WRAP_DIR" ] && rm -rf "$WRAP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# -------- 1) fetch CI-built .so (the native part we cannot link locally) --------
if [ "$DO_FETCH" -eq 1 ]; then
  command -v gh >/dev/null || { echo "gh CLI required (or use --no-fetch)"; exit 2; }
  [ -n "$RUN_ID" ] || RUN_ID="$(gh run list --repo "$REPO_SLUG" --workflow 'Android CI' --status success --limit 1 --json databaseId -q '.[0].databaseId')"
  [ -n "$RUN_ID" ] || { echo "no successful CI run found"; exit 2; }
  RUN_SHA="$(gh run view "$RUN_ID" --repo "$REPO_SLUG" --json headSha -q .headSha 2>/dev/null || true)"
  echo "== CI run $RUN_ID (commit ${RUN_SHA:0:12}) artifact=$ARTIFACT =="
  if [ -n "$RUN_SHA" ]; then
    git cat-file -e "$RUN_SHA^{commit}" 2>/dev/null || git fetch -q origin "$RUN_SHA" 2>/dev/null || true
    if git cat-file -e "$RUN_SHA^{commit}" 2>/dev/null; then
      if ! git diff --quiet "$RUN_SHA" -- "${NATIVE_PATHS[@]}"; then
        echo "!! The CI .so was built from ${RUN_SHA:0:12}, but native code changed since then:"
        git diff --stat "$RUN_SHA" -- "${NATIVE_PATHS[@]}" | tail -5
        if [ "$ALLOW_STALE" -eq 1 ]; then
          echo "!! --allow-stale: continuing (the app may fail to start if JNI signatures differ)"
        else
          echo "!! Push and wait for 'Android CI' to pass, then rerun (or pass --allow-stale)."
          exit 3
        fi
      fi
    else
      echo "(cannot inspect commit ${RUN_SHA:0:12} locally; skipping the staleness check)"
    fi
  fi
  TMP="$(mktemp -d)"
  gh run download "$RUN_ID" --repo "$REPO_SLUG" -n "$ARTIFACT" -D "$TMP"
  CI_APK="$(find "$TMP" -name '*.apk' | head -1)"
  [ -n "$CI_APK" ] || { echo "no APK in artifact $ARTIFACT"; exit 2; }
  rm -rf "$JNILIBS"; mkdir -p "$JNILIBS"
  unzip -o -j "$CI_APK" 'lib/arm64-v8a/*.so' -d "$JNILIBS" >/dev/null
  echo "== placed .so =="; ls -la "$JNILIBS"
fi

# -------- 1b) ...or build the .so locally: NDK clang compiles, a native linker links --------
# Only the NDK's ld.lld is unusable here; the NDK clang runs under qemu and its --ld-path hands
# the link to a native aarch64 linker. Two quirks of the qemu setup are handled:
#   - argv[0] is lost under qemu, so `clang++` runs in C driver mode and would not link libc++.
#     --driver-mode=g++ restores it (the -cc1 compile arguments are identical in both modes).
#   - linker: GNU ld (binutils) unless a native ld.lld is on PATH; LOCAL_LD overrides.
# Everything else mirrors the Gradle modules: same CMakeLists and arguments, minSdk, CMake build
# type (release = RelWithDebInfo, debug = Debug), and AGP-style --strip-unneeded.
if [ "$LOCAL_NATIVE" -eq 1 ]; then
  NDK_VER="$(sed -nE 's/^[[:space:]]*ndkVersion[[:space:]]+"([^"]+)".*/\1/p' "$MODULE/build.gradle" | head -1)"
  MIN_SDK="$(sed -nE 's/^[[:space:]]*minSdk[[:space:]]+([0-9]+).*/\1/p' "$MODULE/build.gradle" | head -1)"
  NDK="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/$NDK_VER}"
  [ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "NDK '$NDK_VER' not found at $NDK"; exit 2; }
  SDK_CMAKE_BIN="$(ls -d "$ANDROID_SDK_ROOT"/cmake/*/bin 2>/dev/null | sort -V | tail -1)"
  CMAKE_BIN="$(command -v cmake || echo "$SDK_CMAKE_BIN/cmake")"
  NINJA_BIN="$(command -v ninja || echo "$SDK_CMAKE_BIN/ninja")"
  LOCAL_LD="${LOCAL_LD:-$(command -v ld.lld || command -v ld.bfd || true)}"
  STRIP_BIN="$(command -v llvm-strip || command -v strip || true)"
  [ -x "$LOCAL_LD" ] && [ -x "$STRIP_BIN" ] || { echo "need a native linker (GNU ld or ld.lld) and strip"; exit 2; }
  NATIVE_TYPE="RelWithDebInfo"; [ "$VARIANT" = "debug" ] && NATIVE_TYPE="Debug"
  echo "== local native build: NDK $NDK_VER, android-$MIN_SDK, $NATIVE_TYPE, linker $LOCAL_LD =="
  bash "$REPO_ROOT/scripts/fetch-sdcpp.sh"   # as in CI: fetch if missing, always (idempotently) patch
  rm -rf "$JNILIBS"; mkdir -p "$JNILIBS"
  for spec in ":sd_jni" "fp16:sd_jni_fp16" "gpu:sd_jni_gpu"; do
    cpu="${spec%%:*}"; target="${spec#*:}"
    bdir="$REPO_ROOT/.build-local-native/$NATIVE_TYPE/${cpu:-base}"; mkdir -p "$bdir"
    echo "== lib$target.so (SD_CPU_VARIANT='$cpu') in $bdir =="
    "$CMAKE_BIN" -S "$REPO_ROOT/$MODULE/src/main/cpp" -B "$bdir" -G Ninja \
      "-DCMAKE_MAKE_PROGRAM=$NINJA_BIN" \
      "-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake" \
      "-DANDROID_NDK=$NDK" -DANDROID_ABI=arm64-v8a "-DANDROID_PLATFORM=android-$MIN_SDK" \
      "-DCMAKE_BUILD_TYPE=$NATIVE_TYPE" \
      "-DCMAKE_C_FLAGS=-fintegrated-cc1" \
      "-DCMAKE_CXX_FLAGS=-std=c++17 --driver-mode=g++ -fintegrated-cc1" \
      "-DCMAKE_AR=$(command -v ar)" "-DCMAKE_RANLIB=$(command -v ranlib)" \
      "-DCMAKE_EXE_LINKER_FLAGS=--ld-path=$LOCAL_LD" \
      "-DCMAKE_SHARED_LINKER_FLAGS=--ld-path=$LOCAL_LD" \
      "-DCMAKE_MODULE_LINKER_FLAGS=--ld-path=$LOCAL_LD" \
      -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON "-DSD_CPU_VARIANT=$cpu" \
      > "$bdir/configure.log" 2>&1 \
      || { tail -30 "$bdir/configure.log"; echo "!! CMake configure failed (see $bdir/configure.log)"; exit 2; }
    "$CMAKE_BIN" --build "$bdir" --target "$target" -j "$(nproc)"
    so="$JNILIBS/lib$target.so"
    "$STRIP_BIN" --strip-unneeded -o "$so" "$bdir/lib$target.so"
    # What the Android loader insists on: arm64, SONAME, no text relocations, 16KB-aligned LOADs.
    hdr="$(readelf -h "$so")"; dyn="$(readelf -d "$so")"; seg="$(readelf -lW "$so")"
    grep -q 'Machine:.*AArch64' <<<"$hdr" || { echo "!! $so is not AArch64"; exit 2; }
    grep -q "soname: \[lib$target\.so\]" <<<"$dyn" || { echo "!! $so has no SONAME"; exit 2; }
    if grep -q TEXTREL <<<"$dyn"; then echo "!! $so has text relocations"; exit 2; fi
    if awk '$1 == "LOAD" && $NF != "0x4000" { bad = 1 } END { exit !bad }' <<<"$seg"; then
      echo "!! $so has LOAD segments that are not 16KB-aligned"; exit 2
    fi
  done
  echo "== built .so =="; ls -la "$JNILIBS"
fi
ls -A "$JNILIBS" 2>/dev/null | grep -q '^libsd_jni\.so$' || { echo "no libsd_jni.so in $JNILIBS (run with --fetch)"; exit 2; }
ls -A "$JNILIBS" 2>/dev/null | grep -q '^libsd_jni_fp16\.so$' \
  || echo "!! libsd_jni_fp16.so missing (older CI run?) — the APK will only have the slower baseline library"
ls -A "$JNILIBS" 2>/dev/null | grep -q '^libsd_jni_gpu\.so$' \
  || echo "!! libsd_jni_gpu.so missing (older CI run?) — the APK will have no GPU (OpenCL) support"
rm -f "$JNILIBS/libOpenCL.so"   # link-time ICD loader only; never package it
# Every `native` method declared in StableDiffusionNative.java must be exported by every .so,
# otherwise the app fails at startup with UnsatisfiedLinkError (stale CI artifact).
JNI_JAVA="$REPO_ROOT/$MODULE/src/main/java/com/micklab/llamaimage/StableDiffusionNative.java"
JNI_METHODS="$(grep -oE 'native [][A-Za-z<>]+ native[A-Za-z0-9]+\(' "$JNI_JAVA" | sed -E 's/.* (native[A-Za-z0-9]+)\(/\1/')"
JNI_MISSING=0
for so in "$JNILIBS"/libsd_jni*.so; do
  SYMS="$(strings "$so")"
  for m in $JNI_METHODS; do
    grep -q "Java_com_micklab_llamaimage_StableDiffusionNative_${m}\$" <<<"$SYMS" \
      || { echo "!! $(basename "$so") lacks JNI method $m"; JNI_MISSING=1; }
  done
done
if [ "$JNI_MISSING" -eq 1 ]; then
  if [ "$ALLOW_STALE" -eq 1 ]; then echo "!! --allow-stale: packaging anyway (the app will fail to start)";
  else echo "!! The .so does not match the Java JNI declarations: push, wait for CI, rerun with --fetch."; exit 3; fi
fi

# -------- 2) explicit-qemu aapt2 wrapper (proot -q default cpu SIGILLs aapt2) --------
AAPT2_REAL="$(find "$ANDROID_SDK_ROOT/build-tools" -name aapt2 -type f | sort -V | tail -1)"
[ -n "$AAPT2_REAL" ] || { echo "aapt2 not found under $ANDROID_SDK_ROOT/build-tools"; exit 2; }
WRAP_DIR="$REPO_ROOT/.build-local-wrap"; mkdir -p "$WRAP_DIR"
cat > "$WRAP_DIR/aapt2" <<EOF
#!/bin/sh
exec env QEMU_CPU=max /usr/bin/qemu-x86_64-static "$AAPT2_REAL" "\$@"
EOF
chmod +x "$WRAP_DIR/aapt2"

# -------- 3) release signing (android_base.sh creds) --------
if [ "$VARIANT" = "release" ]; then
  KS_SRC="${KEYSTORE_PATH:-}"
  [ -n "$KS_SRC" ] || for c in "$REPO_ROOT/keystore.jks" "$HOME/keystore.jks" /root/keystore.jks; do [ -f "$c" ] && { KS_SRC="$c"; break; }; done
  [ -f "${KS_SRC:-}" ] || { echo "release keystore not found (set KEYSTORE_PATH)"; exit 2; }
  cp -f "$KS_SRC" "$REPO_ROOT/$MODULE/keystore.jks"; KS_INSTALLED=1
  export KEYSTORE_PASSWORD="${KEYSTORE_PASSWORD:-micklab}" KEY_ALIAS="${KEY_ALIAS:-mykey}" KEY_PASSWORD="${KEY_PASSWORD:-micklab}"
  echo "== release signing: $KS_SRC -> $MODULE/keystore.jks =="
fi

# -------- 4) local.properties --------
grep -q "^sdk.dir=" "$REPO_ROOT/local.properties" 2>/dev/null || echo "sdk.dir=$ANDROID_SDK_ROOT" > "$REPO_ROOT/local.properties"

# -------- 5) assemble locally with the prebuilt .so (no NDK build) --------
TASK="assembleRelease"; OUT_DIR="$MODULE/build/outputs/apk/release"
[ "$VARIANT" = "debug" ] && { TASK="assembleDebug"; OUT_DIR="$MODULE/build/outputs/apk/debug"; }
echo "== $GRADLE_RUNNER :$MODULE:$TASK -PsdPrebuiltNative=true (QEMU_CPU=$QEMU_CPU) =="
"$GRADLE_RUNNER" --no-daemon --console=plain \
  "-Pandroid.aapt2FromMavenOverride=$WRAP_DIR/aapt2" \
  -PsdPrebuiltNative=true \
  ":$MODULE:$TASK"

# -------- 6) collect --------
APK="$(find "$REPO_ROOT/$OUT_DIR" -name '*.apk' | sort | tail -1)"
[ -n "$APK" ] || { echo "no APK produced under $OUT_DIR"; exit 2; }
echo "== native libs in APK =="; unzip -l "$APK" 'lib/*' | awk 'NR>3 && $4 ~ /\.so$/ {print "   " $4}'
DEST_DIR="${APK_OUTPUT_DIR:-}"; [ -z "$DEST_DIR" ] && { [ -d /sdcard/Download ] && [ -w /sdcard/Download ] && DEST_DIR=/sdcard/Download || DEST_DIR="$HOME/downloads"; }
mkdir -p "$DEST_DIR"
PROJ="$(basename "$REPO_ROOT")"
cp -f "$APK" "$DEST_DIR/$PROJ-app-$VARIANT.apk"
echo "== APK: $DEST_DIR/$PROJ-app-$VARIANT.apk =="
echo "== DONE =="
