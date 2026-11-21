cmake_minimum_required(VERSION 3.22.1)
project(sd_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# CPU variant. The same sources are built once per variant, each in its own Gradle module
# (:app -> baseline, :sdjni-fp16 -> fp16), and StableDiffusionNative loads the best library the
# device supports at runtime (CpuInfo checks the /proc/cpuinfo feature flags).
#
#   SD_CPU_VARIANT=""      -> libsd_jni.so       armv8-a baseline (runs on every arm64 device)
#   SD_CPU_VARIANT="fp16"  -> libsd_jni_fp16.so  armv8.2-a + fp16 + dotprod (Cortex-A55/A75 and later)
#
# Why: this ggml pin selects its SIMD kernels at COMPILE time. With the NDK default (-march=armv8-a)
# the F16 conv/matmul path — most of the UNet and VAE work, since sd.cpp keeps conv weights in F16 —
# widens every element to F32, and Q8_0/Q4_0 dot products cannot use SDOT. Measured on a Snapdragon
# 8 Elite (SD1.5 LCM, 512px, 6 steps, 8 threads): 120s -> 78s per image (VAE decode 44s -> 28s).
# +i8mm and GGML_LLAMAFILE gave no further gain (llamafile was slower), so they are not built.
# Raising the baseline instead would SIGILL on older cores, hence one library per variant.
# ---------------------------------------------------------------------------
set(SD_CPU_VARIANT "" CACHE STRING "CPU feature variant: '' (armv8-a) or fp16")
if(SD_CPU_VARIANT STREQUAL "")
    set(SD_JNI_LIB  sd_jni)
    set(SD_ARCH     "armv8-a")
elseif(SD_CPU_VARIANT STREQUAL "fp16")
    set(SD_JNI_LIB  sd_jni_fp16)
    set(SD_ARCH     "armv8.2-a+fp16+dotprod")
else()
    message(FATAL_ERROR "Unknown SD_CPU_VARIANT '${SD_CPU_VARIANT}'")
endif()
message(STATUS "sd_jni CPU variant: '${SD_CPU_VARIANT}' -> lib${SD_JNI_LIB}.so (-march=${SD_ARCH})")
# Directory-scoped, so it also applies to everything add_subdirectory(sd) below builds.
string(APPEND CMAKE_C_FLAGS   " -march=${SD_ARCH}")
string(APPEND CMAKE_CXX_FLAGS " -march=${SD_ARCH}")
# AGP builds the debug variant with CMAKE_BUILD_TYPE=Debug, for which the NDK adds no -O flag
# (= -O0): ggml then runs many times slower and the debug APK is useless for generation.
# Keep the native code optimized in every variant (release uses RelWithDebInfo = -O2 -g).
string(APPEND CMAKE_C_FLAGS_DEBUG   " -O2")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2")

# ---------------------------------------------------------------------------
# Fetch the pinned stable-diffusion.cpp + ggml sources at configure time.
# They are not committed (see .gitignore); the script is idempotent.
# ---------------------------------------------------------------------------
set(SD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sd)
set(FETCH_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../scripts/fetch-sdcpp.sh)
if(NOT EXISTS ${SD_DIR}/stable-diffusion.h)
    message(STATUS "stable-diffusion.cpp not found — running ${FETCH_SCRIPT}")
    execute_process(
        COMMAND bash ${FETCH_SCRIPT} ${SD_DIR}
        RESULT_VARIABLE SD_FETCH_RC)
    if(NOT SD_FETCH_RC EQUAL 0)
        message(FATAL_ERROR "fetch-sdcpp.sh failed (rc=${SD_FETCH_RC})")
    endif()
endif()

# ---------------------------------------------------------------------------
# ggml / stable-diffusion build options — CPU backend only, cross-compile safe.
# Must be set BEFORE add_subdirectory(sd).
# ---------------------------------------------------------------------------
set(GGML_NATIVE        OFF CACHE BOOL "" FORCE)  # never -march=native when cross-compiling
set(GGML_OPENMP        OFF CACHE BOOL "" FORCE)  # avoid libomp dependency on Android
set(GGML_LLAMAFILE     OFF CACHE BOOL "" FORCE)  # measured slower than ggml's fp16 kernels for SD
set(GGML_CCACHE        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# GGML_CPU_AARCH64 stays at its default. Note: at this pin its Q4_0 -> Q4_0_4_x repacking only
# happens for tensors allocated in the "CPU_AARCH64" buffer type, which sd.cpp does not use.

set(SD_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Pulls in: stable-diffusion (STATIC) + ggml + thirdparty/zip.
add_subdirectory(${SD_DIR} ${CMAKE_BINARY_DIR}/sd-build)

# ---------------------------------------------------------------------------
# JNI shared library
# ---------------------------------------------------------------------------
add_library(${SD_JNI_LIB} SHARED jni/jni_sd.cpp)

target_include_directories(${SD_JNI_LIB} PRIVATE ${SD_DIR})
target_compile_definitions(${SD_JNI_LIB} PRIVATE SD_JNI_VARIANT="${SD_ARCH}")

target_link_libraries(${SD_JNI_LIB} PRIVATE
    stable-diffusion
    log
    android)

# 16KB page-size support (required by Android 15+ / some arm64 devices), mirroring
# the LLM AI Server's native link flags.
target_link_options(${SD_JNI_LIB} PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,-z,common-page-size=16384)
