cmake_minimum_required(VERSION 3.22.1)
project(sd_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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
set(GGML_LLAMAFILE     OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# GGML_CPU and GGML_CPU_AARCH64 stay ON (default): the latter does runtime Q4_0 -> Q4_X
# repacking on aarch64, which is exactly what our Q4_0 model benefits from.

set(SD_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Pulls in: stable-diffusion (STATIC) + ggml + thirdparty/zip.
add_subdirectory(${SD_DIR} ${CMAKE_BINARY_DIR}/sd-build)

# ---------------------------------------------------------------------------
# JNI shared library
# ---------------------------------------------------------------------------
add_library(sd_jni SHARED jni/jni_sd.cpp)

target_include_directories(sd_jni PRIVATE ${SD_DIR})

target_link_libraries(sd_jni PRIVATE
    stable-diffusion
    log
    android)

# 16KB page-size support (required by Android 15+ / some arm64 devices), mirroring
# the LLM AI Server's native link flags.
target_link_options(sd_jni PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,-z,common-page-size=16384)
