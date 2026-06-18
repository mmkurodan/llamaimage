// JNI bridge for stable-diffusion.cpp (SD1.5 text2img PoC).
//
// Mirrors the design of the LLM AI Server's jni_llama.cpp: a single global engine
// context loaded once, dynamic JNI naming (Java_<pkg>_<class>_<method>), String-return
// init that yields "" on success / an error message on failure, and a progress callback
// bridged back to a Java listener. CPU backend only (see CMakeLists.txt).
//
// Builds for Android (arm64-v8a) and also compiles on a host toolchain so the bridge can
// be sanity-checked off-device (android/log.h usage is guarded by __ANDROID__).

#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

#include "stable-diffusion.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "sd_jni"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define ALOGI(...) ((void)0)
#define ALOGE(...) ((void)0)
#endif

// ---------------------------------------------------------------------------
// Global engine state (single context, like ModelManager's single model).
// ---------------------------------------------------------------------------
static std::mutex   g_mutex;          // serializes init / generate / free
static sd_ctx_t*    g_ctx     = nullptr;
static int64_t      g_last_seed = -1;

static JavaVM*      g_vm      = nullptr;
static jobject      g_progress_listener = nullptr;  // global ref, may be null
static std::string  g_log_path;                     // optional log file

// ---------------------------------------------------------------------------
// Logging: forward sd.cpp log lines to logcat and (optionally) a file, matching
// the parent app's "everything also goes to a log file" behaviour.
// ---------------------------------------------------------------------------
static void sd_log_to_file(const char* text) {
    if (g_log_path.empty() || text == nullptr) {
        return;
    }
    FILE* f = fopen(g_log_path.c_str(), "a");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static void sd_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    if (text == nullptr) {
        return;
    }
    if (level == SD_LOG_ERROR) {
        ALOGE("%s", text);
    } else {
        ALOGI("%s", text);
    }
    sd_log_to_file(text);
}

// ---------------------------------------------------------------------------
// Progress: sd.cpp invokes this synchronously on the generating thread (the Java
// worker thread that called txt2img), so we can reach the JVM via that thread.
// ---------------------------------------------------------------------------
static void sd_progress_cb(int step, int steps, float /*time*/, void* /*data*/) {
    if (g_vm == nullptr || g_progress_listener == nullptr) {
        return;
    }
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        // NDK's AttachCurrentThread takes JNIEnv**, desktop OpenJDK's takes void**.
#ifdef __ANDROID__
        jint rc = g_vm->AttachCurrentThread(&env, nullptr);
#else
        jint rc = g_vm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr);
#endif
        if (rc != JNI_OK) {
            return;
        }
        attached = true;
    }
    jclass cls = env->GetObjectClass(g_progress_listener);
    jmethodID mid = env->GetMethodID(cls, "onProgress", "(II)V");
    if (mid != nullptr) {
        env->CallVoidMethod(g_progress_listener, mid, step, steps);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
    env->DeleteLocalRef(cls);
    if (attached) {
        g_vm->DetachCurrentThread();
    }
}

// ---------------------------------------------------------------------------
// JNI lifecycle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_vm = vm;
    sd_set_log_callback(sd_log_cb, nullptr);
    sd_set_progress_callback(sd_progress_cb, nullptr);
    return JNI_VERSION_1_6;
}

static std::string jstr(JNIEnv* env, jstring s) {
    if (s == nullptr) {
        return std::string();
    }
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) {
        env->ReleaseStringUTFChars(s, c);
    }
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSystemInfo(JNIEnv* env, jobject) {
    return env->NewStringUTF(sd_get_system_info());
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSetLogPath(JNIEnv* env, jobject, jstring path) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_log_path = jstr(env, path);
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSetProgressListener(JNIEnv* env, jobject, jobject listener) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_progress_listener != nullptr) {
        env->DeleteGlobalRef(g_progress_listener);
        g_progress_listener = nullptr;
    }
    if (listener != nullptr) {
        g_progress_listener = env->NewGlobalRef(listener);
    }
}

// Load an all-in-one SD1.5 GGUF. Returns "" on success or an error message.
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeInit(JNIEnv* env, jobject, jstring modelPath, jint nThreads) {
    std::lock_guard<std::mutex> lk(g_mutex);
    std::string model = jstr(env, modelPath);
    if (model.empty()) {
        return env->NewStringUTF("model path is empty");
    }

    if (g_ctx != nullptr) {
        free_sd_ctx(g_ctx);
        g_ctx = nullptr;
    }

    int threads = nThreads;
    if (threads <= 0) {
        threads = get_num_physical_cores();
    }

    // Single-file SD1.5 GGUF: only model_path is set, everything else is empty.
    // vae_decode_only=true (txt2img only, no encode), keep weights on the (CPU)
    // compute buffer, no tiling, no flash-attn.
    g_ctx = new_sd_ctx(
        model.c_str(),
        "",            // clip_l_path
        "",            // clip_g_path
        "",            // t5xxl_path
        "",            // diffusion_model_path
        "",            // vae_path
        "",            // taesd_path
        "",            // control_net_path
        "",            // lora_model_dir
        "",            // embed_dir
        "",            // stacked_id_embed_dir
        true,          // vae_decode_only
        false,         // vae_tiling
        false,         // free_params_immediately
        threads,       // n_threads
        SD_TYPE_COUNT, // wtype: keep the model's native (Q4_0) types
        STD_DEFAULT_RNG,
        DEFAULT,       // schedule
        false,         // keep_clip_on_cpu
        false,         // keep_control_net_cpu
        false,         // keep_vae_on_cpu
        false);        // diffusion_flash_attn

    if (g_ctx == nullptr) {
        return env->NewStringUTF("new_sd_ctx failed (see log)");
    }
    return env->NewStringUTF("");
}

// txt2img -> returns width*height*3 RGB bytes, or null on failure.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeTxt2img(
        JNIEnv* env, jobject,
        jstring prompt, jstring negativePrompt,
        jint width, jint height, jint steps,
        jfloat cfgScale, jlong seed, jint sampleMethod, jint clipSkip) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_ctx == nullptr) {
        ALOGE("nativeTxt2img called before init");
        return nullptr;
    }

    std::string p = jstr(env, prompt);
    std::string n = jstr(env, negativePrompt);

    int64_t used_seed = seed;
    if (used_seed < 0) {
        srand((unsigned)time(nullptr));
        used_seed = rand();
    }
    g_last_seed = used_seed;

    sample_method_t method = EULER_A;
    if (sampleMethod >= 0 && sampleMethod < N_SAMPLE_METHODS) {
        method = (sample_method_t)sampleMethod;
    }

    sd_image_t* results = txt2img(
        g_ctx,
        p.c_str(),
        n.c_str(),
        clipSkip,          // clip_skip (<=0 == unspecified)
        cfgScale,          // cfg_scale
        3.5f,              // guidance (flux-only; ignored by SD1.5)
        width,
        height,
        method,
        steps,
        used_seed,
        1,                 // batch_count
        nullptr,           // control_cond
        0.9f,              // control_strength
        20.0f,             // style_strength (pmid-only)
        false,             // normalize_input
        "",                // input_id_images_path
        nullptr,           // skip_layers
        0,                 // skip_layers_count
        0.0f,              // slg_scale
        0.01f,             // skip_layer_start
        0.2f);             // skip_layer_end

    if (results == nullptr || results[0].data == nullptr) {
        ALOGE("txt2img returned no image");
        if (results) {
            free(results);
        }
        return nullptr;
    }

    const uint32_t w = results[0].width;
    const uint32_t h = results[0].height;
    const uint32_t c = results[0].channel;
    const jsize n_bytes = (jsize)(w * h * c);

    jbyteArray arr = env->NewByteArray(n_bytes);
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, n_bytes, reinterpret_cast<const jbyte*>(results[0].data));
    }

    free(results[0].data);
    free(results);
    return arr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeGetLastSeed(JNIEnv*, jobject) {
    return (jlong)g_last_seed;
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeFree(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_ctx != nullptr) {
        free_sd_ctx(g_ctx);
        g_ctx = nullptr;
    }
}
