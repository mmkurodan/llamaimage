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
#include <exception>
#include <mutex>
#include <string>

#include "stable-diffusion.h"
#include "util.h"                     // g_sd_step_latent
#include "ggml/include/ggml.h"        // ggml_tensor for latent preview

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
static std::string  g_last_error;     // last native error (e.g. out-of-memory)

static JavaVM*      g_vm      = nullptr;
static jobject      g_progress_listener = nullptr;  // global ref, may be null
static jobject      g_preview_listener  = nullptr;  // global ref, may be null
static std::string  g_log_path;                     // optional log file

// ---------------------------------------------------------------------------
// Latent → RGB approximation coefficients (A1111 "cheap approximation").
// Maps the 4 SD1.5 latent channels to RGB at latent resolution (W/8 x H/8).
// ---------------------------------------------------------------------------
static const float kLatentRgbCoeff[4][3] = {
    //    R        G        B
    { 0.298f,  0.207f,  0.208f },   // L0
    { 0.187f,  0.286f,  0.173f },   // L1
    {-0.158f,  0.189f,  0.264f },   // L2
    {-0.184f, -0.271f, -0.473f },   // L3
};

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
// worker thread that called txt2img). g_sd_step_latent is set by stable-diffusion.cpp
// to the current denoised latent tensor just before this callback fires (step > 0).
// We use it to produce a lightweight RGB preview via linear approximation.
// ---------------------------------------------------------------------------
static void sd_progress_cb(int step, int steps, float /*time*/, void* /*data*/) {
    if (g_vm == nullptr) {
        return;
    }
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
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

    // --- 1. Fire step progress to ProgressListener ---
    if (g_progress_listener != nullptr) {
        jclass cls = env->GetObjectClass(g_progress_listener);
        jmethodID mid = env->GetMethodID(cls, "onProgress", "(II)V");
        if (mid != nullptr) {
            env->CallVoidMethod(g_progress_listener, mid, step, steps);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
        }
        env->DeleteLocalRef(cls);
    }

    // --- 2. Build latent preview and fire PreviewListener ---
    struct ggml_tensor* lat = g_sd_step_latent;
    if (g_preview_listener != nullptr && lat != nullptr && lat->type == GGML_TYPE_F32) {
        const int W = (int)lat->ne[0];  // latent width  (image_w / 8)
        const int H = (int)lat->ne[1];  // latent height (image_h / 8)
        const int C = (int)lat->ne[2];  // should be 4 for SD1.5

        if (W > 0 && H > 0 && C >= 4) {
            const int n_px = W * H;
            jbyteArray rgb_arr = env->NewByteArray(n_px * 3);
            if (rgb_arr != nullptr) {
                jbyte* rgb = env->GetByteArrayElements(rgb_arr, nullptr);
                if (rgb != nullptr) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            // Read the 4 latent channels for this pixel
                            float L[4];
                            for (int c = 0; c < 4; ++c) {
                                L[c] = *(const float*)((const char*)lat->data
                                        + c * lat->nb[2]
                                        + h * lat->nb[1]
                                        + w * lat->nb[0]);
                            }
                            // Apply approximation coefficients + bias
                            for (int ch = 0; ch < 3; ++ch) {
                                float val = 0.5f;
                                for (int c = 0; c < 4; ++c) {
                                    val += kLatentRgbCoeff[c][ch] * L[c];
                                }
                                int iv = (int)(val * 255.0f + 0.5f);
                                if (iv < 0) iv = 0;
                                if (iv > 255) iv = 255;
                                rgb[(h * W + w) * 3 + ch] = (jbyte)iv;
                            }
                        }
                    }
                    env->ReleaseByteArrayElements(rgb_arr, rgb, 0);

                    jclass pcls = env->GetObjectClass(g_preview_listener);
                    jmethodID pmid = env->GetMethodID(pcls, "onPreview", "([BII)V");
                    if (pmid != nullptr) {
                        env->CallVoidMethod(g_preview_listener, pmid, rgb_arr, W, H);
                        if (env->ExceptionCheck()) {
                            env->ExceptionClear();
                        }
                    }
                    env->DeleteLocalRef(pcls);
                }
                env->DeleteLocalRef(rgb_arr);
            }
        }
    }

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

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSetPreviewListener(JNIEnv* env, jobject, jobject listener) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_preview_listener != nullptr) {
        env->DeleteGlobalRef(g_preview_listener);
        g_preview_listener = nullptr;
    }
    if (listener != nullptr) {
        g_preview_listener = env->NewGlobalRef(listener);
    }
}

// Load an all-in-one SD1.5 GGUF. Returns "" on success or an error message.
// wtype: an sd_type_t value to re-quantize the weights to at load (e.g. Q4_0/Q8_0 for
// faster CPU matmul); pass <0 (or out of range) to keep the model's original types.
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeInit(JNIEnv* env, jobject, jstring modelPath, jint nThreads, jint wtype) {
    std::lock_guard<std::mutex> lk(g_mutex);
    std::string model = jstr(env, modelPath);
    if (model.empty()) {
        return env->NewStringUTF("model path is empty");
    }

    // Free + recreate inside try/catch so a failure (e.g. std::bad_alloc on a large
    // model) surfaces as an error string instead of crashing the process.
    try {
        if (g_ctx != nullptr) {
            free_sd_ctx(g_ctx);
            g_ctx = nullptr;
        }

        int threads = nThreads;
        if (threads <= 0) {
            threads = get_num_physical_cores();
        }

        // <0 or out-of-range -> keep original weight types (SD_TYPE_COUNT).
        sd_type_t wt = (wtype >= 0 && wtype < SD_TYPE_COUNT) ? (sd_type_t)wtype : SD_TYPE_COUNT;

        // Single-file SD1.5 GGUF: only model_path is set, everything else is empty.
        // vae_decode_only=false so the VAE encoder is available for img2img.
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
            false,         // vae_decode_only=false: load the VAE encoder too, so img2img works
            false,         // vae_tiling
            false,         // free_params_immediately
            threads,       // n_threads
            wt,            // wtype: original types, or Q4_0/Q8_0 to speed up CPU matmul
            STD_DEFAULT_RNG,
            DEFAULT,       // schedule
            false,         // keep_clip_on_cpu
            false,         // keep_control_net_cpu
            false,         // keep_vae_on_cpu
            false);        // diffusion_flash_attn
    } catch (const std::exception& e) {
        g_ctx = nullptr;
        g_last_error = std::string("init 例外: ") + e.what();
        ALOGE("%s", g_last_error.c_str());
        return env->NewStringUTF(g_last_error.c_str());
    } catch (...) {
        g_ctx = nullptr;
        g_last_error = "init 不明な例外";
        ALOGE("%s", g_last_error.c_str());
        return env->NewStringUTF(g_last_error.c_str());
    }

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
    g_last_error.clear();
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

    // Generation can throw (e.g. std::bad_alloc on a large request); catch it here so it
    // becomes a clean error+null instead of crashing by escaping the JNI boundary.
    sd_image_t* results = nullptr;
    try {
        results = txt2img(
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
    } catch (const std::exception& e) {
        g_last_error = std::string("txt2img 例外: ") + e.what();
        ALOGE("%s", g_last_error.c_str());
        sd_log_to_file((g_last_error + "\n").c_str());
        return nullptr;
    } catch (...) {
        g_last_error = "txt2img 不明な例外";
        ALOGE("%s", g_last_error.c_str());
        sd_log_to_file((g_last_error + "\n").c_str());
        return nullptr;
    }

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

// img2img -> returns width*height*3 RGB bytes, or null on failure.
// initRgb is an initW*initH*3 RGB image; strength in [0,1] controls how far the
// result departs from it (higher = more change). Requires a context loaded with
// vae_decode_only=false (so the VAE encoder is available).
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeImg2img(
        JNIEnv* env, jobject,
        jbyteArray initRgb, jint initW, jint initH,
        jstring prompt, jstring negativePrompt,
        jint width, jint height, jint steps,
        jfloat cfgScale, jfloat strength, jlong seed, jint sampleMethod, jint clipSkip) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_last_error.clear();
    if (g_ctx == nullptr) {
        ALOGE("nativeImg2img called before init");
        return nullptr;
    }
    if (initRgb == nullptr) {
        ALOGE("nativeImg2img: init image is null");
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

    jbyte* init_data = env->GetByteArrayElements(initRgb, nullptr);
    if (init_data == nullptr) {
        ALOGE("nativeImg2img: failed to access init image bytes");
        return nullptr;
    }

    sd_image_t init_image;
    init_image.width   = (uint32_t)initW;
    init_image.height  = (uint32_t)initH;
    init_image.channel = 3;
    init_image.data    = reinterpret_cast<uint8_t*>(init_data);

    sd_image_t* results = nullptr;
    try {
        results = img2img(
            g_ctx,
            init_image,
            p.c_str(),
            n.c_str(),
            clipSkip,
            cfgScale,
            3.5f,              // guidance (flux-only; ignored by SD1.5)
            width,
            height,
            method,
            steps,
            strength,
            used_seed,
            1,                 // batch_count
            nullptr,           // control_cond
            0.9f,              // control_strength
            20.0f,             // style_strength
            false,             // normalize_input
            "",                // input_id_images_path
            nullptr,           // skip_layers
            0,                 // skip_layers_count
            0.0f,              // slg_scale
            0.01f,             // skip_layer_start
            0.2f);             // skip_layer_end
    } catch (const std::exception& e) {
        g_last_error = std::string("img2img 例外: ") + e.what();
        ALOGE("%s", g_last_error.c_str());
        sd_log_to_file((g_last_error + "\n").c_str());
        env->ReleaseByteArrayElements(initRgb, init_data, JNI_ABORT);
        return nullptr;
    } catch (...) {
        g_last_error = "img2img 不明な例外";
        ALOGE("%s", g_last_error.c_str());
        sd_log_to_file((g_last_error + "\n").c_str());
        env->ReleaseByteArrayElements(initRgb, init_data, JNI_ABORT);
        return nullptr;
    }

    env->ReleaseByteArrayElements(initRgb, init_data, JNI_ABORT);  // read-only, discard changes

    if (results == nullptr || results[0].data == nullptr) {
        ALOGE("img2img returned no image");
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

// Last native error message (empty if none). Set when a generate call throws.
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeFree(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_mutex);
    try {
        if (g_ctx != nullptr) {
            free_sd_ctx(g_ctx);
            g_ctx = nullptr;
        }
    } catch (...) {
        g_ctx = nullptr;
    }
}
