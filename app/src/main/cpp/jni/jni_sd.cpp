// JNI bridge for stable-diffusion.cpp (SD1.x / SD2.x txt2img + img2img).
//
// Mirrors the design of the LLM AI Server's jni_llama.cpp: a single global engine context,
// dynamic JNI naming (Java_<pkg>_<class>_<method>), a String-returning init that yields "" on
// success / an error message on failure, and progress/preview callbacks bridged back to Java.
// CPU backend only (see CMakeLists.txt). The same source is built once per CPU variant
// (SD_JNI_VARIANT, e.g. baseline armv8-a vs. armv8.2-a+fp16+dotprod); Java picks the library.
//
// Relies on the out-of-tree sd.cpp hooks applied by scripts/patch-sdcpp.sh:
//   g_sd_step_latent  current denoised latent during a sampling-step callback (preview)
//   g_sd_phase        1 = sampling steps, 2 = VAE decode tiles (both share the progress callback)
//   sd_unwind_cleanup frees work buffers when the cancel exception unwinds through sd.cpp
//
// Also compiles on a host toolchain so the bridge can be sanity-checked off-device
// (android/log.h usage is guarded by __ANDROID__).

#include <jni.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stable-diffusion.h"
#include "util.h"                     // g_sd_step_latent, g_sd_phase (scripts/patch-sdcpp.sh)
#include "ggml/include/ggml.h"        // ggml_tensor for latent preview

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "sd_jni"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define ALOGD(...) ((void)0)
#define ALOGI(...) ((void)0)
#define ALOGE(...) ((void)0)
#endif

#ifndef SD_JNI_VARIANT
#define SD_JNI_VARIANT "armv8-a"
#endif

// Progress phases reported to Java (must match StableDiffusionNative.PHASE_*).
static const int kPhaseSampling = 1;
[[maybe_unused]] static const int kPhaseDecode = 2;   // VAE-decode tiles (only forwarded)

// ---------------------------------------------------------------------------
// Global engine state (single context, like ModelManager's single model).
// ---------------------------------------------------------------------------
static std::mutex          g_mutex;               // serializes init / generate / free
static sd_ctx_t*           g_ctx = nullptr;
static std::atomic<int64_t> g_last_seed{-1};
static std::atomic<bool>   g_cancel{false};       // set by nativeCancel, cleared by nativeClearCancel
static std::atomic<bool>   g_cancelled{false};    // last generate ended because of a cancel
static std::mutex          g_err_mutex;
static std::string         g_last_error;          // last native error (e.g. out-of-memory)

static JavaVM*             g_vm = nullptr;
static std::mutex          g_listener_mutex;      // guards the listener ref + method ids
static jobject             g_listener = nullptr;  // global ref, may be null
static jmethodID           g_on_progress = nullptr;  // void onProgress(int phase, int step, int steps)
static jmethodID           g_on_preview  = nullptr;  // void onPreview(byte[] rgb, int w, int h)
static std::mutex          g_log_mutex;
static std::string         g_log_path;            // optional log file (INFO and above)

// Thrown from the progress callback to abort sampling; caught in nativeTxt2img/nativeImg2img.
// Deliberately not derived from std::exception so nothing in between can mistake it for an error.
struct sd_jni_cancelled {};

static void set_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    g_last_error = e;
}

// ---------------------------------------------------------------------------
// Latent → RGB approximation (the widely used SD1.x/2.x "cheap approximation";
// keturn, also used by A1111/ComfyUI previews): rgb = latent · coeff, image = rgb * 0.5 + 0.5.
// Maps the 4 latent channels to RGB at latent resolution (W/8 x H/8).
// ---------------------------------------------------------------------------
static const float kLatentRgbCoeff[4][3] = {
    //    R        G        B
    { 0.298f,  0.207f,  0.208f },   // L0
    { 0.187f,  0.286f,  0.173f },   // L1
    {-0.158f,  0.189f,  0.264f },   // L2
    {-0.184f, -0.271f, -0.473f },   // L3
};

// ---------------------------------------------------------------------------
// Logging: forward sd.cpp log lines to logcat and (INFO and above) to a log file,
// matching the parent app's "everything also goes to a log file" behaviour.
// ---------------------------------------------------------------------------
static void sd_log_to_file(const char* text) {
    if (text == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (g_log_path.empty()) {
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
    switch (level) {
        case SD_LOG_DEBUG:
            ALOGD("%s", text);
            return;  // per-tensor/buffer chatter: logcat only
        case SD_LOG_ERROR:
            ALOGE("%s", text);
            break;
        default:
            ALOGI("%s", text);
            break;
    }
    sd_log_to_file(text);
}

// ---------------------------------------------------------------------------
// UTF-16 (Java) -> standard UTF-8. GetStringUTFChars returns *modified* UTF-8 (surrogate pairs
// as 6 bytes), which sd.cpp's tokenizer rejects (utf8_to_utf32 throws), so convert by hand.
// ---------------------------------------------------------------------------
static std::string jstr(JNIEnv* env, jstring s) {
    std::string out;
    if (s == nullptr) {
        return out;
    }
    const jsize len = env->GetStringLength(s);
    std::vector<jchar> buf((size_t)len);
    if (len > 0) {
        env->GetStringRegion(s, 0, len, buf.data());
    }
    out.reserve((size_t)len * 3);
    for (jsize i = 0; i < len; ++i) {
        uint32_t cp = buf[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len && buf[i + 1] >= 0xDC00 && buf[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (uint32_t)(buf[i + 1] - 0xDC00);
            ++i;
        } else if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;  // lone surrogate
        }
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Progress: sd.cpp invokes this synchronously on the generating thread (the Java worker thread
// that called txt2img). g_sd_phase tells sampling steps (1) from VAE-decode tiles (2) — both use
// this callback. During a sampling step g_sd_step_latent holds the current denoised latent,
// which we turn into a lightweight RGB preview.
// ---------------------------------------------------------------------------
static void emit_preview(JNIEnv* env, const struct ggml_tensor* lat) {
    if (g_on_preview == nullptr || lat == nullptr || lat->type != GGML_TYPE_F32) {
        return;
    }
    const int W = (int)lat->ne[0];  // latent width  (image_w / 8)
    const int H = (int)lat->ne[1];  // latent height (image_h / 8)
    const int C = (int)lat->ne[2];  // 4 for SD1.x / SD2.x
    if (W <= 0 || H <= 0 || C < 4) {
        return;
    }
    std::vector<jbyte> rgb((size_t)W * H * 3);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float L[4];
            for (int c = 0; c < 4; ++c) {
                L[c] = *(const float*)((const char*)lat->data + c * lat->nb[2] + h * lat->nb[1] + w * lat->nb[0]);
            }
            for (int ch = 0; ch < 3; ++ch) {
                float v = 0.0f;
                for (int c = 0; c < 4; ++c) {
                    v += kLatentRgbCoeff[c][ch] * L[c];
                }
                int iv = (int)((v * 0.5f + 0.5f) * 255.0f + 0.5f);
                iv = iv < 0 ? 0 : (iv > 255 ? 255 : iv);
                rgb[((size_t)h * W + w) * 3 + ch] = (jbyte)iv;
            }
        }
    }
    jbyteArray arr = env->NewByteArray((jsize)rgb.size());
    if (arr == nullptr) {
        env->ExceptionClear();
        return;
    }
    env->SetByteArrayRegion(arr, 0, (jsize)rgb.size(), rgb.data());
    env->CallVoidMethod(g_listener, g_on_preview, arr, W, H);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    env->DeleteLocalRef(arr);
}

static void sd_progress_cb(int step, int steps, float /*time*/, void* /*data*/) {
    const int phase = g_sd_phase;
    // Cancel only between sampling steps: nothing is mid-flight there, and the unwind is cleaned
    // up by sd_unwind_cleanup (patch-sdcpp.sh). The VAE decode is short and runs to completion.
    if (phase == kPhaseSampling && g_cancel.load()) {
        throw sd_jni_cancelled();
    }
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
    {
        std::lock_guard<std::mutex> lk(g_listener_mutex);
        if (g_listener != nullptr) {
            if (g_on_progress != nullptr) {
                env->CallVoidMethod(g_listener, g_on_progress, phase, step, steps);
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                }
            }
            if (phase == kPhaseSampling && step > 0) {
                emit_preview(env, g_sd_step_latent);
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

extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSystemInfo(JNIEnv* env, jobject) {
    std::string info = std::string("libsd_jni variant: ") + SD_JNI_VARIANT + "\n" + sd_get_system_info();
    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSetLogPath(JNIEnv* env, jobject, jstring path) {
    std::string p = jstr(env, path);
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_log_path = p;
}

// Registers the single callback object (StableDiffusionNative.Callbacks). Does not take the
// engine mutex, so it never blocks behind a running generation.
extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSetListener(JNIEnv* env, jobject, jobject listener) {
    std::lock_guard<std::mutex> lk(g_listener_mutex);
    if (g_listener != nullptr) {
        env->DeleteGlobalRef(g_listener);
        g_listener = nullptr;
    }
    g_on_progress = nullptr;
    g_on_preview = nullptr;
    if (listener == nullptr) {
        return;
    }
    jclass cls = env->GetObjectClass(listener);
    g_on_progress = env->GetMethodID(cls, "onProgress", "(III)V");
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        g_on_progress = nullptr;
    }
    g_on_preview = env->GetMethodID(cls, "onPreview", "([BII)V");
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        g_on_preview = nullptr;
    }
    env->DeleteLocalRef(cls);
    g_listener = env->NewGlobalRef(listener);
}

// Load a single-file SD checkpoint (GGUF/safetensors/ckpt). Returns "" on success or an error.
// wtype: an sd_type_t to re-quantize weights to at load (<0 keeps the file's types).
// taesdPath: optional TAESD weights for a fast, low-memory final VAE decode ("" = full VAE).
// vaeTiling: decode the VAE in tiles (caps peak memory; costs ~2x decode time at 512px).
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeInit(JNIEnv* env, jobject, jstring modelPath,
                                                             jint nThreads, jint wtype, jstring taesdPath,
                                                             jboolean vaeTiling) {
    std::lock_guard<std::mutex> lk(g_mutex);
    std::string model = jstr(env, modelPath);
    if (model.empty()) {
        return env->NewStringUTF("model path is empty");
    }
    std::string taesd = jstr(env, taesdPath);
    set_error("");

    // Free + recreate inside try/catch so a failure (e.g. std::bad_alloc on a large
    // model) surfaces as an error string instead of crashing the process.
    try {
        if (g_ctx != nullptr) {
            free_sd_ctx(g_ctx);
            g_ctx = nullptr;
        }

        int threads = nThreads;
        if (threads <= 0) {
            // Java normally passes an explicit count (performance cores). Note: sd.cpp's own
            // get_num_physical_cores() mis-builds its sysfs path at this pin and returns n/2.
            unsigned hc = std::thread::hardware_concurrency();
            threads = hc > 0 ? (int)hc : 4;
        }

        // <0 or out-of-range -> keep original weight types (SD_TYPE_COUNT).
        sd_type_t wt = (wtype >= 0 && wtype < SD_TYPE_COUNT) ? (sd_type_t)wtype : SD_TYPE_COUNT;

        ALOGI("init: model=%s threads=%d wtype=%d taesd=%s vae_tiling=%d variant=%s",
              model.c_str(), threads, (int)wtype, taesd.empty() ? "-" : taesd.c_str(),
              vaeTiling ? 1 : 0, SD_JNI_VARIANT);

        // Single-file checkpoint: only model_path is set, everything else is empty.
        g_ctx = new_sd_ctx(
            model.c_str(),
            "",            // clip_l_path
            "",            // clip_g_path
            "",            // t5xxl_path
            "",            // diffusion_model_path
            "",            // vae_path
            taesd.c_str(), // taesd_path ("" = full VAE)
            "",            // control_net_path
            "",            // lora_model_dir
            "",            // embed_dir
            "",            // stacked_id_embed_dir
            false,         // vae_decode_only=false: load the VAE encoder too, so img2img works
            vaeTiling == JNI_TRUE,  // vae_tiling: only on low-RAM devices (Java decides)
            false,         // free_params_immediately: keep weights for repeated generations
            threads,       // n_threads
            wt,            // wtype
            STD_DEFAULT_RNG,
            DEFAULT,       // schedule
            false,         // keep_clip_on_cpu
            false,         // keep_control_net_cpu
            false,         // keep_vae_on_cpu
            false);        // diffusion_flash_attn: no gain on CPU at this pin (SD1.x never engages it)
    } catch (const std::exception& e) {
        g_ctx = nullptr;
        set_error(std::string("init 例外: ") + e.what());
    } catch (...) {
        g_ctx = nullptr;
        set_error("init 不明な例外");
    }

    if (g_ctx == nullptr) {
        std::string err;
        {
            std::lock_guard<std::mutex> elk(g_err_mutex);
            err = g_last_error.empty() ? "new_sd_ctx failed (see log)" : g_last_error;
        }
        ALOGE("%s", err.c_str());
        return env->NewStringUTF(err.c_str());
    }
    return env->NewStringUTF("");
}

static int64_t resolve_seed(jlong seed) {
    if (seed >= 0) {
        return (int64_t)seed;
    }
    static std::mt19937_64 rng{std::random_device{}()};  // guarded by g_mutex (callers hold it)
    return (int64_t)(rng() & 0x7FFFFFFFULL);
}

static sample_method_t resolve_method(jint m) {
    return (m >= 0 && m < N_SAMPLE_METHODS) ? (sample_method_t)m : EULER_A;
}

// Copies the first result image into a Java byte[] and frees the sd.cpp result.
static jbyteArray take_result(JNIEnv* env, sd_image_t* results, const char* what) {
    if (results == nullptr || results[0].data == nullptr) {
        set_error(std::string(what) + ": 画像が生成されませんでした（ログ参照）");
        ALOGE("%s returned no image", what);
        if (results) {
            free(results);
        }
        return nullptr;
    }
    const jsize n_bytes = (jsize)(results[0].width * results[0].height * results[0].channel);
    jbyteArray arr = env->NewByteArray(n_bytes);
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, n_bytes, reinterpret_cast<const jbyte*>(results[0].data));
    } else {
        env->ExceptionClear();
        set_error(std::string(what) + ": 結果バッファを確保できませんでした");
    }
    free(results[0].data);
    free(results);
    return arr;
}

// Records an exception escaping sd.cpp. Returns nullptr for convenient `return fail(...)`.
static jbyteArray fail(const std::string& msg) {
    set_error(msg);
    ALOGE("%s", msg.c_str());
    sd_log_to_file((msg + "\n").c_str());
    return nullptr;
}

// txt2img -> returns width*height*3 RGB bytes, or null on failure / cancel.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeTxt2img(
        JNIEnv* env, jobject,
        jstring prompt, jstring negativePrompt,
        jint width, jint height, jint steps,
        jfloat cfgScale, jlong seed, jint sampleMethod, jint clipSkip) {
    std::lock_guard<std::mutex> lk(g_mutex);
    set_error("");
    g_cancelled = false;
    if (g_ctx == nullptr) {
        return fail("txt2img: モデルが読み込まれていません");
    }
    std::string p = jstr(env, prompt);
    std::string n = jstr(env, negativePrompt);
    const int64_t used_seed = resolve_seed(seed);
    g_last_seed = used_seed;

    sd_image_t* results = nullptr;
    g_sd_phase = 0;
    try {
        if (g_cancel.load()) {
            throw sd_jni_cancelled();
        }
        results = txt2img(
            g_ctx,
            p.c_str(),
            n.c_str(),
            clipSkip,          // clip_skip (<=0 == model default)
            cfgScale,          // cfg_scale (1.0 skips the unconditional pass)
            3.5f,              // guidance (flux-only; ignored)
            width,
            height,
            resolve_method(sampleMethod),
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
    } catch (const sd_jni_cancelled&) {
        g_sd_phase = 0;
        g_cancelled = true;
        set_error("cancelled");
        return nullptr;
    } catch (const std::exception& e) {
        g_sd_phase = 0;
        return fail(std::string("txt2img 例外: ") + e.what());
    } catch (...) {
        g_sd_phase = 0;
        return fail("txt2img 不明な例外");
    }
    g_sd_phase = 0;
    return take_result(env, results, "txt2img");
}

// img2img -> returns width*height*3 RGB bytes, or null on failure / cancel.
// initRgb is an initW*initH*3 RGB image (must equal width x height); strength in (0,1] controls
// how far the result departs from it (sd.cpp runs about steps*strength sampling steps).
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeImg2img(
        JNIEnv* env, jobject,
        jbyteArray initRgb, jint initW, jint initH,
        jstring prompt, jstring negativePrompt,
        jint width, jint height, jint steps,
        jfloat cfgScale, jfloat strength, jlong seed, jint sampleMethod, jint clipSkip) {
    std::lock_guard<std::mutex> lk(g_mutex);
    set_error("");
    g_cancelled = false;
    if (g_ctx == nullptr) {
        return fail("img2img: モデルが読み込まれていません");
    }
    if (initRgb == nullptr || env->GetArrayLength(initRgb) < (jsize)(initW * initH * 3)) {
        return fail("img2img: 入力画像が不正です");
    }
    if (initW != width || initH != height) {
        return fail("img2img: 入力画像のサイズが出力サイズと一致しません");
    }
    std::string p = jstr(env, prompt);
    std::string n = jstr(env, negativePrompt);
    const int64_t used_seed = resolve_seed(seed);
    g_last_seed = used_seed;

    std::vector<uint8_t> init_data((size_t)initW * initH * 3);
    env->GetByteArrayRegion(initRgb, 0, (jsize)init_data.size(), reinterpret_cast<jbyte*>(init_data.data()));

    sd_image_t init_image;
    init_image.width   = (uint32_t)initW;
    init_image.height  = (uint32_t)initH;
    init_image.channel = 3;
    init_image.data    = init_data.data();

    sd_image_t* results = nullptr;
    g_sd_phase = 0;
    try {
        if (g_cancel.load()) {
            throw sd_jni_cancelled();
        }
        results = img2img(
            g_ctx,
            init_image,
            p.c_str(),
            n.c_str(),
            clipSkip,
            cfgScale,
            3.5f,              // guidance (flux-only; ignored)
            width,
            height,
            resolve_method(sampleMethod),
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
    } catch (const sd_jni_cancelled&) {
        g_sd_phase = 0;
        g_cancelled = true;
        set_error("cancelled");
        return nullptr;
    } catch (const std::exception& e) {
        g_sd_phase = 0;
        return fail(std::string("img2img 例外: ") + e.what());
    } catch (...) {
        g_sd_phase = 0;
        return fail("img2img 不明な例外");
    }
    g_sd_phase = 0;
    return take_result(env, results, "img2img");
}

// Request cancellation of the running (or next) generation. Lock-free: callable from any thread.
extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeCancel(JNIEnv*, jobject) {
    g_cancel = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeClearCancel(JNIEnv*, jobject) {
    g_cancel = false;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeWasCancelled(JNIEnv*, jobject) {
    return g_cancelled.load() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeGetLastSeed(JNIEnv*, jobject) {
    return (jlong)g_last_seed.load();
}

// Last native error message (empty if none).
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeLastError(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeFree(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_mutex);
    try {
        if (g_ctx != nullptr) {
            free_sd_ctx(g_ctx);
        }
    } catch (...) {
    }
    g_ctx = nullptr;
}
