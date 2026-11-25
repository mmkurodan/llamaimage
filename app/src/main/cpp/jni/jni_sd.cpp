// JNI bridge for stable-diffusion.cpp (SD1.x / SD2.x txt2img + img2img).
//
// Targets the option-based C API of stable-diffusion.cpp (sd_ctx_opt_t / sd_img_gen_opt_t,
// pinned in scripts/fetch-sdcpp.sh). Cancellation, staged progress and latent previews are
// built into that API, so the engine is used unpatched.
//
// One global engine context (like the LLM AI Server's single model), dynamic JNI naming
// (Java_<pkg>_<class>_<method>), a String-returning init that yields "" on success / an error
// message on failure, and progress/preview callbacks bridged back to Java.
//
// The same source is built once per library variant (SD_JNI_VARIANT): CPU baseline, CPU
// fp16+dotprod and fp16+OpenCL (SD_JNI_GPU). The GPU variant can still run on the CPU — the
// backend is chosen per model load (nativeInit useGpu), Java picks the library.

#include <jni.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stable-diffusion.h"

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
static const int kPhaseDecode   = 2;

// ---------------------------------------------------------------------------
// Global engine state (single context).
// ---------------------------------------------------------------------------
static std::mutex             g_mutex;               // serializes init / generate / free
static std::atomic<sd_ctx_t*> g_ctx{nullptr};        // atomic: nativeCancel reads it without g_mutex
static std::atomic<int64_t>   g_last_seed{-1};
static std::atomic<bool>      g_cancel{false};       // set by nativeCancel, cleared by nativeClearCancel
static std::atomic<bool>      g_cancelled{false};    // last generate ended because of a cancel
static std::mutex             g_err_mutex;
static std::string            g_last_error;          // last native error (e.g. out-of-memory)
static std::string            g_backend = "CPU";     // backend of the loaded model (guarded by g_mutex)

static JavaVM*     g_vm = nullptr;
static std::mutex  g_listener_mutex;                 // guards the listener ref + method ids
static jobject     g_listener = nullptr;             // global ref, may be null
static jmethodID   g_on_progress = nullptr;          // void onProgress(int phase, int step, int steps)
static jmethodID   g_on_preview  = nullptr;          // void onPreview(byte[] rgb, int w, int h)
static std::mutex  g_log_mutex;
static std::string g_log_path;                       // optional log file (INFO and above)

static void set_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    g_last_error = e;
}

// ---------------------------------------------------------------------------
// Logging: forward sd.cpp log lines to logcat and (INFO and above) to a log file.
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
        size_t n = strlen(text);
        if (n == 0 || text[n - 1] != '\n') {
            fputc('\n', f);
        }
        fclose(f);
    }
}

static void sd_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    if (text == nullptr) {
        return;
    }
    switch (level) {
        case SD_LOG_DEBUG:
        case SD_LOG_VERBOSE:
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
// as 6 bytes), which sd.cpp's tokenizer rejects, so convert by hand.
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
// Callbacks into Java. sd.cpp calls them synchronously during sd_generate_image(); they may run
// on an engine thread, so attach when needed. They must not throw (C API contract).
// ---------------------------------------------------------------------------
struct JniEnvScope {
    JNIEnv* env = nullptr;
    bool attached = false;
    JniEnvScope() {
        if (g_vm == nullptr) {
            return;
        }
        if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
#ifdef __ANDROID__
            jint rc = g_vm->AttachCurrentThread(&env, nullptr);
#else
            jint rc = g_vm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr);
#endif
            if (rc != JNI_OK) {
                env = nullptr;
                return;
            }
            attached = true;
        }
    }
    ~JniEnvScope() {
        if (attached) {
            g_vm->DetachCurrentThread();
        }
    }
};

// Generation progress events: stage "sampling" = denoising steps, anything after it (VAE
// decode, tiles) is reported as the decode phase. Also where a pending cancel is applied.
static void generation_progress_cb(const sd_generation_progress_t* p, void* /*data*/) {
    if (p == nullptr) {
        return;
    }
    if (g_cancel.load()) {
        sd_ctx_t* ctx = g_ctx.load();
        if (ctx != nullptr) {
            sd_cancel_generation(ctx, SD_CANCEL_ALL);
        }
    }
    const bool sampling = p->stage != nullptr && strcmp(p->stage, "sampling") == 0;
    const bool decoding = p->stage != nullptr && (strstr(p->stage, "decode") != nullptr);
    if (!sampling && !decoding) {
        return;  // loading / conditioning / encode: not shown as steps
    }
    JniEnvScope scope;
    if (scope.env == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_listener_mutex);
    if (g_listener != nullptr && g_on_progress != nullptr) {
        scope.env->CallVoidMethod(g_listener, g_on_progress, sampling ? kPhaseSampling : kPhaseDecode,
                                  (jint)p->completed, (jint)p->total);
        if (scope.env->ExceptionCheck()) {
            scope.env->ExceptionClear();
        }
    }
}

// Latent previews ("proj" mode: a cheap latent -> RGB projection at latent resolution).
// frames[0] is a U8 [C, W, H] tensor, i.e. interleaved RGB rows.
static void preview_cb(int /*step*/, int frame_count, const sd_tensor_t* const* frames, bool /*is_noisy*/,
                       void* /*data*/) {
    if (frame_count < 1 || frames == nullptr || frames[0] == nullptr) {
        return;
    }
    const sd_tensor_t* t = frames[0];
    if (sd_tensor_type(t) != SD_TENSOR_U8 || sd_tensor_rank(t) < 3) {
        return;
    }
    const int C = (int)sd_tensor_dim(t, 0);
    const int W = (int)sd_tensor_dim(t, 1);
    const int H = (int)sd_tensor_dim(t, 2);
    if (C < 3 || W <= 0 || H <= 0) {
        return;
    }
    const uint8_t* src = static_cast<const uint8_t*>(sd_tensor_data(t));
    std::vector<jbyte> rgb((size_t)W * H * 3);
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        rgb[i * 3 + 0] = (jbyte)src[i * C + 0];
        rgb[i * 3 + 1] = (jbyte)src[i * C + 1];
        rgb[i * 3 + 2] = (jbyte)src[i * C + 2];
    }
    JniEnvScope scope;
    if (scope.env == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_listener_mutex);
    if (g_listener == nullptr || g_on_preview == nullptr) {
        return;
    }
    jbyteArray arr = scope.env->NewByteArray((jsize)rgb.size());
    if (arr == nullptr) {
        scope.env->ExceptionClear();
        return;
    }
    scope.env->SetByteArrayRegion(arr, 0, (jsize)rgb.size(), rgb.data());
    scope.env->CallVoidMethod(g_listener, g_on_preview, arr, W, H);
    if (scope.env->ExceptionCheck()) {
        scope.env->ExceptionClear();
    }
    scope.env->DeleteLocalRef(arr);
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

// Lines of "name<TAB>description" from sd_list_devices().
static std::vector<std::pair<std::string, std::string>> list_devices() {
    std::vector<std::pair<std::string, std::string>> out;
    size_t need = sd_list_devices(nullptr, 0);
    if (need == 0) {
        return out;
    }
    std::string buf(need + 1, '\0');
    sd_list_devices(&buf[0], buf.size());
    buf.resize(strlen(buf.c_str()));
    size_t pos = 0;
    while (pos < buf.size()) {
        size_t nl = buf.find('\n', pos);
        std::string line = buf.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = nl == std::string::npos ? buf.size() : nl + 1;
        if (line.empty()) {
            continue;
        }
        size_t tab = line.find('\t');
        out.emplace_back(line.substr(0, tab), tab == std::string::npos ? "" : line.substr(tab + 1));
    }
    return out;
}

// First non-CPU device name (e.g. the Adreno OpenCL device), or "" if none.
static std::string first_gpu_device() {
    for (const auto& d : list_devices()) {
        std::string lower = d.first;
        for (auto& c : lower) c = (char)tolower((unsigned char)c);
        if (lower.rfind("cpu", 0) != 0) {
            return d.first;
        }
    }
    return "";
}

// ---------------------------------------------------------------------------
// JNI lifecycle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_vm = vm;
    sd_set_log_callback(sd_log_cb, nullptr);
    // Preview every denoised step as a cheap latent projection (no extra decoder run).
    sd_set_preview_callback(preview_cb, "proj", 1, true, false, nullptr);
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeSystemInfo(JNIEnv* env, jobject) {
    std::string info = std::string("libsd_jni variant: ") + SD_JNI_VARIANT + " / sd.cpp " + sd_version() + " ("
                       + sd_commit() + ")\n" + sd_get_system_info();
    auto devs = list_devices();
    info += "\ndevices:";
    for (const auto& d : devs) {
        info += " [" + d.first + (d.second.empty() ? "" : ": " + d.second) + "]";
    }
    return env->NewStringUTF(info.c_str());
}

// Name of a usable GPU device in this library ("" = CPU only / no OpenCL).
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeGpuDevice(JNIEnv* env, jobject) {
    return env->NewStringUTF(first_gpu_device().c_str());
}

// Backend the current model was loaded on ("CPU" or the GPU device name).
extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeActiveBackend(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lk(g_mutex);
    return env->NewStringUTF(g_ctx.load() == nullptr ? "" : g_backend.c_str());
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

static std::string last_sd_error(const char* fallback) {
    const char* e = sd_get_last_error();
    return (e != nullptr && *e) ? std::string(e) : std::string(fallback);
}

static void free_ctx_locked() {
    sd_ctx_t* ctx = g_ctx.exchange(nullptr);
    if (ctx != nullptr) {
        sd_set_generation_progress_callback(ctx, nullptr, nullptr);
        sd_ctx_free(ctx);
    }
    g_backend = "CPU";
}

// Load a single-file SD checkpoint (GGUF/safetensors/ckpt). Returns "" on success or an error.
// wtype: an sd_type_t to re-quantize weights to at load (<0 keeps the file's types).
// taesdPath: optional TAESD weights for a fast, low-memory final VAE decode ("" = full VAE).
// vaeTiling: kept for API compatibility; tiling is applied per generation (g_vae_tiling).
// useGpu: run the model graphs on the first GPU device of this library (falls back to an error
// if there is none — Java then reloads on the CPU).
static std::atomic<bool> g_vae_tiling{false};

extern "C" JNIEXPORT jstring JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeInit(JNIEnv* env, jobject, jstring modelPath,
                                                             jint nThreads, jint wtype, jstring taesdPath,
                                                             jboolean vaeTiling, jboolean useGpu) {
    std::lock_guard<std::mutex> lk(g_mutex);
    std::string model = jstr(env, modelPath);
    if (model.empty()) {
        return env->NewStringUTF("model path is empty");
    }
    std::string taesd = jstr(env, taesdPath);
    set_error("");
    free_ctx_locked();
    g_vae_tiling = vaeTiling == JNI_TRUE;

    int threads = nThreads;
    if (threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        threads = hc > 0 ? (int)hc : 4;
    }
    std::string backend = "cpu";
    if (useGpu == JNI_TRUE) {
        backend = first_gpu_device();
        if (backend.empty()) {
            std::string err = "GPU デバイスが見つかりません（OpenCL 非対応）";
            set_error(err);
            return env->NewStringUTF(err.c_str());
        }
    }
    ALOGI("init: model=%s threads=%d wtype=%d taesd=%s vae_tiling=%d backend=%s variant=%s", model.c_str(), threads,
          (int)wtype, taesd.empty() ? "-" : taesd.c_str(), vaeTiling ? 1 : 0, backend.c_str(), SD_JNI_VARIANT);

    std::string err;
    sd_ctx_opt_t* opt = sd_ctx_opt_create();
    sd_ctx_t* ctx = nullptr;
    auto check = [&](sd_status_t st, const char* what) {
        if (st != SD_STATUS_OK && err.empty()) {
            err = std::string(what) + ": " + last_sd_error("error");
        }
    };
    if (opt == nullptr) {
        err = "オプションを確保できません";
    } else {
        check(sd_opt_set_string(opt, "model_path", model.c_str()), "model_path");
        check(sd_opt_set_int(opt, "n_threads", threads), "n_threads");
        check(sd_opt_set_string(opt, "backend", backend.c_str()), "backend");
        check(sd_opt_set_string(opt, "rng_type", "std_default"), "rng_type");
        check(sd_opt_set_bool(opt, "eager_load", true), "eager_load");   // load now, not on first generate
        if (!taesd.empty()) {
            check(sd_opt_set_string(opt, "taesd_path", taesd.c_str()), "taesd_path");
        }
        if (wtype >= 0 && wtype < SD_TYPE_COUNT) {
            check(sd_opt_set_string(opt, "wtype", sd_type_name((sd_type_t)wtype)), "wtype");
        }
        if (err.empty()) {
            check(sd_ctx_create(opt, &ctx), "sd_ctx_create");
        }
        sd_opt_free(opt);
    }
    if (err.empty() && ctx != nullptr && !sd_ctx_supports_image_generation(ctx)) {
        err = "このモデルは画像生成に対応していません: " + last_sd_error(sd_get_model_version_name(ctx));
    }
    if (!err.empty() || ctx == nullptr) {
        if (ctx != nullptr) {
            sd_ctx_free(ctx);
        }
        if (err.empty()) {
            err = "sd_ctx_create failed (see log)";
        }
        set_error(err);
        ALOGE("%s", err.c_str());
        sd_log_to_file(err.c_str());
        return env->NewStringUTF(err.c_str());
    }
    sd_set_generation_progress_callback(ctx, generation_progress_cb, nullptr);
    g_backend = backend == "cpu" ? "CPU" : backend;
    g_ctx = ctx;
    return env->NewStringUTF("");
}

static int64_t resolve_seed(jlong seed) {
    if (seed >= 0) {
        return (int64_t)seed;
    }
    static std::mt19937_64 rng{std::random_device{}()};  // guarded by g_mutex (callers hold it)
    return (int64_t)(rng() & 0x7FFFFFFFULL);
}

// Must match StableDiffusionNative.SAMPLE_*.
static const char* method_name(jint m) {
    switch (m) {
        case 1: return "euler";
        case 5: return "dpm++2m";
        case 9: return "lcm";
        default: return "euler_a";
    }
}

// Records an error. Returns nullptr for convenient `return fail(...)`.
static jbyteArray fail(const std::string& msg) {
    set_error(msg);
    ALOGE("%s", msg.c_str());
    sd_log_to_file(msg.c_str());
    return nullptr;
}

// Runs one generation (init_rgb == nullptr -> txt2img). Returns width*height*3 RGB bytes, or
// null on failure / cancel. Caller holds g_mutex.
static jbyteArray generate_locked(JNIEnv* env, const char* what, const std::string& prompt,
                                  const std::string& negative, int width, int height, int steps, float cfg,
                                  int64_t seed, jint method, int clip_skip, const std::vector<uint8_t>* init_rgb,
                                  float strength) {
    sd_ctx_t* ctx = g_ctx.load();
    if (ctx == nullptr) {
        return fail(std::string(what) + ": モデルが読み込まれていません");
    }
    sd_img_gen_opt_t* opt = sd_img_gen_opt_create();
    if (opt == nullptr) {
        return fail(std::string(what) + ": オプションを確保できません");
    }
    std::string err;
    auto check = [&](sd_status_t st, const char* key) {
        if (st != SD_STATUS_OK && err.empty()) {
            err = std::string(key) + ": " + last_sd_error("error");
        }
    };
    sd_tensor_t* init = nullptr;
    check(sd_opt_set_string(opt, "prompt", prompt.c_str()), "prompt");
    check(sd_opt_set_string(opt, "negative_prompt", negative.c_str()), "negative_prompt");
    check(sd_opt_set_int(opt, "width", width), "width");
    check(sd_opt_set_int(opt, "height", height), "height");
    check(sd_opt_set_int(opt, "sample.steps", steps), "sample.steps");
    check(sd_opt_set_string(opt, "sample.method", method_name(method)), "sample.method");
    check(sd_opt_set_float(opt, "sample.guidance.txt_cfg", cfg), "txt_cfg");
    check(sd_opt_set_int(opt, "seed", seed), "seed");
    if (clip_skip > 0) {
        check(sd_opt_set_int(opt, "clip_skip", clip_skip), "clip_skip");
    }
    if (g_vae_tiling.load()) {
        check(sd_opt_set_bool(opt, "vae_tiling.enabled", true), "vae_tiling.enabled");
    }
    if (init_rgb != nullptr) {
        const int64_t shape[3] = {3, width, height};
        init = sd_tensor_create(SD_TENSOR_U8, 3, shape, init_rgb->data(), init_rgb->size());
        if (init == nullptr) {
            err = "init_image: " + last_sd_error("tensor");
        } else {
            check(sd_opt_set_tensor(opt, "init_image", init), "init_image");
            check(sd_opt_set_float(opt, "strength", strength), "strength");
        }
    }
    if (!err.empty()) {
        if (init) sd_tensor_free(init);
        sd_opt_free(opt);
        return fail(std::string(what) + ": " + err);
    }

    sd_cancel_generation(ctx, SD_CANCEL_RESET);
    sd_result_t* result = nullptr;
    sd_status_t st = g_cancel.load() ? SD_STATUS_CANCELLED : sd_generate_image(ctx, opt, &result);
    if (init) sd_tensor_free(init);
    sd_opt_free(opt);

    if (st == SD_STATUS_CANCELLED || (st != SD_STATUS_OK && g_cancel.load())) {
        sd_result_free(result);
        g_cancelled = true;
        set_error("cancelled");
        return nullptr;
    }
    if (st != SD_STATUS_OK || result == nullptr || sd_result_get_image_count(result) < 1) {
        std::string e = last_sd_error("画像が生成されませんでした（ログ参照）");
        sd_result_free(result);
        return fail(std::string(what) + ": " + e);
    }
    const sd_tensor_t* img = sd_result_get_image(result, 0);
    const int C = (int)sd_tensor_dim(img, 0);
    const int W = (int)sd_tensor_dim(img, 1);
    const int H = (int)sd_tensor_dim(img, 2);
    const uint8_t* src = static_cast<const uint8_t*>(sd_tensor_data(img));
    jbyteArray arr = nullptr;
    if (C >= 3 && W == width && H == height && src != nullptr) {
        std::vector<jbyte> rgb((size_t)W * H * 3);
        for (size_t i = 0; i < (size_t)W * H; ++i) {
            rgb[i * 3 + 0] = (jbyte)src[i * C + 0];
            rgb[i * 3 + 1] = (jbyte)src[i * C + 1];
            rgb[i * 3 + 2] = (jbyte)src[i * C + 2];
        }
        arr = env->NewByteArray((jsize)rgb.size());
        if (arr != nullptr) {
            env->SetByteArrayRegion(arr, 0, (jsize)rgb.size(), rgb.data());
        } else {
            env->ExceptionClear();
            set_error(std::string(what) + ": 結果バッファを確保できませんでした");
        }
    } else {
        set_error(std::string(what) + ": 想定外の出力形式 (" + std::to_string(C) + "x" + std::to_string(W) + "x"
                  + std::to_string(H) + ")");
    }
    sd_result_free(result);
    return arr;
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
    const int64_t used_seed = resolve_seed(seed);
    g_last_seed = used_seed;
    return generate_locked(env, "txt2img", jstr(env, prompt), jstr(env, negativePrompt), width, height, steps,
                           cfgScale, used_seed, sampleMethod, clipSkip, nullptr, 0.0f);
}

// img2img -> returns width*height*3 RGB bytes, or null on failure / cancel.
// initRgb is an initW*initH*3 RGB image (must equal width x height); strength in (0,1] controls
// how far the result departs from it.
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
    if (initRgb == nullptr || env->GetArrayLength(initRgb) < (jsize)(initW * initH * 3)) {
        return fail("img2img: 入力画像が不正です");
    }
    if (initW != width || initH != height) {
        return fail("img2img: 入力画像のサイズが出力サイズと一致しません");
    }
    std::vector<uint8_t> init_data((size_t)initW * initH * 3);
    env->GetByteArrayRegion(initRgb, 0, (jsize)init_data.size(), reinterpret_cast<jbyte*>(init_data.data()));
    const int64_t used_seed = resolve_seed(seed);
    g_last_seed = used_seed;
    return generate_locked(env, "img2img", jstr(env, prompt), jstr(env, negativePrompt), width, height, steps,
                           cfgScale, used_seed, sampleMethod, clipSkip, &init_data, strength);
}

// Request cancellation of the running (or next) generation. Lock-free: callable from any thread.
extern "C" JNIEXPORT void JNICALL
Java_com_micklab_llamaimage_StableDiffusionNative_nativeCancel(JNIEnv*, jobject) {
    g_cancel = true;
    sd_ctx_t* ctx = g_ctx.load();
    if (ctx != nullptr) {
        sd_cancel_generation(ctx, SD_CANCEL_ALL);   // no-op when idle
    }
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
    free_ctx_locked();
}
