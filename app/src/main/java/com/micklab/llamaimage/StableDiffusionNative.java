package com.micklab.llamaimage;

/**
 * JNI interface to the stable-diffusion.cpp engine (libsd_jni*.so).
 *
 * Intentionally mirrors the LLM AI Server's {@code LlamaNative}: a thin Java facade over a single
 * native engine context, an {@code init()} that returns "" on success or an error message, and a
 * callback object for progress. Designed so it can later be merged into the LLM AI Server.
 *
 * The native library is built once per CPU variant (see app/src/main/cpp/CMakeLists.txt); the
 * best one the device supports is loaded here, falling back to the armv8-a baseline.
 */
public class StableDiffusionNative {

    /** Sampler ids — must match enum sample_method_t in stable-diffusion.h. */
    public static final int SAMPLE_EULER_A = 0;
    public static final int SAMPLE_EULER   = 1;
    public static final int SAMPLE_DPMPP2M = 5;
    public static final int SAMPLE_LCM     = 9;   // few-step distilled models (LCM/Turbo/Hyper-SD)

    /** Weight types — must match enum sd_type_t in stable-diffusion.h. */
    public static final int WTYPE_DEFAULT = -1;   // keep the model's original types
    public static final int WTYPE_Q4_0    = 2;
    public static final int WTYPE_Q8_0    = 8;

    /** Progress phases — must match kPhase* in jni_sd.cpp (g_sd_phase in the sd.cpp patch). */
    public static final int PHASE_SAMPLING = 1;   // one callback per denoising step (+ step 0 at start)
    public static final int PHASE_DECODE   = 2;   // VAE decode tiles (only when VAE tiling is on)

    /**
     * Native → Java callbacks, invoked synchronously on the generating thread.
     * {@code onPreview} fires after each sampling step with a W*H*3 RGB approximation of the
     * current latent (W×H = image size / 8; values are unsigned bytes stored as signed).
     */
    public interface Callbacks {
        void onProgress(int phase, int step, int steps);
        void onPreview(byte[] rgb, int w, int h);
    }

    /** Name of the native library that was loaded ("sd_jni_fp16" or the baseline "sd_jni"). */
    public static final String LIBRARY;

    static {
        String loaded = null;
        UnsatisfiedLinkError first = null;
        for (String name : CpuInfo.nativeLibraryCandidates()) {
            try {
                System.loadLibrary(name);
                loaded = name;
                break;
            } catch (UnsatisfiedLinkError e) {   // that variant is not packaged — try the next
                if (first == null) first = e;
            }
        }
        if (loaded == null) {
            throw first != null ? first : new UnsatisfiedLinkError("no sd_jni library");
        }
        LIBRARY = loaded;
    }

    // --- native methods (implemented in jni/jni_sd.cpp) ---
    private native String nativeSystemInfo();
    private native void nativeSetLogPath(String path);
    private native void nativeSetListener(Callbacks listener);
    private native String nativeInit(String modelPath, int nThreads, int wtype, String taesdPath,
                                     boolean vaeTiling);
    private native byte[] nativeTxt2img(String prompt, String negativePrompt,
                                        int width, int height, int steps,
                                        float cfgScale, long seed, int sampleMethod, int clipSkip);
    private native byte[] nativeImg2img(byte[] initRgb, int initW, int initH,
                                        String prompt, String negativePrompt,
                                        int width, int height, int steps,
                                        float cfgScale, float strength, long seed,
                                        int sampleMethod, int clipSkip);
    private native void nativeCancel();
    private native void nativeClearCancel();
    private native boolean nativeWasCancelled();
    private native long nativeGetLastSeed();
    private native String nativeLastError();
    private native void nativeFree();

    // --- thin public wrappers ---

    public String systemInfo() {
        return nativeSystemInfo();
    }

    public void setLogPath(String path) {
        nativeSetLogPath(path);
    }

    /** Registers the single callback object; never blocks behind a running generation. */
    public void setListener(Callbacks listener) {
        nativeSetListener(listener);
    }

    /**
     * Load a single-file SD checkpoint. Returns "" on success or an error message.
     * {@code wtype} re-quantizes weights at load ({@link #WTYPE_DEFAULT} keeps the file's types).
     * {@code taesdPath} points at optional TAESD weights ("" keeps the model's full VAE).
     * {@code vaeTiling} decodes the VAE in tiles: lower peak memory, roughly 2x slower decode.
     */
    public String init(String modelPath, int nThreads, int wtype, String taesdPath, boolean vaeTiling) {
        return nativeInit(modelPath, nThreads, wtype, taesdPath == null ? "" : taesdPath, vaeTiling);
    }

    /**
     * Generate one image. Returns width*height*3 RGB bytes, or {@code null} on failure / cancel.
     * {@code seed < 0} asks native to pick a random seed (retrieve it via {@link #getLastSeed()}).
     */
    public byte[] txt2img(String prompt, String negativePrompt, int width, int height,
                          int steps, float cfgScale, long seed, int sampleMethod, int clipSkip) {
        return nativeTxt2img(prompt, negativePrompt, width, height, steps,
                cfgScale, seed, sampleMethod, clipSkip);
    }

    /**
     * Generate from an existing image. {@code initRgb} is initW*initH*3 RGB bytes and must match
     * width x height; {@code strength} in (0,1] controls how far the result moves from the input.
     */
    public byte[] img2img(byte[] initRgb, int initW, int initH,
                          String prompt, String negativePrompt, int width, int height,
                          int steps, float cfgScale, float strength, long seed,
                          int sampleMethod, int clipSkip) {
        return nativeImg2img(initRgb, initW, initH, prompt, negativePrompt, width, height,
                steps, cfgScale, strength, seed, sampleMethod, clipSkip);
    }

    /** Ask the running (or next) generation to stop at the next sampling step. Thread-safe. */
    public void cancel() {
        nativeCancel();
    }

    public void clearCancel() {
        nativeClearCancel();
    }

    /** True if the most recent generation returned null because it was cancelled. */
    public boolean wasCancelled() {
        return nativeWasCancelled();
    }

    /** Seed actually used by the most recent generation. */
    public long getLastSeed() {
        return nativeGetLastSeed();
    }

    /** Last native error message (e.g. on out-of-memory), or "" if none. */
    public String getLastError() {
        return nativeLastError();
    }

    public void free() {
        nativeFree();
    }
}
