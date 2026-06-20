package com.micklab.llamaimage;

/**
 * JNI interface to the stable-diffusion.cpp engine (libsd_jni.so).
 *
 * Intentionally mirrors the LLM AI Server's {@code LlamaNative}: a thin Java facade over
 * a single native engine context, an {@code init()} that returns "" on success or an error
 * message, and a streaming-style listener for progress. Designed so it can later be merged
 * into the LLM AI Server alongside {@code LlamaNative}.
 */
public class StableDiffusionNative {

    /** Sampler ids — must match enum sample_method_t in stable-diffusion.h. */
    public static final int SAMPLE_EULER_A = 0;
    public static final int SAMPLE_EULER   = 1;
    public static final int SAMPLE_DPMPP2M = 5;

    /** Weight types — must match enum sd_type_t in stable-diffusion.h. */
    public static final int WTYPE_DEFAULT = -1;   // keep the model's original types
    public static final int WTYPE_Q4_0    = 2;
    public static final int WTYPE_Q8_0    = 8;

    /** Progress callback, invoked once per diffusion step from native. */
    public interface ProgressListener {
        void onProgress(int step, int steps);
    }

    /**
     * Preview callback, invoked alongside each progress step.
     * {@code rgb} is a W*H*3 byte array (unsigned values stored as signed bytes);
     * use {@code b & 0xFF} to recover [0,255]. W and H are the latent dimensions
     * (image_width/8 × image_height/8), typically 64×64 for a 512×512 image.
     * The image is a lightweight linear approximation — not a full VAE decode.
     */
    public interface PreviewListener {
        void onPreview(byte[] rgb, int w, int h);
    }

    static {
        System.loadLibrary("sd_jni");
    }

    // --- native methods (implemented in jni/jni_sd.cpp) ---
    private native String nativeSystemInfo();
    private native void nativeSetLogPath(String path);
    private native void nativeSetProgressListener(ProgressListener listener);
    private native void nativeSetPreviewListener(PreviewListener listener);
    private native String nativeInit(String modelPath, int nThreads, int wtype);
    private native byte[] nativeTxt2img(String prompt, String negativePrompt,
                                        int width, int height, int steps,
                                        float cfgScale, long seed, int sampleMethod, int clipSkip);
    private native byte[] nativeImg2img(byte[] initRgb, int initW, int initH,
                                        String prompt, String negativePrompt,
                                        int width, int height, int steps,
                                        float cfgScale, float strength, long seed,
                                        int sampleMethod, int clipSkip);
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

    public void setProgressListener(ProgressListener listener) {
        nativeSetProgressListener(listener);
    }

    public void setPreviewListener(PreviewListener listener) {
        nativeSetPreviewListener(listener);
    }

    /**
     * Load an all-in-one SD1.5 GGUF. Returns "" on success or an error message.
     * {@code wtype} re-quantizes weights at load (e.g. {@link #WTYPE_Q4_0}) for faster CPU
     * matmul; {@link #WTYPE_DEFAULT} keeps the model's original types.
     */
    public String init(String modelPath, int nThreads, int wtype) {
        return nativeInit(modelPath, nThreads, wtype);
    }

    /**
     * Generate one image. Returns width*height*3 RGB bytes, or {@code null} on failure.
     * {@code seed < 0} asks native to pick a random seed (retrieve it via {@link #getLastSeed()}).
     */
    public byte[] txt2img(String prompt, String negativePrompt, int width, int height,
                          int steps, float cfgScale, long seed, int sampleMethod, int clipSkip) {
        return nativeTxt2img(prompt, negativePrompt, width, height, steps,
                cfgScale, seed, sampleMethod, clipSkip);
    }

    /**
     * Generate from an existing image. {@code initRgb} is initW*initH*3 RGB bytes;
     * {@code strength} in [0,1] controls how far the result moves from the input.
     * Returns width*height*3 RGB bytes, or {@code null} on failure.
     */
    public byte[] img2img(byte[] initRgb, int initW, int initH,
                          String prompt, String negativePrompt, int width, int height,
                          int steps, float cfgScale, float strength, long seed,
                          int sampleMethod, int clipSkip) {
        return nativeImg2img(initRgb, initW, initH, prompt, negativePrompt, width, height,
                steps, cfgScale, strength, seed, sampleMethod, clipSkip);
    }

    /** Seed actually used by the most recent {@link #txt2img}. */
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
