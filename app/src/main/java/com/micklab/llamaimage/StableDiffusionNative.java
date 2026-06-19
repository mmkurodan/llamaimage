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
    private native String nativeInit(String modelPath, int nThreads);
    private native byte[] nativeTxt2img(String prompt, String negativePrompt,
                                        int width, int height, int steps,
                                        float cfgScale, long seed, int sampleMethod, int clipSkip);
    private native long nativeGetLastSeed();
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

    /** Load an all-in-one SD1.5 GGUF. Returns "" on success or an error message. */
    public String init(String modelPath, int nThreads) {
        return nativeInit(modelPath, nThreads);
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

    /** Seed actually used by the most recent {@link #txt2img}. */
    public long getLastSeed() {
        return nativeGetLastSeed();
    }

    public void free() {
        nativeFree();
    }
}
