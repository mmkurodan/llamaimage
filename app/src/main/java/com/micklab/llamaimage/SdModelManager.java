package com.micklab.llamaimage;

/**
 * Single owner of the native stable-diffusion context — the image-generation counterpart of
 * the LLM AI Server's {@code ModelManager}. Centralises load / generate / busy state so a UI
 * (and, later, an API service) can drive generation without touching JNI directly.
 *
 * Heavy calls block; callers are expected to invoke them off the main thread. The native layer
 * additionally serialises with its own mutex.
 */
public final class SdModelManager {

    private static volatile SdModelManager instance;

    private final StableDiffusionNative sd = new StableDiffusionNative();

    private volatile boolean modelLoaded = false;
    private volatile boolean busy = false;
    private volatile String loadedModelPath = null;

    private SdModelManager() {}

    public static SdModelManager get() {
        if (instance == null) {
            synchronized (SdModelManager.class) {
                if (instance == null) {
                    instance = new SdModelManager();
                }
            }
        }
        return instance;
    }

    public void setLogPath(String path) {
        sd.setLogPath(path);
    }

    public void setProgressListener(StableDiffusionNative.ProgressListener l) {
        sd.setProgressListener(l);
    }

    public void setPreviewListener(StableDiffusionNative.PreviewListener l) {
        sd.setPreviewListener(l);
    }

    public String systemInfo() {
        return sd.systemInfo();
    }

    public boolean isModelLoaded() {
        return modelLoaded;
    }

    public boolean isBusy() {
        return busy;
    }

    public String getLoadedModelPath() {
        return loadedModelPath;
    }

    /** Load (or reload) the model. Returns "" on success or an error message. Blocking. */
    public synchronized String load(String modelPath, int nThreads) {
        busy = true;
        try {
            modelLoaded = false;
            loadedModelPath = null;
            String err = sd.init(modelPath, nThreads);
            if (err != null && err.isEmpty()) {
                modelLoaded = true;
                loadedModelPath = modelPath;
                return "";
            }
            return (err == null || err.isEmpty()) ? "init failed" : err;
        } finally {
            busy = false;
        }
    }

    /** Generate one image (RGB w*h*3) or {@code null} on failure. Blocking. */
    public synchronized byte[] generate(String prompt, String negativePrompt, int width, int height,
                                        int steps, float cfgScale, long seed) {
        if (!modelLoaded) {
            return null;
        }
        busy = true;
        try {
            return sd.txt2img(prompt, negativePrompt, width, height, steps,
                    cfgScale, seed, StableDiffusionNative.SAMPLE_EULER_A, -1);
        } finally {
            busy = false;
        }
    }

    /** Generate from an existing image (img2img). Returns RGB w*h*3 or {@code null}. Blocking. */
    public synchronized byte[] img2img(byte[] initRgb, int initW, int initH,
                                       String prompt, String negativePrompt, int width, int height,
                                       int steps, float cfgScale, float strength, long seed) {
        if (!modelLoaded) {
            return null;
        }
        busy = true;
        try {
            return sd.img2img(initRgb, initW, initH, prompt, negativePrompt, width, height,
                    steps, cfgScale, strength, seed, StableDiffusionNative.SAMPLE_EULER_A, -1);
        } finally {
            busy = false;
        }
    }

    public long getLastSeed() {
        return sd.getLastSeed();
    }

    /** Last native error message (e.g. on out-of-memory), or "" if none. */
    public String getLastError() {
        return sd.getLastError();
    }

    public synchronized void release() {
        sd.free();
        modelLoaded = false;
        loadedModelPath = null;
    }
}
