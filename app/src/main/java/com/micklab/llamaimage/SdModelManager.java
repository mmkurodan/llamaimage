package com.micklab.llamaimage;

import java.util.Objects;

/**
 * Single owner of the native stable-diffusion context — the image-generation counterpart of
 * the LLM AI Server's {@code ModelManager}. Centralises load / generate / cancel / busy state so a
 * UI (and, later, an API service) can drive generation without touching JNI directly.
 *
 * Heavy calls block; callers invoke them off the main thread. They are serialised here (and in
 * native). The process-wide singleton outlives Activity instances, so a recreated Activity can
 * reuse an already-loaded model: {@link #load} is a no-op when the same configuration is loaded.
 */
public final class SdModelManager {

    /** UI-side listener; called on the generating thread. */
    public interface Listener {
        void onProgress(int phase, int step, int steps);
        void onPreview(byte[] rgb, int w, int h);
    }

    /** Everything that is fixed when the native context is created. */
    public static final class LoadConfig {
        public final String modelPath;
        public final int threads;
        public final int wtype;
        public final String taesdPath;   // "" = full VAE
        public final boolean vaeTiling;
        public final boolean useGpu;     // run on the GPU (OpenCL) if this library has one

        public LoadConfig(String modelPath, int threads, int wtype, String taesdPath, boolean vaeTiling) {
            this(modelPath, threads, wtype, taesdPath, vaeTiling, false);
        }

        public LoadConfig(String modelPath, int threads, int wtype, String taesdPath, boolean vaeTiling,
                          boolean useGpu) {
            this.modelPath = modelPath;
            this.threads = threads;
            this.wtype = wtype;
            this.taesdPath = taesdPath == null ? "" : taesdPath;
            this.vaeTiling = vaeTiling;
            this.useGpu = useGpu;
        }

        @Override public boolean equals(Object o) {
            if (!(o instanceof LoadConfig)) return false;
            LoadConfig c = (LoadConfig) o;
            return threads == c.threads && wtype == c.wtype && vaeTiling == c.vaeTiling
                    && useGpu == c.useGpu && Objects.equals(modelPath, c.modelPath) && Objects.equals(taesdPath, c.taesdPath);
        }

        @Override public int hashCode() {
            return Objects.hash(modelPath, threads, wtype, taesdPath, vaeTiling, useGpu);
        }
    }

    /** Outcome of one generation. Exactly one of {@code rgb != null}, {@code cancelled}, error. */
    public static final class Result {
        public final byte[] rgb;         // width*height*3, or null
        public final int width;
        public final int height;
        public final long seed;
        public final boolean cancelled;
        public final String error;       // "" when ok or cancelled

        Result(byte[] rgb, int width, int height, long seed, boolean cancelled, String error) {
            this.rgb = rgb;
            this.width = width;
            this.height = height;
            this.seed = seed;
            this.cancelled = cancelled;
            this.error = error == null ? "" : error;
        }

        public boolean ok() {
            return rgb != null;
        }
    }

    private static volatile SdModelManager instance;

    private final StableDiffusionNative sd = new StableDiffusionNative();

    private volatile Listener listener;
    private volatile LoadConfig loaded;      // null = no model in memory
    private volatile boolean busy = false;
    private volatile boolean cancelRequested = false;

    private SdModelManager() {
        // Registered once for the process; UI listeners are swapped on the Java side only.
        sd.setListener(new StableDiffusionNative.Callbacks() {
            @Override public void onProgress(int phase, int step, int steps) {
                Listener l = listener;
                if (l != null) l.onProgress(phase, step, steps);
            }

            @Override public void onPreview(byte[] rgb, int w, int h) {
                Listener l = listener;
                if (l != null) l.onPreview(rgb, w, h);
            }
        });
    }

    /** Throws UnsatisfiedLinkError if no native library variant can be loaded. */
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

    public void setListener(Listener l) {
        listener = l;
    }

    /** Clears the listener only if it is still {@code l} (a newer Activity may have replaced it). */
    public void clearListener(Listener l) {
        if (listener == l) listener = null;
    }

    public String systemInfo() {
        return sd.systemInfo();
    }

    public String libraryName() {
        return StableDiffusionNative.LIBRARY;
    }

    /** GPU device of the loaded native library ("" = none: CPU only). */
    public String gpuDevice() {
        return sd.gpuDevice();
    }

    /** Backend the current model runs on ("CPU" or the GPU device name; "" = not loaded). */
    public String activeBackend() {
        return sd.activeBackend();
    }

    public boolean isModelLoaded() {
        return loaded != null;
    }

    public boolean isLoaded(LoadConfig cfg) {
        LoadConfig l = loaded;
        return l != null && l.equals(cfg);
    }

    public LoadConfig getLoadedConfig() {
        return loaded;
    }

    public boolean isBusy() {
        return busy;
    }

    /**
     * Load (or reload) the model. No-op returning "" if {@code cfg} is already loaded.
     * Returns "" on success or an error message. Blocking.
     */
    public synchronized String load(LoadConfig cfg) {
        if (cfg.equals(loaded)) {
            return "";
        }
        busy = true;
        try {
            loaded = null;
            String err = sd.init(cfg.modelPath, cfg.threads, cfg.wtype, cfg.taesdPath, cfg.vaeTiling,
                    cfg.useGpu);
            if (err != null && err.isEmpty()) {
                loaded = cfg;
                return "";
            }
            return (err == null || err.isEmpty()) ? "init failed" : err;
        } finally {
            busy = false;
        }
    }

    /** Generate one image. Blocking. */
    public synchronized Result txt2img(String prompt, String negativePrompt, int width, int height,
                                       int steps, float cfgScale, long seed, int sampleMethod) {
        if (loaded == null) {
            return new Result(null, width, height, seed, false, "モデルが読み込まれていません");
        }
        if (cancelRequested) {
            return new Result(null, width, height, seed, true, "");
        }
        busy = true;
        try {
            byte[] rgb = sd.txt2img(prompt, negativePrompt, width, height, steps,
                    cfgScale, seed, sampleMethod, -1);
            return finish(rgb, width, height);
        } finally {
            busy = false;
        }
    }

    /** Generate from an existing image (img2img); {@code initRgb} must be width x height. Blocking. */
    public synchronized Result img2img(byte[] initRgb, String prompt, String negativePrompt,
                                       int width, int height, int steps, float cfgScale,
                                       float strength, long seed, int sampleMethod) {
        if (loaded == null) {
            return new Result(null, width, height, seed, false, "モデルが読み込まれていません");
        }
        if (cancelRequested) {
            return new Result(null, width, height, seed, true, "");
        }
        busy = true;
        try {
            byte[] rgb = sd.img2img(initRgb, width, height, prompt, negativePrompt, width, height,
                    steps, cfgScale, strength, seed, sampleMethod, -1);
            return finish(rgb, width, height);
        } finally {
            busy = false;
        }
    }

    private Result finish(byte[] rgb, int width, int height) {
        long seed = sd.getLastSeed();
        if (rgb != null) {
            return new Result(rgb, width, height, seed, false, "");
        }
        boolean cancelled = sd.wasCancelled();
        return new Result(null, width, height, seed, cancelled, cancelled ? "" : sd.getLastError());
    }

    /** Ask the running generation to stop at its next sampling step. Never blocks. */
    public void requestCancel() {
        cancelRequested = true;
        sd.cancel();
    }

    /** Call before starting a new generation pipeline. */
    public void clearCancel() {
        cancelRequested = false;
        sd.clearCancel();
    }

    public boolean isCancelRequested() {
        return cancelRequested;
    }

    /** Frees the native context (e.g. before deleting the model file it was loaded from). */
    public synchronized void release() {
        sd.free();
        loaded = null;
    }
}
