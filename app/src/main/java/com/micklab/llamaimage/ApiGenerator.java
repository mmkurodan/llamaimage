package com.micklab.llamaimage;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.io.File;

/**
 * Runs one generation for the HTTP API ({@code api.ImageApiServer}) without the UI.
 *
 * It uses the same settings the user chose in the app (selected model, threads, weight type,
 * TAESD, VAE tiling, sampler, CFG, size, steps, NSFW restriction, LLM for translation), read from
 * the app's SharedPreferences, and the process-wide {@link SdModelManager}, so the UI and the API
 * share one loaded model. A request may override the per-image parameters.
 */
public final class ApiGenerator {

    /** Per-request parameters; null / negative fields fall back to the app's saved settings. */
    public static final class Request {
        public String prompt = "";
        public String negativePrompt = null;
        public int width = 0;              // 0 = saved image size (square)
        public int height = 0;
        public int steps = 0;              // 0 = saved steps
        public float cfgScale = Float.NaN; // NaN = saved CFG
        public long seed = -1;             // -1 = random
        public int sampler = -1;           // -1 = saved sampler
        public boolean translate = true;   // translate non-English prompts with the LLM
        public byte[] initImage = null;    // encoded PNG/JPEG for img2img, or null
        public float strength = Float.NaN; // img2img change amount 0.05..1 (NaN = from 'change')
        public String change = "";         // img2img level "slight" | "normal" | "strong" (per-model strength)
        public boolean jpeg = false;       // response format (default PNG)
        public String modelId = "";        // "" = the model selected in the app
    }

    public static final class Response {
        public final int status;           // HTTP status
        public final String error;         // "" on success
        public final boolean blocked;      // NSFW restriction hit
        public final boolean cancelled;
        public final String imageBase64;
        public final String mime;
        public final int width, height, steps, sampler;
        public final float cfgScale;
        public final long seed;
        public final String prompt, negativePrompt, model, note;
        public final long elapsedMs;
        public float strength;             // img2img strength used (0 for txt2img)

        private Response(int status, String error, boolean blocked, boolean cancelled,
                         String imageBase64, String mime, int width, int height, int steps,
                         int sampler, float cfgScale, long seed, String prompt,
                         String negativePrompt, String model, String note, long elapsedMs) {
            this.status = status;
            this.error = error == null ? "" : error;
            this.blocked = blocked;
            this.cancelled = cancelled;
            this.imageBase64 = imageBase64;
            this.mime = mime;
            this.width = width;
            this.height = height;
            this.steps = steps;
            this.sampler = sampler;
            this.cfgScale = cfgScale;
            this.seed = seed;
            this.prompt = prompt;
            this.negativePrompt = negativePrompt;
            this.model = model;
            this.note = note;
            this.elapsedMs = elapsedMs;
        }

        static Response error(int status, String msg) {
            return new Response(status, msg, false, false, null, null, 0, 0, 0, 0, 0f, -1,
                    "", "", "", "", 0);
        }
    }

    public interface Log {
        void log(String msg);
    }

    /**
     * img2img strength for a change level, per model. The same strength changes the image very
     * differently per model (measured on the device, 512px, same image and seed): SD-Turbo stays close
     * to the original even at 0.65, while DreamShaper 8 LCM is a different-looking subject at 0.65.
     * Values aim at "visible change, same subject" for normal.
     */
    static float strengthFor(String modelId, String level) {
        float[] s; // {slight, normal, strong}
        if ("sd_turbo".equals(modelId)) {
            s = new float[]{0.45f, 0.6f, 0.8f};
        } else if ("dreamshaper8_lcm".equals(modelId)) {
            s = new float[]{0.3f, 0.45f, 0.6f};
        } else {
            s = new float[]{0.35f, 0.5f, 0.7f};   // SD1.5 (CFG 7) and imported models: not measured
        }
        if ("slight".equals(level)) return s[0];
        if ("strong".equals(level)) return s[2];
        return s[1];
    }

    private static final int MIN_SIDE = 128;
    private static final int MAX_SIDE = 1024;

    private final Context app;
    private final Log log;
    private PromptTranslator translator;
    private final Object lock = new Object();
    private volatile boolean running = false;

    public ApiGenerator(Context context, Log log) {
        this.app = context.getApplicationContext();
        this.log = log;
    }

    public boolean isRunning() {
        return running;
    }

    /** True while this API or the app UI is loading / generating. */
    public static boolean engineBusy() {
        try {
            return SdModelManager.get().isBusy();
        } catch (Throwable t) {
            return false;
        }
    }

    public static boolean modelLoaded() {
        try {
            return SdModelManager.get().isModelLoaded();
        } catch (Throwable t) {
            return false;
        }
    }

    /** Stops the running generation (API or UI) at its next sampling step. */
    public static void cancel() {
        try {
            SdModelManager.get().requestCancel();
        } catch (Throwable ignored) {
        }
    }

    /**
     * Image models for GET /models: the catalog (without TAESD) plus an imported file, each with
     * whether it is downloaded and whether it is the app's current selection.
     */
    public org.json.JSONArray listModels() throws org.json.JSONException {
        ModelStore store = new ModelStore(app);
        String selected = selectedModelId();
        java.util.List<ModelCatalog.ModelSpec> specs = new java.util.ArrayList<>(ModelCatalog.MODELS);
        ModelCatalog.ModelSpec imported = store.importedModel();
        if (imported != null) specs.add(imported);
        org.json.JSONArray arr = new org.json.JSONArray();
        for (ModelCatalog.ModelSpec s : specs) {
            if (s == ModelCatalog.TAESD) continue;
            org.json.JSONObject o = new org.json.JSONObject();
            o.put("id", s.id);
            o.put("title", s.title);
            o.put("installed", store.installedFile(s) != null);
            o.put("selected", s.id.equals(selected));
            o.put("steps", s.steps);
            o.put("cfg_scale", (double) s.cfgScale);
            o.put("sampler", s.sampler);
            o.put("size_bytes", s.sizeBytes);
            arr.put(o);
        }
        return arr;
    }

    /** Selected model id from the app's settings ("" when none chosen yet). */
    public String selectedModelId() {
        return prefs().getString(MainActivity.PREFS_MODEL_ID, "");
    }

    /** Blocking. One request at a time: a second concurrent call gets 503. */
    public Response generate(Request req) {
        if (req.prompt == null || req.prompt.trim().isEmpty()) {
            return Response.error(400, "'prompt' is required");
        }
        synchronized (lock) {
            if (running || engineBusy()) {
                return Response.error(503, "busy: another image is being generated");
            }
            running = true;
        }
        try {
            return generateLocked(req);
        } catch (Throwable t) {
            log("API generate error: " + t);
            return Response.error(500, t.getMessage() != null ? t.getMessage() : t.toString());
        } finally {
            running = false;
        }
    }

    private Response generateLocked(Request req) throws Exception {
        long t0 = System.currentTimeMillis();
        SharedPreferences p = prefs();
        ModelStore store = new ModelStore(app);
        SdModelManager manager = SdModelManager.get();
        ensureLogPath(manager);

        // --- model: the request's choice, else the app's selection (same load-time settings as the UI) ---
        String appModelId = p.getString(MainActivity.PREFS_MODEL_ID, "");
        boolean overridden = req.modelId != null && !req.modelId.isEmpty() && !req.modelId.equals(appModelId);
        String modelId = overridden ? req.modelId : appModelId;
        ModelCatalog.ModelSpec spec = ModelCatalog.IMPORTED_ID.equals(modelId)
                ? store.importedModel() : ModelCatalog.findModel(modelId);
        if (overridden && (spec == null || spec == ModelCatalog.TAESD)) {
            return Response.error(400, "unknown model: " + req.modelId + " (see GET /models)");
        }
        File file = spec == null ? null : store.installedFile(spec);
        if (file == null) {
            return Response.error(503, overridden
                    ? "model '" + req.modelId + "' is not downloaded: download it in the llamaimage app first"
                    : "no image model is installed: open the llamaimage app and choose / download a model first");
        }
        boolean useTaesd = p.getBoolean(MainActivity.PREFS_USE_TAESD, true);
        File taesd = useTaesd ? store.installedTaesd() : null;
        int threadsPref = p.getInt(MainActivity.PREFS_THREADS, 0);
        int threads = threadsPref > 0 ? threadsPref : CpuInfo.defaultThreads();
        int wtype = p.getInt(MainActivity.PREFS_WTYPE, StableDiffusionNative.WTYPE_DEFAULT);
        long ram = CpuInfo.totalRamBytes(app);
        boolean vaeTiling = p.getBoolean(MainActivity.PREFS_VAE_TILING,
                ram > 0 && ram < MainActivity.LOW_RAM_BYTES);
        boolean gpu = p.getBoolean(MainActivity.PREFS_USE_GPU, false) && !manager.gpuDevice().isEmpty();
        SdModelManager.LoadConfig cfg = new SdModelManager.LoadConfig(file.getAbsolutePath(),
                threads, wtype, taesd != null ? taesd.getAbsolutePath() : "", vaeTiling, gpu);
        if (!manager.isLoaded(cfg)) {
            log("API: loading model " + spec.title);
            store.ensureVerified(spec, file, pct -> { });
            String err = manager.load(cfg);
            if (!err.isEmpty()) return Response.error(500, "model load failed: " + err);
        }

        // --- per-image parameters (request overrides the saved settings) ---
        int savedSize = p.getInt(MainActivity.PREFS_SIZE, 512);
        int width = MainActivity.snapTo64(clamp(req.width > 0 ? req.width : savedSize, MIN_SIDE, MAX_SIDE));
        int height = MainActivity.snapTo64(clamp(req.height > 0 ? req.height
                : (req.width > 0 ? req.width : savedSize), MIN_SIDE, MAX_SIDE));
        // The app's saved steps / CFG / sampler belong to the app's model; another model gets its own
        // recommended values (e.g. SD-Turbo 1 step vs SD1.5 20 steps).
        boolean specDefaults = overridden && spec.hasRecommendedSettings();
        int steps = clamp(req.steps > 0 ? req.steps
                        : (specDefaults ? spec.steps : p.getInt(MainActivity.PREFS_STEPS, spec.steps)),
                1, MainActivity.MAX_STEPS);
        float cfgScale = !Float.isNaN(req.cfgScale) ? req.cfgScale
                : (specDefaults ? spec.cfgScale : p.getFloat(MainActivity.PREFS_CFG, spec.cfgScale));
        cfgScale = Math.max(1.0f, Math.min(20.0f, cfgScale));
        int sampler = req.sampler >= 0 ? req.sampler
                : (specDefaults ? spec.sampler : p.getInt(MainActivity.PREFS_SAMPLER, spec.sampler));
        long seed = req.seed >= 0 ? req.seed : -1;
        boolean nsfw = p.getBoolean(MainActivity.PREFS_NSFW, false);
        String rawNegative = req.negativePrompt != null ? req.negativePrompt.trim() : "";

        // --- prompt translation / NSFW screening (same as the UI) ---
        String prompt = req.prompt.trim();
        String negative = rawNegative;
        String note = "";
        if (req.translate || nsfw) {
            PromptTranslator.Prepared prep = translator(p).prepare(
                    prompt, rawNegative, req.translate && cfgScale != 1.0f, nsfw);
            if (prep.blocked) {
                return new Response(422, "blocked: " + prep.note, true, false, null, null,
                        width, height, steps, sampler, cfgScale, seed, prep.prompt, prep.negative,
                        spec.id, prep.note, System.currentTimeMillis() - t0);
            }
            if (req.translate) {
                prompt = prep.prompt;
                negative = prep.negative;
            }
            note = prep.note;
        }

        // --- generate ---
        float usedStrength = 0f;
        manager.clearCancel();
        log("API generate: \"" + prompt + "\" " + width + "x" + height + " " + steps + "step"
                + (req.initImage != null ? " img2img" : ""));
        SdModelManager.Result r;
        if (req.initImage != null) {
            Bitmap src = BitmapFactory.decodeByteArray(req.initImage, 0, req.initImage.length);
            if (src == null) return Response.error(400, "could not decode 'init_image'");
            Bitmap fitted = cropToAspectAndScale(src, width, height);
            byte[] init = MainActivity.bitmapToRgb(fitted);
            if (fitted != src) fitted.recycle();
            src.recycle();
            float strength = Float.isNaN(req.strength) ? strengthFor(spec.id, req.change)
                    : Math.max(0.05f, Math.min(1.0f, req.strength));
            usedStrength = strength;
            int[] sched = MainActivity.img2imgSchedule(steps, strength);
            float sdStrength = (sched[1] - 0.5f) / sched[0];
            r = manager.img2img(init, prompt, negative, width, height, sched[0], cfgScale,
                    sdStrength, seed, sampler);
        } else {
            r = manager.txt2img(prompt, negative, width, height, steps, cfgScale, seed, sampler);
        }
        long elapsed = System.currentTimeMillis() - t0;
        if (r.cancelled) {
            return new Response(409, "cancelled", false, true, null, null, width, height, steps,
                    sampler, cfgScale, r.seed, prompt, negative, spec.id, note, elapsed);
        }
        if (!r.ok()) {
            return Response.error(500, r.error.isEmpty() ? "generation failed" : r.error);
        }

        Bitmap bmp = MainActivity.rgbToBitmap(r.rgb, r.width, r.height);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bmp.compress(req.jpeg ? Bitmap.CompressFormat.JPEG : Bitmap.CompressFormat.PNG, 92, out);
        bmp.recycle();
        String b64 = Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP);
        log("API generate done seed=" + r.seed + " " + (elapsed / 1000) + "s"
                + (usedStrength > 0 ? " strength=" + usedStrength : ""));
        Response ok = new Response(200, "", false, false, b64, req.jpeg ? "image/jpeg" : "image/png",
                r.width, r.height, steps, sampler, cfgScale, r.seed, prompt, negative, spec.id,
                note, elapsed);
        ok.strength = usedStrength;
        return ok;
    }

    private PromptTranslator translator(SharedPreferences p) {
        String url = p.getString(MainActivity.PREFS_LLM_URL, MainActivity.DEFAULT_LLM_BASE_URL);
        String model = p.getString(MainActivity.PREFS_LLM_MODEL, "");
        if (translator == null || !translator.sameEndpoint(url, model)) {
            translator = new PromptTranslator(url, model, this::log);
        }
        return translator;
    }

    /** The UI sets the native log path in its own start-up; do it here when the API comes first. */
    private void ensureLogPath(SdModelManager manager) {
        if (manager.isModelLoaded()) return;
        File log = new File(app.getFilesDir(), "sd.log");
        if (log.length() > 2_000_000) {
            log.renameTo(new File(app.getFilesDir(), "sd.log.1"));
        }
        manager.setLogPath(log.getAbsolutePath());
    }

    /** Center-crop to the target aspect ratio, then scale to exactly width x height. */
    private static Bitmap cropToAspectAndScale(Bitmap src, int width, int height) {
        int sw = src.getWidth(), sh = src.getHeight();
        float target = width / (float) height;
        int cw = sw, ch = sh;
        if (sw / (float) sh > target) {
            cw = Math.max(1, Math.round(sh * target));
        } else {
            ch = Math.max(1, Math.round(sw / target));
        }
        Bitmap cropped = (cw == sw && ch == sh) ? src
                : Bitmap.createBitmap(src, (sw - cw) / 2, (sh - ch) / 2, cw, ch);
        Bitmap scaled = Bitmap.createScaledBitmap(cropped, width, height, true);
        if (cropped != src && cropped != scaled) cropped.recycle();
        return scaled;
    }

    private SharedPreferences prefs() {
        return app.getSharedPreferences(MainActivity.PREFS_NAME, Context.MODE_PRIVATE);
    }

    private void log(String msg) {
        if (log != null) log.log(msg);
    }

    private static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}
