package com.micklab.llamaimage.npu;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * SD1.5 txt2img / img2img (LCM, or Euler a / Euler / DPM++ 2M for regular checkpoints) over ONNX models (512x512 static shapes): text_encoder, unet,
 * taesd_encoder, taesd_decoder + CLIP tokenizer files, all in one directory. Runs on the Hexagon
 * NPU through ONNX Runtime's QNN execution provider ({@code qnnOptions} != null), otherwise on the
 * ORT CPU provider (used by the CI test). Independent of Android so it can be tested on a JVM.
 *
 * QNN compiles each graph for the device on the first session creation; the compiled context is
 * written to {@code cacheDir} ("ep.context_*") and loaded directly next time. The UNet comes in
 * parts ({@link #UNET_PARTS}, scripts/npu/unet_parts.py) chained by tensor name: compiling the
 * whole UNet at once needed ~10 GB and the process was killed. Every missing context is compiled
 * and its session closed again (freeing the compiler's memory and the HTP buffers) before the next
 * one; only then are all sessions opened from the cache.
 */
public final class NpuPipeline implements Closeable {
    public static final int SIZE = 512, LATENT = 64, CHANNELS = 4, HIDDEN = 768;
    private static final int LATENT_LEN = CHANNELS * LATENT * LATENT;

    public interface Log {
        void log(String msg);

        /** Session {@code index} (1-based) of {@code total} is being created (compiled on first use). */
        default void stage(int index, int total, String name) {
        }
    }

    public interface Listener {
        void onStep(int step, int steps);

        void onPreview(byte[] rgb, int w, int h);

        boolean isCancelled();
    }

    private static final float[][] LATENT_RGB = {   // same projection as the CPU engine's previews
            {0.298f, 0.207f, 0.208f}, {0.187f, 0.286f, 0.173f}, {-0.158f, 0.189f, 0.264f}, {-0.184f, -0.271f, -0.473f}};

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    /** UNet parts in execution order (= unet_parts.NAMES). */
    public static final String[] UNET_PARTS = {"unet_0", "unet_1", "unet_2"};

    private final OrtSession textEncoder, decoder, encoder;
    private final OrtSession[] unet = new OrtSession[UNET_PARTS.length];
    private final ClipTokenizer tokenizer;
    private final LcmScheduler scheduler = new LcmScheduler();
    private final KDiffusionSampler kSampler = new KDiffusionSampler();
    private long lastSeed = -1;

    public NpuPipeline(File dir, Map<String, String> qnnOptions, File cacheDir, Log log) throws IOException {
        tokenizer = new ClipTokenizer(new File(dir, "tokenizer_vocab.json"), new File(dir, "tokenizer_merges.txt"));
        String[] names = new String[UNET_PARTS.length + 3];
        names[0] = "text_encoder";
        System.arraycopy(UNET_PARTS, 0, names, 1, UNET_PARTS.length);
        names[names.length - 2] = "taesd_decoder";
        names[names.length - 1] = "taesd_encoder";
        OrtSession[] s = new OrtSession[names.length];
        try {
            if (qnnOptions != null && cacheDir != null) {
                // Pass 1: compile what is not cached yet, one graph at a time, closing each session.
                for (int i = 0; i < names.length; i++) {
                    File model = new File(dir, names[i] + ".onnx");
                    if (contextFile(cacheDir, names[i]).isFile()) continue;
                    log.stage(i + 1, names.length, names[i]);
                    s[i] = compile(model, qnnOptions, cacheDir, log);   // null once cached
                }
            }
            // Pass 2: open everything (from the cache on the NPU).
            for (int i = 0; i < names.length; i++) {
                if (s[i] == null) s[i] = open(new File(dir, names[i] + ".onnx"), qnnOptions, cacheDir, log);
            }
        } catch (OrtException | RuntimeException e) {
            for (OrtSession x : s) closeQuietly(x);
            throw new IOException("NPU セッションを作成できません: " + e.getMessage(), e);
        }
        textEncoder = s[0];
        System.arraycopy(s, 1, unet, 0, UNET_PARTS.length);
        decoder = s[names.length - 2];
        encoder = s[names.length - 1];
    }

    private static File contextFile(File cacheDir, String name) {
        return new File(cacheDir, name + "_ctx.onnx");
    }

    private OrtSession open(File model, Map<String, String> qnn, File cacheDir, Log log) throws OrtException {
        String name = model.getName().replace(".onnx", "");
        long t0 = System.currentTimeMillis();
        if (qnn == null) {
            OrtSession s = env.createSession(model.getAbsolutePath(), new OrtSession.SessionOptions());
            log.log("NPU(CPU EP) " + name + ": " + (System.currentTimeMillis() - t0) + "ms");
            return s;
        }
        File ctx = cacheDir == null ? null : contextFile(cacheDir, name);
        if (ctx != null && ctx.isFile()) {
            try {
                OrtSession s = env.createSession(ctx.getAbsolutePath(), qnnOptions(qnn));
                log.log("NPU " + name + ": キャッシュから読込 " + (System.currentTimeMillis() - t0) + "ms");
                return s;
            } catch (OrtException e) {
                log.log("NPU " + name + ": キャッシュが使えないため再コンパイル (" + e.getMessage() + ")");
                deleteContext(cacheDir, name);
            }
        }
        return compileSession(model, qnn, cacheDir, log);   // keeps the (now cached) session
    }

    /** Compiles {@code model} for the HTP (writing the context cache when {@code cacheDir} is set). */
    private OrtSession compileSession(File model, Map<String, String> qnn, File cacheDir, Log log) throws OrtException {
        String name = model.getName().replace(".onnx", "");
        long t0 = System.currentTimeMillis();
        File ctx = cacheDir == null ? null : contextFile(cacheDir, name);
        if (cacheDir != null && !cacheDir.isDirectory()) cacheDir.mkdirs();
        OrtSession.SessionOptions so = qnnOptions(qnn);
        if (cacheDir != null) {
            so.addConfigEntry("ep.context_enable", "1");
            so.addConfigEntry("ep.context_file_path", ctx.getAbsolutePath());
            so.addConfigEntry("ep.context_embed_mode", "0");   // binary beside it (no 2 GB protobuf limit)
        }
        Runtime rt = Runtime.getRuntime();
        log.log("NPU " + name + ": NPU 向け最適化を開始（" + (model.length() + new File(model.getParentFile(),
                name + ".weights").length()) / 1_000_000 + " MB, Java heap 空き " + (rt.maxMemory() - rt.totalMemory()
                + rt.freeMemory()) / 1_000_000 + " MB）");
        OrtSession s = env.createSession(model.getAbsolutePath(), so);
        log.log("NPU " + name + ": 最適化 完了 " + (System.currentTimeMillis() - t0) / 1000 + "s");
        return s;
    }

    /**
     * Compiles and caches; the session is closed again in pass 1 (the cache is opened in pass 2),
     * and kept only when there is no cache to reopen it from.
     */
    private OrtSession compile(File model, Map<String, String> qnn, File cacheDir, Log log) throws OrtException {
        OrtSession s = compileSession(model, qnn, cacheDir, log);
        if (cacheDir == null) return s;
        String name = model.getName().replace(".onnx", "");
        if (!contextFile(cacheDir, name).isFile()) return s;   // no cache written: keep this session
        closeQuietly(s);
        return null;
    }

    private static OrtSession.SessionOptions qnnOptions(Map<String, String> qnn) throws OrtException {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.addQnn(qnn);
        return so;
    }

    private static void deleteContext(File cacheDir, String name) {
        File[] files = cacheDir == null ? null : cacheDir.listFiles();
        if (files == null) return;
        for (File f : files) if (f.getName().startsWith(name + "_ctx")) f.delete();
    }

    public long getLastSeed() {
        return lastSeed;
    }

    /** LCM sampling (the original NPU API; LCM-distilled checkpoints). */
    public byte[] generate(String prompt, String negative, int steps, float cfg, long seed, byte[] initRgb,
                           float strength, Listener listener) throws OrtException {
        return generate(prompt, negative, steps, cfg, seed, initRgb, strength, KDiffusionSampler.LCM, listener);
    }

    /**
     * Returns 512*512*3 RGB bytes, or null if cancelled. {@code initRgb} (512*512*3) switches to
     * img2img with {@code strength} in (0, 1]. cfg &gt; 1 adds an unconditional pass per step.
     * {@code sampler} = KDiffusionSampler id: LCM for LCM checkpoints, EULER_A / EULER / DPMPP2M
     * (same schedules as the CPU engine) for regular SD1.5 checkpoints; anything else = LCM.
     */
    public byte[] generate(String prompt, String negative, int steps, float cfg, long seed, byte[] initRgb,
                           float strength, int sampler, Listener listener) throws OrtException {
        lastSeed = seed >= 0 ? seed : (new Random().nextLong() & 0x7FFFFFFFL);
        Random rng = new Random(lastSeed);
        float[] cond = encodeText(prompt);
        float[] uncond = cfg > 1.0f ? encodeText(negative == null ? "" : negative) : null;
        if (KDiffusionSampler.supports(sampler)) {
            return generateK(sampler, steps, cfg, cond, uncond, rng, initRgb, strength, listener);
        }

        int[] ts = scheduler.timesteps(steps, initRgb != null ? strength : 1.0f);
        float[] latents = new float[LATENT_LEN];
        float[] noise = new float[LATENT_LEN];
        for (int i = 0; i < LATENT_LEN; i++) noise[i] = (float) rng.nextGaussian();
        if (initRgb == null) {
            System.arraycopy(noise, 0, latents, 0, LATENT_LEN);   // init_noise_sigma = 1
        } else {
            latents = encodeImage(initRgb);
            scheduler.addNoise(latents, noise, ts[0]);
        }
        float[] denoised = new float[LATENT_LEN];
        for (int s = 0; s < ts.length; s++) {
            if (listener != null && listener.isCancelled()) return null;
            if (listener != null) listener.onStep(s, ts.length);
            float[] eps = guidedEps(latents, ts[s], cond, uncond, cfg);
            scheduler.step(eps, ts[s], s + 1 < ts.length ? ts[s + 1] : -1, latents, denoised, rng);
            if (listener != null) listener.onPreview(preview(denoised), LATENT, LATENT);
        }
        if (listener != null) listener.onStep(ts.length, ts.length);
        return decode(denoised);
    }

    private byte[] generateK(int sampler, int steps, float cfg, float[] cond, float[] uncond, Random rng,
                             byte[] initRgb, float strength, Listener listener) throws OrtException {
        float[] sigmas = kSampler.schedule(steps, strength, initRgb != null);
        float[] x = new float[LATENT_LEN];
        for (int i = 0; i < LATENT_LEN; i++) x[i] = (float) rng.nextGaussian() * sigmas[0];
        if (initRgb != null) {
            float[] init = encodeImage(initRgb);
            for (int i = 0; i < LATENT_LEN; i++) x[i] += init[i];
        }
        final int total = sigmas.length - 1;
        float[] in = new float[LATENT_LEN];
        if (listener != null) listener.onStep(0, total);
        boolean done;
        try {
            done = kSampler.sample(sampler, x, sigmas, (xs, sigma, t) -> {
                if (listener != null && listener.isCancelled()) throw new CancelledException();
                float cIn = (float) (1 / Math.sqrt(sigma * sigma + 1));
                for (int i = 0; i < LATENT_LEN; i++) in[i] = xs[i] * cIn;
                float[] eps = guidedEps(in, t, cond, uncond, cfg);
                for (int i = 0; i < LATENT_LEN; i++) eps[i] = xs[i] - sigma * eps[i];   // -> x0 estimate
                return eps;
            }, rng, (step, n, den) -> {
                if (listener == null) return true;
                listener.onPreview(preview(den), LATENT, LATENT);
                listener.onStep(step, n);
                return !listener.isCancelled() || step == n;
            });
        } catch (CancelledException e) {
            return null;
        } catch (OrtException | RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return done ? decode(x) : null;
    }

    private static final class CancelledException extends Exception {
    }

    /** eps(x, t) with classifier-free guidance when {@code uncond} != null. */
    private float[] guidedEps(float[] x, float t, float[] cond, float[] uncond, float cfg) throws OrtException {
        float[] eps = unet(x, t, cond);
        if (uncond != null) {
            float[] epsU = unet(x, t, uncond);
            for (int i = 0; i < eps.length; i++) eps[i] = epsU[i] + cfg * (eps[i] - epsU[i]);
        }
        return eps;
    }

    /** session.run with the graph name in the error (the QNN message only names a hash). */
    private static OrtSession.Result run(OrtSession s, String name, Map<String, OnnxTensor> in) throws OrtException {
        try {
            return s.run(in);
        } catch (OrtException e) {
            throw new OrtException("[" + name + "] " + e.getMessage());
        }
    }

    /**
     * True if {@code msg} reports that the Hexagon DSP itself crashed / restarted (SSR) while
     * running a graph: the sessions are dead afterwards, and a broken compiled context is the
     * usual cause when it repeats for one model only.
     */
    public static boolean isDspCrash(String msg) {
        return msg != null && (msg.contains("SSR") || msg.contains("NPU crashed") || msg.contains("Error code: 1007"));
    }

    private float[] encodeText(String text) throws OrtException {
        int[] ids = tokenizer.encode(text);
        try (OnnxTensor in = OnnxTensor.createTensor(env, IntBuffer.wrap(ids), new long[]{1, ClipTokenizer.LENGTH});
             OrtSession.Result r = run(textEncoder, "text_encoder", single("input_ids", in))) {
            return floats(r);
        }
    }

    /** Runs the UNet parts in order; each takes its inputs by name from the earlier tensors. */
    private float[] unet(float[] sample, float t, float[] hidden) throws OrtException {
        Map<String, OnnxTensor> vals = new HashMap<>();
        java.util.List<OrtSession.Result> results = new java.util.ArrayList<>();
        try {
            vals.put("sample", OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), new long[]{1, CHANNELS, LATENT, LATENT}));
            vals.put("timestep", OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{t}), new long[]{1}));
            vals.put("encoder_hidden_states", OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden),
                    new long[]{1, ClipTokenizer.LENGTH, HIDDEN}));
            for (int p = 0; p < unet.length; p++) {
                Map<String, OnnxTensor> in = new HashMap<>();
                for (String n : unet[p].getInputNames()) {
                    OnnxTensor v = vals.get(n);
                    if (v == null) throw new IllegalStateException(UNET_PARTS[p] + ": 入力 " + n + " がありません");
                    in.put(n, v);
                }
                OrtSession.Result r = run(unet[p], UNET_PARTS[p], in);
                results.add(r);
                for (Map.Entry<String, ai.onnxruntime.OnnxValue> e : r) vals.put(e.getKey(), (OnnxTensor) e.getValue());
            }
            OnnxTensor out = vals.get("noise_pred");
            if (out == null) throw new IllegalStateException("noise_pred がありません");
            FloatBuffer fb = out.getFloatBuffer();
            float[] eps = new float[fb.remaining()];
            fb.get(eps);
            return eps;
        } finally {
            for (OrtSession.Result r : results) r.close();
            for (String n : new String[]{"sample", "timestep", "encoder_hidden_states"}) {
                OnnxTensor v = vals.get(n);
                if (v != null) v.close();
            }
        }
    }

    private float[] encodeImage(byte[] rgb) throws OrtException {
        int n = SIZE * SIZE;
        float[] chw = new float[3 * n];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) chw[c * n + i] = (rgb[i * 3 + c] & 0xFF) / 255f;
        }
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, SIZE, SIZE});
             OrtSession.Result r = run(encoder, "taesd_encoder", single("image", in))) {
            return floats(r);
        }
    }

    private byte[] decode(float[] latents) throws OrtException {
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(latents), new long[]{1, CHANNELS, LATENT, LATENT});
             OrtSession.Result r = run(decoder, "taesd_decoder", single("latents", in))) {
            float[] chw = floats(r);
            int n = SIZE * SIZE;
            byte[] rgb = new byte[3 * n];
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < 3; c++) {
                    int v = Math.round(chw[c * n + i] * 255f);
                    rgb[i * 3 + c] = (byte) (v < 0 ? 0 : Math.min(255, v));
                }
            }
            return rgb;
        }
    }

    private static byte[] preview(float[] lat) {
        int n = LATENT * LATENT;
        byte[] rgb = new byte[3 * n];
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < 3; ch++) {
                float v = 0;
                for (int c = 0; c < CHANNELS; c++) v += LATENT_RGB[c][ch] * lat[c * n + i];
                int iv = Math.round((v * 0.5f + 0.5f) * 255f);
                rgb[i * 3 + ch] = (byte) (iv < 0 ? 0 : Math.min(255, iv));
            }
        }
        return rgb;
    }

    private static Map<String, OnnxTensor> single(String name, OnnxTensor t) {
        Map<String, OnnxTensor> m = new HashMap<>();
        m.put(name, t);
        return m;
    }

    private static float[] floats(OrtSession.Result r) throws OrtException {
        FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
        float[] out = new float[fb.remaining()];
        fb.get(out);
        return out;
    }

    private static void closeQuietly(OrtSession s) {
        if (s == null) return;
        try {
            s.close();
        } catch (OrtException ignore) {
        }
    }

    @Override
    public void close() {
        for (OrtSession s : new OrtSession[]{textEncoder, decoder, encoder}) closeQuietly(s);
        for (OrtSession s : unet) closeQuietly(s);
    }
}
