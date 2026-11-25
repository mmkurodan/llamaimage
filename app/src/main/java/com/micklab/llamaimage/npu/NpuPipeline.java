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
 * SD1.5 LCM txt2img / img2img over ONNX models (512x512 static shapes): text_encoder, unet,
 * taesd_encoder, taesd_decoder + CLIP tokenizer files, all in one directory. Runs on the Hexagon
 * NPU through ONNX Runtime's QNN execution provider ({@code qnnOptions} != null), otherwise on the
 * ORT CPU provider (used by the CI test). Independent of Android so it can be tested on a JVM.
 *
 * QNN compiles each graph for the device on the first session creation; the compiled context is
 * written to {@code cacheDir} ("ep.context_*") and loaded directly next time.
 */
public final class NpuPipeline implements Closeable {
    public static final int SIZE = 512, LATENT = 64, CHANNELS = 4, HIDDEN = 768;
    private static final int LATENT_LEN = CHANNELS * LATENT * LATENT;

    public interface Log {
        void log(String msg);
    }

    public interface Listener {
        void onStep(int step, int steps);

        void onPreview(byte[] rgb, int w, int h);

        boolean isCancelled();
    }

    private static final float[][] LATENT_RGB = {   // same projection as the CPU engine's previews
            {0.298f, 0.207f, 0.208f}, {0.187f, 0.286f, 0.173f}, {-0.158f, 0.189f, 0.264f}, {-0.184f, -0.271f, -0.473f}};

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession textEncoder, unet, decoder, encoder;
    private final ClipTokenizer tokenizer;
    private final LcmScheduler scheduler = new LcmScheduler();
    private long lastSeed = -1;

    public NpuPipeline(File dir, Map<String, String> qnnOptions, File cacheDir, Log log) throws IOException {
        tokenizer = new ClipTokenizer(new File(dir, "tokenizer_vocab.json"), new File(dir, "tokenizer_merges.txt"));
        OrtSession te = null, un = null, de = null, en = null;
        try {
            te = open(new File(dir, "text_encoder.onnx"), qnnOptions, cacheDir, log);
            un = open(new File(dir, "unet.onnx"), qnnOptions, cacheDir, log);
            de = open(new File(dir, "taesd_decoder.onnx"), qnnOptions, cacheDir, log);
            en = open(new File(dir, "taesd_encoder.onnx"), qnnOptions, cacheDir, log);
        } catch (OrtException | RuntimeException e) {
            for (OrtSession s : new OrtSession[]{te, un, de, en}) closeQuietly(s);
            throw new IOException("NPU セッションを作成できません: " + e.getMessage(), e);
        }
        textEncoder = te;
        unet = un;
        decoder = de;
        encoder = en;
    }

    private OrtSession open(File model, Map<String, String> qnn, File cacheDir, Log log) throws OrtException {
        String name = model.getName().replace(".onnx", "");
        long t0 = System.currentTimeMillis();
        if (qnn == null) {
            OrtSession s = env.createSession(model.getAbsolutePath(), new OrtSession.SessionOptions());
            log.log("NPU(CPU EP) " + name + ": " + (System.currentTimeMillis() - t0) + "ms");
            return s;
        }
        File ctx = new File(cacheDir, name + "_ctx.onnx");
        if (ctx.isFile()) {
            try {
                OrtSession s = env.createSession(ctx.getAbsolutePath(), qnnOptions(qnn));
                log.log("NPU " + name + ": キャッシュから読込 " + (System.currentTimeMillis() - t0) + "ms");
                return s;
            } catch (OrtException e) {
                log.log("NPU " + name + ": キャッシュが使えないため再コンパイル (" + e.getMessage() + ")");
                deleteContext(cacheDir, name);
            }
        }
        if (cacheDir != null && !cacheDir.isDirectory()) cacheDir.mkdirs();
        OrtSession.SessionOptions so = qnnOptions(qnn);
        if (cacheDir != null) {
            so.addConfigEntry("ep.context_enable", "1");
            so.addConfigEntry("ep.context_file_path", ctx.getAbsolutePath());
            so.addConfigEntry("ep.context_embed_mode", "0");   // binary beside it (no 2 GB protobuf limit)
        }
        OrtSession s = env.createSession(model.getAbsolutePath(), so);
        log.log("NPU " + name + ": コンパイル " + (System.currentTimeMillis() - t0) / 1000 + "s");
        return s;
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

    /**
     * Returns 512*512*3 RGB bytes, or null if cancelled. {@code initRgb} (512*512*3) switches to
     * img2img with {@code strength} in (0, 1]. cfg &gt; 1 adds an unconditional pass per step.
     */
    public byte[] generate(String prompt, String negative, int steps, float cfg, long seed, byte[] initRgb,
                           float strength, Listener listener) throws OrtException {
        lastSeed = seed >= 0 ? seed : (new Random().nextLong() & 0x7FFFFFFFL);
        Random rng = new Random(lastSeed);
        float[] cond = encodeText(prompt);
        float[] uncond = cfg > 1.0f ? encodeText(negative == null ? "" : negative) : null;

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
            float[] eps = unet(latents, ts[s], cond);
            if (uncond != null) {
                float[] epsU = unet(latents, ts[s], uncond);
                for (int i = 0; i < eps.length; i++) eps[i] = epsU[i] + cfg * (eps[i] - epsU[i]);
            }
            scheduler.step(eps, ts[s], s + 1 < ts.length ? ts[s + 1] : -1, latents, denoised, rng);
            if (listener != null) listener.onPreview(preview(denoised), LATENT, LATENT);
        }
        if (listener != null) listener.onStep(ts.length, ts.length);
        return decode(denoised);
    }

    private float[] encodeText(String text) throws OrtException {
        int[] ids = tokenizer.encode(text);
        try (OnnxTensor in = OnnxTensor.createTensor(env, IntBuffer.wrap(ids), new long[]{1, ClipTokenizer.LENGTH});
             OrtSession.Result r = textEncoder.run(single("input_ids", in))) {
            return floats(r);
        }
    }

    private float[] unet(float[] sample, int t, float[] hidden) throws OrtException {
        Map<String, OnnxTensor> in = new HashMap<>();
        try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), new long[]{1, CHANNELS, LATENT, LATENT});
             OnnxTensor ts = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{t}), new long[]{1});
             OnnxTensor h = OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), new long[]{1, ClipTokenizer.LENGTH, HIDDEN})) {
            in.put("sample", x);
            in.put("timestep", ts);
            in.put("encoder_hidden_states", h);
            try (OrtSession.Result r = unet.run(in)) {
                return floats(r);
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
             OrtSession.Result r = encoder.run(single("image", in))) {
            return floats(r);
        }
    }

    private byte[] decode(float[] latents) throws OrtException {
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(latents), new long[]{1, CHANNELS, LATENT, LATENT});
             OrtSession.Result r = decoder.run(single("latents", in))) {
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
        for (OrtSession s : new OrtSession[]{textEncoder, unet, decoder, encoder}) closeQuietly(s);
    }
}
