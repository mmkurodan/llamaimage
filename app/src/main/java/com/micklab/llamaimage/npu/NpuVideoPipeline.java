package com.micklab.llamaimage.npu;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Text-to-video with AnimateDiff-Lightning (ByteDance, CreativeML OpenRAIL-M) on an SD1.5
 * checkpoint: the video UNet ({@link #PARTS}, scripts/npu-video) runs on the NPU, the text encoder
 * and the TAESD decoder (small, run once / per frame) on the ORT CPU provider.
 *
 * Static shapes come from the graph: sample [F, 4, h, w] (the frames are the batch), timestep [1],
 * encoder_hidden_states [1, 77, 768]. Sampler = diffusers' EulerDiscreteScheduler as Lightning
 * prescribes: linear betas 0.00085..0.012, "trailing" timesteps, epsilon prediction, CFG 1 (one
 * UNet pass per step, no negative prompt).
 */
public final class NpuVideoPipeline implements Closeable {
    /** Video UNet parts in execution order (chained by tensor name like NpuPipeline.UNET_PARTS). */
    public static final String[] PARTS = {"vunet_0", "vunet_1", "vunet_2"};
    private static final int TRAIN_STEPS = 1000;

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession[] unet = new OrtSession[PARTS.length];
    private final OrtSession textEncoder, decoder;
    private final ClipTokenizer tokenizer;
    public final int frames, latent, size;
    private final double[] sigmaTable = new double[TRAIN_STEPS];

    /**
     * @param baseDir  the image NPU model dir (text_encoder.onnx + tokenizer files)
     * @param videoDir vunet_*.onnx (+ .weights) and taesd_decoder_video.onnx
     */
    public NpuVideoPipeline(File baseDir, File videoDir, Map<String, String> qnnOptions, File cacheDir,
                            NpuPipeline.Log log) throws IOException {
        tokenizer = new ClipTokenizer(new File(baseDir, "tokenizer_vocab.json"), new File(baseDir, "tokenizer_merges.txt"));
        OrtSession te = null, dec = null;
        try {
            if (qnnOptions != null && cacheDir != null) {
                for (int i = 0; i < PARTS.length; i++) {          // pass 1: compile + cache, one at a time
                    if (NpuPipeline.contextFile(cacheDir, PARTS[i]).isFile()) continue;
                    log.stage(i + 1, PARTS.length, PARTS[i]);
                    unet[i] = NpuPipeline.compile(new File(videoDir, PARTS[i] + ".onnx"), qnnOptions, cacheDir, log);
                }
            }
            for (int i = 0; i < PARTS.length; i++) {              // pass 2: open from the cache
                File model = new File(videoDir, PARTS[i] + ".onnx");
                if (unet[i] == null) unet[i] = qnnOptions == null ? cpuSession(model)   // JVM test
                        : NpuPipeline.open(model, qnnOptions, cacheDir, log);
            }
            te = cpuSession(new File(baseDir, "text_encoder.onnx"));
            dec = cpuSession(new File(videoDir, "taesd_decoder_video.onnx"));
            long[] shape = ((ai.onnxruntime.TensorInfo) unet[0].getInputInfo().get("sample").getInfo()).getShape();
            frames = (int) shape[0];
            latent = (int) shape[2];
        } catch (OrtException | RuntimeException e) {
            for (OrtSession x : unet) NpuPipeline.closeQuietly(x);
            NpuPipeline.closeQuietly(te);
            NpuPipeline.closeQuietly(dec);
            throw new IOException("NPU（動画）セッションを作成できません: " + e.getMessage(), e);
        }
        textEncoder = te;
        decoder = dec;
        size = latent * 8;
        // linear betas (torch.linspace in float32, like diffusers), sigma_t = sqrt((1 - acp) / acp)
        double acp = 1.0;
        for (int t = 0; t < TRAIN_STEPS; t++) {
            float beta = 0.00085f + (0.012f - 0.00085f) * t / (TRAIN_STEPS - 1);
            acp *= 1.0 - beta;
            sigmaTable[t] = Math.sqrt((1.0 - acp) / acp);
        }
    }

    private OrtSession cpuSession(File model) throws OrtException {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        // The fp16 graphs: ORT-CPU has no float16 kernel for the fused com.microsoft.Gelu etc.
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
        return env.createSession(model.getAbsolutePath(), so);
    }

    /** Lightning "trailing" timesteps: round(arange(1000, 0, -1000/steps)) - 1, e.g. 999 749 499 249. */
    public static int[] timesteps(int steps) {
        int[] ts = new int[steps];
        for (int i = 0; i < steps; i++) ts[i] = (int) Math.round(TRAIN_STEPS - i * (double) TRAIN_STEPS / steps) - 1;
        return ts;
    }

    public interface Listener {
        void onStep(int step, int steps, long ms);

        boolean isCancelled();
    }

    /** Returns {@link #frames} RGB frames (size*size*3), or null if cancelled. */
    public List<byte[]> generate(String prompt, int steps, long seed, Listener listener) throws OrtException {
        float[] hidden;
        int[] ids = tokenizer.encode(prompt);
        try (OnnxTensor in = OnnxTensor.createTensor(env, IntBuffer.wrap(ids), new long[]{1, ClipTokenizer.LENGTH});
             OrtSession.Result r = NpuPipeline.run(textEncoder, "text_encoder", NpuPipeline.single("input_ids", in))) {
            hidden = NpuPipeline.floats(r);
        }
        int[] ts = timesteps(steps);
        double[] sig = new double[steps + 1];
        for (int i = 0; i < steps; i++) sig[i] = sigmaTable[ts[i]];
        sig[steps] = 0;
        int n = frames * 4 * latent * latent;
        float[] x = new float[n];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) x[i] = (float) (rng.nextGaussian() * sig[0]);   // init_noise_sigma = max sigma (trailing)
        float[] in = new float[n];
        for (int s = 0; s < steps; s++) {
            if (listener.isCancelled()) return null;
            long t0 = System.currentTimeMillis();
            float scale = (float) (1.0 / Math.sqrt(sig[s] * sig[s] + 1.0));
            for (int i = 0; i < n; i++) in[i] = x[i] * scale;
            float[] eps = unet(in, ts[s], hidden);
            float dt = (float) (sig[s + 1] - sig[s]);
            for (int i = 0; i < n; i++) x[i] += eps[i] * dt;     // Euler, epsilon: d = eps
            listener.onStep(s + 1, steps, System.currentTimeMillis() - t0);
        }
        List<byte[]> out = new ArrayList<>();
        int per = 4 * latent * latent;
        for (int f = 0; f < frames; f++) {
            if (listener.isCancelled()) return null;
            float[] one = new float[per];
            System.arraycopy(x, f * per, one, 0, per);
            out.add(decode(one));
        }
        return out;
    }

    private float[] unet(float[] sample, float t, float[] hidden) throws OrtException {
        Map<String, OnnxTensor> vals = new HashMap<>();
        List<OrtSession.Result> results = new ArrayList<>();
        try {
            vals.put("sample", OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), new long[]{frames, 4, latent, latent}));
            vals.put("timestep", OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{t}), new long[]{1}));
            vals.put("encoder_hidden_states", OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden),
                    new long[]{1, ClipTokenizer.LENGTH, NpuPipeline.HIDDEN}));
            for (int p = 0; p < unet.length; p++) {
                Map<String, OnnxTensor> in = new HashMap<>();
                for (String name : unet[p].getInputNames()) {
                    OnnxTensor v = vals.get(name);
                    if (v == null) throw new IllegalStateException(PARTS[p] + ": 入力 " + name + " がありません");
                    in.put(name, v);
                }
                OrtSession.Result r = NpuPipeline.run(unet[p], PARTS[p], in);
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
            for (String name : new String[]{"sample", "timestep", "encoder_hidden_states"}) {
                OnnxTensor v = vals.get(name);
                if (v != null) v.close();
            }
        }
    }

    private byte[] decode(float[] lat) throws OrtException {
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(lat), new long[]{1, 4, latent, latent});
             OrtSession.Result r = NpuPipeline.run(decoder, "taesd_decoder_video", NpuPipeline.single("latents", in))) {
            float[] chw = NpuPipeline.floats(r);
            int n = size * size;
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

    @Override
    public void close() {
        for (OrtSession s : unet) NpuPipeline.closeQuietly(s);
        NpuPipeline.closeQuietly(textEncoder);
        NpuPipeline.closeQuietly(decoder);
    }
}
