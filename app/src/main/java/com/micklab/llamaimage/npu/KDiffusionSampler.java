package com.micklab.llamaimage.npu;

import java.util.Random;

/**
 * k-diffusion samplers for regular (non-LCM) SD1.5 checkpoints on the NPU path, written to match the
 * CPU engine (stable-diffusion.cpp denoiser.hpp: CompVisDenoiser + DiscreteSchedule + EULER_A /
 * EULER / DPMPP2M, img2img start = sd.cpp's t_enc rule), so the same settings mean the same thing
 * on CPU and NPU. eps-prediction: D(x; sigma) = x - sigma * eps(x / sqrt(sigma^2 + 1), t(sigma)).
 */
public final class KDiffusionSampler {
    /** Sampler ids; same values as StableDiffusionNative.SAMPLE_*. */
    public static final int EULER_A = 0, EULER = 1, DPMPP2M = 5, LCM = 9;

    private static final int TIMESTEPS = 1000;

    /** Denoiser D(x; sigma) -> x0 estimate (the pipeline runs the UNet + CFG behind it). */
    public interface Denoiser {
        float[] denoise(float[] x, float sigma, float t) throws Exception;
    }

    /** Called after each step with the current x0 estimate; return false to cancel. */
    public interface StepCallback {
        boolean onStep(int step, int total, float[] denoised);
    }

    private final float[] logSigmas = new float[TIMESTEPS];

    public KDiffusionSampler() {
        // scaled_linear betas 0.00085..0.012 (same alphas_cumprod as LcmScheduler / sd.cpp).
        double bs = Math.sqrt(0.00085), be = Math.sqrt(0.012), prod = 1.0;
        for (int i = 0; i < TIMESTEPS; i++) {
            double b = bs + (be - bs) * i / (TIMESTEPS - 1);
            prod *= 1.0 - b * b;
            logSigmas[i] = (float) Math.log(Math.sqrt((1 - prod) / prod));
        }
    }

    public static boolean supports(int sampler) {
        return sampler == EULER_A || sampler == EULER || sampler == DPMPP2M;
    }

    public float tToSigma(float t) {
        int lo = (int) Math.floor(t), hi = (int) Math.ceil(t);
        float w = t - lo;
        return (float) Math.exp((1 - w) * logSigmas[lo] + w * logSigmas[hi]);
    }

    public float sigmaToT(float sigma) {
        float ls = (float) Math.log(sigma);
        int lo = 0;
        for (int i = 0; i < TIMESTEPS; i++) if (ls - logSigmas[i] >= 0) lo++;
        lo = Math.min(Math.max(lo - 1, 0), TIMESTEPS - 2);
        int hi = lo + 1;
        float l = logSigmas[lo], h = logSigmas[hi];
        float w = Math.max(0f, Math.min(1f, (l - ls) / (l - h)));
        return (1 - w) * lo + w * hi;
    }

    /** DiscreteSchedule: n sigmas from t = 999 down to 0 (linear in t), then 0. */
    public float[] sigmas(int n) {
        n = Math.max(1, n);
        float[] s = new float[n + 1];
        if (n == 1) {
            s[0] = tToSigma(TIMESTEPS - 1);
            return s;
        }
        float step = (float) (TIMESTEPS - 1) / (n - 1);
        for (int i = 0; i < n; i++) s[i] = tToSigma(TIMESTEPS - 1 - step * i);
        return s;
    }

    /**
     * Sigmas actually run: all of them for txt2img; for img2img (strength &lt; 1) the tail starting
     * t_enc + 1 entries before the end, as sd.cpp does (t_enc = (int)(steps * strength), &lt; steps).
     */
    public float[] schedule(int steps, float strength, boolean img2img) {
        float[] all = sigmas(steps);
        if (!img2img) return all;
        steps = all.length - 1;
        int tEnc = (int) (steps * strength);
        if (tEnc >= steps) tEnc = steps - 1;
        int start = steps - tEnc - 1;
        float[] s = new float[all.length - start];
        System.arraycopy(all, start, s, 0, s.length);
        return s;
    }

    /**
     * Samples in place: {@code x} must already be noise * sigmas[0] (+ the init latent for img2img).
     * Returns false if cancelled. On success {@code x} is the final latent.
     */
    public boolean sample(int method, float[] x, float[] sigmas, Denoiser model, Random rng, StepCallback cb)
            throws Exception {
        int steps = sigmas.length - 1, n = x.length;
        float[] old = method == DPMPP2M ? new float[n] : null;
        for (int i = 0; i < steps; i++) {
            float s = sigmas[i], sn = sigmas[i + 1];
            float[] den = model.denoise(x, s, sigmaToT(s));
            switch (method) {
                case EULER: {
                    float dt = sn - s;
                    for (int j = 0; j < n; j++) x[j] += (x[j] - den[j]) / s * dt;
                    break;
                }
                case DPMPP2M: {
                    float a = sn / s;
                    if (i == 0 || sn == 0) {
                        // b = exp(-h) - 1 with h = log(s / sn); exp(-h) = sn / s.
                        float b = a - 1f;
                        for (int j = 0; j < n; j++) x[j] = a * x[j] - b * den[j];
                    } else {
                        double t = -Math.log(s), tn = -Math.log(sn), tp = -Math.log(sigmas[i - 1]);
                        double h = tn - t, r = (t - tp) / h;
                        float b = (float) (Math.exp(-h) - 1);
                        float c1 = (float) (1 + 1 / (2 * r)), c2 = (float) (1 / (2 * r));
                        for (int j = 0; j < n; j++) x[j] = a * x[j] - b * (c1 * den[j] - c2 * old[j]);
                    }
                    System.arraycopy(den, 0, old, 0, n);
                    break;
                }
                default: {   // EULER_A
                    float up = (float) Math.min(sn, Math.sqrt((double) sn * sn * (s * s - sn * sn) / (s * s)));
                    float down = (float) Math.sqrt(Math.max(0, (double) sn * sn - (double) up * up));
                    float dt = down - s;
                    for (int j = 0; j < n; j++) x[j] += (x[j] - den[j]) / s * dt;
                    if (sn > 0) for (int j = 0; j < n; j++) x[j] += (float) rng.nextGaussian() * up;
                    break;
                }
            }
            if (cb != null && !cb.onStep(i + 1, steps, den)) return false;
        }
        return true;
    }
}
