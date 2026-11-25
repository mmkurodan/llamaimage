package com.micklab.llamaimage.npu;

import java.util.Random;

/**
 * diffusers LCMScheduler for SD1.5 (scaled_linear betas 0.00085..0.012, 1000 train steps,
 * original_inference_steps 50, epsilon prediction, timestep_scaling 10, set_alpha_to_one false).
 * Verified against scripts/npu/reference_pipeline.py: 6 steps -> [999, 839, 679, 499, 339, 179],
 * strength 0.55 -> [539, 459, 359, 279, 179, 99].
 */
public final class LcmScheduler {
    private static final int TRAIN_STEPS = 1000, ORIGINAL_STEPS = 50;
    private static final double SIGMA_DATA = 0.5, TIMESTEP_SCALING = 10.0;

    private final double[] alphasCumprod = new double[TRAIN_STEPS];

    public LcmScheduler() {
        double bs = Math.sqrt(0.00085), be = Math.sqrt(0.012), prod = 1.0;
        for (int i = 0; i < TRAIN_STEPS; i++) {
            double b = bs + (be - bs) * i / (TRAIN_STEPS - 1);
            prod *= 1.0 - b * b;
            alphasCumprod[i] = prod;
        }
    }

    /** Timesteps for {@code steps} LCM steps; strength < 1 starts part-way (img2img). */
    public int[] timesteps(int steps, float strength) {
        int k = TRAIN_STEPS / ORIGINAL_STEPS;
        int n = (int) (ORIGINAL_STEPS * strength);            // python int() truncation
        n = Math.max(1, n);
        steps = Math.max(1, Math.min(steps, n));
        int[] origin = new int[n];                            // reversed lcm_origin_timesteps
        for (int i = 0; i < n; i++) origin[i] = (n - i) * k - 1;
        int[] ts = new int[steps];
        for (int i = 0; i < steps; i++) ts[i] = origin[(int) Math.floor((double) i * n / steps)];
        return ts;
    }

    /** q(x_t | x_0): sqrt(a_t) * x0 + sqrt(1 - a_t) * noise (in place into x0). */
    public void addNoise(float[] x0, float[] noise, int t) {
        double a = alphasCumprod[t];
        float sa = (float) Math.sqrt(a), sb = (float) Math.sqrt(1 - a);
        for (int i = 0; i < x0.length; i++) x0[i] = sa * x0[i] + sb * noise[i];
    }

    /**
     * One LCM step. {@code sample} is replaced by x_{prev}; {@code denoisedOut} receives the
     * consistency-model estimate of x_0. {@code prevT} &lt; 0 marks the last step (no re-noise).
     */
    public void step(float[] eps, int t, int prevT, float[] sample, float[] denoisedOut, Random rng) {
        double aT = alphasCumprod[t];
        double aPrev = prevT >= 0 ? alphasCumprod[prevT] : alphasCumprod[0];
        double scaled = t * TIMESTEP_SCALING;
        double cSkip = SIGMA_DATA * SIGMA_DATA / (scaled * scaled + SIGMA_DATA * SIGMA_DATA);
        double cOut = scaled / Math.sqrt(scaled * scaled + SIGMA_DATA * SIGMA_DATA);
        double sqA = Math.sqrt(aT), sqB = Math.sqrt(1 - aT);
        for (int i = 0; i < sample.length; i++) {
            double x0 = (sample[i] - sqB * eps[i]) / sqA;
            denoisedOut[i] = (float) (cOut * x0 + cSkip * sample[i]);
        }
        if (prevT < 0) {
            System.arraycopy(denoisedOut, 0, sample, 0, sample.length);
            return;
        }
        float sa = (float) Math.sqrt(aPrev), sb = (float) Math.sqrt(1 - aPrev);
        for (int i = 0; i < sample.length; i++) sample[i] = sa * denoisedOut[i] + sb * (float) rng.nextGaussian();
    }
}
