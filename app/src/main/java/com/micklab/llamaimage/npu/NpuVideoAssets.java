package com.micklab.llamaimage.npu;

/**
 * Files for the experimental NPU video path (AnimateDiff-Lightning): weightless templates + weight
 * maps of the video UNet (8 frames x 256 px, NpuVideoPipeline.PARTS) and a TAESD decoder for that
 * latent size, published with GitHub Release npu-video-v1 (scripts/npu-video, workflow
 * "NPU video models"); the motion module itself comes from ByteDance's Hugging Face repo
 * (CreativeML OpenRAIL-M), pinned to a commit.
 */
public final class NpuVideoAssets {
    private NpuVideoAssets() {}

    public static final String BASE =
            "https://github.com/mmkurodan/llamaimage/releases/download/npu-video-v1/";

    /** {name, size, sha256} */
    static final Object[][] FILES = {
            // FILLED FROM THE CI RELEASE (SHA256SUMS)
    };

    static final String MOTION_NAME = "animatediff_lightning_4step_diffusers.safetensors";
    static final String MOTION_URL = "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/"
            + "027c893eec01df7330f5d4b733bc9485ee02e8b2/" + MOTION_NAME;
    static final long MOTION_SIZE = 907702248L;
    static final String MOTION_SHA256 = "8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17";

    /** Identifies the asset set; part of the "built" stamp so a new set rebuilds the video model. */
    static final String VERSION = "npu-video-v1";
}
