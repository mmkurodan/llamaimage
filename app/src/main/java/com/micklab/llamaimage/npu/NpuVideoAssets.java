package com.micklab.llamaimage.npu;

/**
 * Files for the experimental NPU video path (AnimateDiff-Lightning): weightless templates + weight
 * maps of the video UNet (8 frames x 256 px, vunet_0..n-1) and a TAESD decoder for that
 * latent size, published with GitHub Release npu-video-v2 (scripts/npu-video, workflow
 * "NPU video models"); the motion module itself comes from ByteDance's Hugging Face repo
 * (CreativeML OpenRAIL-M), pinned to a commit.
 */
public final class NpuVideoAssets {
    private NpuVideoAssets() {}

    public static final String BASE =
            "https://github.com/mmkurodan/llamaimage/releases/download/npu-video-v2/";

    /** {name, size, sha256} */
    static final Object[][] FILES = {
            {"vunet_0.template.onnx", 636876L, "9bf4c2db62a5a16f584f0e85c7c1716e027e394ddacfc344311283531f91e2c6"},
            {"vunet_0.map.json", 109467L, "d2f31e5a39cd16169bf6a79d263c879ae4d31307f282ce94412754a719d6c7ea"},
            {"vunet_1.template.onnx", 138044L, "fd0729979ff2a57cd7dd528799e375fd62b2ac4174957c8bb21320896b739de7"},
            {"vunet_1.map.json", 23735L, "29675fae2c881b4fa2da4b819692b25211616f44e7f46fcee2cece78a9663ffa"},
            {"vunet_2.template.onnx", 636475L, "0513e25de1c4df62df65fbea86faaf77d087d00c4f81d121de9893c1d2e0efd3"},
            {"vunet_2.map.json", 120185L, "a36835ff9eea765fd3207c13c9d2b8c502d9121326534341a56b8e46ee429cde"},
            {"taesd_decoder_video.onnx", 2468499L, "f8348237cea3977d601d64740a1ba01fbe703b0ce4615f3950c9b0c915480667"},
    };

    static final String MOTION_NAME = "animatediff_lightning_4step_diffusers.safetensors";
    static final String MOTION_URL = "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/"
            + "027c893eec01df7330f5d4b733bc9485ee02e8b2/" + MOTION_NAME;
    static final long MOTION_SIZE = 907702248L;
    static final String MOTION_SHA256 = "8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17";

    /** Identifies the asset set; part of the "built" stamp so a new set rebuilds the video model. */
    static final String VERSION = "npu-video-v1/d2f31e5a";
}
