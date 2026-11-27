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
            {"vunet_0.map.json", 68683L, "ccfb1d8201f33255f6b5cc87574af3d2e9e16b835dbab1afe38132b5baf77c3e"},
            {"vunet_0.template.onnx", 340042L, "44dfec643c91d9cfa39ff327ad5048a3122fac085e51ad66aa9d54529be91f2a"},
            {"vunet_1.map.json", 26281L, "c0ce97c6bc30ead20010d22f0fef025cded68c4c7e1ff32f943a45c67b12232e"},
            {"vunet_1.template.onnx", 139593L, "4b5578a62259ef1fc4f5a366a9b9c7615193555ca75026b69aef82b53ded88ab"},
            {"vunet_2.map.json", 22344L, "493d988a78eae3061f17a0a3415b4eaa70a3eb5c256a22bd8c219b6523bdb038"},
            {"vunet_2.template.onnx", 118772L, "fca7f938824201afb333bdfa18133b1b8fe5b8bc26208158c12fb091ba48b8bc"},
            {"vunet_3.map.json", 15980L, "6a04362c49477a8b58ac9b4fa1b2f1fc1d89b46922fd37d2c8efb3769d1ca79d"},
            {"vunet_3.template.onnx", 88446L, "7bad3300a1fc04020eda2b8d661ad6e0b8382f3fffbb24fe9a058de76d1810f8"},
            {"vunet_4.map.json", 21205L, "fa68d0540b822947dc1da0e35c558226169e6ea6afe204fcd33387cf4c8cac4a"},
            {"vunet_4.template.onnx", 110070L, "3de6bcf99778ec10eb8042a8921b7711a0c58d27b095492e53318cd91c7f090a"},
            {"vunet_5.map.json", 40130L, "7130210498a39a9a63463f56cfca5c9af2685d074cd57c9e8b687fa78a223d2a"},
            {"vunet_5.template.onnx", 204553L, "7e86e94bccb817eabaf8db4e697b03d61385776ca1f66ccda41174b283dfc97f"},
            {"vunet_6.map.json", 58758L, "d7a73ae7ad1c125601cafee9cf950c02c4fd7793b9ba087ab9af29ee9a94940c"},
            {"vunet_6.template.onnx", 276212L, "23895ecf1143fe30f35e2a78471d7d2bc1c50263d4a523562c2b2e27b4a95af6"},
            {"taesd_decoder_video.onnx", 2468499L, "f8348237cea3977d601d64740a1ba01fbe703b0ce4615f3950c9b0c915480667"},
    };

    static final String MOTION_NAME = "animatediff_lightning_4step_diffusers.safetensors";
    static final String MOTION_URL = "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/"
            + "027c893eec01df7330f5d4b733bc9485ee02e8b2/" + MOTION_NAME;
    static final long MOTION_SIZE = 907702248L;
    static final String MOTION_SHA256 = "8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17";

    /** Identifies the asset set; part of the "built" stamp so a new set rebuilds the video model. */
    static final String VERSION = "npu-video-v2/ccfb1d82";
}
