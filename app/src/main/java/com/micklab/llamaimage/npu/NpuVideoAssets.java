package com.micklab.llamaimage.npu;

/**
 * Files for the experimental NPU video path (AnimateDiff-Lightning): weightless templates + weight
 * maps of the video UNet (8 frames x 256 px, vunet_0..n-1) and a TAESD decoder for that
 * latent size, published with GitHub Release npu-video-v6 (scripts/npu-video, workflow
 * "NPU video models"); the motion module itself comes from ByteDance's Hugging Face repo
 * (CreativeML OpenRAIL-M), pinned to a commit.
 */
public final class NpuVideoAssets {
    private NpuVideoAssets() {}

    public static final String BASE =
            "https://github.com/mmkurodan/llamaimage/releases/download/npu-video-v6/";

    /** {name, size, sha256} */
    static final Object[][] FILES = {
            {"vunet_0.map.json", 68683L, "d0d309f57df86f5c57b751d427071c78ea57ba620b569c3cd0846b2e85db4007"},
            {"vunet_0.template.onnx", 684108L, "0c85861cbb4fe783eb51b62fe2aee5b6f6c6f373dc12c9e29a4f6ad497e85170"},
            {"vunet_1.map.json", 26281L, "81af3facd6bba89afe74f913029c9a468ab06de8479dfa61c5f470f68f277f5c"},
            {"vunet_1.template.onnx", 344572L, "c6059bbc7d62d4f198862c48d88f6a704621401fcce334bc45e9a01cd2ad866a"},
            {"vunet_2.map.json", 22344L, "ea975bacc07513087459410e96d75538d7dd5d7b0a6d17e83b2d4aac5538a704"},
            {"vunet_2.template.onnx", 256068L, "fba78a2c9cc7da724b97e8db5755bca77dab5f5f8399af5c3e6bf8d412aec86f"},
            {"vunet_3.map.json", 15980L, "c913b67fd19bc94047a285c4458ec3791cf94559c095f0fb66d9b8f629dfc963"},
            {"vunet_3.template.onnx", 223881L, "42480441469d10d6640bcdafb6271620d89f1dc400b74616ce64fa1ae535cadd"},
            {"vunet_4.map.json", 21205L, "51c2e004cd608d5d89a63d3b46312ba0e28533426b08ef62cadef968172a7136"},
            {"vunet_4.template.onnx", 177824L, "fd7a39f4eadb671b42e01dd562acfd3599a215bb70d3fdeaebe1de4e8dcf7bb2"},
            {"vunet_5.map.json", 40130L, "40c020f1ad00d568b51bee041b33acc8e8f4de50c7d30127e3cc4e5711c1f5e4"},
            {"vunet_5.template.onnx", 407663L, "b4d3fcc5c917f7d180e7ab6c1071756da01e87aaf25ce38534b1e515096f882a"},
            {"vunet_6.map.json", 58758L, "e352306d7ddbf0272c59fead77c513b74c4ca982a1eb42ce595da7c57ac6c41a"},
            {"vunet_6.template.onnx", 618415L, "b4a9eeb2eb042070aa2dcdb8761b652696b9843d0494cc75a0143f981900d022"},
            {"taesd_decoder_video.onnx", 2468499L, "f8348237cea3977d601d64740a1ba01fbe703b0ce4615f3950c9b0c915480667"},
            {"video_parts.json", 609L, "8579ed0e7f2dc76b3a1770d489c2713a5e3b0064f2a2ba35dbd3634f0b4de065"},
    };

    static final String MOTION_NAME = "animatediff_lightning_4step_diffusers.safetensors";
    static final String MOTION_URL = "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/"
            + "027c893eec01df7330f5d4b733bc9485ee02e8b2/" + MOTION_NAME;
    static final long MOTION_SIZE = 907702248L;
    static final String MOTION_SHA256 = "8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17";

    /** Identifies the asset set; part of the "built" stamp so a new set rebuilds the video model. */
    static final String VERSION = "npu-video-v6/d0d309f5";
}
