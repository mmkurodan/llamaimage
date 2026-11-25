package com.micklab.llamaimage.npu;

/**
 * Small files for on-device NPU conversion, published with GitHub Release npu-models-v2 of this
 * project (scripts/npu, workflow "NPU models (ONNX export)"): weightless ONNX templates + weight
 * maps for the SD1.5 UNet (3 parts, NpuPipeline.UNET_PARTS) / text encoder, TAESD (MIT), and the CLIP tokenizer. ~5.6 MB in total.
 */
public final class NpuAssets {
    private NpuAssets() {}

    public static final String BASE =
            "https://github.com/mmkurodan/llamaimage/releases/download/npu-models-v2/";

    /** {name, size, sha256} */
    static final Object[][] FILES = {
            {"unet_0.template.onnx", 318416L, "7429a37aba5ab38777961aa4c9f7ddd85b3507352593106f283d680b929bdef9"},
            {"unet_0.map.json", 60507L, "ca9f35bcacc249ad1c3bf4972d9810b22f61b11b784c969d12f707876965e86c"},
            {"unet_1.template.onnx", 25648L, "772d14f70382eb0a3c5a576a6a3afaa890076a6f28e19eb46604f9ef090c8f3e"},
            {"unet_1.map.json", 7495L, "76d6277ef04a4f8e5e9ee1b0635466fa3704bb6a673208c2546af24f8fbf3e7f"},
            {"unet_2.template.onnx", 374991L, "307c42761aa9acacb56b1715af161695c2f467ec31e3773d36d9c853dc058335"},
            {"unet_2.map.json", 71842L, "1ac4f8827fa8520e76109336eeac10099da505fa1b2283fe8133196775fd0476"},
            {"text_encoder.template.onnx", 146896L, "ce6f3c0bdc0a72b90e009ff0270a7d8adc6005e2f97313ac7f15f904fb703e78"},
            {"text_encoder.map.json", 41643L, "f4a79989a265dfb3bb0b7c99ec677be2d6d2aeca3b4b45179b3b3ce50c6abbf6"},
            {"taesd_decoder.onnx", 2468448L, "96c8e10a043d6ef949f16dd2d44ef9fa94b3a0b89e10671b093bee6e92e5cbfe"},
            {"taesd_encoder.onnx", 2466312L, "2a7a6b97c38b48a8b5d6c3b664e9f44e9e689b96e9b33c5563921fac98a68687"},
            {"tokenizer_vocab.json", 1059962L, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"},
            {"tokenizer_merges.txt", 524619L, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"},
    };

    /** Identifies this asset set; part of the "built" stamp so a new set rebuilds the model. */
    static final String VERSION = "npu-models-v2/ca9f35bc";
}
