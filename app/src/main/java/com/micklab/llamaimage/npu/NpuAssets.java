package com.micklab.llamaimage.npu;

/**
 * Small files for on-device NPU conversion, published with GitHub Release npu-models-v1 of this
 * project (scripts/npu, workflow "NPU models (ONNX export)"): weightless ONNX templates + weight
 * maps for the SD1.5 UNet / text encoder, TAESD (MIT), and the CLIP tokenizer. ~5.6 MB in total.
 */
public final class NpuAssets {
    private NpuAssets() {}

    public static final String BASE =
            "https://github.com/mmkurodan/llamaimage/releases/download/npu-models-v1/";

    /** {name, size, sha256} */
    static final Object[][] FILES = {
            {"unet.template.onnx", 808531L, "8f1d85150afa8df85eaae2daf37da9f6697935f500df8ad45014778c1d474d10"},
            {"unet.map.json", 140165L, "12dd177eccb032be0a968fbbc8e12681756a0a4444c8563f1195fe3bd10ceba0"},
            {"text_encoder.template.onnx", 146896L, "ce6f3c0bdc0a72b90e009ff0270a7d8adc6005e2f97313ac7f15f904fb703e78"},
            {"text_encoder.map.json", 41643L, "f4a79989a265dfb3bb0b7c99ec677be2d6d2aeca3b4b45179b3b3ce50c6abbf6"},
            {"taesd_decoder.onnx", 2468448L, "96c8e10a043d6ef949f16dd2d44ef9fa94b3a0b89e10671b093bee6e92e5cbfe"},
            {"taesd_encoder.onnx", 2466312L, "2a7a6b97c38b48a8b5d6c3b664e9f44e9e689b96e9b33c5563921fac98a68687"},
            {"tokenizer_vocab.json", 1059962L, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"},
            {"tokenizer_merges.txt", 524619L, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"},
    };

    /** Identifies this asset set; part of the "built" stamp so a new set rebuilds the model. */
    static final String VERSION = "npu-models-v1/12dd177e";
}
