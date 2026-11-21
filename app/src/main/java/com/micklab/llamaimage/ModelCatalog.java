package com.micklab.llamaimage;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Built-in list of downloadable models. The app keeps the download URL, exact size and SHA-256
 * of every file, downloads a model the first time it is selected, and verifies it before use.
 *
 * All entries are single-file checkpoints that the pinned stable-diffusion.cpp loads directly,
 * and all share the SD1.x/2.x latent space (same TAESD, same preview coefficients). Sizes and
 * hashes are the Hugging Face LFS metadata of the referenced revision.
 */
public final class ModelCatalog {

    private ModelCatalog() {}

    /** One downloadable file plus the generation settings that suit it. */
    public static final class ModelSpec {
        public final String id;
        public final String title;
        public final String description;
        public final String url;
        public final String fileName;
        public final long sizeBytes;
        public final String sha256;
        // Recommended generation settings, applied when the model is selected.
        public final int sampler;
        public final int steps;
        public final float cfgScale;

        ModelSpec(String id, String title, String description, String url, String fileName,
                  long sizeBytes, String sha256, int sampler, int steps, float cfgScale) {
            this.id = id;
            this.title = title;
            this.description = description;
            this.url = url;
            this.fileName = fileName;
            this.sizeBytes = sizeBytes;
            this.sha256 = sha256;
            this.sampler = sampler;
            this.steps = steps;
            this.cfgScale = cfgScale;
        }

        public String sizeLabel() {
            return formatSize(sizeBytes);
        }

        /** True if selecting this model should apply {@link #sampler}/{@link #steps}/{@link #cfgScale}. */
        public boolean hasRecommendedSettings() {
            return sampler >= 0 && steps > 0;
        }

        public String recommendedLabel() {
            if (!hasRecommendedSettings()) return "現在の生成設定を使用";
            return MainActivity.samplerLabel(sampler) + " / " + steps + "step / CFG " + cfgScale;
        }

        /** Downloadable from the built-in URL (false for a file adopted from an older version). */
        public boolean downloadable() {
            return url != null;
        }
    }

    private static final String HF = "https://huggingface.co/";

    public static final ModelSpec DREAMSHAPER8_LCM = new ModelSpec(
            "dreamshaper8_lcm",
            "DreamShaper 8 LCM（推奨）",
            "高品質・高速。LCM 蒸留済みで 4〜8 step / CFG 1 で仕上がる SD1.5 系モデル。",
            HF + "haven-ai-companion/dreamshaper8-lcm-gguf/resolve/main/DreamShaper8_LCM_q8_0.gguf",
            "DreamShaper8_LCM_q8_0.gguf",
            1802206336L,
            "dd06826d1eca711efa38190395a5ee949d0427fc6422c39433358a815e5b066c",
            StableDiffusionNative.SAMPLE_LCM, 6, 1.0f);

    public static final ModelSpec SD_TURBO = new ModelSpec(
            "sd_turbo",
            "SD-Turbo（最速）",
            "SD2.1 を 1〜4 step に蒸留したモデル。1 step で生成でき最速、細部は DreamShaper に劣る。",
            HF + "Green-Sky/SD-Turbo-GGUF/resolve/main/sd_turbo-f16-q8_0.gguf",
            "sd_turbo-f16-q8_0.gguf",
            2023745376L,
            "d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580",
            StableDiffusionNative.SAMPLE_EULER_A, 1, 1.0f);

    public static final ModelSpec SD15_BASE = new ModelSpec(
            "sd15_base",
            "Stable Diffusion 1.5（標準）",
            "素の SD1.5 (Q4_0)。汎用だが 20 step / CFG 7 が必要で低速。",
            HF + "second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
            "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
            1566768416L,
            "b8944e9fe0b69b36ae1b5bb0185b3a7b8ef14347fe0fa9af6c64c4829022261f",
            StableDiffusionNative.SAMPLE_DPMPP2M, 20, 7.0f);

    /** Tiny AutoEncoder for SD1.x/2.x: much faster, lower-memory final decode (slightly softer). */
    public static final ModelSpec TAESD = new ModelSpec(
            "taesd",
            "TAESD（高速VAEデコーダ）",
            "最終デコードを高速化・省メモリ化する小型 VAE。細部はわずかに甘くなる。",
            HF + "madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors",
            "taesd.safetensors",
            9793292L,
            "db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549",
            0, 0, 0f);

    /** Selectable image-generation models, recommended first. */
    public static final List<ModelSpec> MODELS = Collections.unmodifiableList(Arrays.asList(
            DREAMSHAPER8_LCM, SD_TURBO, SD15_BASE));

    public static final ModelSpec DEFAULT_MODEL = DREAMSHAPER8_LCM;

    /** Id of the model adopted from an older version that matches no catalog entry. */
    public static final String IMPORTED_ID = "imported";

    /** A model the user copied in with an older version of the app (no URL / hash). */
    static ModelSpec imported(String fileName, long size, String title) {
        return new ModelSpec(IMPORTED_ID, title,
                "以前のバージョンでファイル選択から取り込んだモデル。生成設定は現在の値のまま使います。",
                null, fileName, size, null, -1, 0, 0f);
    }

    public static ModelSpec findModel(String id) {
        if (id == null) return null;
        for (ModelSpec m : MODELS) {
            if (m.id.equals(id)) return m;
        }
        return null;
    }

    /** Catalog model with exactly this file size (used to adopt a previously copied model). */
    public static ModelSpec findModelBySize(long size) {
        for (ModelSpec m : MODELS) {
            if (m.sizeBytes == size) return m;
        }
        return null;
    }

    public static String formatSize(long bytes) {
        if (bytes >= 1000L * 1000 * 1000) {
            return String.format(java.util.Locale.US, "%.2f GB", bytes / 1e9);
        }
        return String.format(java.util.Locale.US, "%.1f MB", bytes / 1e6);
    }
}
