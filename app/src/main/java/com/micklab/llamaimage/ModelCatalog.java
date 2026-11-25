package com.micklab.llamaimage;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Built-in list of downloadable models. The app keeps the download URL, exact size and SHA-256
 * of every file, downloads a model the first time it is selected, and verifies it before use.
 *
 * All entries are single-file checkpoints (GGUF or safetensors) that the pinned stable-diffusion.cpp
 * loads directly. Files re-uploaded by third parties were checked against the SHA-256 the original
 * author published on Civitai; HF URLs of re-uploads are pinned to a commit. All share the SD1.x/2.x
 * latent space (same TAESD, same preview coefficients). Sizes and hashes are the HF LFS metadata
 * of the referenced revision, or the Civitai file metadata for Civitai downloads.
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

        /**
         * Can run on the NPU backend: an SD1.5 single-file .safetensors whose tensors fit the NPU
         * templates (checked against every template map key + shape). The NPU path converts it on
         * the device; LCM models sample with LCM, regular ones with Euler a / Euler / DPM++ 2M + CFG.
         */
        public boolean npuCapable() {
            return NPU_IDS.contains(id);
        }

        /** Downloadable from the built-in URL (false for a file adopted from an older version). */
        public boolean downloadable() {
            return url != null;
        }
    }

    private static final String HF = "https://huggingface.co/";

    /**
     * Lykon's official DreamShaper 8 LCM single file (fp16 safetensors) — the checkpoint the NPU
     * templates were made from; also runs on CPU/GPU (sd.cpp loads safetensors).
     */
    public static final ModelSpec DREAMSHAPER8_LCM_ST = new ModelSpec(
            "dreamshaper8_lcm_st",
            "DreamShaper 8 LCM（推奨・NPU 対応）",
            "高品質・高速な SD1.5 系 LCM モデル（公式 fp16 版）。CPU／GPU に加え NPU でも動作。"
                    + "4〜8 step / CFG 1。",
            HF + "Lykon/dreamshaper-8-lcm/resolve/4645d8bc6a8e6b106d21606d63e8460cdad4f1a6/DreamShaper8_LCM.safetensors",
            "DreamShaper8_LCM.safetensors",
            2133804992L,
            "a4f3e1526c5dc4fcbe342f5c410d83ae202c7a415fcefcbb92e0f93fcd0a87c3",
            StableDiffusionNative.SAMPLE_LCM, 6, 1.0f);

    public static final ModelSpec DREAMSHAPER8_LCM = new ModelSpec(
            "dreamshaper8_lcm",
            "DreamShaper 8 LCM（Q8・軽量）",
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

    private static final String CIVITAI = "https://civitai.com/api/download/models/";

    // --- photoreal SD1.5 models. LCM versions are fast (few steps, CFG 1) like DreamShaper LCM;
    // the regular versions need ~20 steps and CFG>1, i.e. several minutes per image on a CPU. ---

    /** Lykon's official LCM version of AbsoluteReality (Civitai only; anonymous download works). */
    public static final ModelSpec ABSOLUTE_REALITY_LCM = new ModelSpec(
            "absolutereality_lcm",
            "AbsoluteReality LCM（写実・高速）",
            "DreamShaper の作者 Lykon による写実特化モデルの公式 LCM 版。肌や髪の質感が得意。"
                    + "4〜8 step / CFG 1〜2、LCM サンプラー専用。NPU 対応。配布元: Civitai。",
            CIVITAI + "256668?fileId=199166",
            "AbsoluteReality_LCM.safetensors",
            2133804992L,
            "0501f9c30cf36a777f0e3722e497b8cc5e8d9fde839e2459b80621e02f8ed98d",
            StableDiffusionNative.SAMPLE_LCM, 6, 1.0f);

    /**
     * Community LCM merge of Photon (Civitai model 306814). Civitai requires a login for this
     * file, so this is an HF re-upload whose SHA-256 equals the one published on Civitai.
     */
    public static final ModelSpec PHOTON_LCM = new ModelSpec(
            "photon_lcm",
            "Photon LCM（写実・高速）",
            "写真表現に強い Photon を LCM 化したモデル（コミュニティ版）。4〜6 step / CFG 1 で仕上がる。"
                    + "NPU 対応。配布元: Hugging Face（Civitai 版と同一ファイル）。",
            HF + "moonshotmillion/Photon_LCM_1.5/resolve/965ff0a137f5baeb46bab41656f52ab1375da6cf/"
                    + "photonLCM_v10.safetensors",
            "Photon_LCM.safetensors",
            2133817176L,
            "f2d45bbd0f7d28b7ff5e33cf5f7dca82ccf7a14fa55a867ec4dc7f701cb2859f",
            StableDiffusionNative.SAMPLE_LCM, 6, 1.0f);

    public static final ModelSpec ABSOLUTE_REALITY = new ModelSpec(
            "absolutereality_181",
            "AbsoluteReality 1.8.1（写実・低速）",
            "Lykon の写実特化モデル（公式リポジトリ）。高品質だが 20 step / CFG 7 が必要で、"
                    + "CPU では 1 枚に数分かかる。NPU 対応。",
            HF + "Lykon/AbsoluteReality/resolve/df7c3efce8f54d02cf73a21fcecd908a0fde2898/"
                    + "AbsoluteReality_1.8.1_pruned.safetensors",
            "AbsoluteReality_1.8.1_pruned.safetensors",
            2132625432L,
            "463d6a9fe8a4b56a4d69ef3692074c0617428dfd8e8f12f9efe3b1e9a71717ce",
            StableDiffusionNative.SAMPLE_DPMPP2M, 20, 7.0f);

    public static final ModelSpec PHOTON = new ModelSpec(
            "photon_v1",
            "Photon v1（写真・低速）",
            "プロンプト追従性の高いフォトリアル SD1.5 モデル。20 step / CFG 6 が必要で、CPU では 1 枚に数分かかる。NPU 対応。",
            HF + "sam749/Photon-v1/resolve/73bb8facb8576b69224315bfc77b14eb11b5591c/photon_v1.safetensors",
            "photon_v1.safetensors",
            2132625918L,
            "ec41bd2a8271acde4ae81cac004d9f3300e7fb0870eae8cbfe0bbc4ef8e27f91",
            StableDiffusionNative.SAMPLE_DPMPP2M, 20, 6.0f);

    public static final ModelSpec JUGGERNAUT_REBORN = new ModelSpec(
            "juggernaut_reborn",
            "Juggernaut Reborn（写実・低速）",
            "人物写真が安定する SD1.5 版 Juggernaut（KandooAI / RunDiffusion）。"
                    + "20 step / CFG 7 が必要で、CPU では 1 枚に数分かかる。NPU 対応。",
            HF + "HyperX-Sentience/Juggernaut-Reborn/resolve/7431787c76a081f7b5fb3bc24d780478bd1e6344/"
                    + "juggernaut_reborn.safetensors",
            "juggernaut_reborn.safetensors",
            2132653060L,
            "338b85bc4f7628bc42cce336242e79154a57c2a4360531436e97f7793568f18c",
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
            DREAMSHAPER8_LCM_ST, DREAMSHAPER8_LCM, ABSOLUTE_REALITY_LCM, PHOTON_LCM, SD_TURBO,
            ABSOLUTE_REALITY, PHOTON, JUGGERNAUT_REBORN, SD15_BASE));

    public static final ModelSpec DEFAULT_MODEL = DREAMSHAPER8_LCM_ST;

    /**
     * Models the NPU backend can convert (SD1.5 fp16 single-file safetensors; all template map keys
     * present with the DreamShaper shapes — verified from the safetensors headers 2026-11-25).
     */
    static final java.util.Set<String> NPU_IDS = new java.util.HashSet<>(Arrays.asList(
            "dreamshaper8_lcm_st", "absolutereality_lcm", "photon_lcm",
            "absolutereality_181", "photon_v1", "juggernaut_reborn"));

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
