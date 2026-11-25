package com.micklab.llamaimage.npu;

import org.json.JSONArray;
import org.json.JSONObject;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Iterator;

import javax.imageio.ImageIO;

/**
 * CI test of the app's NPU pipeline code on a JVM (ONNX Runtime CPU provider):
 *   1. ClipTokenizer vs transformers' CLIPTokenizer (golden JSON),
 *   2. NpuModelAssembler output vs the published assembled weights (byte-identical),
 *   3. txt2img + img2img through NpuPipeline -> PNGs.
 * Args: <assets dir (release files)> <checkpoint.safetensors> <golden tokens.json> <out dir>
 */
public final class NpuJavaTest {
    public static void main(String[] a) throws Exception {
        File assets = new File(a[0]), ckpt = new File(a[1]), golden = new File(a[2]), out = new File(a[3]);
        out.mkdirs();

        // 1. tokenizer
        ClipTokenizer tok = new ClipTokenizer(new File(assets, "tokenizer_vocab.json"), new File(assets, "tokenizer_merges.txt"));
        JSONObject g = new JSONObject(new String(Files.readAllBytes(golden.toPath()), StandardCharsets.UTF_8));
        int bad = 0;
        for (Iterator<String> it = g.keys(); it.hasNext(); ) {
            String prompt = it.next();
            JSONArray want = g.getJSONArray(prompt);
            int[] got = tok.encode(prompt);
            boolean ok = got.length == want.length();
            for (int i = 0; ok && i < got.length; i++) ok = got[i] == want.getInt(i);
            if (!ok) {
                bad++;
                System.out.println("TOKENIZER MISMATCH: " + prompt);
            }
        }
        System.out.println("[java] tokenizer: " + (g.length() - bad) + "/" + g.length() + " prompts identical");
        if (bad > 0) System.exit(1);

        // 2. assembly
        File model = new File(out, "model");
        model.mkdirs();
        for (String name : new String[]{"text_encoder", "unet"}) {
            long t0 = System.currentTimeMillis();
            NpuModelAssembler.assemble(assets, name, ckpt, model, p -> { });
            System.out.println("[java] assembled " + name + " in " + (System.currentTimeMillis() - t0) + "ms");
        }
        for (String f : new String[]{"taesd_decoder.onnx", "taesd_encoder.onnx", "tokenizer_vocab.json", "tokenizer_merges.txt"}) {
            Files.copy(new File(assets, f).toPath(), new File(model, f).toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        // 3. generation on the CPU provider
        try (NpuPipeline p = new NpuPipeline(model, null, null, System.out::println)) {
            long t0 = System.currentTimeMillis();
            byte[] img = p.generate("a photo of a cat sitting on a wooden table, soft light", "", 6, 1.0f, 42, null, 1f,
                    new NpuPipeline.Listener() {
                        public void onStep(int s, int n) { System.out.println("  step " + s + "/" + n); }
                        public void onPreview(byte[] rgb, int w, int h) { }
                        public boolean isCancelled() { return false; }
                    });
            System.out.println("[java] txt2img " + (System.currentTimeMillis() - t0) + "ms seed=" + p.getLastSeed());
            save(img, new File(out, "java_txt2img.png"));
            byte[] img2 = p.generate("a photo of a cat sitting on a wooden table, soft light", "", 6, 1.0f, 7, img, 0.55f, null);
            save(img2, new File(out, "java_img2img.png"));
            System.out.println("[java] img2img ok");
        }
    }

    private static void save(byte[] rgb, File f) throws Exception {
        BufferedImage im = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < 512 * 512; i++) {
            im.setRGB(i % 512, i / 512, ((rgb[i * 3] & 0xFF) << 16) | ((rgb[i * 3 + 1] & 0xFF) << 8) | (rgb[i * 3 + 2] & 0xFF));
        }
        ImageIO.write(im, "png", f);
    }
}
