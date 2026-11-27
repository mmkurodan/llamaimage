package com.micklab.llamaimage.npu;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.List;

/** args: videoTemplates baseAssets checkpoint motion outDir pyAsmDir taesdVideo prompt */
public class NpuVideoJavaTest {
    public static void main(String[] a) throws Exception {
        File vt = new File(a[0]), ba = new File(a[1]), ck = new File(a[2]), mo = new File(a[3]), out = new File(a[4]);
        File base = new File(out, "base"), video = new File(out, "video");
        long t0 = System.currentTimeMillis();
        NpuModelAssembler.assemble(ba, "text_encoder", ck, base, null);
        for (String f : new String[]{"tokenizer_vocab.json", "tokenizer_merges.txt"}) IoUtil.copy(new File(ba, f), new File(base, f));
        for (String p : NpuVideoPipeline.PARTS) {
            NpuModelAssembler.assemble(vt, p, ck, mo, video, null);
            String j = sha(new File(video, p + ".weights")), py = sha(new File(a[5], p + ".weights"));
            System.out.println("[java] " + p + ".weights java=" + j.substring(0, 16) + " python=" + py.substring(0, 16) + (j.equals(py) ? " SAME" : " DIFF"));
            if (!j.equals(py)) throw new IllegalStateException("assembly differs");
        }
        IoUtil.copy(new File(a[6]), new File(video, "taesd_decoder_video.onnx"));
        System.out.println("[java] assembled in " + (System.currentTimeMillis() - t0) / 1000 + "s; timesteps "
                + java.util.Arrays.toString(NpuVideoPipeline.timesteps(4)));
        try (NpuVideoPipeline vp = new NpuVideoPipeline(base, video, null, null, msg -> System.out.println(msg))) {
            long t1 = System.currentTimeMillis();
            List<byte[]> fr = vp.generate(a[7], 4, 42, new NpuVideoPipeline.Listener() {
                public void onStep(int s, int n, long ms) { System.out.println("[java] step " + s + "/" + n + " " + ms + "ms"); }
                public boolean isCancelled() { return false; }
            });
            System.out.println("[java] " + fr.size() + " frames " + vp.size + "px in " + (System.currentTimeMillis() - t1) / 1000 + "s");
            int w = vp.size;
            BufferedImage sheet = new BufferedImage(w * fr.size(), w, BufferedImage.TYPE_INT_RGB);
            for (int k = 0; k < fr.size(); k++) {
                byte[] rgb = fr.get(k);
                for (int i = 0; i < w * w; i++) {
                    int r = rgb[i * 3] & 255, g = rgb[i * 3 + 1] & 255, b = rgb[i * 3 + 2] & 255;
                    sheet.setRGB(k * w + i % w, i / w, (r << 16) | (g << 8) | b);
                }
            }
            javax.imageio.ImageIO.write(sheet, "png", new File(out, "java_sheet.png"));
        }
    }

    static String sha(File f) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(f)) {
            byte[] b = new byte[1 << 16];
            int n;
            while ((n = in.read(b)) > 0) md.update(b, 0, n);
        }
        StringBuilder sb = new StringBuilder();
        for (byte x : md.digest()) sb.append(String.format("%02x", x));
        return sb.toString();
    }
}
