package com.micklab.llamaimage.npu;

import android.content.Context;
import android.os.Build;
import android.system.Os;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Android side of the NPU backend: downloads the small conversion assets (NpuAssets), builds the
 * model for a checkpoint on the device (NpuModelAssembler), and opens NpuPipeline with the QNN
 * HTP execution provider. Directories: assets + built models under the app's external files dir
 * (large), the compiled QNN context cache under the internal files dir.
 */
public final class NpuManager {

    public interface Status {
        void onStatus(String text);
    }

    private final Context ctx;

    public NpuManager(Context ctx) {
        this.ctx = ctx.getApplicationContext();
    }

    private File root() {
        File ext = ctx.getExternalFilesDir("npu");
        return ext != null ? ext : new File(ctx.getFilesDir(), "npu");
    }

    private File assetDir() {
        return new File(root(), "assets");
    }

    public File modelDir(String modelId) {
        return new File(root(), "models/" + modelId);
    }

    private File cacheDir(String modelId) {
        return new File(ctx.getFilesDir(), "npu-ctx/" + modelId);
    }

    /** True if this device has a Snapdragon (Qualcomm) SoC — the only place the HTP EP can work. */
    public static boolean deviceMayHaveHtp() {
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MANUFACTURER : "";
        String hw = (Build.HARDWARE + " " + Build.BOARD + " " + soc).toLowerCase(Locale.ROOT);
        return hw.contains("qcom") || hw.contains("qualcomm") || soc.equalsIgnoreCase("QTI");
    }

    /** The model for {@code modelId} was fully built from the checkpoint with {@code ckptSha}. */
    public boolean isBuilt(String modelId, String ckptSha) {
        File stamp = new File(modelDir(modelId), ".built");
        try {
            return stamp.isFile() && IoUtil.readString(stamp).equals(NpuAssets.VERSION + " " + ckptSha);
        } catch (IOException e) {
            return false;
        }
    }

    /** Downloads (if needed) and verifies the conversion assets. Blocking. */
    public void ensureAssets(Status status) throws IOException {
        File dir = assetDir();
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("作成できません: " + dir);
        for (Object[] f : NpuAssets.FILES) {
            String name = (String) f[0];
            long size = (Long) f[1];
            String sha = (String) f[2];
            File dst = new File(dir, name);
            if (dst.isFile() && dst.length() == size && sha.equals(sha256(dst))) continue;
            status.onStatus("NPU 用ファイルを取得中… " + name);
            File tmp = new File(dir, name + ".tmp");
            download(NpuAssets.BASE + name, tmp);
            if (tmp.length() != size || !sha.equals(sha256(tmp))) {
                tmp.delete();
                throw new IOException("NPU 用ファイルの検証に失敗しました: " + name);
            }
            if (dst.exists()) dst.delete();
            if (!tmp.renameTo(dst)) throw new IOException("配置できません: " + dst);
        }
    }

    /**
     * Builds the NPU model for {@code modelId} from {@code checkpoint} (an SD1.5 .safetensors with
     * SHA-256 {@code ckptSha}) unless already built. ~2 GB written. Blocking.
     */
    public void ensureBuilt(String modelId, File checkpoint, String ckptSha, Status status) throws IOException {
        if (isBuilt(modelId, ckptSha)) return;
        File out = modelDir(modelId);
        File stamp = new File(out, ".built");
        stamp.delete();
        deleteTree(cacheDir(modelId));   // compiled contexts belong to the old weights
        File assets = assetDir();
        NpuModelAssembler.assemble(assets, "text_encoder", checkpoint, out,
                p -> status.onStatus("NPU 用に変換中（テキストエンコーダ） " + p + "%"));
        for (int i = 0; i < NpuPipeline.UNET_PARTS.length; i++) {
            String label = "UNet " + (i + 1) + "/" + NpuPipeline.UNET_PARTS.length;
            NpuModelAssembler.assemble(assets, NpuPipeline.UNET_PARTS[i], checkpoint, out,
                    p -> status.onStatus("NPU 用に変換中（" + label + "） " + p + "%"));
        }
        new File(out, "unet.onnx").delete();      // v1 single-file UNet, if this dir held one
        new File(out, "unet.weights").delete();
        for (String f : new String[]{"taesd_decoder.onnx", "taesd_encoder.onnx", "tokenizer_vocab.json",
                "tokenizer_merges.txt"}) {
            IoUtil.copy(new File(assets, f), new File(out, f));
        }
        IoUtil.writeString(stamp, NpuAssets.VERSION + " " + ckptSha);
    }

    /** Opens the pipeline on the HTP (compiles + caches the graphs on first use: minutes). */
    public NpuPipeline open(String modelId, NpuPipeline.Log log) throws IOException {
        return new NpuPipeline(modelDir(modelId), htpOptions(log), cacheDir(modelId), log);
    }

    private Map<String, String> htpOptions(NpuPipeline.Log log) {
        String libDir = ctx.getApplicationInfo().nativeLibraryDir;
        try {
            // FastRPC loads the HTP skel (libQnnHtpV**Skel.so, extracted from the APK) from here.
            Os.setenv("ADSP_LIBRARY_PATH", libDir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp", true);
        } catch (Exception e) {
            log.log("ADSP_LIBRARY_PATH を設定できません: " + e);
        }
        Map<String, String> qnn = new HashMap<>();
        qnn.put("backend_path", "libQnnHtp.so");
        qnn.put("htp_performance_mode", "burst");
        qnn.put("enable_htp_fp16_precision", "1");
        return qnn;
    }

    // ---- video (AnimateDiff-Lightning, experimental) -------------------------------------------

    private File videoAssetDir() {
        return new File(root(), "video-assets");
    }

    private File motionFile() {
        return new File(root(), "motion/" + NpuVideoAssets.MOTION_NAME);
    }

    private File videoDir(String modelId) {
        return new File(modelDir(modelId), "video");
    }

    private File videoCacheDir(String modelId) {
        return new File(cacheDir(modelId), "video");
    }

    /** Video templates (small) + the motion module (907 MB, Hugging Face). Blocking. */
    public void ensureVideoAssets(Status status) throws IOException {
        File dir = videoAssetDir();
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("作成できません: " + dir);
        java.util.Set<String> wanted = new java.util.HashSet<>();
        for (Object[] f : NpuVideoAssets.FILES) wanted.add((String) f[0]);
        File[] old = dir.listFiles();                      // parts of an older asset set
        if (old != null) for (File f : old) if (!wanted.contains(f.getName())) f.delete();
        for (Object[] f : NpuVideoAssets.FILES) {
            fetch(NpuVideoAssets.BASE + f[0], new File(dir, (String) f[0]), (Long) f[1], (String) f[2], status);
        }
        File m = motionFile();
        File md = m.getParentFile();
        if (md != null && !md.isDirectory() && !md.mkdirs()) throw new IOException("作成できません: " + md);
        fetch(NpuVideoAssets.MOTION_URL, m, NpuVideoAssets.MOTION_SIZE, NpuVideoAssets.MOTION_SHA256, status);
    }

    private void fetch(String url, File dst, long size, String sha, Status status) throws IOException {
        if (dst.isFile() && dst.length() == size && sha.equals(sha256(dst))) return;
        status.onStatus("動画用ファイルを取得中… " + dst.getName() + "（" + size / 1_000_000 + " MB）");
        File tmp = new File(dst.getParentFile(), dst.getName() + ".tmp");
        download(url, tmp, size, status, dst.getName());
        status.onStatus("検証中… " + dst.getName());
        if (tmp.length() != size || !sha.equals(sha256(tmp))) {
            tmp.delete();
            throw new IOException("ファイルの検証に失敗しました: " + dst.getName());
        }
        if (dst.exists()) dst.delete();
        if (!tmp.renameTo(dst)) throw new IOException("配置できません: " + dst);
    }

    /** Builds the video UNet for {@code modelId} (checkpoint + motion module, ~2.6 GB). Blocking. */
    public void ensureVideoBuilt(String modelId, File checkpoint, String ckptSha, Status status) throws IOException {
        File out = videoDir(modelId);
        File stamp = new File(out, ".built");
        String want = NpuVideoAssets.VERSION + " " + ckptSha;
        try {
            if (stamp.isFile() && IoUtil.readString(stamp).equals(want)) return;
        } catch (IOException ignore) {
        }
        deleteTree(out);                      // an older split may have had more parts
        deleteTree(videoCacheDir(modelId));
        File assets = videoAssetDir();
        String[] parts = NpuVideoPipeline.partsIn(assets, ".map.json");
        if (parts.length == 0) throw new IOException("動画用テンプレートがありません");
        for (int i = 0; i < parts.length; i++) {
            String label = "動画 UNet " + (i + 1) + "/" + parts.length;
            NpuModelAssembler.assemble(assets, parts[i], checkpoint, motionFile(), out,
                    p -> status.onStatus("NPU 用に変換中（" + label + "） " + p + "%"));
        }
        IoUtil.copy(new File(assets, "taesd_decoder_video.onnx"), new File(out, "taesd_decoder_video.onnx"));
        IoUtil.writeString(stamp, want);
    }

    /** Size of a built video UNet part's weights in MB (0 if not built). */
    public int videoPartMb(String modelId, String part) {
        return (int) (new File(videoDir(modelId), part + ".weights").length() / 1_000_000);
    }

    /** Opens the video pipeline (compiles + caches the 3 video UNet parts on first use). */
    public NpuVideoPipeline openVideo(String modelId, NpuPipeline.Log log, java.util.Set<String> failed)
            throws IOException {
        return new NpuVideoPipeline(modelDir(modelId), videoDir(modelId), htpOptions(log), videoCacheDir(modelId), log,
                failed);
    }

    /** "d0.mm0（5 MB）" for a video part, from the release's video_parts.json (null if unknown). */
    public String videoPartInfo(String part) {
        try {
            org.json.JSONArray a = new org.json.JSONObject(IoUtil.readString(new File(videoAssetDir(), "video_parts.json")))
                    .getJSONArray("parts");
            for (int i = 0; i < a.length(); i++) {
                org.json.JSONObject o = a.getJSONObject(i);
                if (part.equals(o.getString("name"))) return o.getString("layers") + "（" + o.getInt("mb") + " MB）";
            }
        } catch (Exception ignore) {
        }
        return null;
    }

    /** Removes only the compiled HTP contexts of {@code modelId}: the next open recompiles them. */
    public void deleteCompiled(String modelId) {
        deleteTree(cacheDir(modelId));
    }

    /** Removes the built model and compiled cache for {@code modelId} (e.g. when the model is deleted). */
    public void delete(String modelId) {
        deleteTree(modelDir(modelId));
        deleteTree(cacheDir(modelId));
    }

    private static void deleteTree(File f) {
        File[] kids = f.listFiles();
        if (kids != null) for (File k : kids) deleteTree(k);
        f.delete();
    }

    private static void download(String url, File dst) throws IOException {
        download(url, dst, 0, null, null);
    }

    private static void download(String url, File dst, long size, Status status, String label) throws IOException {
        for (int redirects = 0; redirects < 6; redirects++) {
            HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
            c.setConnectTimeout(20000);
            c.setReadTimeout(60000);
            c.setInstanceFollowRedirects(false);   // GitHub -> objects.githubusercontent.com (https)
            int code = c.getResponseCode();
            if (code >= 300 && code < 400) {
                String loc = c.getHeaderField("Location");
                c.disconnect();
                if (loc == null) throw new IOException("リダイレクト先がありません: " + url);
                url = new URL(new URL(url), loc).toString();
                continue;
            }
            if (code != 200) {
                c.disconnect();
                throw new IOException("ダウンロード失敗 (HTTP " + code + "): " + url);
            }
            try (InputStream in = c.getInputStream(); OutputStream out = new FileOutputStream(dst)) {
                byte[] buf = new byte[65536];
                int n;
                long got = 0;
                int lastPct = -1;
                while ((n = in.read(buf)) > 0) {
                    out.write(buf, 0, n);
                    got += n;
                    if (status != null && size > 0) {
                        int pct = (int) (got * 100 / size);
                        if (pct != lastPct) {
                            lastPct = pct;
                            status.onStatus("ダウンロード中… " + label + " " + pct + "%");
                        }
                    }
                }
            } finally {
                c.disconnect();
            }
            return;
        }
        throw new IOException("リダイレクトが多すぎます: " + url);
    }

    static String sha256(File f) throws IOException {
        try (InputStream in = new FileInputStream(f)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
            StringBuilder sb = new StringBuilder();
            for (byte b : md.digest()) sb.append(String.format(Locale.US, "%02x", b));
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
    }
}
