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
        NpuModelAssembler.assemble(assets, "unet", checkpoint, out,
                p -> status.onStatus("NPU 用に変換中（UNet） " + p + "%"));
        for (String f : new String[]{"taesd_decoder.onnx", "taesd_encoder.onnx", "tokenizer_vocab.json",
                "tokenizer_merges.txt"}) {
            IoUtil.copy(new File(assets, f), new File(out, f));
        }
        IoUtil.writeString(stamp, NpuAssets.VERSION + " " + ckptSha);
    }

    /** Opens the pipeline on the HTP (compiles + caches the graphs on first use: minutes). */
    public NpuPipeline open(String modelId, NpuPipeline.Log log) throws IOException {
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
        return new NpuPipeline(modelDir(modelId), qnn, cacheDir(modelId), log);
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
                while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
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
