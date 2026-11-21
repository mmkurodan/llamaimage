package com.micklab.llamaimage;

import android.app.DownloadManager;
import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.Locale;

/**
 * Model files on the device: where they live, first-time download, verification and deletion.
 *
 * Downloads go through the system {@link DownloadManager} into the app-specific external
 * directory ({@code Android/data/<pkg>/files/models}). That keeps a multi-GB download running
 * while the app is in the background or the screen is off, resumes it after network changes,
 * and needs no storage permission. A finished file is checked against the catalog size and
 * SHA-256 before it is renamed into place, so a partial or corrupt file is never loaded.
 *
 * Files copied in by older versions of the app (files/model.gguf, files/taesd.bin) are adopted
 * into files/models/ instead of being downloaded again.
 */
public final class ModelStore {

    /** Snapshot of a DownloadManager entry. */
    public static final class DownloadState {
        public final int status;        // DownloadManager.STATUS_*
        public final long downloaded;   // bytes so far
        public final long total;        // bytes, or <= 0 if unknown yet
        public final int reason;        // DownloadManager.COLUMN_REASON
        public final String localPath;  // file DownloadManager writes to (may be null)

        DownloadState(int status, long downloaded, long total, int reason, String localPath) {
            this.status = status;
            this.downloaded = downloaded;
            this.total = total;
            this.reason = reason;
            this.localPath = localPath;
        }

        public boolean active() {
            return status == DownloadManager.STATUS_PENDING
                    || status == DownloadManager.STATUS_RUNNING
                    || status == DownloadManager.STATUS_PAUSED;
        }
    }

    public interface Progress {
        void onProgress(int percent);
    }

    private static final String PREFS = "llamaimage_models";
    private static final String KEY_DL = "dl:";             // dl:<specId> -> DownloadManager id
    private static final String KEY_VERIFIED = "verified:"; // verified:<path> -> size:mtime
    private static final String KEY_IMPORTED = "imported_model";
    private static final String LEGACY_MODEL = "model.gguf";
    private static final String LEGACY_TAESD = "taesd.bin";
    private static final String IMPORTED_FILE = "imported_model.gguf";
    private static final String IMPORTED_TAESD = "taesd-imported.bin";
    private static final long SPACE_MARGIN = 200L * 1000 * 1000;

    private final Context ctx;
    private final SharedPreferences prefs;

    public ModelStore(Context context) {
        ctx = context.getApplicationContext();
        prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    // --- locations -------------------------------------------------------------------------

    /** Download destination (app-specific external storage); null if storage is unavailable. */
    public File externalDir() {
        File d = ctx.getExternalFilesDir("models");
        if (d != null && !d.isDirectory()) d.mkdirs();
        return d;
    }

    /** Internal directory holding files adopted from older app versions. */
    public File internalDir() {
        File d = new File(ctx.getFilesDir(), "models");
        if (!d.isDirectory()) d.mkdirs();
        return d;
    }

    /** The complete (right-sized) file for {@code spec}, or null if not installed. */
    public File installedFile(ModelCatalog.ModelSpec spec) {
        for (File dir : new File[]{externalDir(), internalDir()}) {
            if (dir == null) continue;
            File f = new File(dir, spec.fileName);
            if (f.isFile() && (spec.sizeBytes <= 0 || f.length() == spec.sizeBytes)) return f;
        }
        return null;
    }

    public boolean isInstalled(ModelCatalog.ModelSpec spec) {
        return installedFile(spec) != null;
    }

    /** TAESD weights to use: the catalog file, else one adopted from an older version. */
    public File installedTaesd() {
        File f = installedFile(ModelCatalog.TAESD);
        if (f != null) return f;
        File imported = new File(internalDir(), IMPORTED_TAESD);
        return imported.isFile() ? imported : null;
    }

    /** A model copied in by an older version that matches no catalog entry, or null. */
    public ModelCatalog.ModelSpec importedModel() {
        File f = new File(internalDir(), IMPORTED_FILE);
        if (!f.isFile()) return null;
        return ModelCatalog.imported(IMPORTED_FILE, f.length(),
                prefs.getString(KEY_IMPORTED, "以前に読み込んだモデル"));
    }

    // --- adoption of files from older versions ---------------------------------------------

    /**
     * Moves files/model.gguf and files/taesd.bin (copied by the old file-picker flow) into
     * files/models/. A model whose size and SHA-256 match a catalog entry becomes that entry;
     * any other model is kept as the "imported" model. Blocking (hashes the file once).
     * Returns the catalog/imported spec the old model became, or null if there was none.
     */
    public ModelCatalog.ModelSpec adoptLegacyFiles(Progress progress) {
        File legacyTaesd = new File(ctx.getFilesDir(), LEGACY_TAESD);
        if (legacyTaesd.isFile()) {
            boolean isCatalog = legacyTaesd.length() == ModelCatalog.TAESD.sizeBytes
                    && ModelCatalog.TAESD.sha256.equals(sha256Quiet(legacyTaesd, null));
            File dst = new File(internalDir(), isCatalog ? ModelCatalog.TAESD.fileName : IMPORTED_TAESD);
            if (legacyTaesd.renameTo(dst) && isCatalog) markVerified(dst);
        }
        File legacy = new File(ctx.getFilesDir(), LEGACY_MODEL);
        if (!legacy.isFile() || legacy.length() == 0) return null;
        ModelCatalog.ModelSpec match = ModelCatalog.findModelBySize(legacy.length());
        if (match != null && match.sha256.equals(sha256Quiet(legacy, progress))) {
            File dst = new File(internalDir(), match.fileName);
            if (legacy.renameTo(dst)) {
                markVerified(dst);
                return match;
            }
            return null;
        }
        File dst = new File(internalDir(), IMPORTED_FILE);
        if (legacy.renameTo(dst)) {
            prefs.edit().putString(KEY_IMPORTED, "以前に読み込んだモデル ("
                    + ModelCatalog.formatSize(dst.length()) + ")").apply();
            return importedModel();
        }
        return null;
    }

    // --- verification ----------------------------------------------------------------------

    /**
     * Ensures {@code file} is the exact catalog file (SHA-256), hashing it at most once per
     * file version. Files without a catalog hash (imported) are accepted as-is. Blocking.
     */
    public void ensureVerified(ModelCatalog.ModelSpec spec, File file, Progress progress) throws IOException {
        if (spec.sha256 == null || isMarkedVerified(file)) return;
        String actual = sha256(file, progress);
        if (!spec.sha256.equalsIgnoreCase(actual)) {
            throw new IOException("ファイルが破損しています（SHA-256 不一致）: " + file.getName());
        }
        markVerified(file);
    }

    private boolean isMarkedVerified(File f) {
        return stamp(f).equals(prefs.getString(KEY_VERIFIED + f.getAbsolutePath(), ""));
    }

    private void markVerified(File f) {
        prefs.edit().putString(KEY_VERIFIED + f.getAbsolutePath(), stamp(f)).apply();
    }

    private static String stamp(File f) {
        return f.length() + ":" + f.lastModified();
    }

    private static String sha256Quiet(File f, Progress p) {
        try {
            return sha256(f, p);
        } catch (IOException e) {
            return "";
        }
    }

    static String sha256(File f, Progress progress) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (Exception e) {
            throw new IOException(e);
        }
        long total = Math.max(1, f.length());
        long done = 0;
        int lastPct = -1;
        byte[] buf = new byte[1 << 20];
        try (InputStream in = new FileInputStream(f)) {
            int n;
            while ((n = in.read(buf)) > 0) {
                md.update(buf, 0, n);
                done += n;
                int pct = (int) (done * 100 / total);
                if (progress != null && pct != lastPct) {
                    lastPct = pct;
                    progress.onProgress(pct);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : md.digest()) sb.append(String.format(Locale.US, "%02x", b));
        return sb.toString();
    }

    // --- downloads -------------------------------------------------------------------------

    private DownloadManager dm() {
        return (DownloadManager) ctx.getSystemService(Context.DOWNLOAD_SERVICE);
    }

    private File partFile(ModelCatalog.ModelSpec spec) {
        File dir = externalDir();
        return dir == null ? null : new File(dir, spec.fileName + ".part");
    }

    /** Starts the first-time download of {@code spec}. Returns the DownloadManager id. */
    public long startDownload(ModelCatalog.ModelSpec spec) throws IOException {
        if (spec.url == null) throw new IOException("このモデルにはダウンロード先がありません");
        File dir = externalDir();
        if (dir == null) throw new IOException("ストレージ（アプリ専用領域）を利用できません");
        long free = dir.getUsableSpace();
        if (free < spec.sizeBytes + SPACE_MARGIN) {
            throw new IOException("空き容量が不足しています（必要 " + ModelCatalog.formatSize(spec.sizeBytes)
                    + " / 空き " + ModelCatalog.formatSize(free) + "）");
        }
        File part = partFile(spec);
        if (part.exists()) part.delete();   // DownloadManager refuses an existing destination

        DownloadManager.Request req = new DownloadManager.Request(Uri.parse(spec.url));
        req.setTitle(spec.title);
        req.setDescription("llamaimage: 画像生成モデルをダウンロード中");
        req.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE);
        // Same path as partFile(): Android/data/<pkg>/files/models/<name>.part (no permission needed).
        req.setDestinationInExternalFilesDir(ctx, "models", part.getName());
        req.setMimeType("application/octet-stream");
        req.setAllowedOverMetered(true);
        req.setAllowedOverRoaming(false);
        long id;
        try {
            id = dm().enqueue(req);
        } catch (RuntimeException e) {
            throw new IOException("ダウンロードを開始できません（ダウンロードマネージャ無効？）: " + e.getMessage());
        }
        prefs.edit().putLong(KEY_DL + spec.id, id).apply();
        return id;
    }

    /** The DownloadManager id for {@code spec}, or -1 if no download is recorded. */
    public long downloadId(ModelCatalog.ModelSpec spec) {
        return prefs.getLong(KEY_DL + spec.id, -1);
    }

    /** Current state of {@code spec}'s download, or null if none (or the entry vanished). */
    public DownloadState queryDownload(ModelCatalog.ModelSpec spec) {
        long id = downloadId(spec);
        if (id < 0) return null;
        try (Cursor c = dm().query(new DownloadManager.Query().setFilterById(id))) {
            if (c == null || !c.moveToFirst()) {
                prefs.edit().remove(KEY_DL + spec.id).apply();
                return null;
            }
            String uri = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI));
            String path = null;
            if (uri != null) {
                try {
                    path = Uri.parse(uri).getPath();
                } catch (Exception ignore) {
                }
            }
            return new DownloadState(
                    c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)),
                    c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)),
                    c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)),
                    c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)),
                    path);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** First catalog item (models, then TAESD) with a download still in progress, or null. */
    public ModelCatalog.ModelSpec activeDownload() {
        for (ModelCatalog.ModelSpec s : ModelCatalog.MODELS) {
            DownloadState st = queryDownload(s);
            if (st != null && st.active()) return s;
        }
        DownloadState st = queryDownload(ModelCatalog.TAESD);
        return st != null && st.active() ? ModelCatalog.TAESD : null;
    }

    /**
     * Turns a SUCCESSFUL download into the installed file: size + SHA-256 check, then rename.
     * Deletes the download on mismatch. Blocking (hashes the file). Returns the installed file.
     */
    public File finishDownload(ModelCatalog.ModelSpec spec, DownloadState st, Progress progress) throws IOException {
        long id = downloadId(spec);
        File part = (st != null && st.localPath != null) ? new File(st.localPath) : partFile(spec);
        try {
            if (part == null || !part.isFile()) {
                throw new IOException("ダウンロードしたファイルが見つかりません");
            }
            if (part.length() != spec.sizeBytes) {
                throw new IOException("サイズが一致しません（" + part.length() + " / " + spec.sizeBytes + " bytes）");
            }
            String actual = sha256(part, progress);
            if (!spec.sha256.equalsIgnoreCase(actual)) {
                throw new IOException("SHA-256 が一致しません（破損したダウンロード）");
            }
            File dst = new File(part.getParentFile(), spec.fileName);
            if (dst.exists()) dst.delete();
            if (!part.renameTo(dst)) throw new IOException("ファイルを配置できません: " + dst);
            markVerified(dst);
            return dst;
        } catch (IOException e) {
            if (part != null) part.delete();
            throw e;
        } finally {
            // Drop the DownloadManager entry (its file was renamed away or deleted above).
            if (id >= 0) {
                try {
                    dm().remove(id);
                } catch (RuntimeException ignore) {
                }
            }
            prefs.edit().remove(KEY_DL + spec.id).apply();
        }
    }

    /** Cancels and forgets {@code spec}'s download (DownloadManager deletes the partial file). */
    public void cancelDownload(ModelCatalog.ModelSpec spec) {
        long id = downloadId(spec);
        if (id >= 0) {
            try {
                dm().remove(id);
            } catch (RuntimeException ignore) {
            }
        }
        prefs.edit().remove(KEY_DL + spec.id).apply();
        File part = partFile(spec);
        if (part != null && part.exists()) part.delete();
    }

    /** Deletes every installed copy of {@code spec}. */
    public void delete(ModelCatalog.ModelSpec spec) {
        for (File dir : new File[]{externalDir(), internalDir()}) {
            if (dir == null) continue;
            File f = new File(dir, spec.fileName);
            if (f.exists()) {
                prefs.edit().remove(KEY_VERIFIED + f.getAbsolutePath()).apply();
                f.delete();
            }
        }
        if (ModelCatalog.TAESD == spec) {
            new File(internalDir(), IMPORTED_TAESD).delete();
        }
    }

    /** Human-readable DownloadManager pause/failure reason. */
    public static String reasonText(DownloadState st) {
        if (st == null) return "";
        if (st.status == DownloadManager.STATUS_PAUSED) {
            switch (st.reason) {
                case DownloadManager.PAUSED_WAITING_FOR_NETWORK: return "ネットワーク待ち";
                case DownloadManager.PAUSED_QUEUED_FOR_WIFI: return "Wi-Fi 待ち";
                case DownloadManager.PAUSED_WAITING_TO_RETRY: return "再試行待ち";
                default: return "一時停止中";
            }
        }
        if (st.status == DownloadManager.STATUS_FAILED) {
            switch (st.reason) {
                case DownloadManager.ERROR_INSUFFICIENT_SPACE: return "空き容量不足";
                case DownloadManager.ERROR_CANNOT_RESUME: return "再開できません";
                case DownloadManager.ERROR_HTTP_DATA_ERROR: return "通信エラー";
                case DownloadManager.ERROR_TOO_MANY_REDIRECTS: return "リダイレクト過多";
                case DownloadManager.ERROR_FILE_ALREADY_EXISTS: return "ファイルが既に存在";
                case DownloadManager.ERROR_DEVICE_NOT_FOUND: return "ストレージが見つかりません";
                default: return st.reason >= 100 && st.reason < 600 ? "HTTP " + st.reason : "エラー " + st.reason;
            }
        }
        return "";
    }
}
