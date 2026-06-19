package com.micklab.llamaimage;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal SD1.5 text2img PoC UI.
 *
 * Robustness notes (learned the hard way): the native library is arm64-v8a only, so on a
 * non-arm64 device/emulator {@code System.loadLibrary} fails. We therefore (1) wire all UI
 * listeners BEFORE touching native code so the screen is never dead, (2) initialise native
 * lazily on a worker thread wrapped in try/catch(Throwable) and surface any error on screen,
 * and (3) load the model straight from its content-URI file descriptor (no 1.5GB copy), with
 * a copy-to-internal-storage fallback.
 */
public class MainActivity extends Activity {

    private static final int REQ_PICK_MODEL = 1001;
    private static final String MODEL_FILENAME = "model.gguf";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private volatile SdModelManager manager;     // null until native init succeeds
    private volatile boolean nativeReady = false;
    private ParcelFileDescriptor modelPfd;       // kept open while a fd-backed model is loaded

    private TextView statusText;
    private TextView logText;
    private EditText promptEdit;
    private EditText negativeEdit;
    private EditText stepsEdit;
    private EditText sizeEdit;
    private Button pickButton;
    private Button generateButton;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(buildUi());

        // Wire listeners FIRST — the UI must stay responsive even if native init fails.
        pickButton.setOnClickListener(v -> pickModel());
        generateButton.setOnClickListener(v -> generate());
        generateButton.setEnabled(false);

        statusText.setText("ネイティブ初期化中…");
        worker.submit(this::initNative);
    }

    /** Load libsd_jni.so and query the engine. Any failure is shown, never crashes the app. */
    private void initNative() {
        try {
            SdModelManager m = SdModelManager.get();   // triggers System.loadLibrary("sd_jni")
            m.setLogPath(new File(getFilesDir(), "sd.log").getAbsolutePath());
            m.setProgressListener((step, steps) ->
                    runOnUiThread(() -> statusText.setText("生成中… step " + step + "/" + steps)));
            String info = m.systemInfo();
            manager = m;
            nativeReady = true;
            appendLogUi("ネイティブ準備完了");
            appendLogUi(info);
            runOnUiThread(() -> statusText.setText("準備完了 — モデルを選択してください"));
        } catch (Throwable t) {
            appendLogUi("ネイティブ初期化失敗: " + t);
            appendLogUi("この端末の ABI が arm64-v8a でない可能性があります（エミュレータ等）。");
            runOnUiThread(() -> statusText.setText("ネイティブ初期化失敗: " + t.getClass().getSimpleName()));
        }
    }

    // --- UI construction (no XML, mirrors the scaffold style) ---
    private View buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(12);
        root.setPadding(pad, pad, pad, pad);

        pickButton = new Button(this);
        pickButton.setText("モデル選択 (.gguf を Download から)");
        root.addView(pickButton);

        statusText = new TextView(this);
        statusText.setText("起動中…");
        root.addView(statusText);

        promptEdit = new EditText(this);
        promptEdit.setHint("プロンプト");
        promptEdit.setText("a photo of a cat");
        root.addView(promptEdit);

        negativeEdit = new EditText(this);
        negativeEdit.setHint("ネガティブプロンプト (任意)");
        root.addView(negativeEdit);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        stepsEdit = new EditText(this);
        stepsEdit.setHint("steps");
        stepsEdit.setText("20");
        stepsEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        sizeEdit = new EditText(this);
        sizeEdit.setHint("size");
        sizeEdit.setText("512");
        sizeEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        row.addView(stepsEdit);
        row.addView(sizeEdit);
        root.addView(row);

        generateButton = new Button(this);
        generateButton.setText("生成");
        root.addView(generateButton);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(imageView);

        logText = new TextView(this);
        logText.setTextSize(11);
        root.addView(logText);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        return scroll;
    }

    // --- model picking + load ---
    private void pickModel() {
        if (!nativeReady) {
            Toast.makeText(this, "ネイティブ初期化が完了していません（ログ参照）", Toast.LENGTH_SHORT).show();
            return;
        }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        // Best-effort: open the picker already in the Download folder (API 26+, ignored otherwise).
        try {
            intent.putExtra(android.provider.DocumentsContract.EXTRA_INITIAL_URI,
                    Uri.parse("content://com.android.externalstorage.documents/document/primary:Download"));
        } catch (Throwable ignore) {
        }
        startActivityForResult(intent, REQ_PICK_MODEL);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQ_PICK_MODEL || resultCode != RESULT_OK || data == null) {
            return;
        }
        final Uri uri = data.getData();
        if (uri == null) {
            return;
        }
        setBusyUi(true);
        statusText.setText("モデルを準備中…");
        worker.submit(() -> loadModel(uri));
    }

    private void loadModel(Uri uri) {
        try {
            // Primary path: hand the engine the file descriptor directly via /proc/self/fd,
            // so no 1.5GB copy and no app-storage space requirement.
            ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "r");
            if (pfd != null) {
                String fdPath = "/proc/self/fd/" + pfd.getFd();
                appendLogUi("fd 経由で読込: " + fdPath);
                String err = manager.load(fdPath, 0);
                if (err.isEmpty()) {
                    closePfd();
                    modelPfd = pfd;   // keep open while loaded
                    onModelLoaded();
                    return;
                }
                appendLogUi("fd 経由の読込失敗: " + err + " → コピー方式にフォールバック");
                pfd.close();
            }

            // Fallback: copy into app-private storage, then load from the real path.
            long srcSize = querySize(uri);
            File dest = new File(getFilesDir(), MODEL_FILENAME);
            if (dest.exists() && srcSize > 0 && dest.length() == srcSize) {
                appendLogUi("既存コピーを再利用: " + dest.getName());
            } else {
                appendLogUi("コピー開始 (" + srcSize + " bytes)…");
                copyUriToFile(uri, dest, srcSize);
            }
            String err = manager.load(dest.getAbsolutePath(), 0);
            if (err.isEmpty()) {
                onModelLoaded();
            } else {
                appendLogUi("モデル読込失敗: " + err);
                runOnUiThread(() -> {
                    statusText.setText("読込失敗: " + err);
                    setBusyUi(false);
                });
            }
        } catch (Throwable t) {
            appendLogUi("モデル準備エラー: " + t);
            runOnUiThread(() -> {
                statusText.setText("エラー: " + t.getClass().getSimpleName());
                setBusyUi(false);
            });
        }
    }

    private void onModelLoaded() {
        appendLogUi("モデル読込成功");
        runOnUiThread(() -> {
            statusText.setText("モデル読込済み — 生成できます");
            setBusyUi(false);
        });
    }

    private long querySize(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[]{OpenableColumns.SIZE}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.SIZE);
                if (idx >= 0 && !c.isNull(idx)) {
                    return c.getLong(idx);
                }
            }
        } catch (Exception ignore) {
        }
        return -1;
    }

    private void copyUriToFile(Uri uri, File dest, long total) throws Exception {
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(dest)) {
            if (in == null) {
                throw new IllegalStateException("openInputStream returned null");
            }
            byte[] buf = new byte[1 << 20];
            long copied = 0;
            int lastPct = -1;
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
                copied += n;
                if (total > 0) {
                    int pct = (int) (copied * 100 / total);
                    if (pct != lastPct && pct % 5 == 0) {
                        lastPct = pct;
                        final int p = pct;
                        runOnUiThread(() -> statusText.setText("コピー中… " + p + "%"));
                    }
                }
            }
            out.flush();
        }
    }

    // --- generation ---
    private void generate() {
        if (manager == null || !manager.isModelLoaded()) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            return;
        }
        final String prompt = promptEdit.getText().toString();
        final String negative = negativeEdit.getText().toString();
        final int steps = parseInt(stepsEdit.getText().toString(), 20);
        final int size = parseInt(sizeEdit.getText().toString(), 512);

        setBusyUi(true);
        statusText.setText("生成中…");
        worker.submit(() -> {
            try {
                long t0 = System.currentTimeMillis();
                byte[] rgb = manager.generate(prompt, negative, size, size, steps, 7.0f, -1);
                long ms = System.currentTimeMillis() - t0;
                if (rgb == null) {
                    appendLogUi("生成失敗");
                    runOnUiThread(() -> {
                        statusText.setText("生成失敗");
                        setBusyUi(false);
                    });
                    return;
                }
                final Bitmap bmp = rgbToBitmap(rgb, size, size);
                final long seed = manager.getLastSeed();
                appendLogUi("生成完了 seed=" + seed + " " + ms + "ms");
                runOnUiThread(() -> {
                    imageView.setImageBitmap(bmp);
                    statusText.setText("完了 (seed=" + seed + ", " + ms + "ms)");
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                appendLogUi("生成エラー: " + t);
                runOnUiThread(() -> {
                    statusText.setText("生成エラー: " + t.getClass().getSimpleName());
                    setBusyUi(false);
                });
            }
        });
    }

    private static Bitmap rgbToBitmap(byte[] rgb, int w, int h) {
        int[] pixels = new int[w * h];
        for (int i = 0; i < w * h; i++) {
            int r = rgb[i * 3] & 0xff;
            int g = rgb[i * 3 + 1] & 0xff;
            int b = rgb[i * 3 + 2] & 0xff;
            pixels[i] = 0xff000000 | (r << 16) | (g << 8) | b;
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
    }

    // --- helpers ---
    private void setBusyUi(boolean busy) {
        pickButton.setEnabled(!busy && nativeReady);
        generateButton.setEnabled(!busy && manager != null && manager.isModelLoaded());
    }

    private static int parseInt(String s, int def) {
        try {
            int v = Integer.parseInt(s.trim());
            return v > 0 ? v : def;
        } catch (Exception e) {
            return def;
        }
    }

    private void appendLog(String msg) {
        if (msg == null) {
            return;
        }
        logText.append(msg.endsWith("\n") ? msg : msg + "\n");
    }

    private void appendLogUi(String msg) {
        runOnUiThread(() -> appendLog(msg));
    }

    private void closePfd() {
        if (modelPfd != null) {
            try {
                modelPfd.close();
            } catch (Exception ignore) {
            }
            modelPfd = null;
        }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        closePfd();
        worker.shutdownNow();
    }
}
