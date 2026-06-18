package com.micklab.llamaimage;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
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
 * Flow: pick the GGUF already sitting in the device Download folder via the Storage Access
 * Framework -> copy it once into app-private storage -> load it into the native engine ->
 * type a prompt -> generate -> show the resulting bitmap. CPU backend, single image.
 */
public class MainActivity extends Activity {

    private static final int REQ_PICK_MODEL = 1001;
    private static final String MODEL_FILENAME = "model.gguf";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private SdModelManager manager;

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

        manager = SdModelManager.get();
        manager.setLogPath(new File(getFilesDir(), "sd.log").getAbsolutePath());
        manager.setProgressListener((step, steps) ->
                runOnUiThread(() -> statusText.setText("生成中… step " + step + "/" + steps)));

        appendLog(manager.systemInfo());
        refreshButtons();

        pickButton.setOnClickListener(v -> pickModel());
        generateButton.setOnClickListener(v -> generate());
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
        statusText.setText("モデル未ロード");
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

    // --- model picking + copy + load ---
    private void pickModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
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
        worker.submit(() -> prepareAndLoad(uri));
    }

    private void prepareAndLoad(Uri uri) {
        try {
            long srcSize = querySize(uri);
            File dest = new File(getFilesDir(), MODEL_FILENAME);

            if (dest.exists() && srcSize > 0 && dest.length() == srcSize) {
                appendLogUi("既存のコピーを再利用: " + dest.getName() + " (" + srcSize + " bytes)");
            } else {
                appendLogUi("コピー開始 (" + srcSize + " bytes)…");
                copyUriToFile(uri, dest, srcSize);
                appendLogUi("コピー完了: " + dest.getAbsolutePath());
            }

            runOnUiThread(() -> statusText.setText("モデル読込中…"));
            String err = manager.load(dest.getAbsolutePath(), 0 /* native picks physical cores */);
            if (err.isEmpty()) {
                appendLogUi("モデル読込成功");
                runOnUiThread(() -> statusText.setText("モデル読込済み: " + dest.getName()));
            } else {
                appendLogUi("モデル読込失敗: " + err);
                runOnUiThread(() -> statusText.setText("読込失敗: " + err));
            }
        } catch (Exception e) {
            appendLogUi("エラー: " + e);
            runOnUiThread(() -> statusText.setText("エラー: " + e.getMessage()));
        } finally {
            runOnUiThread(() -> setBusyUi(false));
        }
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
        if (!manager.isModelLoaded()) {
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
        pickButton.setEnabled(!busy);
        generateButton.setEnabled(!busy && manager.isModelLoaded());
    }

    private void refreshButtons() {
        setBusyUi(false);
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

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        worker.shutdownNow();
    }
}
