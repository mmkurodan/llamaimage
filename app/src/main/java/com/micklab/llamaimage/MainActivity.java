package com.micklab.llamaimage;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
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
 * Actions live in the ActionBar (model picker / generate / save) and the file picker auto-opens
 * after init — this is resilient to ROMs that render an in-layout header oddly (observed). Live
 * status + per-step progress go to the ActionBar subtitle and a visible progress bar, since CPU
 * SD1.5 generation takes minutes and the user needs to see it advancing.
 *
 * Native interaction is lazy, on a worker thread, wrapped in try/catch(Throwable). The model is
 * copied once into app storage (cached by size) and loaded from there — loading straight from a
 * content-URI fd does not work with this engine.
 */
public class MainActivity extends Activity {

    private static final int REQ_PICK_MODEL = 1001;
    private static final String MODEL_FILENAME = "model.gguf";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private volatile SdModelManager manager;     // null until native init succeeds
    private volatile boolean nativeReady = false;
    private volatile long genStart = 0;          // generation start time for elapsed/ETA

    private LinearLayout rootBody;
    private EditText promptEdit;
    private EditText negativeEdit;
    private EditText stepsEdit;
    private EditText sizeEdit;
    private Button generateButton;
    private ProgressBar progressBar;
    private ImageView imageView;
    private TextView logText;
    private Bitmap lastBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Keep the keyboard closed on launch, but allow it to open when the user taps a field
        // (STATE_HIDDEN, not STATE_ALWAYS_HIDDEN — the latter blocks input entirely).
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_HIDDEN
                | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        setContentView(buildUi());
        rootBody.requestFocus();   // hold initial focus off the prompt (no auto-scroll under the title bar)

        generateButton.setOnClickListener(v -> generate());
        generateButton.setEnabled(false);

        setStatus("ネイティブ初期化中…");
        worker.submit(this::initNative);
    }

    /** Load libsd_jni.so and query the engine. Any failure is shown, never crashes the app. */
    private void initNative() {
        try {
            SdModelManager m = SdModelManager.get();   // triggers System.loadLibrary("sd_jni")
            m.setLogPath(new File(getFilesDir(), "sd.log").getAbsolutePath());
            m.setProgressListener(this::onGenProgress);
            String info = m.systemInfo();
            manager = m;
            nativeReady = true;
            appendLogUi("ネイティブ準備完了");
            appendLogUi(info);
            setStatusUi("準備完了");
            autoStartModel();
        } catch (Throwable t) {
            appendLogUi("ネイティブ初期化失敗: " + t);
            appendLogUi("この端末の ABI が arm64-v8a でない可能性があります（エミュレータ等）。");
            runOnUiThread(() -> setStatus("ネイティブ初期化失敗: " + t.getClass().getSimpleName()));
        }
    }

    /**
     * After native init: if a model was already copied in a previous run, load that cached copy
     * automatically (no picker). Only the first time (no cached file) do we open the picker.
     * Runs on the worker thread (called from initNative).
     */
    private void autoStartModel() {
        File cached = new File(getFilesDir(), MODEL_FILENAME);
        if (cached.exists() && cached.length() > 0) {
            appendLogUi("キャッシュ済みモデルを自動ロード (" + cached.length() + " bytes)");
            runOnUiThread(() -> {
                setStatus("モデル読込中（キャッシュ）…");
                setBusyUi(true);
            });
            try {
                String err = manager.load(cached.getAbsolutePath(), 0);
                if (err.isEmpty()) {
                    appendLogUi("モデル読込成功（キャッシュ）");
                    runOnUiThread(() -> {
                        setStatus("モデル読込済み — 「② 生成」できます");
                        setBusyUi(false);
                    });
                } else {
                    appendLogUi("キャッシュ読込失敗: " + err);
                    runOnUiThread(() -> {
                        setStatus("キャッシュ読込失敗 — モデルを選択してください");
                        setBusyUi(false);
                        pickModel();
                    });
                }
            } catch (Throwable t) {
                appendLogUi("キャッシュ読込エラー: " + t);
                runOnUiThread(() -> {
                    setBusyUi(false);
                    pickModel();
                });
            }
        } else {
            runOnUiThread(this::pickModel);   // first run: open the picker
        }
    }

    /** Per-diffusion-step progress callback (called from the worker/native thread). */
    private void onGenProgress(int step, int steps) {
        runOnUiThread(() -> {
            if (steps > 0) {
                progressBar.setVisibility(View.VISIBLE);
                progressBar.setMax(steps);
                progressBar.setProgress(step);
            }
            long el = (System.currentTimeMillis() - genStart) / 1000;
            String eta = "";
            if (step > 0 && step < steps) {
                eta = " 残り~" + (el * (steps - step) / step) + "s";
            }
            setStatus("生成中 " + step + "/" + steps + " (" + el + "s" + eta + ")");
        });
    }

    // --- UI construction (no XML). Single scrolling body; actions are in the ActionBar. ---
    private View buildUi() {
        int pad = dp(12);

        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(pad, pad, pad, pad);
        body.setFocusable(true);
        body.setFocusableInTouchMode(true);   // hold focus so the prompt doesn't open the keyboard
        rootBody = body;

        // Non-focusable header line — also keeps the first input clear of the title bar.
        TextView guide = new TextView(this);
        guide.setText("プロンプトを入力して「② 生成」。モデル選択は上部メニューから。");
        guide.setPadding(0, 0, 0, dp(8));
        body.addView(guide);

        promptEdit = new EditText(this);
        promptEdit.setHint("プロンプト");
        promptEdit.setText("a photo of a cat");
        body.addView(promptEdit);

        negativeEdit = new EditText(this);
        negativeEdit.setHint("ネガティブプロンプト (任意)");
        body.addView(negativeEdit);

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
        body.addView(row);

        generateButton = new Button(this);
        generateButton.setText("② 生成");
        body.addView(generateButton);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setVisibility(View.GONE);
        body.addView(progressBar);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        body.addView(imageView);

        logText = new TextView(this);
        logText.setTextSize(11);
        body.addView(logText);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(body);
        return scroll;
    }

    // ActionBar menu — always visible, independent of the content layout.
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, "モデル選択").setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        menu.add(0, 2, 1, "生成").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        menu.add(0, 3, 2, "画像を保存").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case 1: pickModel(); return true;
            case 2: generate(); return true;
            case 3: saveImage(); return true;
            default: return super.onOptionsItemSelected(item);
        }
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
        setStatus("モデルを準備中…");
        worker.submit(() -> loadModel(uri));
    }

    private void loadModel(Uri uri) {
        try {
            // Copy once into app storage (cached by size), then load from the real path.
            // (Loading from a /proc/self/fd path is not supported by this engine.)
            long srcSize = querySize(uri);
            File dest = new File(getFilesDir(), MODEL_FILENAME);
            if (dest.exists() && srcSize > 0 && dest.length() == srcSize) {
                appendLogUi("既存コピーを再利用: " + dest.getName());
            } else {
                appendLogUi("コピー開始 (" + srcSize + " bytes)…");
                copyUriToFile(uri, dest, srcSize);
                appendLogUi("コピー完了");
            }
            setStatusUi("モデル読込中…");
            String err = manager.load(dest.getAbsolutePath(), 0);
            if (err.isEmpty()) {
                appendLogUi("モデル読込成功");
                runOnUiThread(() -> {
                    setStatus("モデル読込済み — 「② 生成」できます");
                    setBusyUi(false);
                });
            } else {
                appendLogUi("モデル読込失敗: " + err);
                runOnUiThread(() -> {
                    setStatus("読込失敗: " + err);
                    setBusyUi(false);
                });
            }
        } catch (Throwable t) {
            appendLogUi("モデル準備エラー: " + t);
            runOnUiThread(() -> {
                setStatus("エラー: " + t.getClass().getSimpleName());
                setBusyUi(false);
            });
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
                        setStatusUi("コピー中… " + pct + "%");
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
        genStart = System.currentTimeMillis();
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setMax(steps);
        progressBar.setProgress(0);
        setStatus("生成開始… (CPU: " + size + "px/" + steps + "step は数分かかります)");
        appendLog("生成開始: \"" + prompt + "\" " + size + "px " + steps + "step");

        worker.submit(() -> {
            try {
                byte[] rgb = manager.generate(prompt, negative, size, size, steps, 7.0f, -1);
                long ms = System.currentTimeMillis() - genStart;
                if (rgb == null) {
                    appendLogUi("生成失敗");
                    runOnUiThread(() -> {
                        setStatus("生成失敗");
                        progressBar.setVisibility(View.GONE);
                        setBusyUi(false);
                    });
                    return;
                }
                final Bitmap bmp = rgbToBitmap(rgb, size, size);
                final long seed = manager.getLastSeed();
                appendLogUi("生成完了 seed=" + seed + " " + (ms / 1000) + "s");
                runOnUiThread(() -> {
                    lastBitmap = bmp;
                    imageView.setImageBitmap(bmp);
                    progressBar.setProgress(progressBar.getMax());
                    setStatus("完了 seed=" + seed + " (" + (ms / 1000) + "s)");
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                appendLogUi("生成エラー: " + t);
                runOnUiThread(() -> {
                    setStatus("生成エラー: " + t.getClass().getSimpleName());
                    progressBar.setVisibility(View.GONE);
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

    // --- save the last generated image to the gallery ---
    private void saveImage() {
        final Bitmap bmp = lastBitmap;
        if (bmp == null) {
            Toast.makeText(this, "保存できる画像がありません", Toast.LENGTH_SHORT).show();
            return;
        }
        worker.submit(() -> {
            try {
                String name = "llamaimage_" + System.currentTimeMillis() + ".png";
                if (Build.VERSION.SDK_INT >= 29) {
                    ContentValues cv = new ContentValues();
                    cv.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                    cv.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
                    cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/llamaimage");
                    Uri u = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                    try (OutputStream os = getContentResolver().openOutputStream(u)) {
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, os);
                    }
                    appendLogUi("保存: Pictures/llamaimage/" + name);
                } else {
                    File dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                    File f = new File(dir, name);
                    try (OutputStream os = new FileOutputStream(f)) {
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, os);
                    }
                    appendLogUi("保存: " + f.getAbsolutePath());
                }
                runOnUiThread(() -> Toast.makeText(this, "画像を保存しました", Toast.LENGTH_SHORT).show());
            } catch (Throwable t) {
                appendLogUi("保存失敗: " + t);
                runOnUiThread(() -> Toast.makeText(this, "保存失敗", Toast.LENGTH_SHORT).show());
            }
        });
    }

    // --- helpers ---
    private void setBusyUi(boolean busy) {
        generateButton.setEnabled(!busy && manager != null && manager.isModelLoaded());
    }

    /** Live status shown in the always-visible ActionBar subtitle. */
    private void setStatus(String s) {
        if (getActionBar() != null) {
            getActionBar().setSubtitle(s);
        }
    }

    private void setStatusUi(String s) {
        runOnUiThread(() -> setStatus(s));
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
