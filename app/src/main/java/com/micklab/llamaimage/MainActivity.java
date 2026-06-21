package com.micklab.llamaimage;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {

    private static final int REQ_PICK_MODEL = 1001;
    private static final int REQ_PICK_INIT_IMAGE = 1002;
    private static final String MODEL_FILENAME = "model.gguf";
    private static final String PREFS_NAME = "llamaimage_prefs";
    private static final String PREFS_LLM_URL = "llm_base_url";
    private static final String PREFS_LLM_MODEL = "llm_model";
    private static final String PREFS_WTYPE = "wtype";
    private static final String PREFS_SAMPLER = "sampler";
    private static final String PREFS_THREADS = "threads";
    private static final String PREFS_NSFW = "nsfw_restriction";
    private static final String PREFS_SIZE = "image_size";

    private static final int[] SIZE_VALUES = {512, 384, 256};
    private static final String[] SIZE_LABELS = {"標準 (512px)", "小 (384px)", "最小 (256px)"};
    private static final String DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434";
    private static final String LLM_PACKAGE = "com.micklab.llama";
    private static final String LLM_SERVICE_CLASS = "com.micklab.llama.OllamaForegroundService";
    private static final String LLM_SERVICE_ACTION = "com.micklab.llama.START_SERVICE";
    private static final String LLM_PLAY_STORE_URL =
            "https://play.google.com/store/apps/details?id=com.micklab.llama";

    private static final String[] NSFW_WORDS = {
        "nsfw", "nude", "naked", "nipple", "nipples", "penis", "vagina",
        "breast", "genitalia", "genitals", "erotic", "pornographic",
        "pornography", "porn", "xxx", "hentai", "topless", "bottomless",
        "pussy", "cock", "dick", "boobs", "tits", "genital",
        "masturbat", "lewd", "obscene", "intercourse", "fetish"
    };

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private volatile SdModelManager manager;
    private volatile boolean nativeReady = false;
    private volatile long genStart = 0;

    private String llmBaseUrl = DEFAULT_LLM_BASE_URL;
    private String llmModel = "";
    // Generation settings (load-time: wtype/threads need a model reload; sampler is per-run).
    private int wtype = StableDiffusionNative.WTYPE_DEFAULT;
    private int sampler = StableDiffusionNative.SAMPLE_DPMPP2M;
    private int nThreads = 0;                 // 0 = auto (physical cores)
    private boolean nsfwRestriction = false;

    private LinearLayout rootBody;
    private EditText promptEdit;
    private TextView translatedPromptView;
    private EditText negativeEdit;
    private EditText stepsEdit;
    private Button sizeButton;
    private int selectedSize = 512;
    private EditText strengthEdit;
    private Button generateButton;
    private Button pickInitButton;
    private Button img2imgButton;
    private ImageView initThumb;
    private ProgressBar progressBar;
    private ImageView imageView;
    private TextView logText;
    private Bitmap lastBitmap;
    private Bitmap initBitmap;               // selected img2img input, null until picked

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_HIDDEN
                | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        loadSettings();
        setContentView(buildUi());
        rootBody.requestFocus();

        generateButton.setOnClickListener(v -> generate());
        generateButton.setEnabled(false);
        pickInitButton.setOnClickListener(v -> pickInitImage());
        img2imgButton.setOnClickListener(v -> generateImg2img());
        img2imgButton.setEnabled(false);

        setStatus("ネイティブ初期化中…");
        worker.submit(this::initNative);
    }

    private void initNative() {
        try {
            SdModelManager m = SdModelManager.get();
            m.setLogPath(new File(getFilesDir(), "sd.log").getAbsolutePath());
            m.setProgressListener(this::onGenProgress);
            m.setPreviewListener(this::onGenPreview);
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

    private void autoStartModel() {
        File cached = new File(getFilesDir(), MODEL_FILENAME);
        if (cached.exists() && cached.length() > 0) {
            appendLogUi("キャッシュ済みモデルを自動ロード (" + cached.length() + " bytes)");
            runOnUiThread(() -> {
                setStatus("モデル読込中（キャッシュ）…");
                setBusyUi(true);
            });
            try {
                String err = manager.load(cached.getAbsolutePath(), nThreads, wtype);
                if (err.isEmpty()) {
                    appendLogUi("モデル読込成功（キャッシュ）" + wtypeSuffix());
                    runOnUiThread(() -> {
                        setStatus("モデル読込済み — 「生成」できます");
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
            runOnUiThread(this::pickModel);
        }
    }

    /** Per-step progress callback from native (called on worker thread). */
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

    /** Latent-approximation preview — fires alongside each progress step (step > 0). */
    private void onGenPreview(byte[] rgb, int w, int h) {
        final Bitmap bmp = rgbToBitmap(rgb, w, h);
        runOnUiThread(() -> imageView.setImageBitmap(bmp));
    }

    private View buildUi() {
        int pad = dp(12);

        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(pad, pad, pad, pad);
        body.setFocusable(true);
        body.setFocusableInTouchMode(true);
        rootBody = body;

        TextView guide = new TextView(this);
        guide.setText("プロンプトを入力（日本語可）→「生成」で自動翻訳＆画像生成（LLM設定時）。モデル・LLM・生成設定は上部メニューから。");
        guide.setPadding(0, 0, 0, dp(8));
        body.addView(guide);

        promptEdit = new EditText(this);
        promptEdit.setHint("プロンプト（日本語・英語）");
        promptEdit.setText("a photo of a cat");
        body.addView(promptEdit);

        translatedPromptView = new TextView(this);
        translatedPromptView.setTextSize(11);
        translatedPromptView.setTextColor(0xFF888888);
        translatedPromptView.setPadding(0, dp(2), 0, dp(4));
        translatedPromptView.setVisibility(View.GONE);
        body.addView(translatedPromptView);

        negativeEdit = new EditText(this);
        negativeEdit.setHint("ネガティブプロンプト (任意)");
        body.addView(negativeEdit);

        LinearLayout paramRow = new LinearLayout(this);
        paramRow.setOrientation(LinearLayout.HORIZONTAL);

        stepsEdit = new EditText(this);
        stepsEdit.setHint("steps");
        stepsEdit.setText("15");
        stepsEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        sizeButton = new Button(this);
        sizeButton.setText(sizeLabelFor(selectedSize));
        sizeButton.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        sizeButton.setOnClickListener(v -> showSizePicker());

        strengthEdit = new EditText(this);
        strengthEdit.setHint("strength");
        strengthEdit.setText("0.6");
        strengthEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        paramRow.addView(stepsEdit);
        paramRow.addView(sizeButton);
        paramRow.addView(strengthEdit);
        body.addView(paramRow);

        generateButton = new Button(this);
        generateButton.setText("生成");
        generateButton.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        body.addView(generateButton);

        // img2img row: input-image thumbnail + pick + generate-from-image.
        LinearLayout img2imgRow = new LinearLayout(this);
        img2imgRow.setOrientation(LinearLayout.HORIZONTAL);
        img2imgRow.setGravity(android.view.Gravity.CENTER_VERTICAL);

        initThumb = new ImageView(this);
        initThumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams thumbParams = new LinearLayout.LayoutParams(dp(48), dp(48));
        thumbParams.rightMargin = dp(8);
        initThumb.setLayoutParams(thumbParams);

        pickInitButton = new Button(this);
        pickInitButton.setText("入力画像");
        pickInitButton.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        img2imgButton = new Button(this);
        img2imgButton.setText("画像から生成");
        img2imgButton.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        img2imgRow.addView(initThumb);
        img2imgRow.addView(pickInitButton);
        img2imgRow.addView(img2imgButton);
        body.addView(img2imgRow);

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

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, "モデル選択").setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        menu.add(0, 2, 1, "生成").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        menu.add(0, 3, 2, "画像を保存").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        menu.add(0, 4, 3, "LLM設定").setShowAsAction(MenuItem.SHOW_AS_ACTION_NEVER);
        menu.add(0, 5, 4, "生成設定").setShowAsAction(MenuItem.SHOW_AS_ACTION_NEVER);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case 1: pickModel(); return true;
            case 2: generate(); return true;
            case 3: saveImage(); return true;
            case 4: showLlmSettingsDialog(); return true;
            case 5: showGenSettingsDialog(); return true;
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
        if (resultCode != RESULT_OK || data == null) return;
        final Uri uri = data.getData();
        if (uri == null) return;
        if (requestCode == REQ_PICK_MODEL) {
            setBusyUi(true);
            setStatus("モデルを準備中…");
            worker.submit(() -> loadModel(uri));
        } else if (requestCode == REQ_PICK_INIT_IMAGE) {
            setBusyUi(true);
            setStatus("入力画像を読込中…");
            loadInitImage(uri);
        }
    }

    private void loadModel(Uri uri) {
        try {
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
            String err = manager.load(dest.getAbsolutePath(), nThreads, wtype);
            if (err.isEmpty()) {
                appendLogUi("モデル読込成功" + wtypeSuffix());
                runOnUiThread(() -> {
                    setStatus("モデル読込済み — 「生成」できます");
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
        try (android.database.Cursor c = getContentResolver().query(
                uri, new String[]{OpenableColumns.SIZE}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.SIZE);
                if (idx >= 0 && !c.isNull(idx)) return c.getLong(idx);
            }
        } catch (Exception ignore) {
        }
        return -1;
    }

    private void copyUriToFile(Uri uri, File dest, long total) throws Exception {
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(dest)) {
            if (in == null) throw new IllegalStateException("openInputStream returned null");
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

    // --- image generation ---

    private void generate() {
        if (manager == null || !manager.isModelLoaded()) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            return;
        }
        final String rawPrompt = promptEdit.getText().toString().trim();
        final String negative = negativeEdit.getText().toString();
        final int steps = parseInt(stepsEdit.getText().toString(), 20);
        final int size = selectedSize;

        setBusyUi(true);
        translatedPromptView.setVisibility(View.GONE);
        setStatus("準備中…");

        worker.submit(() -> {
            final String prompt = translateOrOriginal(rawPrompt);

            if (nsfwRestriction && isNsfwContent(prompt)) {
                runOnUiThread(() -> {
                    setStatus("生成中止: 性的コンテンツを検出");
                    Toast.makeText(this,
                            "コンテンツがモデレートされました（性的表現を検出）。生成を中止します。",
                            Toast.LENGTH_LONG).show();
                    setBusyUi(false);
                });
                return;
            }

            genStart = System.currentTimeMillis();
            runOnUiThread(() -> {
                progressBar.setVisibility(View.VISIBLE);
                progressBar.setMax(steps);
                progressBar.setProgress(0);
                setStatus("生成開始… (" + size + "px/" + steps + "step)");
            });
            appendLogUi("生成開始: \"" + prompt + "\" " + size + "px " + steps + "step");

            try {
                byte[] rgb = manager.generate(prompt, negative, size, size, steps, 7.0f, -1, sampler);
                long ms = System.currentTimeMillis() - genStart;
                if (rgb == null) {
                    final String nativeErr = manager.getLastError();
                    appendLogUi("生成失敗" + (nativeErr.isEmpty() ? "" : ": " + nativeErr));
                    runOnUiThread(() -> {
                        setStatus("生成失敗" + (nativeErr.isEmpty() ? "" : " — " + nativeErr));
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

    // --- img2img: generate from a selected input image ---

    private void pickInitImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try {
            startActivityForResult(intent, REQ_PICK_INIT_IMAGE);
        } catch (ActivityNotFoundException e) {
            Toast.makeText(this, "画像ピッカーを開けませんでした", Toast.LENGTH_SHORT).show();
        }
    }

    private void loadInitImage(Uri uri) {
        worker.submit(() -> {
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                final Bitmap bmp = BitmapFactory.decodeStream(in);
                if (bmp == null) {
                    appendLogUi("入力画像の読込に失敗しました");
                    runOnUiThread(() -> Toast.makeText(this, "画像を読み込めませんでした", Toast.LENGTH_SHORT).show());
                    return;
                }
                appendLogUi("入力画像を選択 (" + bmp.getWidth() + "x" + bmp.getHeight() + ")");
                runOnUiThread(() -> {
                    initBitmap = bmp;
                    initThumb.setImageBitmap(bmp);
                    setStatus("入力画像を選択しました — 「画像から生成」できます");
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                appendLogUi("入力画像エラー: " + t);
                runOnUiThread(() -> {
                    Toast.makeText(this, "画像読込エラー", Toast.LENGTH_SHORT).show();
                    setBusyUi(false);
                });
            }
        });
    }

    private void generateImg2img() {
        if (manager == null || !manager.isModelLoaded()) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            return;
        }
        if (initBitmap == null) {
            Toast.makeText(this, "先に入力画像を選択してください", Toast.LENGTH_SHORT).show();
            pickInitImage();
            return;
        }
        final String rawPrompt = promptEdit.getText().toString().trim();
        final String negative = negativeEdit.getText().toString();
        final int steps = parseInt(stepsEdit.getText().toString(), 20);
        final int size = snapTo64(selectedSize);
        final float strength = clamp(parseFloat(strengthEdit.getText().toString(), 0.6f), 0.05f, 1.0f);
        final Bitmap src = initBitmap;

        setBusyUi(true);
        translatedPromptView.setVisibility(View.GONE);
        setStatus("準備中…");

        worker.submit(() -> {
            final String prompt = translateOrOriginal(rawPrompt);

            if (nsfwRestriction && isNsfwContent(prompt)) {
                runOnUiThread(() -> {
                    setStatus("生成中止: 性的コンテンツを検出");
                    Toast.makeText(this,
                            "コンテンツがモデレートされました（性的表現を検出）。生成を中止します。",
                            Toast.LENGTH_LONG).show();
                    setBusyUi(false);
                });
                return;
            }

            genStart = System.currentTimeMillis();
            runOnUiThread(() -> {
                progressBar.setVisibility(View.VISIBLE);
                progressBar.setMax(steps);
                progressBar.setProgress(0);
                setStatus("img2img 生成開始… (" + size + "px/" + steps + "step, strength=" + strength + ")");
            });
            appendLogUi("img2img生成: \"" + prompt + "\" " + size + "px " + steps + "step strength=" + strength);

            try {
                Bitmap scaled = Bitmap.createScaledBitmap(src, size, size, true);
                byte[] init = bitmapToRgb(scaled);
                byte[] rgb = manager.img2img(init, size, size, prompt, negative,
                        size, size, steps, 7.0f, strength, -1, sampler);
                long ms = System.currentTimeMillis() - genStart;
                if (rgb == null) {
                    final String nativeErr = manager.getLastError();
                    appendLogUi("img2img 失敗" + (nativeErr.isEmpty() ? "" : ": " + nativeErr));
                    runOnUiThread(() -> {
                        setStatus("img2img 失敗" + (nativeErr.isEmpty() ? "" : " — " + nativeErr));
                        progressBar.setVisibility(View.GONE);
                        setBusyUi(false);
                    });
                    return;
                }
                final Bitmap bmp = rgbToBitmap(rgb, size, size);
                final long seed = manager.getLastSeed();
                appendLogUi("img2img 完了 seed=" + seed + " " + (ms / 1000) + "s");
                runOnUiThread(() -> {
                    lastBitmap = bmp;
                    imageView.setImageBitmap(bmp);
                    progressBar.setProgress(progressBar.getMax());
                    setStatus("img2img 完了 seed=" + seed + " (" + (ms / 1000) + "s)");
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                appendLogUi("img2img エラー: " + t);
                runOnUiThread(() -> {
                    setStatus("img2img エラー: " + t.getClass().getSimpleName());
                    progressBar.setVisibility(View.GONE);
                    setBusyUi(false);
                });
            }
        });
    }

    // --- translation helpers (called from worker thread) ---

    private String translateOrOriginal(String rawPrompt) {
        String baseUrl = normalizeBaseUrl(llmBaseUrl);
        if (baseUrl.isEmpty()) return rawPrompt;
        String model = llmModel.isEmpty() ? "default" : llmModel;
        try {
            String result = requestLlmText(baseUrl, model, buildTranslationPrompt(rawPrompt), 30000);
            if (result == null || result.isEmpty()) {
                appendLogUi("翻訳スキップ: LLMからの応答が空");
                return rawPrompt;
            }
            String translated = result.trim();
            appendLogUi("翻訳: \"" + rawPrompt + "\" → \"" + translated + "\"");
            runOnUiThread(() -> {
                translatedPromptView.setText("英訳: " + translated);
                translatedPromptView.setVisibility(View.VISIBLE);
            });
            return translated;
        } catch (Throwable t) {
            appendLogUi("翻訳エラー（スキップ）: " + t.getMessage());
            return rawPrompt;
        }
    }

    private String buildTranslationPrompt(String input) {
        StringBuilder sb = new StringBuilder(
            "Translate the following text to English for use as a Stable Diffusion image generation prompt. "
            + "Output ONLY the English translation — no preamble, no explanation, no extra text.");
        if (nsfwRestriction) {
            sb.append(" Only if the input explicitly and unambiguously describes pornographic or sexually explicit content,"
                    + " output only the single word: NSFW."
                    + " Animals, people, food, nature, landscapes, and everyday subjects are NOT NSFW.");
        }
        sb.append('\n').append(input);
        return sb.toString();
    }

    private boolean isNsfwContent(String prompt) {
        if (prompt == null || prompt.isEmpty()) return false;
        if ("nsfw".equalsIgnoreCase(prompt.trim())) return true;
        String lower = prompt.toLowerCase(Locale.US);
        for (String word : NSFW_WORDS) {
            if (lower.contains(word)) return true;
        }
        return false;
    }

    // --- generation settings (weight precision / sampler / threads / NSFW) ---

    private void showGenSettingsDialog() {
        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(16);
        container.setPadding(pad, pad, pad, pad);

        final int[] selWtype = {wtype};
        TextView precLabel = new TextView(this);
        precLabel.setText("重み精度（量子化）");
        container.addView(precLabel);
        Button precBtn = new Button(this);
        precBtn.setText("精度: " + wtypeLabel(selWtype[0]));
        precBtn.setOnClickListener(v -> {
            selWtype[0] = nextWtype(selWtype[0]);
            precBtn.setText("精度: " + wtypeLabel(selWtype[0]));
        });
        container.addView(precBtn);
        TextView precHint = new TextView(this);
        precHint.setText("Q4_0=最速・省メモリ / Q8_0=高品質 / 元のまま。変更時はモデル再読込");
        precHint.setTextSize(11);
        container.addView(precHint);

        final int[] selSampler = {sampler};
        TextView samLabel = new TextView(this);
        samLabel.setText("サンプラー");
        samLabel.setPadding(0, dp(12), 0, 0);
        container.addView(samLabel);
        Button samBtn = new Button(this);
        samBtn.setText("サンプラー: " + samplerLabel(selSampler[0]));
        samBtn.setOnClickListener(v -> {
            selSampler[0] = (selSampler[0] == StableDiffusionNative.SAMPLE_DPMPP2M)
                    ? StableDiffusionNative.SAMPLE_EULER_A : StableDiffusionNative.SAMPLE_DPMPP2M;
            samBtn.setText("サンプラー: " + samplerLabel(selSampler[0]));
        });
        container.addView(samBtn);
        TextView samHint = new TextView(this);
        samHint.setText("DPM++ 2M は少ないステップ（〜15）で収束しやすい");
        samHint.setTextSize(11);
        container.addView(samHint);

        TextView thLabel = new TextView(this);
        thLabel.setText("スレッド数（0=自動）");
        thLabel.setPadding(0, dp(12), 0, 0);
        container.addView(thLabel);
        EditText thInput = new EditText(this);
        thInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        thInput.setText(String.valueOf(nThreads));
        container.addView(thInput);

        final boolean[] selNsfw = {nsfwRestriction};
        TextView nsfwLabel = new TextView(this);
        nsfwLabel.setText("性表現制限");
        nsfwLabel.setPadding(0, dp(12), 0, 0);
        container.addView(nsfwLabel);
        Button nsfwBtn = new Button(this);
        nsfwBtn.setText("性表現制限: " + (selNsfw[0] ? "ON" : "OFF"));
        nsfwBtn.setOnClickListener(v -> {
            selNsfw[0] = !selNsfw[0];
            nsfwBtn.setText("性表現制限: " + (selNsfw[0] ? "ON" : "OFF"));
        });
        container.addView(nsfwBtn);
        TextView nsfwHint = new TextView(this);
        nsfwHint.setText("ONの場合、LLMへ「性的表現があればNSFWとだけ返す」よう指示します。また生成プロンプトにNSFWワードが含まれていれば生成を中止します。");
        nsfwHint.setTextSize(11);
        container.addView(nsfwHint);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(container);

        new AlertDialog.Builder(this)
                .setTitle("生成設定")
                .setView(scroll)
                .setPositiveButton("保存", (d, w) -> {
                    int newThreads = Math.max(0, parseInt(thInput.getText().toString(), 0));
                    boolean reloadNeeded = (selWtype[0] != wtype) || (newThreads != nThreads);
                    wtype = selWtype[0];
                    sampler = selSampler[0];
                    nThreads = newThreads;
                    nsfwRestriction = selNsfw[0];
                    saveGenSettings();
                    if (reloadNeeded) {
                        reloadModel();
                    } else {
                        Toast.makeText(this, "生成設定を保存しました", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("キャンセル", null)
                .show();
    }

    private void showSizePicker() {
        int current = 0;
        for (int i = 0; i < SIZE_VALUES.length; i++) {
            if (SIZE_VALUES[i] == selectedSize) { current = i; break; }
        }
        final int[] picked = {current};
        new AlertDialog.Builder(this)
                .setTitle("サイズ選択")
                .setSingleChoiceItems(SIZE_LABELS, current, (d, which) -> picked[0] = which)
                .setPositiveButton("選択", (d, w) -> {
                    selectedSize = SIZE_VALUES[picked[0]];
                    sizeButton.setText(SIZE_LABELS[picked[0]]);
                    saveGenSettings();
                })
                .setNegativeButton("キャンセル", null)
                .show();
    }

    private static String sizeLabelFor(int size) {
        for (int i = 0; i < SIZE_VALUES.length; i++) {
            if (SIZE_VALUES[i] == size) return SIZE_LABELS[i];
        }
        return SIZE_LABELS[0];
    }

    /** Reload the cached model with the current weight precision / thread count. */
    private void reloadModel() {
        File cached = new File(getFilesDir(), MODEL_FILENAME);
        if (!cached.exists() || cached.length() == 0) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            return;
        }
        setBusyUi(true);
        setStatus("設定変更のため再読込中…" + wtypeSuffix());
        worker.submit(() -> {
            try {
                String err = manager.load(cached.getAbsolutePath(), nThreads, wtype);
                runOnUiThread(() -> {
                    if (err.isEmpty()) {
                        setStatus("再読込完了" + wtypeSuffix());
                        appendLog("再読込完了" + wtypeSuffix());
                    } else {
                        setStatus("再読込失敗: " + err);
                    }
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                appendLogUi("再読込エラー: " + t);
                runOnUiThread(() -> {
                    setStatus("再読込エラー: " + t.getClass().getSimpleName());
                    setBusyUi(false);
                });
            }
        });
    }

    private String wtypeSuffix() {
        return wtype == StableDiffusionNative.WTYPE_DEFAULT ? "" : " [" + wtypeLabel(wtype) + "]";
    }

    private static String wtypeLabel(int w) {
        if (w == StableDiffusionNative.WTYPE_Q4_0) return "Q4_0";
        if (w == StableDiffusionNative.WTYPE_Q8_0) return "Q8_0";
        return "元のまま";
    }

    private static int nextWtype(int w) {
        if (w == StableDiffusionNative.WTYPE_DEFAULT) return StableDiffusionNative.WTYPE_Q8_0;
        if (w == StableDiffusionNative.WTYPE_Q8_0) return StableDiffusionNative.WTYPE_Q4_0;
        return StableDiffusionNative.WTYPE_DEFAULT;
    }

    private static String samplerLabel(int s) {
        return s == StableDiffusionNative.SAMPLE_DPMPP2M ? "DPM++ 2M" : "Euler a";
    }

    // --- LLM API ---

    private String requestLlmText(String baseUrl, String model, String prompt, int timeoutMs) throws Exception {
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(baseUrl + "/api/generate").openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(timeoutMs);
            conn.setDoOutput(true);

            JSONObject req = new JSONObject();
            req.put("model", model);
            req.put("prompt", prompt);
            req.put("stream", false);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(req.toString().getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            String body = readStream(code >= 400 ? conn.getErrorStream() : conn.getInputStream());
            if (code >= 400) throw new Exception("HTTP " + code + ": " + body);

            JSONObject root = new JSONObject(body);
            String response = root.optString("response", "").trim();
            if (response.isEmpty()) {
                JSONObject message = root.optJSONObject("message");
                if (message != null) response = message.optString("content", "").trim();
            }
            return response;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    // --- LLM settings dialog ---

    private void showLlmSettingsDialog() {
        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(12);
        container.setPadding(pad, pad, pad, pad);

        Button launchBtn = new Button(this);
        launchBtn.setText("LLMを起動 (com.micklab.llama)");
        container.addView(launchBtn);

        GradientDrawable statusBg = new GradientDrawable();
        statusBg.setColor(0xFFFFF59D);
        statusBg.setCornerRadius(dp(4));
        TextView apiStatusView = new TextView(this);
        apiStatusView.setText("確認中…");
        apiStatusView.setTextColor(0xFF000000);
        apiStatusView.setPadding(dp(8), dp(6), dp(8), dp(6));
        apiStatusView.setBackground(statusBg);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.topMargin = dp(6);
        container.addView(apiStatusView, statusParams);

        TextView urlLabel = new TextView(this);
        urlLabel.setText("LLM API URL");
        urlLabel.setPadding(0, dp(10), 0, 0);
        container.addView(urlLabel);

        EditText urlInput = new EditText(this);
        urlInput.setHint(DEFAULT_LLM_BASE_URL);
        urlInput.setText(llmBaseUrl);
        container.addView(urlInput);

        launchBtn.setOnClickListener(v -> {
            String url = normalizeBaseUrl(urlInput.getText().toString());
            openLlmOrStore(url.isEmpty() ? DEFAULT_LLM_BASE_URL : url);
        });

        final String[] selectedModel = {llmModel};
        TextView modelView = new TextView(this);
        modelView.setText("モデル: " + (llmModel.isEmpty() ? "(未選択)" : llmModel));
        modelView.setPadding(0, dp(10), 0, 0);
        container.addView(modelView);

        Button modelBtn = new Button(this);
        modelBtn.setText("api/tags から選択");
        modelBtn.setOnClickListener(v -> {
            String url = normalizeBaseUrl(urlInput.getText().toString());
            if (url.isEmpty()) {
                Toast.makeText(this, "URLを入力してください", Toast.LENGTH_SHORT).show();
                return;
            }
            checkAndUpdateApiStatus(url, apiStatusView, statusBg);
            loadModelsAndShowChooser(url, selectedModel, modelView);
        });
        container.addView(modelBtn);

        ScrollView scrollView = new ScrollView(this);
        scrollView.addView(container);

        new AlertDialog.Builder(this)
                .setTitle("LLM翻訳設定")
                .setView(scrollView)
                .setPositiveButton("保存", (dialog, which) -> {
                    llmBaseUrl = normalizeBaseUrl(urlInput.getText().toString());
                    if (llmBaseUrl.isEmpty()) llmBaseUrl = DEFAULT_LLM_BASE_URL;
                    llmModel = selectedModel[0] == null ? "" : selectedModel[0].trim();
                    saveLlmSettings();
                    Toast.makeText(this, "LLM設定を保存しました", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("キャンセル", null)
                .show();

        checkAndUpdateApiStatus(normalizeBaseUrl(llmBaseUrl), apiStatusView, statusBg);
    }

    private void checkAndUpdateApiStatus(String baseUrl, TextView statusView, GradientDrawable bg) {
        if (baseUrl == null || baseUrl.isEmpty()) return;
        new Thread(() -> {
            boolean available = false;
            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(baseUrl + "/api/tags").openConnection();
                conn.setConnectTimeout(5000);
                conn.setReadTimeout(5000);
                available = conn.getResponseCode() < 400;
                conn.disconnect();
            } catch (Exception ignore) {}
            final boolean ok = available;
            runOnUiThread(() -> {
                statusView.setText(ok ? "LLM API 利用可能" : "LLM API 利用不可");
                bg.setColor(ok ? 0xFF81C784 : 0xFFFFF59D);
            });
        }).start();
    }

    private void loadModelsAndShowChooser(String baseUrl, String[] selectedModel, TextView modelView) {
        modelView.setText("モデル: 読み込み中…");
        new Thread(() -> {
            try {
                List<String> models = fetchModelNames(baseUrl);
                runOnUiThread(() -> {
                    if (models.isEmpty()) {
                        modelView.setText("モデル: " + labelOf(selectedModel[0]));
                        Toast.makeText(this, "モデルが見つかりませんでした", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    int checked = models.indexOf(selectedModel[0]);
                    if (checked < 0) checked = 0;
                    final int[] picked = {checked};
                    ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                            android.R.layout.simple_list_item_single_choice, models);
                    ListView listView = new ListView(this);
                    listView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
                    listView.setAdapter(adapter);
                    listView.setItemChecked(checked, true);
                    listView.setOnItemClickListener((parent, view, which, id) -> picked[0] = which);
                    new AlertDialog.Builder(this)
                            .setTitle("モデル選択")
                            .setView(listView)
                            .setPositiveButton("選択", (dialog, which) -> {
                                selectedModel[0] = models.get(picked[0]);
                                modelView.setText("モデル: " + selectedModel[0]);
                            })
                            .setNegativeButton("キャンセル", (dialog, which) ->
                                    modelView.setText("モデル: " + labelOf(selectedModel[0])))
                            .show();
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    modelView.setText("モデル: " + labelOf(selectedModel[0]));
                    Toast.makeText(this, "モデル取得に失敗: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                });
            }
        }).start();
    }

    private String labelOf(String model) {
        return (model == null || model.isEmpty()) ? "(未選択)" : model;
    }

    private List<String> fetchModelNames(String baseUrl) throws Exception {
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(baseUrl + "/api/tags").openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(15000);
            int code = conn.getResponseCode();
            String body = readStream(code >= 400 ? conn.getErrorStream() : conn.getInputStream());
            if (code >= 400) throw new Exception("HTTP " + code);
            JSONObject root = new JSONObject(body);
            JSONArray arr = root.optJSONArray("models");
            List<String> names = new ArrayList<>();
            if (arr != null) {
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject m = arr.optJSONObject(i);
                    if (m == null) continue;
                    String name = m.optString("name", "").trim();
                    if (name.isEmpty()) name = m.optString("model", "").trim();
                    if (!name.isEmpty() && !names.contains(name)) names.add(name);
                }
            }
            return names;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private void openLlmOrStore(String baseUrl) {
        Intent launchIntent = getPackageManager().getLaunchIntentForPackage(LLM_PACKAGE);
        if (launchIntent == null) {
            try {
                startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(LLM_PLAY_STORE_URL)));
            } catch (ActivityNotFoundException e) {
                Toast.makeText(this, "Google Playを開けませんでした", Toast.LENGTH_SHORT).show();
            }
            return;
        }
        try {
            Intent serviceIntent = new Intent();
            serviceIntent.setClassName(LLM_PACKAGE, LLM_SERVICE_CLASS);
            serviceIntent.setAction(LLM_SERVICE_ACTION);
            try {
                Uri uri = Uri.parse(baseUrl);
                int port = uri.getPort();
                serviceIntent.putExtra("port", port > 0 ? port : 11434);
            } catch (Exception ignore) {}
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }
            } catch (Exception ignore) {}
            launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(launchIntent);
        } catch (ActivityNotFoundException e) {
            try {
                startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(LLM_PLAY_STORE_URL)));
            } catch (ActivityNotFoundException ignore) {}
        }
    }

    // --- SharedPreferences ---

    private void loadSettings() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        llmBaseUrl = prefs.getString(PREFS_LLM_URL, DEFAULT_LLM_BASE_URL);
        llmModel = prefs.getString(PREFS_LLM_MODEL, "");
        wtype = prefs.getInt(PREFS_WTYPE, StableDiffusionNative.WTYPE_DEFAULT);
        sampler = prefs.getInt(PREFS_SAMPLER, StableDiffusionNative.SAMPLE_DPMPP2M);
        nThreads = prefs.getInt(PREFS_THREADS, 0);
        nsfwRestriction = prefs.getBoolean(PREFS_NSFW, false);
        selectedSize = prefs.getInt(PREFS_SIZE, 512);
    }

    private void saveLlmSettings() {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                .putString(PREFS_LLM_URL, llmBaseUrl)
                .putString(PREFS_LLM_MODEL, llmModel)
                .apply();
    }

    private void saveGenSettings() {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                .putInt(PREFS_WTYPE, wtype)
                .putInt(PREFS_SAMPLER, sampler)
                .putInt(PREFS_THREADS, nThreads)
                .putBoolean(PREFS_NSFW, nsfwRestriction)
                .putInt(PREFS_SIZE, selectedSize)
                .apply();
    }

    // --- image save ---

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
        boolean modelReady = manager != null && manager.isModelLoaded();
        generateButton.setEnabled(!busy && modelReady);
        img2imgButton.setEnabled(!busy && modelReady);
        pickInitButton.setEnabled(!busy);
    }

    private void setStatus(String s) {
        if (getActionBar() != null) getActionBar().setSubtitle(s);
    }

    private void setStatusUi(String s) {
        runOnUiThread(() -> setStatus(s));
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

    private static int parseInt(String s, int def) {
        try {
            int v = Integer.parseInt(s.trim());
            return v > 0 ? v : def;
        } catch (Exception e) {
            return def;
        }
    }

    private static float parseFloat(String s, float def) {
        try {
            return Float.parseFloat(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /** SD1.5 needs dimensions that are multiples of 64; snap (min 64). */
    private static int snapTo64(int v) {
        int snapped = (v / 64) * 64;
        return snapped < 64 ? 64 : snapped;
    }

    /** Pack a Bitmap into tightly-packed RGB (w*h*3) bytes for the native img2img call. */
    private static byte[] bitmapToRgb(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int[] px = new int[w * h];
        bmp.getPixels(px, 0, w, 0, 0, w, h);
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++) {
            int c = px[i];
            rgb[i * 3]     = (byte) ((c >> 16) & 0xff);
            rgb[i * 3 + 1] = (byte) ((c >> 8) & 0xff);
            rgb[i * 3 + 2] = (byte) (c & 0xff);
        }
        return rgb;
    }

    private String normalizeBaseUrl(String url) {
        if (url == null) return "";
        String s = url.trim();
        while (s.endsWith("/")) s = s.substring(0, s.length() - 1);
        return s;
    }

    private String readStream(InputStream in) throws Exception {
        if (in == null) return "";
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) sb.append(line).append('\n');
        }
        return sb.toString().trim();
    }

    private void appendLog(String msg) {
        if (msg == null) return;
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
