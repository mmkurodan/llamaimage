package com.micklab.llamaimage;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.Gravity;
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {

    private static final int REQ_PICK_INIT_IMAGE = 1002;

    private static final String PREFS_NAME = "llamaimage_prefs";
    private static final String PREFS_LLM_URL = "llm_base_url";
    private static final String PREFS_LLM_MODEL = "llm_model";
    private static final String PREFS_WTYPE = "wtype";
    private static final String PREFS_SAMPLER = "sampler";
    private static final String PREFS_CFG = "cfg_scale";
    private static final String PREFS_THREADS = "threads";
    private static final String PREFS_NSFW = "nsfw_restriction";
    private static final String PREFS_SIZE = "image_size";
    private static final String PREFS_TAESD_LEGACY = "taesd_path";   // old versions: picked file path
    private static final String PREFS_USE_TAESD = "use_taesd";
    private static final String PREFS_VAE_TILING = "vae_tiling";
    private static final String PREFS_MODEL_ID = "model_id";
    private static final String PREFS_PENDING_MODEL = "pending_model_id";  // select after download
    private static final String PREFS_STEPS = "steps";
    private static final String PREFS_STRENGTH = "strength";
    private static final String PREFS_SEED = "seed";
    private static final String PREFS_PROMPT = "prompt";
    private static final String PREFS_NEGATIVE = "negative";

    private static final int[] SIZE_VALUES = {512, 448, 384, 256};
    private static final String[] SIZE_LABELS = {"512px（標準）", "448px", "384px（高速）", "256px（最速）"};
    private static final int MAX_STEPS = 50;
    private static final int MAX_LOG_CHARS = 20000;
    private static final long LOW_RAM_BYTES = 6L * 1024 * 1024 * 1024;

    private static final String DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434";
    private static final String LLM_PACKAGE = "com.micklab.llama";
    private static final String LLM_SERVICE_CLASS = "com.micklab.llama.OllamaForegroundService";
    private static final String LLM_SERVICE_ACTION = "com.micklab.llama.START_SERVICE";
    private static final String LLM_PLAY_STORE_URL =
            "https://play.google.com/store/apps/details?id=com.micklab.llama";

    private static final int MENU_MODEL = 1, MENU_GENERATE = 2, MENU_SAVE = 3, MENU_SETTINGS = 4, MENU_LLM = 5;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private volatile SdModelManager manager;
    private ModelStore store;
    private PromptTranslator translator;

    // --- settings (persisted) ---
    private String llmBaseUrl = DEFAULT_LLM_BASE_URL;
    private String llmModel = "";
    private String modelId = "";
    private int wtype = StableDiffusionNative.WTYPE_DEFAULT;
    private int sampler = StableDiffusionNative.SAMPLE_LCM;
    private float cfgScale = 1.0f;
    private int nThreads = 0;                 // 0 = auto (performance cores)
    private boolean nsfwRestriction = false;
    private boolean useTaesd = false;
    private boolean vaeTiling = false;
    private int selectedSize = 512;

    // --- UI state (main thread) ---
    private boolean nativeReady = false;
    private boolean loading = false;          // model verify/load in progress
    private boolean generating = false;
    private boolean finalizing = false;       // download verification in progress
    private boolean started = false;          // between onStart and onStop
    private boolean chooserShownThisLaunch = false;
    private ModelCatalog.ModelSpec monitoredDownload;
    private long dlLastBytes = -1;
    private long dlLastTime = 0;
    private double dlSpeed = 0;
    private long genStartMs, samplingStartMs, samplingEndMs;
    private long lastDecodeMs = 0;            // measured on the previous generation (ETA)
    private int sampleSteps = 0;

    // --- views ---
    private LinearLayout rootBody;
    private TextView modelText;
    private LinearLayout downloadRow;
    private ProgressBar downloadBar;
    private TextView downloadText;
    private Button downloadCancelButton;
    private EditText promptEdit;
    private TextView translatedPromptView;
    private EditText negativeEdit;
    private EditText stepsEdit;
    private Button sizeButton;
    private EditText seedEdit;
    private EditText strengthEdit;
    private Button generateButton;
    private Button cancelButton;
    private Button pickInitButton;
    private Button img2imgButton;
    private ImageView initThumb;
    private ProgressBar progressBar;
    private TextView statusText;
    private ImageView imageView;
    private TextView logText;
    private Bitmap lastBitmap;
    private long lastSeed = -1;
    private Bitmap initBitmap;                // selected img2img input (square, cropped)

    private final SdModelManager.Listener genListener = new SdModelManager.Listener() {
        @Override public void onProgress(int phase, int step, int steps) {
            final long now = SystemClock.elapsedRealtime();
            runOnUiThread(() -> handleProgress(phase, step, steps, now));
        }

        @Override public void onPreview(byte[] rgb, int w, int h) {
            final Bitmap bmp = rgbToBitmap(rgb, w, h);
            runOnUiThread(() -> {
                if (generating) imageView.setImageBitmap(bmp);
            });
        }
    };

    private final Runnable pollTask = this::pollDownload;

    // =====================================================================================
    // Lifecycle
    // =====================================================================================

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_HIDDEN
                | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        store = new ModelStore(this);
        loadSettings();
        setContentView(buildUi());
        rootBody.requestFocus();

        generateButton.setOnClickListener(v -> startGeneration(false));
        img2imgButton.setOnClickListener(v -> startGeneration(true));
        pickInitButton.setOnClickListener(v -> pickInitImage());
        cancelButton.setOnClickListener(v -> cancelGeneration());
        downloadCancelButton.setOnClickListener(v -> confirmCancelDownload());
        imageView.setOnLongClickListener(v -> {
            if (lastSeed >= 0) {
                seedEdit.setText(String.valueOf(lastSeed));
                Toast.makeText(this, "シード " + lastSeed + " を固定しました（-1 でランダムに戻ります）",
                        Toast.LENGTH_SHORT).show();
            }
            return true;
        });

        refreshControls();
        setStatus("ネイティブ初期化中…");
        worker.submit(this::initNative);
    }

    @Override
    protected void onStart() {
        super.onStart();
        started = true;
        if (nativeReady) checkDownloads();
    }

    @Override
    protected void onStop() {
        super.onStop();
        started = false;
        ui.removeCallbacks(pollTask);
        persistInputs();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        ui.removeCallbacksAndMessages(null);
        SdModelManager m = manager;
        if (m != null) m.clearListener(genListener);
        worker.shutdown();   // queued work finishes; native calls cannot be interrupted anyway
    }

    private void initNative() {
        try {
            File log = new File(getFilesDir(), "sd.log");
            if (log.length() > 2_000_000) {
                log.renameTo(new File(getFilesDir(), "sd.log.1"));
            }
            SdModelManager m = SdModelManager.get();
            m.setLogPath(log.getAbsolutePath());
            m.setListener(genListener);
            appendLogUi("ネイティブ準備完了: lib" + m.libraryName() + ".so / 自動スレッド数 "
                    + CpuInfo.defaultThreads());
            appendLogUi(m.systemInfo().trim());
            manager = m;
        } catch (Throwable t) {
            appendLogUi("ネイティブ初期化失敗: " + t);
            appendLogUi("この端末の ABI が arm64-v8a でない可能性があります（エミュレータ等）。");
            runOnUiThread(() -> setStatus("ネイティブ初期化失敗: " + t.getClass().getSimpleName()));
            return;
        }

        // Files copied by the old file-picker flow become catalog entries (no re-download).
        try {
            ModelCatalog.ModelSpec adopted = store.adoptLegacyFiles(
                    pct -> setStatusUi("以前のモデルを確認中… " + pct + "%"));
            if (adopted != null) {
                appendLogUi("以前のモデルを引き継ぎ: " + adopted.title);
                runOnUiThread(() -> {
                    if (modelId.isEmpty() || ModelCatalog.findModel(modelId) == null) {
                        // Old defaults (LCM/CFG1/6step) were wrong for e.g. vanilla SD1.5.
                        selectModel(adopted, false);
                    }
                });
            }
        } catch (Throwable t) {
            appendLogUi("以前のモデルの引き継ぎに失敗: " + t);
        }

        runOnUiThread(() -> {
            nativeReady = true;
            refreshControls();
            startupModelFlow();
        });
    }

    /** Decide what to do once native is ready: resume a download, load a model, or ask. */
    private void startupModelFlow() {
        boolean downloading = checkDownloads();
        if (finalizing) {
            return;   // a finished download is being verified; it selects + loads when done
        }
        ModelCatalog.ModelSpec spec = currentSpec();
        if (spec == null || store.installedFile(spec) == null) {
            // Fall back to any installed model.
            spec = null;
            for (ModelCatalog.ModelSpec s : allSpecs()) {
                if (store.installedFile(s) != null) {
                    spec = s;
                    break;
                }
            }
            if (spec != null) selectModel(spec, true);
        }
        if (spec != null) {
            ensureModelLoaded();   // a running download of another model continues meanwhile
        } else if (downloading) {
            setStatus("ダウンロード中… 完了後に自動で読み込みます");
        } else {
            setStatus("モデルを選択してください（初回のみダウンロード）");
            showModelChooser();
        }
    }

    // =====================================================================================
    // UI
    // =====================================================================================

    private View buildUi() {
        int pad = dp(12);
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(pad, pad, pad, pad);
        body.setFocusable(true);
        body.setFocusableInTouchMode(true);
        rootBody = body;

        // Model row: current model + change button.
        LinearLayout modelRow = new LinearLayout(this);
        modelRow.setOrientation(LinearLayout.HORIZONTAL);
        modelRow.setGravity(Gravity.CENTER_VERTICAL);
        modelText = new TextView(this);
        modelText.setTextSize(13);
        modelText.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button changeModel = new Button(this);
        changeModel.setText("モデル");
        changeModel.setOnClickListener(v -> showModelChooser());
        modelRow.addView(modelText);
        modelRow.addView(changeModel);
        body.addView(modelRow);

        // Download row (visible only while a model downloads).
        downloadRow = new LinearLayout(this);
        downloadRow.setOrientation(LinearLayout.VERTICAL);
        downloadRow.setVisibility(View.GONE);
        downloadBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        downloadBar.setMax(1000);
        downloadText = new TextView(this);
        downloadText.setTextSize(12);
        downloadCancelButton = new Button(this);
        downloadCancelButton.setText("ダウンロード中止");
        downloadRow.addView(downloadBar);
        downloadRow.addView(downloadText);
        downloadRow.addView(downloadCancelButton);
        body.addView(downloadRow);

        promptEdit = new EditText(this);
        promptEdit.setHint("プロンプト（日本語・英語）");
        promptEdit.setMinLines(2);
        body.addView(promptEdit);

        translatedPromptView = new TextView(this);
        translatedPromptView.setTextSize(11);
        translatedPromptView.setTextColor(0xFF888888);
        translatedPromptView.setPadding(0, dp(2), 0, dp(4));
        translatedPromptView.setVisibility(View.GONE);
        body.addView(translatedPromptView);

        negativeEdit = new EditText(this);
        negativeEdit.setHint("ネガティブプロンプト（CFG>1 のとき有効）");
        body.addView(negativeEdit);

        LinearLayout paramRow = new LinearLayout(this);
        paramRow.setOrientation(LinearLayout.HORIZONTAL);
        stepsEdit = new EditText(this);
        stepsEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        sizeButton = new Button(this);
        sizeButton.setOnClickListener(v -> showSizePicker());
        seedEdit = new EditText(this);
        seedEdit.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
        paramRow.addView(labeled("ステップ数", stepsEdit));
        paramRow.addView(labeled("サイズ", sizeButton));
        paramRow.addView(labeled("シード（-1=ランダム）", seedEdit));
        body.addView(paramRow);

        generateButton = new Button(this);
        generateButton.setText("生成");
        generateButton.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        body.addView(generateButton);

        cancelButton = new Button(this);
        cancelButton.setText("中止");
        cancelButton.setVisibility(View.GONE);
        cancelButton.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        body.addView(cancelButton);

        // img2img row: input-image thumbnail + pick + strength + generate-from-image.
        LinearLayout img2imgRow = new LinearLayout(this);
        img2imgRow.setOrientation(LinearLayout.HORIZONTAL);
        img2imgRow.setGravity(Gravity.CENTER_VERTICAL);
        initThumb = new ImageView(this);
        initThumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams thumbParams = new LinearLayout.LayoutParams(dp(48), dp(48));
        thumbParams.rightMargin = dp(8);
        initThumb.setLayoutParams(thumbParams);
        pickInitButton = new Button(this);
        pickInitButton.setText("入力画像");
        strengthEdit = new EditText(this);
        strengthEdit.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        img2imgButton = new Button(this);
        img2imgButton.setText("画像から生成");
        img2imgRow.addView(initThumb);
        img2imgRow.addView(labeled("img2img", pickInitButton));
        img2imgRow.addView(labeled("変化量 0.05〜1", strengthEdit));
        img2imgRow.addView(labeled(" ", img2imgButton));
        body.addView(img2imgRow);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setVisibility(View.GONE);
        body.addView(progressBar);

        statusText = new TextView(this);
        statusText.setTextSize(13);
        statusText.setPadding(0, dp(2), 0, dp(6));
        body.addView(statusText);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        body.addView(imageView);

        TextView guide = new TextView(this);
        guide.setText("日本語のプロンプトは LLM（LLM設定）で英訳してから生成します。"
                + "モデルを選ぶと推奨のサンプラー／ステップ数／CFG が適用されます（「生成設定」で変更可）。"
                + "画像を長押しするとそのシードを固定できます。");
        guide.setTextSize(11);
        guide.setTextColor(0xFF777777);
        guide.setPadding(0, dp(8), 0, dp(4));
        body.addView(guide);

        logText = new TextView(this);
        logText.setTextSize(11);
        body.addView(logText);

        // Restore persisted inputs.
        SharedPreferences p = prefs();
        promptEdit.setText(p.getString(PREFS_PROMPT, "a photo of a cat"));
        negativeEdit.setText(p.getString(PREFS_NEGATIVE, ""));
        stepsEdit.setText(String.valueOf(p.getInt(PREFS_STEPS, 6)));
        seedEdit.setText(String.valueOf(p.getLong(PREFS_SEED, -1)));
        strengthEdit.setText(String.valueOf(p.getFloat(PREFS_STRENGTH, 0.6f)));
        sizeButton.setText(sizeLabelFor(selectedSize));
        updateModelText();

        ScrollView scroll = new ScrollView(this);
        scroll.addView(body);
        return scroll;
    }

    private LinearLayout labeled(String label, View field) {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        TextView t = new TextView(this);
        t.setText(label);
        t.setTextSize(11);
        t.setTextColor(0xFF666666);
        col.addView(t);
        col.addView(field, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        col.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        return col;
    }

    private boolean isBusy() {
        return loading || generating || finalizing;
    }

    /** Single place that enables/disables controls for the current state. */
    private void refreshControls() {
        boolean modelReady = manager != null && manager.isModelLoaded();
        boolean idle = nativeReady && !isBusy();
        generateButton.setEnabled(idle && modelReady);
        img2imgButton.setEnabled(idle && modelReady);
        pickInitButton.setEnabled(!generating);
        cancelButton.setVisibility(generating ? View.VISIBLE : View.GONE);
        invalidateOptionsMenu();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, MENU_MODEL, 0, "モデル").setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        menu.add(0, MENU_GENERATE, 1, "生成").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        menu.add(0, MENU_SAVE, 2, "画像を保存").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        menu.add(0, MENU_SETTINGS, 3, "生成設定").setShowAsAction(MenuItem.SHOW_AS_ACTION_NEVER);
        menu.add(0, MENU_LLM, 4, "LLM設定").setShowAsAction(MenuItem.SHOW_AS_ACTION_NEVER);
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean modelReady = manager != null && manager.isModelLoaded();
        boolean idle = nativeReady && !isBusy();
        setMenuEnabled(menu, MENU_GENERATE, idle && modelReady);
        setMenuEnabled(menu, MENU_SETTINGS, idle);
        setMenuEnabled(menu, MENU_SAVE, lastBitmap != null);
        return super.onPrepareOptionsMenu(menu);
    }

    private static void setMenuEnabled(Menu menu, int id, boolean enabled) {
        MenuItem item = menu.findItem(id);
        if (item != null) item.setEnabled(enabled);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case MENU_MODEL: showModelChooser(); return true;
            case MENU_GENERATE: startGeneration(false); return true;
            case MENU_SAVE: saveImage(); return true;
            case MENU_SETTINGS: showGenSettingsDialog(); return true;
            case MENU_LLM: showLlmSettingsDialog(); return true;
            default: return super.onOptionsItemSelected(item);
        }
    }

    // =====================================================================================
    // Model selection / download / load
    // =====================================================================================

    private List<ModelCatalog.ModelSpec> allSpecs() {
        List<ModelCatalog.ModelSpec> list = new ArrayList<>(ModelCatalog.MODELS);
        ModelCatalog.ModelSpec imported = store.importedModel();
        if (imported != null) list.add(imported);
        return list;
    }

    private ModelCatalog.ModelSpec currentSpec() {
        if (ModelCatalog.IMPORTED_ID.equals(modelId)) return store.importedModel();
        return ModelCatalog.findModel(modelId);
    }

    private void updateModelText() {
        ModelCatalog.ModelSpec s = currentSpec();
        modelText.setText(s == null ? "モデル: 未選択" : "モデル: " + s.title);
    }

    /** Model chooser: one card per model with download / use / delete actions. */
    private void showModelChooser() {
        if (!nativeReady) {
            Toast.makeText(this, "ネイティブ初期化が完了していません（ログ参照）", Toast.LENGTH_SHORT).show();
            return;
        }
        chooserShownThisLaunch = true;
        LinearLayout list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        list.setPadding(dp(16), dp(8), dp(16), dp(8));
        final AlertDialog[] dialog = new AlertDialog[1];

        ModelCatalog.ModelSpec active = currentSpec();
        boolean loadedNow = manager != null && manager.isModelLoaded();
        for (ModelCatalog.ModelSpec s : allSpecs()) {
            boolean installed = store.installedFile(s) != null;
            ModelStore.DownloadState st = s.downloadable() ? store.queryDownload(s) : null;
            boolean downloadingThis = st != null && st.active();
            boolean isActive = active != null && active.id.equals(s.id);

            LinearLayout card = new LinearLayout(this);
            card.setOrientation(LinearLayout.VERTICAL);
            card.setPadding(dp(12), dp(10), dp(12), dp(10));
            GradientDrawable bg = new GradientDrawable();
            bg.setCornerRadius(dp(6));
            bg.setStroke(dp(1), isActive ? 0xFF3F51B5 : 0xFFCCCCCC);
            bg.setColor(isActive ? 0xFFE8EAF6 : 0xFFFFFFFF);
            card.setBackground(bg);
            LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            cardLp.bottomMargin = dp(10);

            TextView title = new TextView(this);
            title.setText(s.title + (isActive && loadedNow ? "（使用中）" : ""));
            title.setTypeface(Typeface.DEFAULT_BOLD);
            title.setTextSize(15);
            card.addView(title);

            TextView info = new TextView(this);
            String state = installed ? "✓ 端末に保存済み"
                    : downloadingThis ? "ダウンロード中…"
                    : "未ダウンロード（初回のみ " + s.sizeLabel() + "）";
            info.setText(state + "\n" + s.sizeLabel() + " ・ 推奨: " + s.recommendedLabel());
            info.setTextSize(12);
            card.addView(info);

            TextView desc = new TextView(this);
            desc.setText(s.description);
            desc.setTextSize(12);
            desc.setTextColor(0xFF666666);
            card.addView(desc);

            LinearLayout buttons = new LinearLayout(this);
            buttons.setOrientation(LinearLayout.HORIZONTAL);
            buttons.setGravity(Gravity.END);
            if (installed) {
                if (!(isActive && loadedNow)) {
                    Button use = new Button(this);
                    use.setText("このモデルを使う");
                    use.setEnabled(!isBusy());
                    use.setOnClickListener(v -> {
                        dialog[0].dismiss();
                        selectModel(s, true);
                        ensureModelLoaded();
                    });
                    buttons.addView(use);
                }
                Button del = new Button(this);
                del.setText("削除");
                del.setEnabled(!isBusy());
                del.setOnClickListener(v -> {
                    dialog[0].dismiss();
                    confirmDeleteModel(s);
                });
                buttons.addView(del);
            } else if (downloadingThis) {
                Button stop = new Button(this);
                stop.setText("ダウンロード中止");
                stop.setOnClickListener(v -> {
                    dialog[0].dismiss();
                    confirmCancelDownload();
                });
                buttons.addView(stop);
            } else if (s.downloadable()) {
                Button dl = new Button(this);
                dl.setText("ダウンロードして使う");
                dl.setEnabled(monitoredDownload == null);
                dl.setOnClickListener(v -> {
                    dialog[0].dismiss();
                    confirmDownload(s);
                });
                buttons.addView(dl);
            }
            card.addView(buttons);
            list.addView(card, cardLp);
        }

        TextView where = new TextView(this);
        File ext = store.externalDir();
        where.setText("ダウンロード元は Hugging Face（URL はアプリに内蔵）。保存先: "
                + (ext != null ? ext.getAbsolutePath() : "(利用不可)")
                + "\n高速VAEデコード（TAESD, 既定 ON）の切替は「生成設定」から。");
        where.setTextSize(11);
        where.setTextColor(0xFF777777);
        list.addView(where);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(list);
        dialog[0] = new AlertDialog.Builder(this)
                .setTitle("画像生成モデル")
                .setView(scroll)
                .setNegativeButton("閉じる", null)
                .show();
    }

    /** Persist the selection and (optionally) apply the model's recommended settings. */
    private void selectModel(ModelCatalog.ModelSpec spec, boolean announce) {
        boolean changed = !spec.id.equals(modelId);
        modelId = spec.id;
        prefs().edit().putString(PREFS_MODEL_ID, modelId).apply();
        updateModelText();
        if (spec.hasRecommendedSettings() && (changed || !announce)) {
            sampler = spec.sampler;
            cfgScale = spec.cfgScale;
            stepsEdit.setText(String.valueOf(spec.steps));
            saveGenSettings();
            persistInputs();
            appendLog("推奨設定を適用: " + spec.recommendedLabel());
            if (announce) {
                Toast.makeText(this, spec.title + "\n推奨設定: " + spec.recommendedLabel(),
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void confirmDownload(ModelCatalog.ModelSpec spec) {
        new AlertDialog.Builder(this)
                .setTitle("モデルのダウンロード")
                .setMessage("「" + spec.title + "」(" + spec.sizeLabel() + ") をダウンロードします。"
                        + (useTaesd && store.installedTaesd() == null
                            ? "（高速デコード用 TAESD " + ModelCatalog.TAESD.sizeLabel() + " も取得）" : "")
                        + "\n初回のみ必要です。モバイル通信の場合は通信量にご注意ください（Wi-Fi 推奨）。\n"
                        + "アプリを閉じてもダウンロードは続きます。")
                .setPositiveButton("ダウンロード", (d, w) -> startDownload(spec, true))
                .setNegativeButton("キャンセル", null)
                .show();
    }

    private void startDownload(ModelCatalog.ModelSpec spec, boolean selectWhenDone) {
        try {
            store.startDownload(spec);
        } catch (IOException e) {
            new AlertDialog.Builder(this).setTitle("ダウンロードできません")
                    .setMessage(e.getMessage()).setPositiveButton("OK", null).show();
            return;
        }
        if (selectWhenDone) {
            prefs().edit().putString(PREFS_PENDING_MODEL, spec.id).apply();
        }
        appendLog("ダウンロード開始: " + spec.title + " (" + spec.sizeLabel() + ")");
        if (spec != ModelCatalog.TAESD && useTaesd && store.installedTaesd() == null) {
            ModelStore.DownloadState t = store.queryDownload(ModelCatalog.TAESD);
            if (t == null || !t.active()) {
                try {
                    store.startDownload(ModelCatalog.TAESD);   // 9.8 MB; finalized after the model
                } catch (IOException e) {
                    appendLog("TAESD のダウンロードを開始できません: " + e.getMessage());
                }
            }
        }
        monitor(spec);
    }

    private void monitor(ModelCatalog.ModelSpec spec) {
        monitoredDownload = spec;
        dlLastBytes = -1;
        dlSpeed = 0;
        downloadRow.setVisibility(View.VISIBLE);
        downloadText.setText(spec.title + " を準備中…");
        downloadBar.setIndeterminate(true);
        ui.removeCallbacks(pollTask);
        if (started) ui.post(pollTask);
    }

    /**
     * Looks at every recorded download: finishes completed ones, reports failures and resumes
     * monitoring a running one. Returns true if a download is running or being finalized.
     */
    private boolean checkDownloads() {
        if (finalizing) return true;
        // TAESD first: when both finish together the model is then loaded once, with TAESD.
        List<ModelCatalog.ModelSpec> specs = new ArrayList<>();
        specs.add(ModelCatalog.TAESD);
        specs.addAll(ModelCatalog.MODELS);
        ModelCatalog.ModelSpec running = null;
        for (ModelCatalog.ModelSpec s : specs) {
            ModelStore.DownloadState st = store.queryDownload(s);
            if (st == null) continue;
            if (st.status == DownloadManager.STATUS_SUCCESSFUL) {
                finalizeDownload(s, st);
                return true;
            }
            if (st.status == DownloadManager.STATUS_FAILED) {
                reportDownloadFailure(s, st);
                continue;
            }
            // Show a model download rather than the tiny TAESD one when both are running.
            if (st.active() && (running == null || running == ModelCatalog.TAESD)) running = s;
        }
        if (running != null) {
            monitor(running);
            return true;
        }
        hideDownloadRow();
        return false;
    }

    private void pollDownload() {
        ModelCatalog.ModelSpec spec = monitoredDownload;
        if (spec == null || !started) return;
        ModelStore.DownloadState st = store.queryDownload(spec);
        if (st == null) {
            appendLog("ダウンロードが取り消されました: " + spec.title);
            hideDownloadRow();
            if (manager != null && !manager.isModelLoaded() && !isBusy()) startupModelFlow();
            return;
        }
        switch (st.status) {
            case DownloadManager.STATUS_SUCCESSFUL:
                finalizeDownload(spec, st);
                return;
            case DownloadManager.STATUS_FAILED:
                reportDownloadFailure(spec, st);
                return;
            default:
                break;
        }
        long now = SystemClock.elapsedRealtime();
        if (dlLastBytes >= 0 && now > dlLastTime) {
            double inst = (st.downloaded - dlLastBytes) * 1000.0 / (now - dlLastTime);
            dlSpeed = dlSpeed <= 0 ? inst : dlSpeed * 0.7 + inst * 0.3;
        }
        dlLastBytes = st.downloaded;
        dlLastTime = now;
        long total = st.total > 0 ? st.total : spec.sizeBytes;
        String paused = ModelStore.reasonText(st);
        StringBuilder sb = new StringBuilder(spec.title).append(": ");
        if (st.downloaded > 0 && total > 0) {
            int permille = (int) (st.downloaded * 1000 / total);
            downloadBar.setIndeterminate(false);
            downloadBar.setProgress(permille);
            sb.append(permille / 10).append("% (").append(ModelCatalog.formatSize(st.downloaded))
                    .append(" / ").append(ModelCatalog.formatSize(total)).append(")");
            if (dlSpeed > 0 && st.status == DownloadManager.STATUS_RUNNING) {
                long etaSec = (long) ((total - st.downloaded) / dlSpeed);
                sb.append(String.format(Locale.US, " %.1f MB/s 残り約%s", dlSpeed / 1e6, formatDuration(etaSec)));
            }
        } else {
            downloadBar.setIndeterminate(true);
            sb.append("接続中…");
        }
        if (!paused.isEmpty()) sb.append(" [").append(paused).append("]");
        downloadText.setText(sb.toString());
        if (!generating && !loading && (manager == null || !manager.isModelLoaded())) {
            setStatus("ダウンロード中… 完了後に自動で読み込みます");
        }
        ui.postDelayed(pollTask, 1000);
    }

    private void hideDownloadRow() {
        monitoredDownload = null;
        ui.removeCallbacks(pollTask);
        downloadRow.setVisibility(View.GONE);
        refreshControls();
    }

    private void reportDownloadFailure(ModelCatalog.ModelSpec spec, ModelStore.DownloadState st) {
        String why = ModelStore.reasonText(st);
        store.cancelDownload(spec);
        hideDownloadRow();
        appendLog("ダウンロード失敗: " + spec.title + " (" + why + ")");
        setStatus("ダウンロード失敗: " + why);
        new AlertDialog.Builder(this)
                .setTitle("ダウンロード失敗")
                .setMessage(spec.title + "\n理由: " + why + "\n\nもう一度試しますか？")
                .setPositiveButton("再試行", (d, w) -> startDownload(spec,
                        spec.id.equals(prefs().getString(PREFS_PENDING_MODEL, ""))))
                .setNegativeButton("閉じる", null)
                .show();
    }

    private void confirmCancelDownload() {
        final ModelCatalog.ModelSpec spec = monitoredDownload;
        if (spec == null) return;
        new AlertDialog.Builder(this)
                .setTitle("ダウンロード中止")
                .setMessage(spec.title + " のダウンロードを中止しますか？（途中までのデータは削除されます）")
                .setPositiveButton("中止する", (d, w) -> {
                    store.cancelDownload(spec);
                    prefs().edit().remove(PREFS_PENDING_MODEL).apply();
                    appendLog("ダウンロードを中止: " + spec.title);
                    hideDownloadRow();
                    if (manager != null && !manager.isModelLoaded()) {
                        setStatus("モデルを選択してください");
                    }
                })
                .setNegativeButton("続ける", null)
                .show();
    }

    /** Verify (size + SHA-256) and install a completed download, then use it. */
    private void finalizeDownload(ModelCatalog.ModelSpec spec, ModelStore.DownloadState st) {
        if (finalizing) return;
        finalizing = true;
        ui.removeCallbacks(pollTask);
        downloadRow.setVisibility(View.VISIBLE);
        downloadBar.setIndeterminate(false);
        downloadText.setText(spec.title + ": ダウンロード完了 — 検証中…");
        refreshControls();
        worker.submit(() -> {
            String error = null;
            try {
                store.finishDownload(spec, st, pct -> runOnUiThread(() -> {
                    downloadBar.setProgress(pct * 10);
                    downloadText.setText(spec.title + ": 検証中（SHA-256） " + pct + "%");
                }));
            } catch (IOException e) {
                error = e.getMessage();
            }
            final String err = error;
            runOnUiThread(() -> {
                finalizing = false;
                hideDownloadRow();
                if (err != null) {
                    appendLog("検証失敗: " + spec.title + " — " + err);
                    new AlertDialog.Builder(this).setTitle("ダウンロードしたファイルが不正です")
                            .setMessage(err + "\n\nもう一度ダウンロードしますか？")
                            .setPositiveButton("再ダウンロード", (d, w) -> startDownload(spec,
                                    spec.id.equals(prefs().getString(PREFS_PENDING_MODEL, ""))))
                            .setNegativeButton("閉じる", null).show();
                    checkDownloads();
                    return;
                }
                appendLog("ダウンロード完了・検証OK: " + spec.title);
                if (spec == ModelCatalog.TAESD) {
                    useTaesd = true;
                    saveGenSettings();
                } else {
                    String pending = prefs().getString(PREFS_PENDING_MODEL, "");
                    if (spec.id.equals(pending) || currentSpec() == null || !manager.isModelLoaded()) {
                        prefs().edit().remove(PREFS_PENDING_MODEL).apply();
                        selectModel(spec, true);
                    } else {
                        Toast.makeText(this, spec.title + " のダウンロードが完了しました", Toast.LENGTH_SHORT).show();
                    }
                }
                checkDownloads();   // finish / keep monitoring any other download
                ModelCatalog.ModelSpec cur = currentSpec();
                // If another finished download is now being verified, its completion loads instead
                // (so e.g. model + TAESD finishing together load the model once).
                if (!finalizing && cur != null && store.installedFile(cur) != null) ensureModelLoaded();
                refreshControls();
            });
        });
    }

    private void confirmDeleteModel(ModelCatalog.ModelSpec spec) {
        new AlertDialog.Builder(this)
                .setTitle("モデルを削除")
                .setMessage("「" + spec.title + "」(" + spec.sizeLabel() + ") を端末から削除しますか？"
                        + (spec.downloadable() ? "\n（再度使うときは再ダウンロードします）" : "\n（再取得はできません）"))
                .setPositiveButton("削除", (d, w) -> {
                    boolean wasActive = spec.id.equals(modelId);
                    loading = true;
                    refreshControls();
                    worker.submit(() -> {
                        SdModelManager.LoadConfig lc = manager.getLoadedConfig();
                        File f = store.installedFile(spec);
                        if (lc != null && f != null && lc.modelPath.equals(f.getAbsolutePath())) {
                            manager.release();
                        }
                        store.delete(spec);
                        runOnUiThread(() -> {
                            loading = false;
                            appendLog("削除しました: " + spec.title);
                            if (wasActive) {
                                modelId = "";
                                prefs().edit().putString(PREFS_MODEL_ID, "").apply();
                                updateModelText();
                                setStatus("モデルを選択してください");
                            }
                            refreshControls();
                            if (wasActive) showModelChooser();
                        });
                    });
                })
                .setNegativeButton("キャンセル", null)
                .show();
    }

    private SdModelManager.LoadConfig buildLoadConfig(File modelFile) {
        File taesd = useTaesd ? store.installedTaesd() : null;
        int threads = nThreads > 0 ? nThreads : CpuInfo.defaultThreads();
        return new SdModelManager.LoadConfig(modelFile.getAbsolutePath(), threads, wtype,
                taesd != null ? taesd.getAbsolutePath() : "", vaeTiling);
    }

    /**
     * Makes sure the selected model is loaded with the current load-time settings. A no-op if it
     * already is (e.g. after the Activity was recreated), so the model is never reloaded needlessly.
     */
    private void ensureModelLoaded() {
        if (manager == null) return;
        ModelCatalog.ModelSpec spec = currentSpec();
        File file = spec == null ? null : store.installedFile(spec);
        if (file == null) {
            setStatus("モデルを選択してください");
            if (!chooserShownThisLaunch) showModelChooser();
            return;
        }
        if (useTaesd && store.installedTaesd() == null) {
            // TAESD enabled but not on the device yet: fetch it (small) and load meanwhile.
            ModelStore.DownloadState st = store.queryDownload(ModelCatalog.TAESD);
            if ((st == null || !st.active()) && monitoredDownload == null) {
                startDownload(ModelCatalog.TAESD, false);
            }
        }
        final SdModelManager.LoadConfig cfg = buildLoadConfig(file);
        if (manager.isLoaded(cfg)) {
            setStatus("モデル読込済み — 「生成」できます");
            refreshControls();
            return;
        }
        if (loading) return;
        loading = true;
        refreshControls();
        setStatus("モデル読込中… " + spec.title);
        worker.submit(() -> {
            String err;
            long t0 = SystemClock.elapsedRealtime();
            try {
                store.ensureVerified(spec, file, pct -> setStatusUi("モデルを検証中… " + pct + "%"));
                setStatusUi("モデル読込中… " + spec.title);
                err = manager.load(cfg);
            } catch (Throwable t) {
                err = t.getMessage() != null ? t.getMessage() : t.toString();
            }
            final String e = err;
            final long ms = SystemClock.elapsedRealtime() - t0;
            runOnUiThread(() -> {
                loading = false;
                if (e.isEmpty()) {
                    appendLog("モデル読込成功: " + spec.title + " (" + (ms / 1000) + "s, "
                            + cfg.threads + "スレッド" + wtypeSuffix()
                            + (cfg.taesdPath.isEmpty() ? "" : ", TAESD") + (cfg.vaeTiling ? ", VAEタイル" : "") + ")");
                    setStatus("モデル読込済み — 「生成」できます");
                } else {
                    appendLog("モデル読込失敗: " + e);
                    setStatus("読込失敗: " + e);
                }
                refreshControls();
                // The selection or load-time settings may have changed meanwhile: converge to them
                // (never retry the config that was just attempted, so a failing load cannot loop).
                ModelCatalog.ModelSpec cur = currentSpec();
                File f = cur == null ? null : store.installedFile(cur);
                if (f != null) {
                    SdModelManager.LoadConfig want = buildLoadConfig(f);
                    if (!want.equals(cfg) && !manager.isLoaded(want)) ensureModelLoaded();
                }
            });
        });
    }

    // =====================================================================================
    // Generation
    // =====================================================================================

    private void startGeneration(boolean fromImage) {
        if (isBusy()) {
            Toast.makeText(this, "処理中です", Toast.LENGTH_SHORT).show();
            return;
        }
        if (manager == null || !manager.isModelLoaded()) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            if (currentSpec() == null || store.installedFile(currentSpec()) == null) showModelChooser();
            return;
        }
        if (fromImage && initBitmap == null) {
            Toast.makeText(this, "先に入力画像を選択してください", Toast.LENGTH_SHORT).show();
            pickInitImage();
            return;
        }
        final String rawPrompt = promptEdit.getText().toString().trim();
        if (rawPrompt.isEmpty()) {
            Toast.makeText(this, "プロンプトを入力してください", Toast.LENGTH_SHORT).show();
            return;
        }
        final String rawNegative = negativeEdit.getText().toString().trim();
        final int steps = Math.min(MAX_STEPS, parseInt(stepsEdit.getText().toString(), 6));
        stepsEdit.setText(String.valueOf(steps));
        final int size = snapTo64(selectedSize);
        final long seed = parseLong(seedEdit.getText().toString(), -1);
        final float strength = clamp(parseFloat(strengthEdit.getText().toString(), 0.6f), 0.05f, 1.0f);
        final float cfg = cfgScale;
        final int method = sampler;
        final boolean nsfw = nsfwRestriction;
        final Bitmap src = initBitmap;
        persistInputs();

        generating = true;
        sampleSteps = 0;
        samplingStartMs = 0;
        samplingEndMs = 0;
        genStartMs = SystemClock.elapsedRealtime();
        manager.clearCancel();
        cancelButton.setEnabled(true);
        cancelButton.setText("中止");
        translatedPromptView.setVisibility(View.GONE);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setIndeterminate(true);
        refreshControls();
        setStatus(PromptTranslator.needsTranslation(rawPrompt) ? "翻訳中…" : "準備中…");
        final PromptTranslator tr = translator();

        worker.submit(() -> {
            SdModelManager.Result r;
            PromptTranslator.Prepared prep = null;
            try {
                prep = tr.prepare(rawPrompt, rawNegative, cfg != 1.0f, nsfw);
                if (prep.blocked) {
                    final String why = prep.note;
                    runOnUiThread(() -> {
                        Toast.makeText(this, "コンテンツがモデレートされました（" + why + "）。生成を中止します。",
                                Toast.LENGTH_LONG).show();
                        finishGeneration("生成中止: " + why);
                    });
                    return;
                }
                final PromptTranslator.Prepared p = prep;
                runOnUiThread(() -> {
                    if (!p.prompt.equals(rawPrompt)) {
                        translatedPromptView.setText("英訳: " + p.prompt);
                        translatedPromptView.setVisibility(View.VISIBLE);
                    }
                    if (!p.note.isEmpty()) appendLog(p.note);
                    setStatus("生成開始… (" + size + "px / " + steps + "step"
                            + (fromImage ? " / 変化量 " + strength : "") + ")");
                });
                appendLogUi((fromImage ? "img2img: \"" : "生成: \"") + p.prompt + "\" " + size + "px "
                        + steps + "step " + samplerLabel(method) + " CFG " + cfg
                        + (seed >= 0 ? " seed=" + seed : ""));
                if (fromImage) {
                    byte[] init = bitmapToRgb(Bitmap.createScaledBitmap(src, size, size, true));
                    int[] sched = img2imgSchedule(steps, strength);
                    float sdStrength = (sched[1] - 0.5f) / sched[0];
                    r = manager.img2img(init, p.prompt, p.negative, size, size, sched[0], cfg, sdStrength, seed, method);
                } else {
                    r = manager.txt2img(p.prompt, p.negative, size, size, steps, cfg, seed, method);
                }
            } catch (Throwable t) {
                r = null;
                final String msg = t.toString();
                appendLogUi("生成エラー: " + msg);
                runOnUiThread(() -> finishGeneration("生成エラー: " + t.getClass().getSimpleName()));
                return;
            }
            final SdModelManager.Result res = r;
            final Bitmap bmp = res.ok() ? rgbToBitmap(res.rgb, res.width, res.height) : null;
            runOnUiThread(() -> onGenerationResult(res, bmp));
        });
    }

    private void onGenerationResult(SdModelManager.Result r, Bitmap bmp) {
        long now = SystemClock.elapsedRealtime();
        double totalSec = (now - genStartMs) / 1000.0;
        if (r.ok()) {
            if (samplingEndMs > 0) lastDecodeMs = now - samplingEndMs;
            lastBitmap = bmp;
            lastSeed = r.seed;
            imageView.setImageBitmap(bmp);
            String timing = String.format(Locale.US, "%.1fs", totalSec);
            if (samplingStartMs > 0 && samplingEndMs > 0) {
                timing += String.format(Locale.US, "（サンプリング %.1fs / デコード %.1fs）",
                        (samplingEndMs - samplingStartMs) / 1000.0, (now - samplingEndMs) / 1000.0);
            }
            appendLog("生成完了 seed=" + r.seed + " " + timing);
            finishGeneration("完了 seed=" + r.seed + " (" + String.format(Locale.US, "%.1fs", totalSec) + ")");
        } else if (r.cancelled) {
            appendLog("生成を中止しました");
            finishGeneration("中止しました");
        } else {
            String e = r.error.isEmpty() ? "原因不明（ログ参照）" : r.error;
            appendLog("生成失敗: " + e);
            finishGeneration("生成失敗 — " + e);
        }
    }

    /** Common end of a generation. Shows the last finished image (drops a leftover preview). */
    private void finishGeneration(String status) {
        generating = false;
        imageView.setImageBitmap(lastBitmap);   // null clears the view
        progressBar.setIndeterminate(false);
        progressBar.setVisibility(View.GONE);
        setStatus(status);
        refreshControls();
    }

    private void cancelGeneration() {
        if (!generating || manager == null) return;
        manager.requestCancel();
        cancelButton.setEnabled(false);
        cancelButton.setText("中止しています…");
        setStatus("中止しています…（現在のステップの終了後に停止）");
    }

    /** Native progress, already on the UI thread. */
    private void handleProgress(int phase, int step, int steps, long now) {
        if (!generating) return;
        if (phase == StableDiffusionNative.PHASE_SAMPLING) {
            if (step == 0 || samplingStartMs == 0) samplingStartMs = now;
            sampleSteps = steps;
            progressBar.setIndeterminate(false);
            progressBar.setMax(steps + 1);           // the last slot is the VAE decode
            progressBar.setProgress(step);
            long elapsed = (now - genStartMs) / 1000;
            if (step >= steps) {
                samplingEndMs = now;
                if (cancelButton.isEnabled()) {
                    cancelButton.setEnabled(false);
                    cancelButton.setText("仕上げ中…");
                }
                setStatus("仕上げ（VAEデコード）中… " + elapsed + "s"
                        + (lastDecodeMs > 0 ? " 残り約" + formatDuration(lastDecodeMs / 1000) : ""));
                return;
            }
            String eta = "";
            if (step > 0) {
                long perStep = (now - samplingStartMs) / step;
                long remainMs = perStep * (steps - step) + lastDecodeMs;
                eta = " 残り約" + formatDuration(remainMs / 1000);
            }
            setStatus("生成中 " + step + "/" + steps + "（" + elapsed + "s" + eta + "）");
        } else if (phase == StableDiffusionNative.PHASE_DECODE) {
            // VAE tiling reports its tiles through the same callback; keep it as one final phase.
            progressBar.setIndeterminate(false);
            progressBar.setMax(sampleSteps + 1);
            progressBar.setProgress(sampleSteps);
            setStatus("仕上げ（VAEデコード）中… タイル " + Math.min(step, steps) + "/" + steps);
        }
    }

    // --- img2img input ---

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

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQ_PICK_INIT_IMAGE || resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        final Uri uri = data.getData();
        setStatus("入力画像を読込中…");
        // Decoding a large photo is independent of the native worker; keep it off the UI thread.
        new Thread(() -> {
            try {
                Bitmap bmp = loadSquareImage(uri, 512);
                appendLogUi("入力画像を選択 (" + bmp.getWidth() + "x" + bmp.getHeight() + "、中央を正方形に切り抜き)");
                runOnUiThread(() -> {
                    initBitmap = bmp;
                    initThumb.setImageBitmap(bmp);
                    setStatus("入力画像を選択しました — 「画像から生成」できます");
                });
            } catch (Throwable t) {
                appendLogUi("入力画像エラー: " + t);
                runOnUiThread(() -> {
                    setStatus("入力画像を読み込めませんでした");
                    Toast.makeText(this, "画像を読み込めませんでした", Toast.LENGTH_SHORT).show();
                });
            }
        }, "init-image").start();
    }

    /**
     * Decodes {@code uri} (subsampled so a huge photo does not exhaust memory), applies its EXIF
     * rotation, and center-crops it to a square of at most {@code maxSide} px — img2img needs a
     * square input and stretching would distort it.
     */
    private Bitmap loadSquareImage(Uri uri, int maxSide) throws IOException {
        ContentResolver cr = getContentResolver();
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = cr.openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw new IOException("画像を解析できません");
        int sample = 1;
        while (Math.min(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2;
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        Bitmap bmp;
        try (InputStream in = cr.openInputStream(uri)) {
            bmp = BitmapFactory.decodeStream(in, null, opts);
        }
        if (bmp == null) throw new IOException("画像をデコードできません");
        int rotation = 0;
        try (InputStream in = cr.openInputStream(uri)) {
            if (in != null) {
                int o = new ExifInterface(in).getAttributeInt(ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL);
                rotation = o == ExifInterface.ORIENTATION_ROTATE_90 ? 90
                        : o == ExifInterface.ORIENTATION_ROTATE_180 ? 180
                        : o == ExifInterface.ORIENTATION_ROTATE_270 ? 270 : 0;
            }
        } catch (Exception ignore) {
        }
        int w = bmp.getWidth(), h = bmp.getHeight(), s = Math.min(w, h);
        Matrix m = new Matrix();
        if (rotation != 0) m.postRotate(rotation);
        float scale = s > maxSide ? (float) maxSide / s : 1f;
        m.postScale(scale, scale);
        return Bitmap.createBitmap(bmp, (w - s) / 2, (h - s) / 2, s, s, m, true);
    }

    // =====================================================================================
    // Settings dialogs
    // =====================================================================================

    private void showGenSettingsDialog() {
        if (isBusy()) {
            Toast.makeText(this, "処理中は変更できません", Toast.LENGTH_SHORT).show();
            return;
        }
        LinearLayout c = new LinearLayout(this);
        c.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(16);
        c.setPadding(pad, pad, pad, pad);

        final int[] selSampler = {sampler};
        addLabel(c, "サンプラー", 0);
        Button samBtn = new Button(this);
        samBtn.setText("サンプラー: " + samplerLabel(selSampler[0]));
        samBtn.setOnClickListener(v -> {
            selSampler[0] = nextSampler(selSampler[0]);
            samBtn.setText("サンプラー: " + samplerLabel(selSampler[0]));
        });
        c.addView(samBtn);
        addHint(c, "LCM=LCM 蒸留モデル用 ／ Euler a=SD-Turbo ／ DPM++ 2M=通常の SD1.5（〜20step）");

        addLabel(c, "CFG スケール", dp(12));
        final EditText cfgInput = new EditText(this);
        cfgInput.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        cfgInput.setText(String.valueOf(cfgScale));
        c.addView(cfgInput);
        addHint(c, "1.0 = 最速（ネガティブプロンプト無効）／ 1 より大きいと計算量が約 2 倍。通常の SD1.5 は 7.0");

        ModelCatalog.ModelSpec spec = currentSpec();
        if (spec != null && spec.hasRecommendedSettings()) {
            Button rec = new Button(this);
            rec.setText("このモデルの推奨値に戻す（" + spec.recommendedLabel() + "）");
            rec.setOnClickListener(v -> {
                selSampler[0] = spec.sampler;
                samBtn.setText("サンプラー: " + samplerLabel(selSampler[0]));
                cfgInput.setText(String.valueOf(spec.cfgScale));
                stepsEdit.setText(String.valueOf(spec.steps));
            });
            c.addView(rec);
        }

        final boolean[] selTaesd = {useTaesd};
        addLabel(c, "高速VAEデコード（TAESD）", dp(12));
        Button taesdBtn = new Button(this);
        taesdBtn.setText(onOff("TAESD", selTaesd[0]));
        taesdBtn.setOnClickListener(v -> {
            selTaesd[0] = !selTaesd[0];
            taesdBtn.setText(onOff("TAESD", selTaesd[0]));
        });
        c.addView(taesdBtn);
        addHint(c, "仕上げの VAE デコードが数十倍速くなり省メモリ（細部はわずかに甘くなる）。"
                + (store.installedTaesd() == null ? "初回のみ " + ModelCatalog.TAESD.sizeLabel() + " をダウンロード。" : ""));

        final boolean[] selTiling = {vaeTiling};
        addLabel(c, "省メモリ VAE（タイル処理）", dp(12));
        Button tileBtn = new Button(this);
        tileBtn.setText(onOff("タイル処理", selTiling[0]));
        tileBtn.setOnClickListener(v -> {
            selTiling[0] = !selTiling[0];
            tileBtn.setText(onOff("タイル処理", selTiling[0]));
        });
        c.addView(tileBtn);
        addHint(c, "VAE を分割して処理しピークメモリを下げる（デコードは約 2 倍遅い）。メモリ不足で落ちる端末向け。"
                + "この端末の RAM: " + ModelCatalog.formatSize(CpuInfo.totalRamBytes(this)));

        final int[] selWtype = {wtype};
        addLabel(c, "重み精度（読込時に再量子化）", dp(12));
        Button precBtn = new Button(this);
        precBtn.setText("精度: " + wtypeLabel(selWtype[0]));
        precBtn.setOnClickListener(v -> {
            selWtype[0] = nextWtype(selWtype[0]);
            precBtn.setText("精度: " + wtypeLabel(selWtype[0]));
        });
        c.addView(precBtn);
        addHint(c, "通常は「元のまま」。Q4_0 は省メモリ（約 0.2GB 減）だが画質が落ち、速度も上がらない"
                + "（conv 層は常に F16。実測では Q8_0 の方が速い）");

        addLabel(c, "スレッド数（0=自動: " + CpuInfo.defaultThreads() + "）", dp(12));
        EditText thInput = new EditText(this);
        thInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        thInput.setText(String.valueOf(nThreads));
        c.addView(thInput);

        final boolean[] selNsfw = {nsfwRestriction};
        addLabel(c, "性表現制限", dp(12));
        Button nsfwBtn = new Button(this);
        nsfwBtn.setText(onOff("性表現制限", selNsfw[0]));
        nsfwBtn.setOnClickListener(v -> {
            selNsfw[0] = !selNsfw[0];
            nsfwBtn.setText(onOff("性表現制限", selNsfw[0]));
        });
        c.addView(nsfwBtn);
        addHint(c, "ON: 性的な語を含むプロンプトは生成を中止し、LLM 接続時は LLM にも判定させます。");

        ScrollView scroll = new ScrollView(this);
        scroll.addView(c);
        new AlertDialog.Builder(this)
                .setTitle("生成設定")
                .setView(scroll)
                .setPositiveButton("保存", (d, w) -> {
                    sampler = selSampler[0];
                    cfgScale = clamp(parseFloat(cfgInput.getText().toString(), cfgScale), 1.0f, 20.0f);
                    wtype = selWtype[0];
                    nThreads = Math.max(0, Math.min(64, parseIntAllowZero(thInput.getText().toString(), 0)));
                    nsfwRestriction = selNsfw[0];
                    useTaesd = selTaesd[0];
                    vaeTiling = selTiling[0];
                    saveGenSettings();
                    persistInputs();
                    Toast.makeText(this, "生成設定を保存しました", Toast.LENGTH_SHORT).show();
                    ensureModelLoaded();   // reloads only if a load-time setting changed
                })
                .setNegativeButton("キャンセル", null)
                .show();
    }

    private void addLabel(LinearLayout c, String text, int topPad) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setPadding(0, topPad, 0, 0);
        c.addView(t);
    }

    private void addHint(LinearLayout c, String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextSize(11);
        t.setTextColor(0xFF666666);
        c.addView(t);
    }

    private static String onOff(String label, boolean on) {
        return label + ": " + (on ? "ON" : "OFF");
    }

    private void showSizePicker() {
        int current = 0;
        for (int i = 0; i < SIZE_VALUES.length; i++) {
            if (SIZE_VALUES[i] == selectedSize) {
                current = i;
                break;
            }
        }
        final int[] picked = {current};
        new AlertDialog.Builder(this)
                .setTitle("サイズ（正方形）")
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
        return size + "px";
    }

    private String wtypeSuffix() {
        return wtype == StableDiffusionNative.WTYPE_DEFAULT ? "" : ", " + wtypeLabel(wtype);
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

    static String samplerLabel(int s) {
        switch (s) {
            case StableDiffusionNative.SAMPLE_LCM: return "LCM";
            case StableDiffusionNative.SAMPLE_DPMPP2M: return "DPM++ 2M";
            case StableDiffusionNative.SAMPLE_EULER: return "Euler";
            default: return "Euler a";
        }
    }

    /** Cycle LCM → Euler a → DPM++ 2M → Euler → LCM for the settings toggle. */
    private static int nextSampler(int s) {
        switch (s) {
            case StableDiffusionNative.SAMPLE_LCM: return StableDiffusionNative.SAMPLE_EULER_A;
            case StableDiffusionNative.SAMPLE_EULER_A: return StableDiffusionNative.SAMPLE_DPMPP2M;
            case StableDiffusionNative.SAMPLE_DPMPP2M: return StableDiffusionNative.SAMPLE_EULER;
            default: return StableDiffusionNative.SAMPLE_LCM;
        }
    }

    // --- LLM settings dialog ---

    private PromptTranslator translator() {
        if (translator == null || !translator.sameEndpoint(llmBaseUrl, llmModel)) {
            translator = new PromptTranslator(llmBaseUrl, llmModel, this::appendLogUi);
        }
        return translator;
    }

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
        urlLabel.setText("LLM API URL（日本語プロンプトの英訳に使用。英語のみなら不要）");
        urlLabel.setPadding(0, dp(10), 0, 0);
        container.addView(urlLabel);

        EditText urlInput = new EditText(this);
        urlInput.setHint(DEFAULT_LLM_BASE_URL);
        urlInput.setText(llmBaseUrl);
        container.addView(urlInput);

        launchBtn.setOnClickListener(v -> {
            String url = PromptTranslator.normalizeBaseUrl(urlInput.getText().toString());
            openLlmOrStore(url.isEmpty() ? DEFAULT_LLM_BASE_URL : url);
        });

        final String[] selectedModel = {llmModel};
        TextView modelView = new TextView(this);
        modelView.setText("モデル: " + labelOf(llmModel));
        modelView.setPadding(0, dp(10), 0, 0);
        container.addView(modelView);

        Button modelBtn = new Button(this);
        modelBtn.setText("api/tags から選択");
        modelBtn.setOnClickListener(v -> {
            String url = PromptTranslator.normalizeBaseUrl(urlInput.getText().toString());
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
                    llmBaseUrl = PromptTranslator.normalizeBaseUrl(urlInput.getText().toString());
                    llmModel = selectedModel[0] == null ? "" : selectedModel[0].trim();
                    prefs().edit().putString(PREFS_LLM_URL, llmBaseUrl).putString(PREFS_LLM_MODEL, llmModel).apply();
                    Toast.makeText(this, "LLM設定を保存しました", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("キャンセル", null)
                .show();

        checkAndUpdateApiStatus(PromptTranslator.normalizeBaseUrl(llmBaseUrl), apiStatusView, statusBg);
    }

    private void checkAndUpdateApiStatus(String baseUrl, TextView statusView, GradientDrawable bg) {
        if (baseUrl == null || baseUrl.isEmpty()) {
            statusView.setText("LLM 未設定（翻訳しない）");
            return;
        }
        new Thread(() -> {
            final boolean ok = PromptTranslator.isAvailable(baseUrl);
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
                List<String> models = PromptTranslator.fetchModelNames(baseUrl);
                runOnUiThread(() -> {
                    if (models.isEmpty()) {
                        modelView.setText("モデル: " + labelOf(selectedModel[0]));
                        Toast.makeText(this, "モデルが見つかりませんでした", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    int checked = Math.max(0, models.indexOf(selectedModel[0]));
                    final int[] picked = {checked};
                    ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                            android.R.layout.simple_list_item_single_choice, models);
                    ListView listView = new ListView(this);
                    listView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
                    listView.setAdapter(adapter);
                    listView.setItemChecked(checked, true);
                    listView.setOnItemClickListener((parent, view, which, id) -> picked[0] = which);
                    new AlertDialog.Builder(this)
                            .setTitle("LLMモデル選択")
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

    private static String labelOf(String model) {
        return (model == null || model.isEmpty()) ? "(未選択)" : model;
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
                int port = Uri.parse(baseUrl).getPort();
                serviceIntent.putExtra("port", port > 0 ? port : 11434);
            } catch (Exception ignore) {
            }
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }
            } catch (Exception ignore) {
            }
            launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(launchIntent);
        } catch (ActivityNotFoundException e) {
            try {
                startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(LLM_PLAY_STORE_URL)));
            } catch (ActivityNotFoundException ignore) {
            }
        }
    }

    // =====================================================================================
    // Persistence
    // =====================================================================================

    private SharedPreferences prefs() {
        return getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
    }

    private void loadSettings() {
        SharedPreferences p = prefs();
        llmBaseUrl = p.getString(PREFS_LLM_URL, DEFAULT_LLM_BASE_URL);
        llmModel = p.getString(PREFS_LLM_MODEL, "");
        modelId = p.getString(PREFS_MODEL_ID, "");
        wtype = p.getInt(PREFS_WTYPE, StableDiffusionNative.WTYPE_DEFAULT);
        sampler = p.getInt(PREFS_SAMPLER, ModelCatalog.DEFAULT_MODEL.sampler);
        cfgScale = p.getFloat(PREFS_CFG, ModelCatalog.DEFAULT_MODEL.cfgScale);
        nThreads = p.getInt(PREFS_THREADS, 0);
        nsfwRestriction = p.getBoolean(PREFS_NSFW, false);
        selectedSize = p.getInt(PREFS_SIZE, 512);
        // On by default: the final decode drops from ~28s to ~2s at 512px with near-identical output.
        // (Old versions stored a picked TAESD file path; ModelStore adopts that file.)
        useTaesd = p.getBoolean(PREFS_USE_TAESD, true);
        vaeTiling = p.getBoolean(PREFS_VAE_TILING, CpuInfo.totalRamBytes(this) > 0
                && CpuInfo.totalRamBytes(this) < LOW_RAM_BYTES);
    }

    private void saveGenSettings() {
        prefs().edit()
                .putInt(PREFS_WTYPE, wtype)
                .putInt(PREFS_SAMPLER, sampler)
                .putFloat(PREFS_CFG, cfgScale)
                .putInt(PREFS_THREADS, nThreads)
                .putBoolean(PREFS_NSFW, nsfwRestriction)
                .putInt(PREFS_SIZE, selectedSize)
                .putBoolean(PREFS_USE_TAESD, useTaesd)
                .putBoolean(PREFS_VAE_TILING, vaeTiling)
                .remove(PREFS_TAESD_LEGACY)
                .apply();
    }

    private void persistInputs() {
        if (promptEdit == null) return;
        prefs().edit()
                .putString(PREFS_PROMPT, promptEdit.getText().toString())
                .putString(PREFS_NEGATIVE, negativeEdit.getText().toString())
                .putInt(PREFS_STEPS, Math.min(MAX_STEPS, parseInt(stepsEdit.getText().toString(), 6)))
                .putLong(PREFS_SEED, parseLong(seedEdit.getText().toString(), -1))
                .putFloat(PREFS_STRENGTH, clamp(parseFloat(strengthEdit.getText().toString(), 0.6f), 0.05f, 1.0f))
                .apply();
    }

    // =====================================================================================
    // Image save
    // =====================================================================================

    private void saveImage() {
        final Bitmap bmp = lastBitmap;
        final long seed = lastSeed;
        if (bmp == null) {
            Toast.makeText(this, "保存できる画像がありません", Toast.LENGTH_SHORT).show();
            return;
        }
        new Thread(() -> {
            String name = "llamaimage_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date())
                    + (seed >= 0 ? "_s" + seed : "") + ".png";
            try {
                String where;
                if (Build.VERSION.SDK_INT >= 29) {
                    ContentResolver cr = getContentResolver();
                    ContentValues cv = new ContentValues();
                    cv.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                    cv.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
                    cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/llamaimage");
                    cv.put(MediaStore.Images.Media.IS_PENDING, 1);
                    Uri u = cr.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                    if (u == null) throw new IOException("MediaStore に登録できません");
                    try (OutputStream os = cr.openOutputStream(u)) {
                        if (os == null || !bmp.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                            throw new IOException("PNG 書き込み失敗");
                        }
                    } catch (IOException e) {
                        cr.delete(u, null, null);
                        throw e;
                    }
                    cv.clear();
                    cv.put(MediaStore.Images.Media.IS_PENDING, 0);
                    cr.update(u, cv, null, null);
                    where = "Pictures/llamaimage/" + name;
                } else {
                    File dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                    File f = new File(dir, name);
                    try (OutputStream os = new FileOutputStream(f)) {
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, os);
                    }
                    where = f.getAbsolutePath();
                }
                appendLogUi("保存: " + where);
                runOnUiThread(() -> Toast.makeText(this, "画像を保存しました", Toast.LENGTH_SHORT).show());
            } catch (Throwable t) {
                appendLogUi("保存失敗: " + t);
                runOnUiThread(() -> Toast.makeText(this, "保存失敗", Toast.LENGTH_SHORT).show());
            }
        }, "save-image").start();
    }

    // =====================================================================================
    // Helpers
    // =====================================================================================

    private void setStatus(String s) {
        if (getActionBar() != null) getActionBar().setSubtitle(s);
        if (statusText != null) statusText.setText(s);
    }

    private void setStatusUi(String s) {
        runOnUiThread(() -> setStatus(s));
    }

    private static String formatDuration(long sec) {
        if (sec < 60) return sec + "s";
        return (sec / 60) + "分" + (sec % 60) + "s";
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

    /** Pack a Bitmap into tightly-packed RGB (w*h*3) bytes for the native img2img call. */
    private static byte[] bitmapToRgb(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int[] px = new int[w * h];
        bmp.getPixels(px, 0, w, 0, 0, w, h);
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++) {
            int c = px[i];
            rgb[i * 3] = (byte) ((c >> 16) & 0xff);
            rgb[i * 3 + 1] = (byte) ((c >> 8) & 0xff);
            rgb[i * 3 + 2] = (byte) (c & 0xff);
        }
        return rgb;
    }

    /** Positive int, else {@code def}. */
    private static int parseInt(String s, int def) {
        try {
            int v = Integer.parseInt(s.trim());
            return v > 0 ? v : def;
        } catch (Exception e) {
            return def;
        }
    }

    private static int parseIntAllowZero(String s, int def) {
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    /** Seed: any value >= 0 is used as-is; negative or invalid means random (-1). */
    private static long parseLong(String s, long def) {
        try {
            long v = Long.parseLong(s.trim());
            return v >= 0 ? v : -1;
        } catch (Exception e) {
            return def;
        }
    }

    private static float parseFloat(String s, float def) {
        try {
            float v = Float.parseFloat(s.trim());
            return Float.isNaN(v) ? def : v;
        } catch (Exception e) {
            return def;
        }
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /**
     * img2img schedule: {total schedule steps, denoising steps actually run}.
     * sd.cpp runs floor(total*strength)+1 steps from part-way down a {@code total}-step schedule,
     * so the effective strength is (floor(total*s)+1)/total — never below 1/total, and always 1.0
     * (input ignored) for a 1-step schedule such as SD-Turbo's. Use at least 4 schedule steps and
     * run the number of steps closest to the requested strength; the caller passes
     * strength' = (run - 0.5) / total so that sd.cpp's floor() lands exactly on it.
     */
    static int[] img2imgSchedule(int steps, float strength) {
        int total = Math.max(steps, 4);
        int run = Math.max(1, Math.min(total, Math.round(total * strength)));
        return new int[]{total, run};
    }

    /** SD needs dimensions that are multiples of 64; snap (min 64). */
    private static int snapTo64(int v) {
        int snapped = (v / 64) * 64;
        return snapped < 64 ? 64 : snapped;
    }

    private void appendLog(String msg) {
        if (msg == null || logText == null) return;
        logText.append(msg.endsWith("\n") ? msg : msg + "\n");
        CharSequence t = logText.getText();
        if (t.length() > MAX_LOG_CHARS) {
            logText.setText(t.subSequence(t.length() - MAX_LOG_CHARS / 2, t.length()));
        }
    }

    private void appendLogUi(String msg) {
        runOnUiThread(() -> appendLog(msg));
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
