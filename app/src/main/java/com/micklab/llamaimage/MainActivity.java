package com.micklab.llamaimage;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
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
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
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

    static final String PREFS_NAME = "llamaimage_prefs";
    static final String PREFS_LLM_URL = "llm_base_url";
    static final String PREFS_LLM_MODEL = "llm_model";
    static final String PREFS_WTYPE = "wtype";
    static final String PREFS_SAMPLER = "sampler";
    static final String PREFS_CFG = "cfg_scale";
    static final String PREFS_THREADS = "threads";
    static final String PREFS_NSFW = "nsfw_restriction";
    static final String PREFS_SIZE = "image_size";
    static final String PREFS_TAESD_LEGACY = "taesd_path";   // old versions: picked file path
    static final String PREFS_USE_TAESD = "use_taesd";
    static final String PREFS_VAE_TILING = "vae_tiling";
    static final String PREFS_USE_GPU = "use_gpu";   // legacy (before the NPU): true = GPU
    static final String PREFS_BACKEND = "backend";   // BACKEND_*
    static final int BACKEND_CPU = 0, BACKEND_GPU = 1, BACKEND_NPU = 2;
    static final String[] BACKEND_LABELS = {"CPU", "GPU（OpenCL）", "NPU（Hexagon）"};
    static final String PREFS_MODEL_ID = "model_id";
    static final String PREFS_PENDING_MODEL = "pending_model_id";  // select after download
    static final String PREFS_STEPS = "steps";
    static final String PREFS_STRENGTH = "strength";
    static final String PREFS_SEED = "seed";
    static final String PREFS_PROMPT = "prompt";
    static final String PREFS_NEGATIVE = "negative";
    static final String PREFS_TAB = "main_tab";

    private static final int TAB_TXT2IMG = 0, TAB_IMG2IMG = 1, TAB_API = 2;
    private static final int STRENGTH_MIN_PCT = 5, STRENGTH_MAX_PCT = 100;

    private static final int[] SIZE_VALUES = {512, 448, 384, 256};
    private static final String[] SIZE_LABELS = {"512px（標準）", "448px", "384px（高速）", "256px（最速）"};
    static final int MAX_STEPS = 50;
    private static final int MAX_LOG_CHARS = 20000;
    static final long LOW_RAM_BYTES = 6L * 1024 * 1024 * 1024;

    static final String DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434";
    private static final String LLM_PACKAGE = "com.micklab.llama";
    private static final String LLM_SERVICE_CLASS = "com.micklab.llama.OllamaForegroundService";
    private static final String LLM_SERVICE_ACTION = "com.micklab.llama.START_SERVICE";
    private static final String LLM_PLAY_STORE_URL =
            "https://play.google.com/store/apps/details?id=com.micklab.llama";

    private static final int MENU_LLM = 1, MENU_MANUAL = 2, MENU_RIGHTS = 3, MENU_PRIVACY = 4;

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
    private int backend = BACKEND_CPU;        // preferred backend; falls back when unavailable
    private com.micklab.llamaimage.npu.NpuManager npuManager;
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
    private TextView modelStateText;
    private View modelDot;
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
    private SeekBar strengthBar;
    private TextView strengthValueText;
    private Button generateButton;
    private Button cancelButton;
    private View pickInitButton;
    private ImageView initThumb;
    private TextView initThumbHint;
    private TextView[] tabViews;
    private LinearLayout tabBar;
    private ScrollView mainScroll;
    private LinearLayout genArea;
    private LinearLayout apiArea;
    private LinearLayout img2imgPanel;
    private int currentTab = TAB_TXT2IMG;
    private View apiDot;
    private TextView apiStateText;
    private TextView apiAddressText;
    private Switch apiSwitch;
    private long apiToggleAt = 0;
    private ProgressBar progressBar;
    private TextView statusText;
    private ImageView imageView;
    private LinearLayout placeholderView;
    private LinearLayout imageActions;
    private Button saveButton;
    private TextView logText;
    private ScrollView logScroll;
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
                if (generating) showImage(bmp);
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
        String prevLog = previousLogTail(80);
        boolean npuCrashed = prefs().getBoolean(PREFS_NPU_PENDING, false);
        if (npuCrashed) {
            // The previous NPU preparation never finished (the process died): don't retry it
            // automatically, or the app is stuck in "NPU を準備中…" on every launch.
            backend = BACKEND_CPU;
            prefs().edit().putBoolean(PREFS_NPU_PENDING, false).apply();
            saveGenSettings();
        }
        setContentView(buildUi());
        rootBody.requestFocus();
        if (!prevLog.isEmpty()) {
            logText.append("— 前回のログ（末尾） —\n" + prevLog + "\n— ここから今回 —\n");
        }
        persistLog("==== 起動 ====");
        if (npuCrashed) {
            appendLog("前回 NPU の準備中にアプリが終了したため、CPU に切り替えました（生成設定で再度 NPU を選べます）");
        }

        generateButton.setOnClickListener(v -> startGeneration(currentTab == TAB_IMG2IMG));
        cancelButton.setOnClickListener(v -> cancelGeneration());
        downloadCancelButton.setOnClickListener(v -> confirmCancelDownload());
        imageView.setOnLongClickListener(v -> {
            lockSeed();
            return true;
        });

        refreshControls();
        setStatus("ネイティブ初期化中…");
        worker.submit(this::initNative);
        // Bring the image API back if it was left enabled (e.g. after the process was killed).
        com.micklab.llamaimage.api.ImageApiService.startIfEnabled(this);
    }

    @Override
    protected void onStart() {
        super.onStart();
        started = true;
        if (nativeReady) checkDownloads();
        if (currentTab == TAB_API) apiPollTask.run();
    }

    @Override
    protected void onStop() {
        super.onStop();
        started = false;
        ui.removeCallbacks(pollTask);
        ui.removeCallbacks(apiPollTask);
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
    // --- NPU preparation progress -----------------------------------------------------------
    // ORT compiles each graph in one blocking call without progress, so the bar shows the stage
    // (weighted by its typical share of the time: the UNet dominates) plus the elapsed time.
    private static final int[] NPU_STAGE_WEIGHT = {0, 8, 80, 6, 6};   // index 1..4, sums to 100
    private static final String[] NPU_STAGE_LABEL = {"", "テキストエンコーダ", "UNet", "デコーダ", "エンコーダ"};
    private int npuStage = 0, npuStages = 0;
    private long npuStageStart = 0, npuPrepStart = 0;
    private final Runnable npuTicker = new Runnable() {
        @Override public void run() {
            if (npuStage <= 0) return;
            long now = SystemClock.elapsedRealtime();
            setStatus("NPU 向け最適化中 " + npuStage + "/" + npuStages + "（" + NPU_STAGE_LABEL[npuStage] + "）… "
                    + formatDuration((now - npuStageStart) / 1000) + "（合計 " + formatDuration((now - npuPrepStart) / 1000)
                    + "・初回のみ数分）");
            ui.postDelayed(this, 1000);
        }
    };

    /** Process RSS / device available memory, written to the persisted log while compiling. */
    private Thread memWatch;

    private String memLine() {
        long rssKb = -1;
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader("/proc/self/status"))) {
            String l;
            while ((l = r.readLine()) != null) {
                if (l.startsWith("VmRSS:")) rssKb = Long.parseLong(l.replaceAll("[^0-9]", ""));
            }
        } catch (Exception ignore) {
        }
        android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
        ((android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(mi);
        return "メモリ: アプリ " + (rssKb / 1024) + " MB / 端末の空き " + (mi.availMem >> 20) + " MB"
                + (mi.lowMemory ? "（低メモリ）" : "");
    }

    private void startMemWatch() {
        if (memWatch != null) return;
        memWatch = new Thread(() -> {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    persistLog("  " + memLine());   // file only: survives a kill, no UI spam
                    Thread.sleep(3000);
                }
            } catch (InterruptedException ignore) {
            }
        }, "npu-memwatch");
        memWatch.start();
    }

    private void stopMemWatch() {
        if (memWatch != null) {
            memWatch.interrupt();
            memWatch = null;
        }
    }

    private void startNpuStage(int index, int total, String name) {
        startMemWatch();
        appendLog(memLine());
        if (npuPrepStart == 0) npuPrepStart = SystemClock.elapsedRealtime();
        npuStage = Math.min(index, 4);
        npuStages = total;
        npuStageStart = SystemClock.elapsedRealtime();
        int done = 0;
        for (int i = 1; i < npuStage; i++) done += NPU_STAGE_WEIGHT[i];
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setIndeterminate(false);
        progressBar.setMax(100);
        progressBar.setProgress(done);
        progressBar.setSecondaryProgress(done + NPU_STAGE_WEIGHT[npuStage]);
        ui.removeCallbacks(npuTicker);
        npuTicker.run();
    }

    private void stopNpuStage() {
        if (npuStage > 0) appendLog(memLine());
        stopMemWatch();
        npuStage = 0;
        npuPrepStart = 0;
        ui.removeCallbacks(npuTicker);
        if (!generating) progressBar.setVisibility(View.GONE);
    }

    /** "NPU 用に変換中（UNet） 42%" -> status + determinate bar. */
    private void showBuildProgress(String status) {
        setStatus(status);
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d+)%").matcher(status);
        if (m.find()) {
            progressBar.setVisibility(View.VISIBLE);
            progressBar.setIndeterminate(false);
            progressBar.setMax(100);
            progressBar.setProgress(Integer.parseInt(m.group(1)));
            progressBar.setSecondaryProgress(0);
        }
    }

    /** Keeps the screen on (app stays foreground, not killed by OEM memory governors) while busy. */
    private void updateKeepScreenOn() {
        if (loading || generating) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

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
            // Nothing on the device yet: keep (or pre-select) a downloadable model so the main
            // button offers its download — no dialog in the way of the prompt → generate flow.
            ModelCatalog.ModelSpec want = currentSpec();
            if (want == null || !want.downloadable()) selectModel(ModelCatalog.DEFAULT_MODEL, false);
            setStatus("「" + currentSpec().title + "」を使います — 初回のみダウンロードが必要です");
            refreshControls();
        }
    }

    // =====================================================================================
    // UI
    // =====================================================================================

    // --- palette: black base, violet→pink accent ---
    private static final int C_BG = 0xFF000000;
    private static final int C_SURFACE = 0xFF111114;
    private static final int C_SURFACE2 = 0xFF1A1A20;
    private static final int C_STROKE = 0xFF26262E;
    private static final int C_TEXT = 0xFFF4F4F6;
    private static final int C_TEXT2 = 0xFFA1A1AA;
    private static final int C_TEXT3 = 0xFF63636E;
    private static final int C_ACCENT = 0xFFA78BFA;
    private static final int C_GRAD_A = 0xFF7C3AED;
    private static final int C_GRAD_B = 0xFFDB2777;
    private static final int C_OK = 0xFF34D399;
    private static final int C_WARN = 0xFFFBBF24;

    private static final int BTN_PRIMARY = 0, BTN_SECONDARY = 1, BTN_GHOST = 2;

    /**
     * Main screen, top to bottom: header, model card, tabs (テキスト→画像 / 画像→画像 / API).
     * The two generation tabs share everything from the prompt down (generate, image, save,
     * 詳細設定); 画像→画像 only adds the input image + strength slider above the prompt.
     * The API tab swaps that area for the API switch. The log box is always at the bottom.
     */
    private View buildUi() {
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(20), dp(14), dp(20), dp(28));
        body.setFocusable(true);
        body.setFocusableInTouchMode(true);
        rootBody = body;

        // --- header: wordmark + overflow menu ---
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout brand = new LinearLayout(this);
        brand.setOrientation(LinearLayout.VERTICAL);
        TextView title = text("llamaimage", 24, C_TEXT);
        title.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        title.setLetterSpacing(-0.02f);
        TextView tagline = text("ON-DEVICE  IMAGE  GENERATION", 9, C_TEXT3);
        tagline.setLetterSpacing(0.22f);
        brand.addView(title);
        brand.addView(tagline);
        header.addView(brand, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView menuButton = text("⋮", 24, C_TEXT2);
        menuButton.setGravity(Gravity.CENTER);
        menuButton.setBackground(new RippleDrawable(ColorStateList.valueOf(0x33FFFFFF), null, null));
        menuButton.setOnClickListener(this::showMainMenu);
        header.addView(menuButton, new LinearLayout.LayoutParams(dp(44), dp(44)));
        body.addView(header, marginLp(0, 0, 0, 18));

        // --- model card (tap to change) ---
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.HORIZONTAL);
        card.setGravity(Gravity.CENTER_VERTICAL);
        card.setPadding(dp(16), dp(12), dp(14), dp(12));
        card.setBackground(ripple(rounded(C_SURFACE, 16, C_STROKE)));
        card.setOnClickListener(v -> showModelChooser());
        modelDot = new View(this);
        LinearLayout.LayoutParams dotLp = new LinearLayout.LayoutParams(dp(8), dp(8));
        dotLp.rightMargin = dp(12);
        card.addView(modelDot, dotLp);
        LinearLayout modelCol = new LinearLayout(this);
        modelCol.setOrientation(LinearLayout.VERTICAL);
        TextView modelLabel = text("MODEL", 9, C_TEXT3);
        modelLabel.setLetterSpacing(0.2f);
        modelText = text("", 15, C_TEXT);
        modelText.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        modelStateText = text("", 12, C_TEXT2);
        modelCol.addView(modelLabel);
        modelCol.addView(modelText);
        modelCol.addView(modelStateText);
        card.addView(modelCol, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        card.addView(text("変更 ›", 13, C_ACCENT));
        body.addView(card, marginLp(0, 0, 0, 10));

        // --- download progress (only while a model downloads) ---
        downloadRow = new LinearLayout(this);
        downloadRow.setOrientation(LinearLayout.VERTICAL);
        downloadRow.setPadding(dp(16), dp(12), dp(16), dp(8));
        downloadRow.setBackground(rounded(C_SURFACE, 16, C_STROKE));
        downloadRow.setVisibility(View.GONE);
        downloadBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        downloadBar.setMax(1000);
        tintProgress(downloadBar);
        downloadText = text("", 12, C_TEXT2);
        downloadCancelButton = button("ダウンロード中止", BTN_GHOST);
        downloadRow.addView(downloadBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(4)));
        downloadRow.addView(downloadText, marginLp(0, 8, 0, 4));
        LinearLayout.LayoutParams dcLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(36));
        dcLp.gravity = Gravity.END;
        downloadRow.addView(downloadCancelButton, dcLp);
        body.addView(downloadRow, marginLp(0, 0, 0, 10));

        // --- tabs (segmented control) ---
        LinearLayout tabBar = new LinearLayout(this);
        tabBar.setOrientation(LinearLayout.HORIZONTAL);
        tabBar.setPadding(dp(4), dp(4), dp(4), dp(4));
        tabBar.setBackground(rounded(C_SURFACE, 14, C_STROKE));
        String[] tabLabels = {"テキスト→画像", "画像→画像", "API"};
        tabViews = new TextView[tabLabels.length];
        for (int i = 0; i < tabLabels.length; i++) {
            final int index = i;
            TextView t = text(tabLabels[i], 13, C_TEXT2);
            t.setGravity(Gravity.CENTER);
            t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
            t.setOnClickListener(v -> selectTab(index));
            tabViews[i] = t;
            tabBar.addView(t, new LinearLayout.LayoutParams(0, dp(40), 1f));
        }
        body.addView(tabBar, marginLp(0, 6, 0, 14));
        this.tabBar = tabBar;

        // ===== generation area (tabs 0 and 1) =====
        genArea = new LinearLayout(this);
        genArea.setOrientation(LinearLayout.VERTICAL);

        // --- img2img only: input image + strength slider ---
        img2imgPanel = new LinearLayout(this);
        img2imgPanel.setOrientation(LinearLayout.HORIZONTAL);
        img2imgPanel.setGravity(Gravity.CENTER_VERTICAL);
        img2imgPanel.setPadding(dp(12), dp(12), dp(16), dp(12));
        img2imgPanel.setBackground(rounded(C_SURFACE, 16, C_STROKE));
        FrameLayout thumbFrame = new FrameLayout(this);
        thumbFrame.setBackground(ripple(rounded(C_SURFACE2, 12, C_STROKE)));
        thumbFrame.setClipToOutline(true);
        initThumbHint = text("＋\n入力画像", 11, C_TEXT3);
        initThumbHint.setGravity(Gravity.CENTER);
        thumbFrame.addView(initThumbHint, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        initThumb = new ImageView(this);
        initThumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumbFrame.addView(initThumb, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        thumbFrame.setOnClickListener(v -> pickInitImage());
        pickInitButton = thumbFrame;
        LinearLayout.LayoutParams thumbLp = new LinearLayout.LayoutParams(dp(84), dp(84));
        thumbLp.rightMargin = dp(14);
        img2imgPanel.addView(thumbFrame, thumbLp);
        LinearLayout strengthCol = new LinearLayout(this);
        strengthCol.setOrientation(LinearLayout.VERTICAL);
        LinearLayout strengthHead = new LinearLayout(this);
        strengthHead.setOrientation(LinearLayout.HORIZONTAL);
        strengthHead.addView(text("変化量", 13, C_TEXT),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        strengthValueText = text("", 13, C_ACCENT);
        strengthValueText.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        strengthHead.addView(strengthValueText);
        strengthCol.addView(strengthHead);
        strengthBar = new SeekBar(this);
        strengthBar.setMax(STRENGTH_MAX_PCT - STRENGTH_MIN_PCT);
        strengthBar.setProgressTintList(ColorStateList.valueOf(C_ACCENT));
        strengthBar.setProgressBackgroundTintList(ColorStateList.valueOf(0x33FFFFFF));
        strengthBar.setThumbTintList(ColorStateList.valueOf(0xFFFFFFFF));
        strengthBar.setPadding(dp(4), dp(10), dp(4), dp(10));
        strengthBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int progress, boolean fromUser) {
                updateStrengthLabel();
            }
            @Override public void onStartTrackingTouch(SeekBar s) {}
            @Override public void onStopTrackingTouch(SeekBar s) {}
        });
        strengthCol.addView(strengthBar, marginLp(-4, 2, -4, 0));
        LinearLayout strengthEnds = new LinearLayout(this);
        strengthEnds.setOrientation(LinearLayout.HORIZONTAL);
        strengthEnds.addView(text("元画像に近い", 10, C_TEXT3),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        strengthEnds.addView(text("大きく変える", 10, C_TEXT3));
        strengthCol.addView(strengthEnds);
        img2imgPanel.addView(strengthCol,
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        genArea.addView(img2imgPanel, marginLp(0, 0, 0, 10));

        // --- prompt ---
        promptEdit = field("どんな画像を作りますか？（日本語・英語）");
        promptEdit.setTextSize(16);
        promptEdit.setMinLines(3);
        promptEdit.setGravity(Gravity.TOP | Gravity.START);
        promptEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        promptEdit.setBackground(rounded(C_SURFACE, 16, C_STROKE));
        promptEdit.setPadding(dp(16), dp(14), dp(16), dp(14));
        genArea.addView(promptEdit, marginLp(0, 0, 0, 0));

        translatedPromptView = text("", 12, C_TEXT3);
        translatedPromptView.setPadding(dp(4), dp(6), dp(4), 0);
        translatedPromptView.setVisibility(View.GONE);
        genArea.addView(translatedPromptView);

        // --- generate / cancel (same slot) ---
        generateButton = button("生成する", BTN_PRIMARY);
        genArea.addView(generateButton, sizedLp(ViewGroup.LayoutParams.MATCH_PARENT, dp(56), 0, 12, 0, 0));
        cancelButton = button("中止", BTN_SECONDARY);
        cancelButton.setVisibility(View.GONE);
        genArea.addView(cancelButton, sizedLp(ViewGroup.LayoutParams.MATCH_PARENT, dp(56), 0, 12, 0, 0));

        statusText = text("", 12, C_TEXT2);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(0, dp(10), 0, dp(10));
        genArea.addView(statusText);

        // --- canvas: square image, placeholder, progress along the bottom edge ---
        FrameLayout canvas = new FrameLayout(this);
        canvas.setBackground(rounded(C_SURFACE, 20, C_STROKE));
        canvas.setClipToOutline(true);
        imageView = new ImageView(this) {
            @Override
            protected void onMeasure(int widthSpec, int heightSpec) {
                super.onMeasure(widthSpec, widthSpec);   // square, like every generated image
            }
        };
        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        canvas.addView(imageView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        placeholderView = new LinearLayout(this);
        placeholderView.setOrientation(LinearLayout.VERTICAL);
        placeholderView.setGravity(Gravity.CENTER);
        TextView spark = text("✦", 34, C_ACCENT);
        spark.setGravity(Gravity.CENTER);
        TextView hint = text("プロンプトを入力して「生成する」", 13, C_TEXT3);
        hint.setGravity(Gravity.CENTER);
        placeholderView.addView(spark);
        placeholderView.addView(hint, marginLp(0, 6, 0, 0));
        canvas.addView(placeholderView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setVisibility(View.GONE);
        tintProgress(progressBar);
        canvas.addView(progressBar, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(4), Gravity.BOTTOM));
        genArea.addView(canvas, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // --- actions under the image ---
        imageActions = new LinearLayout(this);
        imageActions.setOrientation(LinearLayout.HORIZONTAL);
        imageActions.setVisibility(View.GONE);
        saveButton = button("↓  保存", BTN_SECONDARY);
        saveButton.setOnClickListener(v -> saveImage());
        Button reworkButton = button("↻  再加工", BTN_SECONDARY);
        reworkButton.setOnClickListener(v -> useResultAsInput());
        Button seedButton = button("シード固定", BTN_GHOST);
        seedButton.setOnClickListener(v -> lockSeed());
        LinearLayout.LayoutParams saveLp = new LinearLayout.LayoutParams(0, dp(48), 1f);
        LinearLayout.LayoutParams reworkLp = new LinearLayout.LayoutParams(0, dp(48), 1f);
        reworkLp.leftMargin = dp(8);
        LinearLayout.LayoutParams seedLp = new LinearLayout.LayoutParams(0, dp(48), 1f);
        seedLp.leftMargin = dp(8);
        imageActions.addView(saveButton, saveLp);
        imageActions.addView(reworkButton, reworkLp);
        imageActions.addView(seedButton, seedLp);
        genArea.addView(imageActions, marginLp(0, 12, 0, 0));

        // --- folded: negative prompt and the other parameters ---
        LinearLayout adv = new LinearLayout(this);
        adv.setOrientation(LinearLayout.VERTICAL);

        negativeEdit = field("例: blurry, low quality");
        adv.addView(labeled("ネガティブプロンプト（CFG>1 のとき有効）", negativeEdit, false));

        LinearLayout paramRow = new LinearLayout(this);
        paramRow.setOrientation(LinearLayout.HORIZONTAL);
        stepsEdit = field("");
        stepsEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        sizeButton = button("", BTN_GHOST);
        sizeButton.setBackground(ripple(rounded(C_SURFACE2, 12, C_STROKE)));
        sizeButton.setTextSize(14);
        sizeButton.setOnClickListener(v -> showSizePicker());
        seedEdit = field("-1");
        seedEdit.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
        paramRow.addView(labeled("ステップ数", stepsEdit, true));
        paramRow.addView(labeled("サイズ", sizeButton, true));
        paramRow.addView(labeled("シード（-1=ランダム）", seedEdit, true));
        ((LinearLayout.LayoutParams) paramRow.getChildAt(1).getLayoutParams()).leftMargin = dp(8);
        ((LinearLayout.LayoutParams) paramRow.getChildAt(2).getLayoutParams()).leftMargin = dp(8);
        adv.addView(paramRow, marginLp(0, 14, 0, 0));

        Button more = button("サンプラー・CFG・TAESD などの生成設定  ›", BTN_GHOST);
        more.setOnClickListener(v -> showGenSettingsDialog());
        adv.addView(more, sizedLp(ViewGroup.LayoutParams.MATCH_PARENT, dp(44), 0, 18, 0, 0));

        TextView guide = text("日本語のプロンプトは LLM（⋮ → LLM設定）で英訳してから生成します。"
                + "モデルを選ぶと推奨のサンプラー／ステップ数／CFG が適用されます。", 11, C_TEXT3);
        guide.setLineSpacing(0, 1.2f);
        adv.addView(guide, marginLp(0, 12, 0, 0));

        genArea.addView(section("詳細設定", "ネガティブプロンプト・ステップ数・サイズ・シード",
                adv, false), marginLp(0, 24, 0, 0));
        body.addView(genArea);

        // ===== API area (tab 2) =====
        apiArea = buildApiPanel();
        body.addView(apiArea);

        // --- log: fixed-height scrollable box ---
        logText = text("", 11, C_TEXT2);
        logText.setTypeface(Typeface.MONOSPACE);
        logText.setTextIsSelectable(true);
        logText.setLineSpacing(0, 1.15f);
        // The page is itself a ScrollView. Claim the gesture as soon as a finger lands on the log
        // box (dispatchTouchEvent runs even when the selectable TextView consumes the touch, which
        // an OnTouchListener on the box does not), and use nested scrolling so the page does not
        // intercept the drag.
        logScroll = new ScrollView(this) {
            @Override
            public boolean dispatchTouchEvent(MotionEvent e) {
                int a = e.getActionMasked();
                if (a == MotionEvent.ACTION_DOWN) {
                    getParent().requestDisallowInterceptTouchEvent(canScrollVertically(1) || canScrollVertically(-1));
                } else if (a == MotionEvent.ACTION_UP || a == MotionEvent.ACTION_CANCEL) {
                    getParent().requestDisallowInterceptTouchEvent(false);
                }
                return super.dispatchTouchEvent(e);
            }
        };
        logScroll.setNestedScrollingEnabled(true);
        logScroll.setBackground(rounded(0xFF08080A, 12, C_STROKE));
        logScroll.setPadding(dp(12), dp(10), dp(12), dp(10));
        logScroll.setClipToPadding(false);
        logScroll.setScrollbarFadingEnabled(false);
        logScroll.addView(logText);
        body.addView(section("ログ", null, logScroll, true, dp(180)), marginLp(0, 12, 0, 0));

        // Restore persisted inputs.
        SharedPreferences p = prefs();
        promptEdit.setText(p.getString(PREFS_PROMPT, "a photo of a cat"));
        negativeEdit.setText(p.getString(PREFS_NEGATIVE, ""));
        stepsEdit.setText(String.valueOf(p.getInt(PREFS_STEPS, 6)));
        seedEdit.setText(String.valueOf(p.getLong(PREFS_SEED, -1)));
        setStrength(p.getFloat(PREFS_STRENGTH, 0.6f));
        sizeButton.setText(sizeLabelFor(selectedSize));
        updateModelText();
        selectTab(Math.max(0, Math.min(TAB_API, p.getInt(PREFS_TAB, TAB_TXT2IMG))));

        ScrollView scroll = new ScrollView(this);
        mainScroll = scroll;
        scroll.setBackgroundColor(C_BG);
        scroll.setFillViewport(true);
        scroll.addView(body);
        return scroll;
    }

    /** API tab: on/off switch, live state, and how other apps call it. */
    private LinearLayout buildApiPanel() {
        LinearLayout area = new LinearLayout(this);
        area.setOrientation(LinearLayout.VERTICAL);

        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(16), dp(14), dp(16), dp(16));
        card.setBackground(rounded(C_SURFACE, 16, C_STROKE));

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        TextView t = text("画像生成 API", 16, C_TEXT);
        t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        col.addView(t);
        LinearLayout stateRow = new LinearLayout(this);
        stateRow.setOrientation(LinearLayout.HORIZONTAL);
        stateRow.setGravity(Gravity.CENTER_VERTICAL);
        apiDot = new View(this);
        LinearLayout.LayoutParams dlp = new LinearLayout.LayoutParams(dp(8), dp(8));
        dlp.rightMargin = dp(8);
        stateRow.addView(apiDot, dlp);
        apiStateText = text("", 12, C_TEXT2);
        stateRow.addView(apiStateText);
        col.addView(stateRow, marginLp(0, 4, 0, 0));
        row.addView(col, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        apiSwitch = new Switch(this);
        int[][] states = {{android.R.attr.state_checked}, {}};
        apiSwitch.setThumbTintList(new ColorStateList(states, new int[]{0xFFFFFFFF, 0xFF8A8A94}));
        apiSwitch.setTrackTintList(new ColorStateList(states, new int[]{C_GRAD_A, 0xFF3A3A44}));
        apiSwitch.setOnClickListener(v -> setApiEnabled(apiSwitch.isChecked()));
        row.addView(apiSwitch);
        card.addView(row);

        apiAddressText = text("", 13, C_TEXT);
        apiAddressText.setTypeface(Typeface.MONOSPACE);
        apiAddressText.setTextIsSelectable(true);
        apiAddressText.setBackground(rounded(C_SURFACE2, 10, 0));
        apiAddressText.setPadding(dp(12), dp(10), dp(12), dp(10));
        card.addView(apiAddressText, marginLp(0, 14, 0, 0));
        TextView scope = text("この端末内のアプリからのみ接続できます（127.0.0.1）。"
                + "Assistant など他のアプリから呼ばれたときは自動で起動します。", 11, C_TEXT3);
        scope.setLineSpacing(0, 1.2f);
        card.addView(scope, marginLp(0, 8, 0, 0));
        area.addView(card);

        LinearLayout ref = new LinearLayout(this);
        ref.setOrientation(LinearLayout.VERTICAL);
        ref.setPadding(dp(16), dp(14), dp(16), dp(16));
        ref.setBackground(rounded(C_SURFACE, 16, C_STROKE));
        TextView rt = text("エンドポイント", 14, C_TEXT);
        rt.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        ref.addView(rt);
        TextView eps = text("POST /generate\n  {\"prompt\": \"…\"} → {\"image\": PNG(Base64), \"seed\", …}\n"
                + "POST /v1/images/generations   OpenAI 互換\n"
                + "GET  /health ・ GET /models ・ POST /cancel", 11, C_TEXT2);
        eps.setTypeface(Typeface.MONOSPACE);
        eps.setTextIsSelectable(true);
        eps.setLineSpacing(0, 1.25f);
        ref.addView(eps, marginLp(0, 8, 0, 0));
        TextView note = text("モデル・サイズ・ステップ数などは、指定がなければこのアプリの設定を使います。"
                + "日本語のプロンプトは LLM設定の LLM で英訳されます。詳しくは ⋮ → マニュアル。", 11, C_TEXT3);
        note.setLineSpacing(0, 1.2f);
        ref.addView(note, marginLp(0, 10, 0, 0));
        area.addView(ref, marginLp(0, 10, 0, 0));
        return area;
    }

    private void selectTab(int tab) {
        currentTab = tab;
        prefs().edit().putInt(PREFS_TAB, tab).apply();
        for (int i = 0; i < tabViews.length; i++) {
            boolean on = i == tab;
            tabViews[i].setTextColor(on ? C_TEXT : C_TEXT3);
            if (on) {
                GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT,
                        new int[]{0xFF2A1E4A, 0xFF3A1A33});
                g.setCornerRadius(dp(10));
                g.setStroke(dp(1), 0x557C3AED);
                tabViews[i].setBackground(g);
            } else {
                tabViews[i].setBackground(ripple(rounded(0x00000000, 10, 0)));
            }
        }
        genArea.setVisibility(tab == TAB_API ? View.GONE : View.VISIBLE);
        img2imgPanel.setVisibility(tab == TAB_IMG2IMG ? View.VISIBLE : View.GONE);
        apiArea.setVisibility(tab == TAB_API ? View.VISIBLE : View.GONE);
        ui.removeCallbacks(apiPollTask);
        if (tab == TAB_API) apiPollTask.run();
        if (generateButton != null && store != null) refreshControls();
    }

    /** Refreshes the API tab once a second while it is visible (the service starts async). */
    private final Runnable apiPollTask = new Runnable() {
        @Override public void run() {
            refreshApiState();
            if (currentTab == TAB_API && started) ui.postDelayed(this, 1000);
        }
    };

    private void refreshApiState() {
        boolean enabled = com.micklab.llamaimage.api.ImageApiService.isEnabled(this);
        boolean listening = com.micklab.llamaimage.api.ImageApiService.isListening();
        int port = com.micklab.llamaimage.api.ImageApiService.port(this);
        String err = com.micklab.llamaimage.api.ImageApiService.lastError();
        boolean pending = SystemClock.elapsedRealtime() - apiToggleAt < 3000;
        boolean want = pending ? apiSwitch.isChecked() : enabled;
        if (!pending) apiSwitch.setChecked(enabled);
        String state = listening && want ? "稼働中" : want ? "起動中…" : listening ? "停止中…" : "停止";
        if (!err.isEmpty() && want && !listening && !pending) state = "エラー: " + err;
        enabled = want;
        apiStateText.setText(state);
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(listening && enabled ? C_OK : enabled ? C_WARN : C_TEXT3);
        apiDot.setBackground(d);
        apiAddressText.setText("http://127.0.0.1:" + port);
    }

    private void setApiEnabled(boolean on) {
        apiToggleAt = SystemClock.elapsedRealtime();
        int port = com.micklab.llamaimage.api.ImageApiService.port(this);
        if (on) {
            com.micklab.llamaimage.api.ImageApiService.start(this, -1);
            appendLog("画像生成 API を開始（ポート " + port + "）");
        } else {
            try {
                startService(com.micklab.llamaimage.api.ImageApiService.stopIntent(this));
                appendLog("画像生成 API を停止");
            } catch (Exception e) {
                Toast.makeText(this, "停止できませんでした: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
        // The service updates "enabled" itself; show the result on the next polls.
        ui.postDelayed(this::refreshApiState, 300);
    }

    private float strengthValue() {
        return (strengthBar.getProgress() + STRENGTH_MIN_PCT) / 100f;
    }

    private void setStrength(float v) {
        int pct = Math.round(clamp(v, 0.05f, 1.0f) * 100);
        strengthBar.setProgress(pct - STRENGTH_MIN_PCT);
        updateStrengthLabel();
    }

    private void updateStrengthLabel() {
        strengthValueText.setText(String.format(Locale.US, "%.2f", strengthValue()));
    }

    // --- view helpers (code-built UI, no XML) ---

    private TextView text(String s, float sp, int color) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        return t;
    }

    private EditText field(String hint) {
        EditText e = new EditText(this);
        e.setHint(hint);
        e.setTextSize(14);
        e.setTextColor(C_TEXT);
        e.setHintTextColor(C_TEXT3);
        e.setBackground(rounded(C_SURFACE2, 12, C_STROKE));
        e.setPadding(dp(12), dp(10), dp(12), dp(10));
        return e;
    }

    private Button button(String label, int style) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setStateListAnimator(null);
        b.setElevation(0);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setPadding(dp(16), 0, dp(16), 0);
        b.setTextSize(style == BTN_PRIMARY ? 16 : 14);
        b.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        b.setSingleLine(true);
        if (style == BTN_PRIMARY) {
            GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT,
                    new int[]{C_GRAD_A, C_GRAD_B});
            g.setCornerRadius(dp(28));
            b.setBackground(ripple(g));
            b.setTextColor(0xFFFFFFFF);
            b.setLetterSpacing(0.04f);
        } else if (style == BTN_SECONDARY) {
            b.setBackground(ripple(rounded(C_SURFACE2, 28, 0xFF3A3A44)));
            b.setTextColor(C_TEXT);
        } else {
            b.setBackground(ripple(rounded(0x00000000, 12, C_STROKE)));
            b.setTextColor(C_TEXT2);
        }
        return b;
    }

    private GradientDrawable rounded(int color, int radiusDp, int strokeColor) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(dp(radiusDp));
        if (strokeColor != 0) g.setStroke(dp(1), strokeColor);
        return g;
    }

    private static RippleDrawable ripple(Drawable content) {
        return new RippleDrawable(ColorStateList.valueOf(0x22FFFFFF), content, null);
    }

    private void tintProgress(ProgressBar bar) {
        bar.setProgressTintList(ColorStateList.valueOf(C_ACCENT));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(0x33FFFFFF));
        bar.setIndeterminateTintList(ColorStateList.valueOf(C_ACCENT));
        bar.setSecondaryProgressTintList(ColorStateList.valueOf(0x667C3AED));   // current NPU stage span
    }

    private LinearLayout.LayoutParams marginLp(int l, int t, int r, int b) {
        return sizedLp(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, l, t, r, b);
    }

    private LinearLayout.LayoutParams sizedLp(int w, int h, int l, int t, int r, int b) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(w, h);
        lp.setMargins(dp(l), dp(t), dp(r), dp(b));
        return lp;
    }

    private LinearLayout labeled(String label, View field, boolean weighted) {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        TextView t = text(label, 11, C_TEXT3);
        t.setSingleLine(true);
        col.addView(t, marginLp(2, 0, 0, 6));
        col.addView(field, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(44)));
        col.setLayoutParams(weighted
                ? new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                : new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));
        return col;
    }

    private View section(String title, String subtitle, View content, boolean expanded) {
        return section(title, subtitle, content, expanded, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    /** A card with a tappable header that folds {@code content} away. */
    private View section(String title, String subtitle, View content, boolean expanded, int contentHeight) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(rounded(C_SURFACE, 16, C_STROKE));

        LinearLayout head = new LinearLayout(this);
        head.setOrientation(LinearLayout.HORIZONTAL);
        head.setGravity(Gravity.CENTER_VERTICAL);
        head.setPadding(dp(16), dp(14), dp(14), dp(14));
        head.setBackground(new RippleDrawable(ColorStateList.valueOf(0x22FFFFFF), null,
                rounded(0xFFFFFFFF, 16, 0)));
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        TextView t = text(title, 14, C_TEXT);
        t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        col.addView(t);
        if (subtitle != null) col.addView(text(subtitle, 11, C_TEXT3), marginLp(0, 2, 0, 0));
        head.addView(col, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView chevron = text("›", 22, C_TEXT2);
        chevron.setRotation(expanded ? 90 : 0);
        head.addView(chevron);
        card.addView(head);

        FrameLayout holder = new FrameLayout(this);
        holder.setPadding(dp(16), 0, dp(16), dp(16));
        holder.addView(content, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, contentHeight));
        holder.setVisibility(expanded ? View.VISIBLE : View.GONE);
        card.addView(holder);

        head.setOnClickListener(v -> {
            boolean open = holder.getVisibility() != View.VISIBLE;
            holder.setVisibility(open ? View.VISIBLE : View.GONE);
            chevron.animate().rotation(open ? 90 : 0).setDuration(150).start();
        });
        return card;
    }

    /** Read-only scrollable text (manual / rights / privacy policy). */
    private void showDocument(String title, String body) {
        TextView t = text(body, 13, C_TEXT2);
        t.setTextIsSelectable(true);
        t.setLineSpacing(0, 1.3f);
        t.setPadding(dp(22), dp(8), dp(22), dp(8));
        ScrollView sv = new ScrollView(this);
        sv.addView(t);
        new AlertDialog.Builder(this).setTitle(title).setView(sv).setPositiveButton("閉じる", null).show();
    }

    private static void setEnabledStyled(View v, boolean enabled) {
        v.setEnabled(enabled);
        v.setAlpha(enabled ? 1f : 0.35f);
    }

    /** Shows {@code bmp} in the canvas (null = empty canvas with the placeholder). */
    private void showImage(Bitmap bmp) {
        imageView.setImageBitmap(bmp);
        placeholderView.setVisibility(bmp == null ? View.VISIBLE : View.GONE);
    }

    /**
     * 「再加工」: the generated image becomes the img2img input. Switches to the 画像→画像 tab and
     * scrolls up to the input / strength slider so the user can tweak the prompt and regenerate.
     */
    private void useResultAsInput() {
        if (lastBitmap == null || generating) return;
        initBitmap = lastBitmap;   // already square; scaled to the target size at generation time
        initThumb.setImageBitmap(initBitmap);
        initThumbHint.setVisibility(View.GONE);
        selectTab(TAB_IMG2IMG);
        appendLog("生成画像を入力画像に設定 (" + initBitmap.getWidth() + "x" + initBitmap.getHeight()
                + (lastSeed >= 0 ? ", seed=" + lastSeed : "") + ")");
        setStatus("生成画像を入力にしました — プロンプトと変化量を調整して「生成する」");
        mainScroll.post(() -> mainScroll.smoothScrollTo(0, Math.max(0, tabBar.getTop() - dp(8))));
    }

    private void lockSeed() {
        if (lastSeed < 0) return;
        seedEdit.setText(String.valueOf(lastSeed));
        Toast.makeText(this, "シード " + lastSeed + " を固定しました（詳細設定で -1 にするとランダム）",
                Toast.LENGTH_SHORT).show();
    }

    private boolean isBusy() {
        return loading || generating || finalizing;
    }

    /** Single place that enables/disables controls for the current state. */
    private void refreshControls() {
        updateKeepScreenOn();
        boolean modelReady = manager != null && manager.isModelLoaded();
        boolean idle = nativeReady && !isBusy();
        ModelCatalog.ModelSpec spec = currentSpec();
        boolean installed = spec != null && store.installedFile(spec) != null;
        boolean canDownload = spec != null && !installed && spec.downloadable() && monitoredDownload == null;

        String label;
        if (loading) label = "モデル読み込み中…";
        else if (modelReady || installed) label = "生成する";
        else if (monitoredDownload != null || finalizing) label = "ダウンロード中…";
        else if (canDownload) label = "モデルをダウンロード（" + spec.sizeLabel() + "）";
        else label = "生成する";
        generateButton.setText(label);
        setEnabledStyled(generateButton, idle && (modelReady || canDownload));
        generateButton.setVisibility(generating ? View.GONE : View.VISIBLE);
        cancelButton.setVisibility(generating ? View.VISIBLE : View.GONE);
        setEnabledStyled(pickInitButton, !generating);
        imageActions.setVisibility(lastBitmap != null && !generating ? View.VISIBLE : View.GONE);
        updateModelText();
    }

    private void showMainMenu(View anchor) {
        PopupMenu pm = new PopupMenu(this, anchor);
        Menu m = pm.getMenu();
        m.add(0, MENU_LLM, 0, "LLM設定（翻訳）");
        m.add(0, MENU_MANUAL, 1, "マニュアル");
        m.add(0, MENU_RIGHTS, 2, "権利・ライセンス");
        m.add(0, MENU_PRIVACY, 3, "Privacy Policy");
        pm.setOnMenuItemClickListener(item -> {
            switch (item.getItemId()) {
                case MENU_LLM: showLlmSettingsDialog(); return true;
                case MENU_MANUAL: showDocument("マニュアル", AppDocs.manual(this)); return true;
                case MENU_RIGHTS: showDocument("権利・ライセンス", AppDocs.rights(this)); return true;
                case MENU_PRIVACY: showDocument("Privacy Policy", AppDocs.privacy()); return true;
                default: return false;
            }
        });
        pm.show();
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
        modelText.setText(s == null ? "未選択" : s.title);
        String state;
        int dot;
        if (s == null) {
            state = "タップしてモデルを選択";
            dot = C_TEXT3;
        } else if (loading) {
            state = "読み込み中…";
            dot = C_WARN;
        } else if (manager != null && manager.isModelLoaded()) {
            String be = manager.activeBackend();
            state = "準備完了 ・ " + ("NPU".equals(be) ? "NPU" : be.isEmpty() || "CPU".equals(be) ? "CPU" : "GPU") + " ・ "
                    + s.recommendedLabel();
            dot = C_OK;
        } else if (store.installedFile(s) != null) {
            state = "端末に保存済み";
            dot = C_WARN;
        } else if (monitoredDownload == s || finalizing) {
            state = "ダウンロード中…";
            dot = C_WARN;
        } else {
            state = "未ダウンロード ・ " + s.sizeLabel();
            dot = C_TEXT3;
        }
        modelStateText.setText(state);
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(dot);
        modelDot.setBackground(d);
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
            card.setBackground(rounded(isActive ? 0x267C3AED : C_SURFACE2, 14,
                    isActive ? C_ACCENT : C_STROKE));
            LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            cardLp.bottomMargin = dp(10);

            TextView title = new TextView(this);
            title.setText(s.title + (isActive && loadedNow ? "（使用中）" : ""));
            title.setTypeface(Typeface.DEFAULT_BOLD);
            title.setTextSize(15);
            title.setTextColor(C_TEXT);
            card.addView(title);

            TextView info = new TextView(this);
            String state = installed ? "✓ 端末に保存済み"
                    : downloadingThis ? "ダウンロード中…"
                    : "未ダウンロード（初回のみ " + s.sizeLabel() + "）";
            info.setText(state + "\n" + s.sizeLabel() + " ・ 推奨: " + s.recommendedLabel());
            info.setTextSize(12);
            info.setTextColor(installed ? C_OK : C_TEXT2);
            card.addView(info);

            TextView desc = new TextView(this);
            desc.setText(s.description);
            desc.setTextSize(12);
            desc.setTextColor(C_TEXT2);
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
        where.setText("ダウンロード元は Hugging Face / Civitai（URL と SHA-256 はアプリに内蔵）。保存先: "
                + (ext != null ? ext.getAbsolutePath() : "(利用不可)")
                + "\n高速VAEデコード（TAESD, 既定 ON）の切替は「生成設定」から。");
        where.setTextSize(11);
        where.setTextColor(C_TEXT3);
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
        refreshControls();
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
                        npu().delete(spec.id);
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
        ModelCatalog.ModelSpec spec = currentSpec();
        String npuId = backend == BACKEND_NPU && npuUsable(spec) ? spec.id : null;
        boolean gpu = npuId == null && backend == BACKEND_GPU && manager != null && !manager.gpuDevice().isEmpty();
        return new SdModelManager.LoadConfig(modelFile.getAbsolutePath(), threads, wtype,
                taesd != null ? taesd.getAbsolutePath() : "", vaeTiling, gpu, npuId);
    }

    private com.micklab.llamaimage.npu.NpuManager npu() {
        if (npuManager == null) npuManager = new com.micklab.llamaimage.npu.NpuManager(this);
        return npuManager;
    }

    /** NPU backend possible for this model on this device (SD1.5 LCM safetensors, Snapdragon). */
    static boolean npuUsable(ModelCatalog.ModelSpec spec) {
        return spec != null && spec.npuCapable() && com.micklab.llamaimage.npu.NpuManager.deviceMayHaveHtp();
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
            if (spec == null) {
                setStatus("モデルを選択してください");
                if (!chooserShownThisLaunch) showModelChooser();
            } else {
                setStatus("「" + spec.title + "」をダウンロードすると生成できます");
            }
            refreshControls();
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
            setStatus("準備完了 — プロンプトを入力して「生成する」");
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
                if (cfg.npuModelId != null) {
                    prefs().edit().putBoolean(PREFS_NPU_PENDING, true).commit();   // crash guard
                    com.micklab.llamaimage.npu.NpuManager nm = npu();
                    appendLogUi("NPU: 準備開始（" + spec.title + "）");
                    nm.ensureAssets(this::setStatusUi);
                    appendLogUi("NPU: 補助ファイル OK");
                    if (!nm.isBuilt(spec.id, spec.sha256)) {
                        appendLogUi("NPU 用モデルを端末で作成します（初回のみ・約 2GB）");
                        nm.ensureBuilt(spec.id, file, spec.sha256, s -> runOnUiThread(() -> showBuildProgress(s)));
                        appendLogUi("NPU: モデル作成 完了");
                    }
                    setStatusUi("NPU を準備中…（初回は最適化に数分かかります）");
                    appendLogUi("NPU: セッション作成（初回は NPU 向け最適化）開始");
                    err = manager.loadNpu(cfg, nm, new com.micklab.llamaimage.npu.NpuPipeline.Log() {
                        @Override public void log(String msg) {
                            appendLogUi(msg);
                        }

                        @Override public void stage(int index, int total, String name) {
                            runOnUiThread(() -> startNpuStage(index, total, name));
                        }
                    });
                    runOnUiThread(this::stopNpuStage);
                    prefs().edit().putBoolean(PREFS_NPU_PENDING, false).commit();
                } else {
                    setStatusUi("モデル読込中… " + spec.title);
                    err = manager.load(cfg);
                }
            } catch (Throwable t) {
                err = t.getMessage() != null ? t.getMessage() : t.toString();
            }
            final String e = err;
            final long ms = SystemClock.elapsedRealtime() - t0;
            prefs().edit().putBoolean(PREFS_NPU_PENDING, false).commit();   // finished (ok or error)
            runOnUiThread(() -> {
                loading = false;
                stopNpuStage();
                if (e.isEmpty()) {
                    appendLog("モデル読込成功: " + spec.title + " (" + (ms / 1000) + "s, "
                            + (cfg.npuModelId != null ? "NPU, " : cfg.useGpu ? "GPU " + manager.activeBackend() + ", " : "CPU ")
                            + cfg.threads + "スレッド" + wtypeSuffix()
                            + (cfg.taesdPath.isEmpty() ? "" : ", TAESD") + (cfg.vaeTiling ? ", VAEタイル" : "") + ")");
                    setStatus("準備完了 — プロンプトを入力して「生成する」");
                } else if (cfg.useGpu || cfg.npuModelId != null) {
                    // GPU / NPU load failed (driver, memory, unsupported SoC): fall back to the CPU.
                    appendLog((cfg.npuModelId != null ? "NPU" : "GPU") + " での読込に失敗したため CPU に切り替えます: " + e);
                    backend = BACKEND_CPU;
                    saveGenSettings();
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
            ModelCatalog.ModelSpec s = currentSpec();
            if (s != null && store.installedFile(s) == null && s.downloadable() && monitoredDownload == null) {
                confirmDownload(s);   // first run: the main button doubles as "download the model"
            } else if (s == null) {
                showModelChooser();
            } else {
                Toast.makeText(this, "モデルの準備中です", Toast.LENGTH_SHORT).show();
            }
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
        SdModelManager.LoadConfig lc = manager.getLoadedConfig();
        final boolean onNpu = lc != null && lc.npuModelId != null;
        final int size = onNpu ? com.micklab.llamaimage.npu.NpuPipeline.SIZE : snapTo64(selectedSize);
        if (onNpu && selectedSize != size) appendLog("NPU は 512px 固定のため 512px で生成します");
        final long seed = parseLong(seedEdit.getText().toString(), -1);
        final float strength = strengthValue();
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
            showImage(bmp);
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
        showImage(lastBitmap);   // null clears the view
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
                    initThumbHint.setVisibility(View.GONE);
                    setStatus("入力画像を選択しました — 変化量を調整して「生成する」");
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

        final String gpuDev = manager != null ? manager.gpuDevice() : "";
        final boolean npuOk = npuUsable(spec);
        final int[] selBackend = {backend};
        addLabel(c, "実行デバイス", dp(12));
        Button beBtn = new Button(this);
        beBtn.setText("実行: " + BACKEND_LABELS[selBackend[0]]);
        beBtn.setOnClickListener(v -> {
            int nb = selBackend[0];
            for (int i = 0; i < 3; i++) {
                nb = (nb + 1) % 3;
                if (nb == BACKEND_CPU || (nb == BACKEND_GPU && !gpuDev.isEmpty()) || (nb == BACKEND_NPU && npuOk)) break;
            }
            selBackend[0] = nb;
            beBtn.setText("実行: " + BACKEND_LABELS[nb]);
        });
        c.addView(beBtn);
        addHint(c, "CPU：標準。 GPU：" + (gpuDev.isEmpty() ? "この端末では利用不可" : gpuDev + "（この端末では CPU より遅い場合あり）")
                + "。 NPU：" + (npuOk ? "Snapdragon の NPU で高速生成（512px 固定・LCM）。初回のみ端末で変換（約 2GB）と最適化に数分"
                : "このモデル／端末では利用不可（SD1.5 LCM の safetensors モデル＋Snapdragon が必要）")
                + "。読込に失敗した場合は自動で CPU に戻ります。");

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
                    backend = selBackend[0];
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
        t.setTextColor(C_TEXT3);
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
        backend = p.getInt(PREFS_BACKEND, p.getBoolean(PREFS_USE_GPU, false) ? BACKEND_GPU : BACKEND_CPU);
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
                .putInt(PREFS_BACKEND, backend)
                .remove(PREFS_USE_GPU)
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
                .putFloat(PREFS_STRENGTH, strengthValue())
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
        if (statusText != null) statusText.setText(s);
    }

    private void setStatusUi(String s) {
        runOnUiThread(() -> setStatus(s));
    }

    private static String formatDuration(long sec) {
        if (sec < 60) return sec + "s";
        return (sec / 60) + "分" + (sec % 60) + "s";
    }

    static Bitmap rgbToBitmap(byte[] rgb, int w, int h) {
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
    static byte[] bitmapToRgb(Bitmap bmp) {
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
    static int snapTo64(int v) {
        int snapped = (v / 64) * 64;
        return snapped < 64 ? 64 : snapped;
    }

    private static final String APP_LOG = "app_log.txt";
    private static final String PREFS_NPU_PENDING = "npu_load_pending";   // set while an NPU load runs

    /** Also appended to files/app_log.txt so the next launch can show what happened before a crash. */
    private void persistLog(String line) {
        try {
            File f = new File(getFilesDir(), APP_LOG);
            if (f.length() > 200_000) {
                File old = new File(getFilesDir(), APP_LOG + ".1");
                old.delete();
                f.renameTo(old);
            }
            try (java.io.FileWriter w = new java.io.FileWriter(f, true)) {
                w.write(new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + " " + line + "\n");
            }
        } catch (IOException ignore) {
        }
    }

    /** Last lines of the previous run's log (shown at startup), or "". */
    private String previousLogTail(int maxLines) {
        File f = new File(getFilesDir(), APP_LOG);
        if (!f.isFile()) return "";
        try {
            java.util.List<String> lines = new java.util.ArrayList<>();
            try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(f))) {
                String l;
                while ((l = r.readLine()) != null) {
                    lines.add(l);
                    if (lines.size() > maxLines) lines.remove(0);
                }
            }
            return android.text.TextUtils.join("\n", lines);
        } catch (IOException e) {
            return "";
        }
    }

    private void appendLog(String msg) {
        if (msg == null || logText == null) return;
        persistLog(msg.trim());
        logText.append(msg.endsWith("\n") ? msg : msg + "\n");
        CharSequence t = logText.getText();
        if (t.length() > MAX_LOG_CHARS) {
            logText.setText(t.subSequence(t.length() - MAX_LOG_CHARS / 2, t.length()));
        }
        logScroll.post(() -> logScroll.scrollTo(0, logText.getHeight()));
    }

    private void appendLogUi(String msg) {
        runOnUiThread(() -> appendLog(msg));
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
