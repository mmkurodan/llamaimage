package com.micklab.llamaimage;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.drawable.GradientDrawable;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {

    private static final int REQ_PICK_MODEL = 1001;
    private static final String MODEL_FILENAME = "model.gguf";
    private static final String PREFS_NAME = "llamaimage_prefs";
    private static final String PREFS_LLM_URL = "llm_base_url";
    private static final String PREFS_LLM_MODEL = "llm_model";
    private static final String DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434";
    private static final String LLM_PACKAGE = "com.micklab.llama";
    private static final String LLM_SERVICE_CLASS = "com.micklab.llama.OllamaForegroundService";
    private static final String LLM_SERVICE_ACTION = "com.micklab.llama.START_SERVICE";
    private static final String LLM_PLAY_STORE_URL =
            "https://play.google.com/store/apps/details?id=com.micklab.llama";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private volatile SdModelManager manager;
    private volatile boolean nativeReady = false;
    private volatile long genStart = 0;
    private volatile long imgStart = 0;
    private volatile int currentImageIndex = 0;
    private volatile int totalImages = 1;

    private String llmBaseUrl = DEFAULT_LLM_BASE_URL;
    private String llmModel = "";

    private LinearLayout rootBody;
    private EditText promptEdit;
    private EditText negativeEdit;
    private EditText stepsEdit;
    private EditText sizeEdit;
    private EditText countEdit;
    private Button generateButton;
    private Button translateButton;
    private ProgressBar progressBar;
    private ImageView imageView;
    private TextView logText;
    private Bitmap lastBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_HIDDEN
                | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        loadLlmSettings();
        setContentView(buildUi());
        rootBody.requestFocus();

        generateButton.setOnClickListener(v -> generate());
        generateButton.setEnabled(false);
        translateButton.setOnClickListener(v -> translatePrompt());

        setStatus("ネイティブ初期化中…");
        worker.submit(this::initNative);
    }

    private void initNative() {
        try {
            SdModelManager m = SdModelManager.get();
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
            long el = (System.currentTimeMillis() - imgStart) / 1000;
            String eta = "";
            if (step > 0 && step < steps) {
                eta = " 残り~" + (el * (steps - step) / step) + "s";
            }
            String imgInfo = (totalImages > 1) ? " (" + (currentImageIndex + 1) + "/" + totalImages + ")" : "";
            setStatus("生成中" + imgInfo + " " + step + "/" + steps + " (" + el + "s" + eta + ")");
        });
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
        guide.setText("プロンプトを入力（日本語可）→「翻訳」で英語化→「生成」。モデル・LLM設定は上部メニューから。");
        guide.setPadding(0, 0, 0, dp(8));
        body.addView(guide);

        promptEdit = new EditText(this);
        promptEdit.setHint("プロンプト（日本語・英語）");
        promptEdit.setText("a photo of a cat");
        body.addView(promptEdit);

        negativeEdit = new EditText(this);
        negativeEdit.setHint("ネガティブプロンプト (任意)");
        body.addView(negativeEdit);

        LinearLayout paramRow = new LinearLayout(this);
        paramRow.setOrientation(LinearLayout.HORIZONTAL);

        stepsEdit = new EditText(this);
        stepsEdit.setHint("steps");
        stepsEdit.setText("20");
        stepsEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        sizeEdit = new EditText(this);
        sizeEdit.setHint("size");
        sizeEdit.setText("512");
        sizeEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        countEdit = new EditText(this);
        countEdit.setHint("回数");
        countEdit.setText("1");
        countEdit.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        paramRow.addView(stepsEdit);
        paramRow.addView(sizeEdit);
        paramRow.addView(countEdit);
        body.addView(paramRow);

        LinearLayout btnRow = new LinearLayout(this);
        btnRow.setOrientation(LinearLayout.HORIZONTAL);

        translateButton = new Button(this);
        translateButton.setText("翻訳");
        translateButton.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        generateButton = new Button(this);
        generateButton.setText("生成");
        generateButton.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 2f));

        btnRow.addView(translateButton);
        btnRow.addView(generateButton);
        body.addView(btnRow);

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
        menu.add(0, 4, 3, "LLM設定").setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case 1: pickModel(); return true;
            case 2: generate(); return true;
            case 3: saveImage(); return true;
            case 4: showLlmSettingsDialog(); return true;
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
        if (requestCode != REQ_PICK_MODEL || resultCode != RESULT_OK || data == null) return;
        final Uri uri = data.getData();
        if (uri == null) return;
        setBusyUi(true);
        setStatus("モデルを準備中…");
        worker.submit(() -> loadModel(uri));
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
            String err = manager.load(dest.getAbsolutePath(), 0);
            if (err.isEmpty()) {
                appendLogUi("モデル読込成功");
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

    // --- image generation (supports count > 1, shows each image as it completes) ---

    private void generate() {
        if (manager == null || !manager.isModelLoaded()) {
            Toast.makeText(this, "先にモデルを読み込んでください", Toast.LENGTH_SHORT).show();
            return;
        }
        final String prompt = promptEdit.getText().toString();
        final String negative = negativeEdit.getText().toString();
        final int steps = parseInt(stepsEdit.getText().toString(), 20);
        final int size = parseInt(sizeEdit.getText().toString(), 512);
        final int count = Math.max(1, parseInt(countEdit.getText().toString(), 1));

        setBusyUi(true);
        totalImages = count;
        currentImageIndex = 0;
        genStart = System.currentTimeMillis();
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setMax(steps);
        progressBar.setProgress(0);
        setStatus("生成開始… (" + size + "px/" + steps + "step" + (count > 1 ? " x" + count + "枚" : "") + ")");
        appendLog("生成開始: \"" + prompt + "\" " + size + "px " + steps + "step" + (count > 1 ? " x" + count : ""));

        worker.submit(() -> {
            for (int i = 0; i < count; i++) {
                final int imgIdx = i;
                currentImageIndex = i;
                imgStart = System.currentTimeMillis();
                // Reset progress bar for each image so onGenProgress starts fresh
                runOnUiThread(() -> {
                    progressBar.setMax(steps);
                    progressBar.setProgress(0);
                });
                try {
                    byte[] rgb = manager.generate(prompt, negative, size, size, steps, 7.0f, -1);
                    long ms = System.currentTimeMillis() - imgStart;
                    if (rgb == null) {
                        appendLogUi("生成失敗 (" + (imgIdx + 1) + "/" + count + ")");
                        if (imgIdx == count - 1) {
                            runOnUiThread(() -> {
                                setStatus("生成失敗");
                                progressBar.setVisibility(View.GONE);
                                setBusyUi(false);
                            });
                        }
                        continue;
                    }
                    final Bitmap bmp = rgbToBitmap(rgb, size, size);
                    final long seed = manager.getLastSeed();
                    final long imgMs = ms;
                    appendLogUi("生成完了 (" + (imgIdx + 1) + "/" + count + ") seed=" + seed + " " + (ms / 1000) + "s");
                    final boolean isLast = (imgIdx == count - 1);
                    runOnUiThread(() -> {
                        lastBitmap = bmp;
                        imageView.setImageBitmap(bmp);
                        progressBar.setProgress(progressBar.getMax());
                        if (isLast) {
                            long totalMs = System.currentTimeMillis() - genStart;
                            setStatus("完了" + (count > 1 ? " " + count + "枚" : "")
                                    + " seed=" + seed + " (" + (totalMs / 1000) + "s)");
                            setBusyUi(false);
                        } else {
                            setStatus("完了 (" + (imgIdx + 1) + "/" + count + ") seed=" + seed
                                    + " (" + (imgMs / 1000) + "s) — 次を生成中…");
                        }
                    });
                } catch (Throwable t) {
                    appendLogUi("生成エラー (" + (imgIdx + 1) + "/" + count + "): " + t);
                    runOnUiThread(() -> {
                        setStatus("生成エラー: " + t.getClass().getSimpleName());
                        progressBar.setVisibility(View.GONE);
                        setBusyUi(false);
                    });
                    return;
                }
            }
        });
    }

    // --- Japanese → English prompt translation via LLM API ---

    private void translatePrompt() {
        String baseUrl = normalizeBaseUrl(llmBaseUrl);
        if (baseUrl.isEmpty()) {
            Toast.makeText(this, "上部メニュー「LLM設定」でURLを設定してください", Toast.LENGTH_LONG).show();
            return;
        }
        String input = promptEdit.getText().toString().trim();
        if (input.isEmpty()) {
            Toast.makeText(this, "プロンプトを入力してください", Toast.LENGTH_SHORT).show();
            return;
        }
        String model = llmModel.isEmpty() ? "default" : llmModel;

        setBusyUi(true);
        setStatus("翻訳中…");

        final String translationPrompt =
                "Translate the following Japanese text into English for use as a Stable Diffusion "
                + "image generation prompt. Output only the English translation, nothing else:\n"
                + input;

        worker.submit(() -> {
            try {
                String result = requestLlmText(baseUrl, model, translationPrompt, 30000);
                if (result == null || result.isEmpty()) {
                    appendLogUi("翻訳失敗: 空のレスポンス");
                    runOnUiThread(() -> {
                        setStatus("翻訳失敗 — LLMからの応答が空です");
                        setBusyUi(false);
                    });
                    return;
                }
                appendLogUi("翻訳: \"" + input + "\" → \"" + result + "\"");
                runOnUiThread(() -> {
                    promptEdit.setText(result);
                    setStatus("翻訳完了 — 内容を確認して「生成」を押してください");
                    setBusyUi(false);
                });
            } catch (Throwable t) {
                String msg = t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName();
                appendLogUi("翻訳エラー: " + t);
                runOnUiThread(() -> {
                    setStatus("翻訳エラー: " + msg);
                    Toast.makeText(this, "翻訳エラー: " + msg, Toast.LENGTH_LONG).show();
                    setBusyUi(false);
                });
            }
        });
    }

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

    // --- SharedPreferences for LLM config ---

    private void loadLlmSettings() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        llmBaseUrl = prefs.getString(PREFS_LLM_URL, DEFAULT_LLM_BASE_URL);
        llmModel = prefs.getString(PREFS_LLM_MODEL, "");
    }

    private void saveLlmSettings() {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                .putString(PREFS_LLM_URL, llmBaseUrl)
                .putString(PREFS_LLM_MODEL, llmModel)
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
        generateButton.setEnabled(!busy && manager != null && manager.isModelLoaded());
        translateButton.setEnabled(!busy);
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
