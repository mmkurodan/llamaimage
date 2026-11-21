package com.micklab.llamaimage;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Prompt pre-processing through an Ollama-compatible LLM (the LLM AI Server, com.micklab.llama):
 * Japanese → English translation and an optional NSFW check, plus a local keyword filter.
 *
 * Only prompts that contain non-ASCII text are sent for translation — an English prompt goes to
 * CLIP unchanged (no LLM round-trip, and no risk of the LLM rewording it). Translations are cached
 * so re-generating with the same prompt costs nothing. All network calls block; call them off the
 * main thread.
 */
public final class PromptTranslator {

    /** Result of {@link #prepare}. */
    public static final class Prepared {
        public final String prompt;
        public final String negative;
        public final boolean blocked;      // NSFW restriction hit
        public final String note;          // user-facing remark ("" if none)

        Prepared(String prompt, String negative, boolean blocked, String note) {
            this.prompt = prompt;
            this.negative = negative;
            this.blocked = blocked;
            this.note = note;
        }
    }

    public interface Log {
        void log(String msg);
    }

    // Matched against whole words (exact) or word prefixes (stems), so e.g. "cocktail",
    // "peacock" or "Dickens" no longer trip the filter.
    private static final String[] NSFW_EXACT = {
        "nsfw", "nude", "nudes", "naked", "nipple", "nipples", "penis", "vagina", "breast", "breasts",
        "genital", "genitals", "genitalia", "porn", "xxx", "hentai", "topless", "bottomless",
        "pussy", "cock", "cocks", "dick", "dicks", "boobs", "tits", "lewd", "fetish", "intercourse",
    };
    private static final String[] NSFW_STEMS = {
        "pornograph", "erotic", "masturbat", "obscen", "nudity",
    };
    private static final Pattern NON_WORD = Pattern.compile("[^a-z0-9]+");

    private static final int CACHE_SIZE = 32;
    private final Map<String, String> cache = new LinkedHashMap<String, String>(16, 0.75f, true) {
        @Override protected boolean removeEldestEntry(Map.Entry<String, String> e) {
            return size() > CACHE_SIZE;
        }
    };

    private final String baseUrl;
    private final String model;
    private final Log log;

    public PromptTranslator(String baseUrl, String model, Log log) {
        this.baseUrl = normalizeBaseUrl(baseUrl);
        this.model = (model == null || model.trim().isEmpty()) ? "default" : model.trim();
        this.log = log;
    }

    public boolean sameEndpoint(String url, String m) {
        return baseUrl.equals(normalizeBaseUrl(url))
                && model.equals((m == null || m.trim().isEmpty()) ? "default" : m.trim());
    }

    /**
     * Translates (if needed) and screens the prompt pair. {@code translateNegative} should be
     * false when CFG is 1 (the negative prompt is ignored then, so it is not worth an LLM call).
     */
    public Prepared prepare(String rawPrompt, String rawNegative, boolean translateNegative,
                            boolean nsfwRestriction) {
        List<String> notes = new ArrayList<>();
        String prompt = translateIfNeeded(rawPrompt, notes);
        String negative = translateNegative ? translateIfNeeded(rawNegative, notes) : rawNegative;
        if (nsfwRestriction) {
            if (containsNsfwWord(prompt)) {
                return new Prepared(prompt, negative, true, "性的表現を含む語を検出しました");
            }
            if (!baseUrl.isEmpty() && checkNsfwWithLlm(prompt)) {
                return new Prepared(prompt, negative, true, "LLM が性的コンテンツと判定しました");
            }
        }
        return new Prepared(prompt, negative, false, android.text.TextUtils.join(" / ", notes));
    }

    /** True if the text has non-ASCII letters (e.g. Japanese), which CLIP's vocabulary lacks. */
    public static boolean needsTranslation(String s) {
        if (s == null) return false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c > 0x7F && Character.isLetter(c)) return true;
        }
        return false;
    }

    private String translateIfNeeded(String text, List<String> notes) {
        if (text == null || text.trim().isEmpty() || !needsTranslation(text)) return text;
        synchronized (cache) {
            String hit = cache.get(text);
            if (hit != null) return hit;
        }
        if (baseUrl.isEmpty()) {
            notes.add("LLM 未設定のため翻訳できません");
            return text;
        }
        try {
            String out = cleanLlmOutput(requestLlmText(buildTranslationPrompt(text), 30000));
            if (out.isEmpty()) {
                notes.add("翻訳結果が空のため原文で生成");
                return text;
            }
            log.log("翻訳: \"" + text + "\" → \"" + out + "\"");
            synchronized (cache) {
                cache.put(text, out);
            }
            return out;
        } catch (Exception e) {
            log.log("翻訳エラー（原文で生成）: " + e.getMessage());
            notes.add("LLM に接続できず原文で生成（日本語は理解されません）");
            return text;
        }
    }

    private static String buildTranslationPrompt(String input) {
        return "Translate the following text to English for use as a Stable Diffusion image generation prompt. "
                + "Keep comma-separated tags as tags. "
                + "Output ONLY the English translation — no preamble, no explanation, no quotes.\n"
                + input;
    }

    private static String buildNsfwCheckPrompt(String text) {
        return "You are a content moderator. Determine if the following text describes pornographic or sexually explicit content. "
                + "Output only the single word NSFW if it does, or OK if it does not. "
                + "Animals, people, food, nature, landscapes, and everyday subjects are NOT NSFW.\n"
                + text;
    }

    private boolean checkNsfwWithLlm(String text) {
        try {
            String r = cleanLlmOutput(requestLlmText(buildNsfwCheckPrompt(text), 15000));
            return r.toUpperCase(Locale.US).startsWith("NSFW");
        } catch (Exception e) {
            log.log("NSFWチェックをスキップ（LLM 接続不可）: " + e.getMessage());
            return false;
        }
    }

    /** Strips reasoning blocks, quotes and "Translation:"-style preambles; keeps the first line. */
    static String cleanLlmOutput(String s) {
        if (s == null) return "";
        String t = s.replaceAll("(?is)<think>.*?</think>", "").trim();
        for (String line : t.split("\\r?\\n")) {
            String l = line.trim();
            if (l.isEmpty()) continue;
            l = l.replaceFirst("(?i)^(english|translation|prompt)\\s*:\\s*", "");
            if (l.length() >= 2 && (l.startsWith("\"") && l.endsWith("\"")
                    || l.startsWith("'") && l.endsWith("'")
                    || l.startsWith("「") && l.endsWith("」"))) {
                l = l.substring(1, l.length() - 1).trim();
            }
            return l;
        }
        return "";
    }

    public static boolean containsNsfwWord(String prompt) {
        if (prompt == null || prompt.isEmpty()) return false;
        for (String w : NON_WORD.split(prompt.toLowerCase(Locale.US))) {
            if (w.isEmpty()) continue;
            for (String x : NSFW_EXACT) if (w.equals(x)) return true;
            for (String x : NSFW_STEMS) if (w.startsWith(x)) return true;
        }
        return false;
    }

    // --- Ollama-compatible HTTP API ---

    private String requestLlmText(String prompt, int timeoutMs) throws Exception {
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(baseUrl + "/api/generate").openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(timeoutMs);
            conn.setDoOutput(true);

            JSONObject req = new JSONObject();
            req.put("model", model);
            req.put("prompt", prompt);
            req.put("stream", false);
            req.put("think", false);                                   // no reasoning preamble
            req.put("options", new JSONObject().put("temperature", 0)); // deterministic

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

    /** GET /api/tags reachability. */
    public static boolean isAvailable(String baseUrl) {
        String b = normalizeBaseUrl(baseUrl);
        if (b.isEmpty()) return false;
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(b + "/api/tags").openConnection();
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);
            return conn.getResponseCode() < 400;
        } catch (Exception e) {
            return false;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    /** Model names from GET /api/tags. */
    public static List<String> fetchModelNames(String baseUrl) throws Exception {
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(normalizeBaseUrl(baseUrl) + "/api/tags").openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(15000);
            int code = conn.getResponseCode();
            String body = readStream(code >= 400 ? conn.getErrorStream() : conn.getInputStream());
            if (code >= 400) throw new Exception("HTTP " + code);
            JSONArray arr = new JSONObject(body).optJSONArray("models");
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

    public static String normalizeBaseUrl(String url) {
        if (url == null) return "";
        String s = url.trim();
        while (s.endsWith("/")) s = s.substring(0, s.length() - 1);
        return s;
    }

    private static String readStream(InputStream in) throws Exception {
        if (in == null) return "";
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) sb.append(line).append('\n');
        }
        return sb.toString().trim();
    }
}
