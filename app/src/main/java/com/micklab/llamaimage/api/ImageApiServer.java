package com.micklab.llamaimage.api;

import android.util.Base64;
import android.util.Log;

import com.micklab.llamaimage.ApiGenerator;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Minimal HTTP/1.1 server for image generation, modelled after OcrApiServer in the companion
 * pdf app (raw ServerSocket + ThreadPoolExecutor, no external library). Bound to loopback only.
 *
 * Endpoints:
 *  GET  /health                   {"status":"ok","busy":..,"model_loaded":..,"model":..}
 *  GET  /models                   {"models":[{"id","title","installed","selected",...}]}
 *  POST /generate                 JSON request -> {"image": base64, "mime", "seed", ...}
 *  POST /v1/images/generations    OpenAI-compatible subset (n=1, size "WxH", b64_json)
 *  POST /cancel                   stop the running generation at its next step
 *
 * POST /generate fields (all optional except prompt; unset = the app's saved settings):
 *  prompt, negative_prompt, width, height (or size), steps, cfg_scale, seed (-1 random),
 *  sampler, translate (default true), init_image (base64 PNG/JPEG -> img2img), strength,
 *  change ("slight" | "normal" | "strong": img2img strength chosen per model; "strength" overrides),
 *  format ("png" | "jpeg"), model (catalog id from GET /models; default = the app's selection).
 */
public final class ImageApiServer {

    public static final int DEFAULT_PORT = 8766;
    private static final String TAG = "ImageApiServer";
    private static final int MAX_BODY_BYTES = 30 * 1024 * 1024;   // base64 init images

    public interface Listener {
        void onServerStarted(int port);
        void onServerStopped();
        void onServerError(String message);
        void onRequest(String method, String path);
        /** True while an image is being generated (used to hold a wake lock). */
        void onBusy(boolean busy);
    }

    private final ApiGenerator generator;
    private int port = DEFAULT_PORT;
    private volatile ServerSocket serverSocket;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final List<Socket> connections = new ArrayList<>();
    private volatile Listener listener;

    private final ThreadPoolExecutor pool = new ThreadPoolExecutor(
            2, 8, 60L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(32),
            new ThreadPoolExecutor.CallerRunsPolicy());

    public ImageApiServer(ApiGenerator generator) {
        this.generator = generator;
    }

    public void setPort(int p) {
        port = p;
    }

    public void setListener(Listener l) {
        listener = l;
    }

    public boolean isRunning() {
        return running.get();
    }

    public void start() {
        Thread t = new Thread(this::runServer, "ImageApiServer-accept");
        t.setDaemon(true);
        t.start();
    }

    public void stop() {
        running.set(false);
        try {
            ServerSocket s = serverSocket;
            if (s != null) s.close();
        } catch (IOException ignored) {
        }
        synchronized (connections) {
            for (Socket s : connections) {
                try { s.close(); } catch (IOException ignored) { }
            }
            connections.clear();
        }
        Listener l = listener;
        if (l != null) l.onServerStopped();
    }

    // ── Server loop ────────────────────────────────────────────────────────────

    private void runServer() {
        try {
            // Loopback only: the API is for apps on this device (no LAN exposure).
            serverSocket = new ServerSocket(port, 16, InetAddress.getByName("127.0.0.1"));
            running.set(true);
            Log.i(TAG, "Listening on 127.0.0.1:" + port);
            Listener l = listener;
            if (l != null) l.onServerStarted(port);
            while (running.get()) {
                final Socket sock;
                try {
                    sock = serverSocket.accept();
                } catch (IOException e) {
                    break;
                }
                synchronized (connections) { connections.add(sock); }
                pool.execute(() -> {
                    try {
                        handleConnection(sock);
                    } catch (Exception e) {
                        Log.d(TAG, "Connection closed: " + e.getMessage());
                    } finally {
                        synchronized (connections) { connections.remove(sock); }
                        try { sock.close(); } catch (IOException ignored) { }
                    }
                });
            }
        } catch (Exception e) {
            // Only reachable when binding fails (e.g. port in use); accept() errors break the loop.
            Log.e(TAG, "Server error", e);
            Listener l = listener;
            if (l != null) l.onServerError(e.getMessage() != null ? e.getMessage() : "error");
        } finally {
            running.set(false);
        }
    }

    // ── Request handling ─────────────────────────────────────────────────────

    private void handleConnection(Socket sock) throws IOException {
        InputStream in = sock.getInputStream();
        OutputStream out = sock.getOutputStream();

        String reqLine = readLine(in);
        if (reqLine == null) return;
        String[] parts = reqLine.split(" ");
        if (parts.length < 2) return;
        String method = parts[0].toUpperCase(Locale.US);
        String path = parts[1];
        int q = path.indexOf('?');
        if (q >= 0) path = path.substring(0, q);
        path = path.toLowerCase(Locale.US);

        Map<String, String> headers = new HashMap<>();
        while (true) {
            String line = readLine(in);
            if (line == null || line.isEmpty()) break;
            int idx = line.indexOf(':');
            if (idx > 0) headers.put(line.substring(0, idx).trim().toLowerCase(Locale.US),
                    line.substring(idx + 1).trim());
        }

        Listener l = listener;
        if (l != null) l.onRequest(method, path);

        if ("OPTIONS".equals(method)) {
            send(out, 204, "text/plain", "");
        } else if ("GET".equals(method) && "/health".equals(path)) {
            sendJson(out, 200, health());
        } else if ("GET".equals(method) && "/models".equals(path)) {
            try {
                sendJson(out, 200, new JSONObject().put("models", generator.listModels()).toString());
            } catch (JSONException e) {
                sendJson(out, 500, jsonError(e.getMessage()));
            }
        } else if ("POST".equals(method) && "/cancel".equals(path)) {
            ApiGenerator.cancel();
            sendJson(out, 200, "{\"status\":\"cancel requested\"}");
        } else if ("POST".equals(method) && ("/generate".equals(path)
                || "/v1/images/generations".equals(path))) {
            boolean openAi = path.startsWith("/v1/");
            String body = readBody(in, headers.get("content-length"));
            handleGenerate(out, body, openAi);
        } else {
            sendJson(out, 404, "{\"error\":\"Not found\"}");
        }
    }

    private String health() {
        try {
            JSONObject o = new JSONObject();
            o.put("status", "ok");
            o.put("busy", generator.isRunning() || ApiGenerator.engineBusy());
            o.put("model_loaded", ApiGenerator.modelLoaded());
            o.put("model", generator.selectedModelId());
            return o.toString();
        } catch (JSONException e) {
            return "{\"status\":\"ok\"}";
        }
    }

    private void handleGenerate(OutputStream out, String body, boolean openAi) throws IOException {
        ApiGenerator.Request req;
        try {
            req = parseRequest(body, openAi);
        } catch (IllegalArgumentException | JSONException e) {
            sendJson(out, 400, jsonError(e.getMessage() != null ? e.getMessage() : "bad request"));
            return;
        }
        Listener l = listener;
        if (l != null) l.onBusy(true);
        ApiGenerator.Response r;
        try {
            r = generator.generate(req);
        } finally {
            if (l != null) l.onBusy(false);
        }
        try {
            if (r.status != 200) {
                JSONObject err = new JSONObject();
                err.put("error", r.error);
                if (r.blocked) err.put("blocked", true);
                if (r.cancelled) err.put("cancelled", true);
                if (!r.note.isEmpty()) err.put("note", r.note);
                sendJson(out, r.status, err.toString());
                return;
            }
            JSONObject o = new JSONObject();
            if (openAi) {
                JSONObject item = new JSONObject();
                item.put("b64_json", r.imageBase64);
                item.put("revised_prompt", r.prompt);
                o.put("created", System.currentTimeMillis() / 1000);
                o.put("data", new JSONArray().put(item));
            } else {
                o.put("image", r.imageBase64);
                o.put("mime", r.mime);
                o.put("width", r.width);
                o.put("height", r.height);
                o.put("seed", r.seed);
                o.put("steps", r.steps);
                o.put("cfg_scale", (double) r.cfgScale);
                o.put("sampler", r.sampler);
                o.put("prompt", r.prompt);
                o.put("negative_prompt", r.negativePrompt);
                o.put("model", r.model);
                o.put("note", r.note);
                o.put("elapsed_ms", r.elapsedMs);
                if (r.strength > 0) o.put("strength", (double) r.strength);
            }
            sendJson(out, 200, o.toString());
        } catch (JSONException e) {
            sendJson(out, 500, jsonError(e.getMessage()));
        }
    }

    private static ApiGenerator.Request parseRequest(String body, boolean openAi) throws JSONException {
        if (body == null || body.trim().isEmpty()) throw new IllegalArgumentException("empty body");
        JSONObject j = new JSONObject(body);
        ApiGenerator.Request r = new ApiGenerator.Request();
        r.prompt = j.optString("prompt", "");
        if (r.prompt.trim().isEmpty()) throw new IllegalArgumentException("'prompt' is required");
        if (j.has("negative_prompt")) r.negativePrompt = j.optString("negative_prompt", "");
        Object size = j.opt("size");
        if (size instanceof String) {
            // OpenAI style "512x512"
            String[] wh = ((String) size).toLowerCase(Locale.US).split("x");
            if (wh.length == 2) {
                r.width = parsePositive(wh[0]);
                r.height = parsePositive(wh[1]);
            }
        } else if (size instanceof Number) {
            r.width = r.height = ((Number) size).intValue();
        }
        if (j.has("width")) r.width = j.optInt("width", 0);
        if (j.has("height")) r.height = j.optInt("height", 0);
        r.steps = j.optInt("steps", 0);
        if (j.has("cfg_scale")) r.cfgScale = (float) j.optDouble("cfg_scale", Double.NaN);
        r.seed = j.optLong("seed", -1);
        r.sampler = j.optInt("sampler", -1);
        r.translate = j.optBoolean("translate", true);
        r.modelId = j.optString("model", "").trim();
        r.strength = j.has("strength") ? (float) j.optDouble("strength", Double.NaN) : Float.NaN;
        r.change = j.optString("change", "normal");
        String format = openAi ? "png" : j.optString("format", "png");
        r.jpeg = "jpeg".equalsIgnoreCase(format) || "jpg".equalsIgnoreCase(format);
        if (openAi && j.optInt("n", 1) != 1) throw new IllegalArgumentException("only n=1 is supported");
        String init = j.optString("init_image", "");
        if (!init.isEmpty()) {
            int comma = init.indexOf(',');
            if (init.startsWith("data:") && comma > 0) init = init.substring(comma + 1);
            try {
                r.initImage = Base64.decode(init, Base64.DEFAULT);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("'init_image' is not valid base64");
            }
        }
        return r;
    }

    private static int parsePositive(String s) {
        try {
            return Math.max(0, Integer.parseInt(s.trim()));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    // ── HTTP helpers ─────────────────────────────────────────────────────────

    private static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int prev = -1;
        while (true) {
            int b = in.read();
            if (b < 0) return sb.length() == 0 ? null : sb.toString();
            if (b == '\n' && prev == '\r') {
                sb.setLength(sb.length() - 1);
                return sb.toString();
            }
            sb.append((char) b);
            prev = b;
        }
    }

    private static String readBody(InputStream in, String contentLength) throws IOException {
        long len = -1;
        try {
            if (contentLength != null) len = Long.parseLong(contentLength.trim());
        } catch (NumberFormatException ignored) {
        }
        if (len > MAX_BODY_BYTES) throw new IOException("body too large");
        if (len >= 0) {
            byte[] buf = new byte[(int) len];
            int read = 0;
            while (read < buf.length) {
                int n = in.read(buf, read, buf.length - read);
                if (n < 0) break;
                read += n;
            }
            return new String(buf, 0, read, StandardCharsets.UTF_8);
        }
        // No Content-Length: the request has no body we can delimit (no chunked support).
        return "";
    }

    private static void sendJson(OutputStream out, int status, String body) throws IOException {
        send(out, status, "application/json", body);
    }

    private static void send(OutputStream out, int status, String ct, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder();
        sb.append("HTTP/1.1 ").append(status).append(' ').append(statusText(status)).append("\r\n");
        sb.append("Content-Type: ").append(ct).append("; charset=utf-8\r\n");
        sb.append("Content-Length: ").append(bytes.length).append("\r\n");
        sb.append("Access-Control-Allow-Origin: *\r\n");
        if (status == 204) {
            sb.append("Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n");
            sb.append("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
        }
        sb.append("Connection: close\r\n\r\n");
        out.write(sb.toString().getBytes(StandardCharsets.US_ASCII));
        out.write(bytes);
        out.flush();
    }

    private static String statusText(int code) {
        switch (code) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 409: return "Conflict";
            case 422: return "Unprocessable Entity";
            case 500: return "Internal Server Error";
            case 503: return "Service Unavailable";
            default: return "Error";
        }
    }

    private static String jsonError(String msg) {
        return "{\"error\":" + JSONObject.quote(msg == null ? "error" : msg) + "}";
    }
}
