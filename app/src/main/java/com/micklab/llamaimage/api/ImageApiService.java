package com.micklab.llamaimage.api;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import com.micklab.llamaimage.ApiGenerator;
import com.micklab.llamaimage.MainActivity;

/**
 * Foreground service hosting {@link ImageApiServer}, after OcrApiService in the companion pdf app.
 *
 * Exported so other apps (e.g. the Assistant) can switch the API on/off with an explicit Intent:
 * action {@link #ACTION_START} (optional int extra {@link #EXTRA_PORT}) or {@link #ACTION_STOP},
 * sent with {@code startForegroundService()}. The "enabled" choice is persisted, so the API comes
 * back after a sticky restart or when the app is launched again, until it is explicitly stopped.
 */
public class ImageApiService extends Service {

    private static final String TAG = "ImageApiService";
    private static final String CHANNEL_ID = "image_api_channel";
    private static final int NOTIF_ID = 3001;

    /** Public contract for other apps (mirrors com.micklab.pdf.START_SERVICE). */
    public static final String ACTION_START = "com.micklab.llamaimage.START_SERVICE";
    public static final String ACTION_STOP = "com.micklab.llamaimage.STOP_SERVICE";
    public static final String EXTRA_PORT = "port";

    private static final String PREFS = "image_api";
    private static final String KEY_ENABLED = "api_enabled";
    private static final String KEY_PORT = "api_port";

    /** Liveness in this process. */
    private static volatile boolean serviceRunning = false;
    private static volatile boolean serverListening = false;
    private static volatile String lastError = "";

    private ImageApiServer server;
    private int serverPort = -1;
    private PowerManager.WakeLock wakeLock;
    private int busyCount = 0;

    public static Intent startIntent(Context ctx, int port) {
        Intent i = new Intent(ctx, ImageApiService.class).setAction(ACTION_START);
        if (port > 0) i.putExtra(EXTRA_PORT, port);
        return i;
    }

    public static Intent stopIntent(Context ctx) {
        return new Intent(ctx, ImageApiService.class).setAction(ACTION_STOP);
    }

    public static boolean isEnabled(Context ctx) {
        return prefs(ctx).getBoolean(KEY_ENABLED, false);
    }

    public static int port(Context ctx) {
        return prefs(ctx).getInt(KEY_PORT, ImageApiServer.DEFAULT_PORT);
    }

    public static boolean isListening() {
        return serviceRunning && serverListening;
    }

    public static String lastError() {
        return lastError;
    }

    /** Starts the service iff the API was left enabled (app launch). Refused starts are ignored. */
    public static void startIfEnabled(Context ctx) {
        if (!isEnabled(ctx)) return;
        start(ctx, -1);
    }

    public static void start(Context ctx, int port) {
        try {
            Intent i = startIntent(ctx.getApplicationContext(), port);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.getApplicationContext().startForegroundService(i);
            } else {
                ctx.getApplicationContext().startService(i);
            }
        } catch (Exception e) {
            Log.w(TAG, "start not allowed right now", e);
        }
    }

    private static SharedPreferences prefs(Context ctx) {
        return ctx.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (pm != null) {
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "llamaimage:api");
            wakeLock.setReferenceCounted(false);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            // Explicit disable: clear the persisted intent so nothing restarts us.
            prefs(this).edit().putBoolean(KEY_ENABLED, false).apply();
            // startForegroundService() obliges us to enter the foreground once, even for "stop".
            try {
                startForegroundCompat(buildNotif("画像生成 API を停止しました"));
            } catch (Exception ignored) {
            }
            stopForegroundCompat();
            stopSelf();
            return START_NOT_STICKY;
        }

        // A sticky restart delivers a null intent: use the persisted port.
        int requested = intent != null ? intent.getIntExtra(EXTRA_PORT, -1) : -1;
        SharedPreferences.Editor ed = prefs(this).edit().putBoolean(KEY_ENABLED, true);
        if (requested >= 1024 && requested <= 65535) ed.putInt(KEY_PORT, requested);
        ed.apply();
        int port = requested >= 1024 && requested <= 65535 ? requested : port(this);

        try {
            startForegroundCompat(buildNotif("画像生成 API を起動中…"));
        } catch (Exception e) {
            // Background-start restriction: stay "enabled" and let the next launch try again.
            Log.w(TAG, "startForeground not allowed in this context", e);
            stopSelf();
            return START_STICKY;
        }
        serviceRunning = true;
        startServer(port);
        return START_STICKY;
    }

    @Override
    public void onTaskRemoved(Intent rootIntent) {
        // Swiped from recents: some OEMs kill without a sticky restart, so re-request.
        startIfEnabled(getApplicationContext());
        super.onTaskRemoved(rootIntent);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        serviceRunning = false;
        serverListening = false;
        if (server != null) {
            server.stop();
            server = null;
        }
        releaseWake(true);
        super.onDestroy();
    }

    private void startServer(int port) {
        if (server != null && server.isRunning() && serverPort == port) return;
        if (server != null) server.stop();
        serverPort = port;
        ApiGenerator generator = new ApiGenerator(this, msg -> Log.i(TAG, msg));
        server = new ImageApiServer(generator);
        server.setPort(port);
        server.setListener(new ImageApiServer.Listener() {
            @Override public void onServerStarted(int p) {
                serverListening = true;
                lastError = "";
                updateNotif("画像生成 API 稼働中（127.0.0.1:" + p + "）");
            }

            @Override public void onServerStopped() {
                serverListening = false;
            }

            @Override public void onServerError(String message) {
                serverListening = false;
                lastError = message;
                updateNotif("画像生成 API エラー: " + message);
            }

            @Override public void onRequest(String method, String path) {
                Log.d(TAG, method + " " + path);
            }

            @Override public void onBusy(boolean busy) {
                if (busy) acquireWake(); else releaseWake(false);
            }
        });
        server.start();
    }

    private synchronized void acquireWake() {
        if (busyCount++ == 0 && wakeLock != null && !wakeLock.isHeld()) {
            try {
                wakeLock.acquire(30 * 60 * 1000L);   // generation is minutes at most
            } catch (Exception e) {
                Log.w(TAG, "wake lock failed", e);
            }
        }
    }

    private synchronized void releaseWake(boolean all) {
        if (all) busyCount = 0; else if (busyCount > 0) busyCount--;
        if (busyCount == 0 && wakeLock != null && wakeLock.isHeld()) {
            try { wakeLock.release(); } catch (Exception ignored) { }
        }
    }

    private void startForegroundCompat(Notification n) {
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIF_ID, n);
        }
    }

    @SuppressWarnings("deprecation")
    private void stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= 24) {
            stopForeground(STOP_FOREGROUND_REMOVE);
        } else {
            stopForeground(true);
        }
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(CHANNEL_ID, "画像生成 API",
                    NotificationManager.IMPORTANCE_LOW);
            ch.setDescription("他のアプリから画像生成を受け付ける API の稼働状態");
            ch.setShowBadge(false);
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) nm.createNotificationChannel(ch);
        }
    }

    private Notification buildNotif(String content) {
        int piFlags = PendingIntent.FLAG_UPDATE_CURRENT
                | (Build.VERSION.SDK_INT >= 23 ? PendingIntent.FLAG_IMMUTABLE : 0);
        PendingIntent open = PendingIntent.getActivity(this, 0,
                new Intent(this, MainActivity.class), piFlags);
        PendingIntent stop = PendingIntent.getService(this, 1, stopIntent(this), piFlags);
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return b.setContentTitle("llamaimage 画像生成 API")
                .setContentText(content)
                .setSmallIcon(android.R.drawable.ic_menu_gallery)
                .setContentIntent(open)
                .addAction(android.R.drawable.ic_media_pause, "停止", stop)
                .setOngoing(true)
                .build();
    }

    private void updateNotif(String content) {
        if (!serviceRunning) return;
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) nm.notify(NOTIF_ID, buildNotif(content));
    }
}
