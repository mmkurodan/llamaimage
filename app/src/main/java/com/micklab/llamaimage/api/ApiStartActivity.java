package com.micklab.llamaimage.api;

import android.app.Activity;
import android.os.Bundle;

/**
 * Invisible trampoline that switches the image API on from another app.
 *
 * Some OEM ROMs (e.g. ZTE / Nubia "AutoLaunch") refuse a cross-app startForegroundService()
 * but still allow a user-visible activity launch. Once this activity is in front, this app is
 * foreground and may start its own foreground service.
 *
 * Contract: explicit Intent to com.micklab.llamaimage/.api.ApiStartActivity with action
 * com.micklab.llamaimage.START_SERVICE (optional int extra "port").
 */
public class ApiStartActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        int port = getIntent() != null ? getIntent().getIntExtra(ImageApiService.EXTRA_PORT, -1) : -1;
        ImageApiService.start(getApplicationContext(), port);
        finish();
        overridePendingTransition(0, 0);
    }
}
