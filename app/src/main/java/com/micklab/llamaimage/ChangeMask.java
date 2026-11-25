package com.micklab.llamaimage;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.media.FaceDetector;

/**
 * "Where may the image change" masks for img2img / video: 0 = keep the input pixel, 255 = take the
 * generated pixel. Applied after generation as a feathered composite, so it works the same for the
 * CPU / GPU engine and the NPU pipeline.
 */
public final class ChangeMask {
    private ChangeMask() {}

    /** Ellipses around detected faces (android.media.FaceDetector), or null if none was found. */
    public static byte[] faces(Bitmap src, int maxFaces) {
        int w = src.getWidth() & ~1, h = src.getHeight();
        Bitmap b565 = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
        new Canvas(b565).drawBitmap(src, 0, 0, null);
        FaceDetector.Face[] found = new FaceDetector.Face[maxFaces];
        int n = new FaceDetector(w, h, maxFaces).findFaces(b565, found);
        if (n <= 0) return null;
        Bitmap m = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ALPHA_8);
        Canvas c = new Canvas(m);
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(0xFFFFFFFF);
        PointF mid = new PointF();
        for (int i = 0; i < n; i++) {
            FaceDetector.Face f = found[i];
            if (f == null) continue;
            f.getMidPoint(mid);   // between the eyes
            float e = f.eyesDistance();
            float cy = mid.y + 0.45f * e;
            c.drawOval(new RectF(mid.x - 1.25f * e, cy - 1.75f * e, mid.x + 1.25f * e, cy + 1.6f * e), p);
        }
        return fromAlpha(m);
    }

    /** True if at least one face is detected. */
    public static boolean hasFace(Bitmap src) {
        return faces(src, 1) != null;
    }

    public static byte[] fromAlpha(Bitmap alpha) {
        int w = alpha.getWidth(), h = alpha.getHeight();
        int[] px = new int[w * h];
        alpha.getPixels(px, 0, w, 0, 0, w, h);
        byte[] m = new byte[w * h];
        for (int i = 0; i < px.length; i++) m[i] = (byte) (px[i] >>> 24);
        return m;
    }

    public static Bitmap toAlpha(byte[] m, int w, int h) {
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) px[i] = (m[i] & 0xFF) << 24;
        Bitmap b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        b.setPixels(px, 0, w, 0, 0, w, h);
        return b;
    }

    /** Mask resized (bilinear) to w x h. */
    public static byte[] resize(byte[] m, int mw, int mh, int w, int h) {
        if (mw == w && mh == h) return m;
        return fromAlpha(Bitmap.createScaledBitmap(toAlpha(m, mw, mh), w, h, true));
    }

    /** Same center crop + rescale as MainActivity.zoomBitmap, so a painted mask follows the zoom. */
    public static byte[] zoom(byte[] m, int w, int h, float zoom) {
        if (zoom <= 0f) return m;
        return fromAlpha(MainActivity.zoomBitmap(toAlpha(m, w, h), zoom));
    }

    public static boolean isEmpty(byte[] m) {
        if (m == null) return true;
        for (byte b : m) if (b != 0) return false;
        return true;
    }

    /** Soft edge: box blur (3 passes ~ gaussian) with radius r. */
    public static byte[] feather(byte[] m, int w, int h, int r) {
        if (r <= 0) return m;
        int[] a = new int[w * h], t = new int[w * h];
        for (int i = 0; i < a.length; i++) a[i] = m[i] & 0xFF;
        for (int pass = 0; pass < 3; pass++) {
            for (int y = 0; y < h; y++) {          // horizontal
                int sum = 0, row = y * w;
                for (int x = -r; x <= r; x++) sum += a[row + Math.max(0, Math.min(w - 1, x))];
                for (int x = 0; x < w; x++) {
                    t[row + x] = sum / (2 * r + 1);
                    sum += a[row + Math.min(w - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
                }
            }
            for (int x = 0; x < w; x++) {          // vertical
                int sum = 0;
                for (int y = -r; y <= r; y++) sum += t[Math.max(0, Math.min(h - 1, y)) * w + x];
                for (int y = 0; y < h; y++) {
                    a[y * w + x] = sum / (2 * r + 1);
                    sum += t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
                }
            }
        }
        byte[] out = new byte[w * h];
        for (int i = 0; i < out.length; i++) out[i] = (byte) a[i];
        return out;
    }

    /** gen * m + orig * (1 - m), feathered; {@code m} is w x h of gen (orig scaled if needed). */
    public static Bitmap composite(Bitmap gen, Bitmap orig, byte[] m) {
        int w = gen.getWidth(), h = gen.getHeight();
        if (orig.getWidth() != w || orig.getHeight() != h) orig = Bitmap.createScaledBitmap(orig, w, h, true);
        byte[] f = feather(m, w, h, Math.max(2, Math.min(w, h) / 48));
        int[] g = new int[w * h], o = new int[w * h];
        gen.getPixels(g, 0, w, 0, 0, w, h);
        orig.getPixels(o, 0, w, 0, 0, w, h);
        for (int i = 0; i < g.length; i++) {
            int a = f[i] & 0xFF;
            if (a == 255) continue;
            int gc = g[i], oc = o[i], out = 0xFF000000;
            for (int sh = 0; sh <= 16; sh += 8) {
                int v = (((gc >> sh) & 0xFF) * a + ((oc >> sh) & 0xFF) * (255 - a) + 127) / 255;
                out |= v << sh;
            }
            g[i] = out;
        }
        Bitmap r = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        r.setPixels(g, 0, w, 0, 0, w, h);
        return r;
    }
}
