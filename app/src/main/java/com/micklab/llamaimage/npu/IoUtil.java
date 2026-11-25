package com.micklab.llamaimage.npu;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** java.io helpers (java.nio.file needs API 26; the app's minSdk is 24). */
final class IoUtil {
    private IoUtil() {}

    static byte[] readAll(File f) throws IOException {
        try (InputStream in = new FileInputStream(f)) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream((int) Math.min(f.length(), Integer.MAX_VALUE - 8));
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
            return bo.toByteArray();
        }
    }

    static String readString(File f) throws IOException {
        return new String(readAll(f), StandardCharsets.UTF_8);
    }

    static List<String> readLines(File f) throws IOException {
        List<String> out = new ArrayList<>();
        for (String s : readString(f).split("\n", -1)) out.add(s.endsWith("\r") ? s.substring(0, s.length() - 1) : s);
        return out;
    }

    static void writeString(File f, String s) throws IOException {
        try (OutputStream out = new FileOutputStream(f)) {
            out.write(s.getBytes(StandardCharsets.UTF_8));
        }
    }

    static void copy(File src, File dst) throws IOException {
        try (InputStream in = new FileInputStream(src); OutputStream out = new FileOutputStream(dst)) {
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
    }

    /** Rename replacing {@code dst}. */
    static void move(File src, File dst) throws IOException {
        if (dst.exists() && !dst.delete()) throw new IOException("削除できません: " + dst);
        if (!src.renameTo(dst)) throw new IOException("移動できません: " + src + " -> " + dst);
    }
}
