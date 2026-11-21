package com.micklab.llamaimage;

import android.app.ActivityManager;
import android.content.Context;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Device CPU / memory facts used to pick the native library variant, the default thread count
 * and whether the VAE should decode in tiles. Pure reads of /proc and /sys; no permissions.
 */
public final class CpuInfo {

    private CpuInfo() {}

    private static volatile Set<String> cachedFeatures;

    /**
     * ARM HWCAP feature names (as printed in /proc/cpuinfo "Features") supported by EVERY core.
     * Empty if unreadable. Typical names: asimdhp (fp16 SIMD), asimddp (dotprod), i8mm.
     */
    public static Set<String> features() {
        Set<String> f = cachedFeatures;
        if (f != null) return f;
        Set<String> common = null;
        try (BufferedReader r = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
            String line;
            while ((line = r.readLine()) != null) {
                int colon = line.indexOf(':');
                if (colon < 0 || !line.substring(0, colon).trim().equalsIgnoreCase("Features")) continue;
                Set<String> s = new HashSet<>(Arrays.asList(
                        line.substring(colon + 1).trim().toLowerCase(Locale.US).split("\\s+")));
                if (common == null) common = s; else common.retainAll(s);
            }
        } catch (Exception ignore) {
        }
        f = common == null ? new HashSet<>() : common;
        cachedFeatures = f;
        return f;
    }

    public static boolean hasFp16AndDotprod() {
        Set<String> f = features();
        return f.contains("asimdhp") && f.contains("asimddp");
    }

    /**
     * Native libraries to try, best first. Each name must match a library built by
     * app/src/main/cpp/CMakeLists.txt (SD_CPU_VARIANT); the baseline always comes last.
     */
    public static List<String> nativeLibraryCandidates() {
        List<String> libs = new ArrayList<>();
        if (hasFp16AndDotprod()) libs.add("sd_jni_fp16");
        libs.add("sd_jni");
        return libs;
    }

    /** In-order efficiency cores: Cortex-A35/A53/A55/A510/A520 and Qualcomm "Silver" Kryo cores. */
    private static final Set<String> LITTLE_PARTS = new HashSet<>(Arrays.asList(
            "0x41:0xd04", "0x41:0xd03", "0x41:0xd05", "0x41:0xd46", "0x41:0xd80",
            "0x51:0x801", "0x51:0x803", "0x51:0x805"));

    /**
     * Default worker thread count: the performance cores. ggml splits every op evenly across
     * threads and waits at a barrier, so efficiency ("LITTLE") cores hold everyone back. A core is
     * LITTLE if /proc/cpuinfo names a known in-order part, or if its maximum frequency is at most
     * 70% of the fastest core's. All-big designs (e.g. Snapdragon 8 Elite: 6+2 Oryon) use them all.
     */
    public static int defaultThreads() {
        int n = Runtime.getRuntime().availableProcessors();
        String[] parts = cpuParts(n);
        long[] maxFreq = new long[n];
        long best = 0;
        for (int i = 0; i < n; i++) {
            maxFreq[i] = readLong("/sys/devices/system/cpu/cpu" + i + "/cpufreq/cpuinfo_max_freq");
            best = Math.max(best, maxFreq[i]);
        }
        int big = 0;
        for (int i = 0; i < n; i++) {
            boolean little = parts[i] != null && LITTLE_PARTS.contains(parts[i]);
            if (best > 0 && maxFreq[i] > 0 && maxFreq[i] * 10 <= best * 7) little = true;
            if (!little) big++;
        }
        return big > 0 ? big : Math.max(1, n);
    }

    /** "implementer:part" per logical CPU from /proc/cpuinfo (entries may be null). */
    private static String[] cpuParts(int n) {
        String[] out = new String[n];
        String[] impl = new String[n];
        int cur = -1;
        try (BufferedReader r = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
            String line;
            while ((line = r.readLine()) != null) {
                int colon = line.indexOf(':');
                if (colon < 0) continue;
                String key = line.substring(0, colon).trim().toLowerCase(Locale.US);
                String val = line.substring(colon + 1).trim().toLowerCase(Locale.US);
                if (key.equals("processor")) {
                    try { cur = Integer.parseInt(val); } catch (NumberFormatException e) { cur = -1; }
                } else if (cur >= 0 && cur < n && key.equals("cpu implementer")) {
                    impl[cur] = val;
                } else if (cur >= 0 && cur < n && key.equals("cpu part") && impl[cur] != null) {
                    out[cur] = impl[cur] + ":" + val;
                }
            }
        } catch (Exception ignore) {
        }
        return out;
    }

    /** Total device RAM in bytes (0 if unknown). */
    public static long totalRamBytes(Context ctx) {
        try {
            ActivityManager am = (ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(mi);
            return mi.totalMem;
        } catch (Exception e) {
            return 0;
        }
    }

    private static long readLong(String path) {
        File f = new File(path);
        if (!f.canRead()) return -1;
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String s = r.readLine();
            return s == null ? -1 : Long.parseLong(s.trim());
        } catch (Exception e) {
            return -1;
        }
    }
}
