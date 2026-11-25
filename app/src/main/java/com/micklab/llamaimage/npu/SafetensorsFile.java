package com.micklab.llamaimage.npu;

import org.json.JSONObject;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/** Minimal reader for .safetensors (8-byte LE header length, JSON header, raw little-endian data). */
final class SafetensorsFile implements Closeable {

    static final class Entry {
        final String dtype;     // "F16", "F32", "BF16", ...
        final long[] shape;
        final long begin, end;  // byte offsets relative to the data section

        Entry(String dtype, long[] shape, long begin, long end) {
            this.dtype = dtype;
            this.shape = shape;
            this.begin = begin;
            this.end = end;
        }

        long elements() {
            long n = 1;
            for (long d : shape) n *= d;
            return n;
        }
    }

    private final RandomAccessFile raf;
    private final FileChannel ch;
    private final long dataStart;
    final Map<String, Entry> entries = new HashMap<>();

    SafetensorsFile(File f) throws IOException {
        raf = new RandomAccessFile(f, "r");
        ch = raf.getChannel();
        ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(lenBuf, 0);
        long headerLen = lenBuf.getLong(0);
        if (headerLen <= 0 || headerLen > 100_000_000L) throw new IOException("safetensors ヘッダが不正です");
        ByteBuffer hb = ByteBuffer.allocate((int) headerLen);
        readFully(hb, 8);
        dataStart = 8 + headerLen;
        try {
            JSONObject h = new JSONObject(new String(hb.array(), StandardCharsets.UTF_8));
            for (Iterator<String> it = h.keys(); it.hasNext(); ) {
                String k = it.next();
                if (k.equals("__metadata__")) continue;
                JSONObject t = h.getJSONObject(k);
                org.json.JSONArray sh = t.getJSONArray("shape");
                long[] shape = new long[sh.length()];
                for (int i = 0; i < shape.length; i++) shape[i] = sh.getLong(i);
                org.json.JSONArray off = t.getJSONArray("data_offsets");
                entries.put(k, new Entry(t.getString("dtype"), shape, off.getLong(0), off.getLong(1)));
            }
        } catch (org.json.JSONException e) {
            throw new IOException("safetensors ヘッダを解析できません: " + e.getMessage(), e);
        }
    }

    private void readFully(ByteBuffer b, long pos) throws IOException {
        while (b.hasRemaining()) {
            int n = ch.read(b, pos + b.position());
            if (n < 0) throw new IOException("ファイルが途中で終わっています");
        }
        b.flip();
    }

    /** Tensor as fp16 values (converted from F32 / BF16 if needed). */
    short[] readFp16(String key) throws IOException {
        Entry e = entries.get(key);
        if (e == null) throw new IOException("チェックポイントに " + key + " がありません");
        long n = e.elements();
        if (n > Integer.MAX_VALUE / 4) throw new IOException("テンソルが大きすぎます: " + key);
        ByteBuffer b = ByteBuffer.allocate((int) (e.end - e.begin)).order(ByteOrder.LITTLE_ENDIAN);
        readFully(b, dataStart + e.begin);
        short[] out = new short[(int) n];
        switch (e.dtype) {
            case "F16":
                b.asShortBuffer().get(out);
                break;
            case "F32":
                for (int i = 0; i < n; i++) out[i] = Fp16.fromFloat(b.getFloat(i * 4));
                break;
            case "BF16":
                for (int i = 0; i < n; i++) out[i] = Fp16.fromFloat(Float.intBitsToFloat((b.getShort(i * 2) & 0xFFFF) << 16));
                break;
            default:
                throw new IOException("未対応の型 " + e.dtype + ": " + key);
        }
        return out;
    }

    @Override
    public void close() throws IOException {
        raf.close();
    }
}
