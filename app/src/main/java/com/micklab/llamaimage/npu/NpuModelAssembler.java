package com.micklab.llamaimage.npu;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Builds a runnable ONNX model on the device from a weightless template + weight map (published by
 * scripts/npu/make_template.py) and an SD1.5 single-file .safetensors checkpoint. Same algorithm as
 * scripts/npu/assemble.py: for each map entry read tensor `key` as fp16, transpose (2-D view) if
 * `transpose`, and append it at `offset` of the weights file the template references.
 */
public final class NpuModelAssembler {

    public interface Progress {
        void onProgress(int percent);
    }

    private NpuModelAssembler() {}

    /**
     * Writes {@code <outDir>/<name>.onnx} (template copy) and its weights file. Returns the model.
     * Writes to temporary names first, so an interrupted run never leaves a half model behind.
     */
    public static File assemble(File templateDir, String name, File checkpoint, File outDir, Progress progress)
            throws IOException {
        JSONObject map;
        try {
            map = new JSONObject(new String(Files.readAllBytes(new File(templateDir, name + ".map.json").toPath()),
                    StandardCharsets.UTF_8));
        } catch (org.json.JSONException e) {
            throw new IOException(name + ".map.json を解析できません", e);
        }
        if (!outDir.isDirectory() && !outDir.mkdirs()) throw new IOException("作成できません: " + outDir);
        String weightsName;
        long total;
        JSONArray tensors;
        try {
            weightsName = map.getString("weights_file");
            total = map.getLong("total_bytes");
            tensors = map.getJSONArray("tensors");
        } catch (org.json.JSONException e) {
            throw new IOException(name + ".map.json の形式が不正です", e);
        }
        File weights = new File(outDir, weightsName);
        File tmpWeights = new File(outDir, weightsName + ".tmp");
        try (SafetensorsFile ck = new SafetensorsFile(checkpoint);
             FileOutputStream fos = new FileOutputStream(tmpWeights);
             FileChannel out = fos.getChannel()) {
            long written = 0;
            int lastPct = -1;
            for (int i = 0; i < tensors.length(); i++) {
                JSONObject t = tensors.getJSONObject(i);
                String key = t.getString("key");
                JSONArray dims = t.getJSONArray("dims");
                long offset = t.getLong("offset"), length = t.getLong("length");
                if (offset != written) throw new IOException("map のオフセットが不連続です: " + t.getString("name"));
                short[] v = ck.readFp16(key);
                if ((long) v.length * 2 != length) {
                    throw new IOException("テンソルの大きさが一致しません（SD1.5 系ではないモデル？）: " + key);
                }
                if (t.getBoolean("transpose")) {
                    // Source viewed as [rows, cols] with rows = first dim; result is its transpose.
                    SafetensorsFile.Entry e = ck.entries.get(key);
                    int rows = (int) e.shape[0];
                    int cols = v.length / rows;
                    short[] tr = new short[v.length];
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) tr[c * rows + r] = v[r * cols + c];
                    }
                    v = tr;
                }
                long expect = 2;
                for (int d = 0; d < dims.length(); d++) expect *= dims.getLong(d);
                if (expect != length) throw new IOException("map の dims が不正です: " + t.getString("name"));
                ByteBuffer bb = ByteBuffer.allocate(v.length * 2).order(ByteOrder.LITTLE_ENDIAN);
                bb.asShortBuffer().put(v);
                while (bb.hasRemaining()) out.write(bb);
                written += length;
                int pct = (int) (written * 100 / Math.max(1, total));
                if (progress != null && pct != lastPct) {
                    progress.onProgress(pct);
                    lastPct = pct;
                }
            }
            if (written != total) throw new IOException("重みの合計が一致しません");
            out.force(false);
        } catch (org.json.JSONException e) {
            tmpWeights.delete();
            throw new IOException(name + ".map.json の形式が不正です", e);
        } catch (IOException | RuntimeException e) {
            tmpWeights.delete();
            throw e;
        }
        File model = new File(outDir, name + ".onnx");
        File tmpModel = new File(outDir, name + ".onnx.tmp");
        Files.copy(new File(templateDir, name + ".template.onnx").toPath(), tmpModel.toPath(),
                StandardCopyOption.REPLACE_EXISTING);
        Files.move(tmpWeights.toPath(), weights.toPath(), StandardCopyOption.REPLACE_EXISTING);
        Files.move(tmpModel.toPath(), model.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return model;
    }
}
