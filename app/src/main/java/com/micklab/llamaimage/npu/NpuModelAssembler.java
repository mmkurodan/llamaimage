package com.micklab.llamaimage.npu;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

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
        return assemble(templateDir, name, checkpoint, null, outDir, progress);
    }

    /** Map keys starting with this are read from the motion module file (video UNet, AnimateDiff). */
    public static final String MOTION_PREFIX = "motion:";

    /**
     * Same, with a second weight source: keys "motion:&lt;key&gt;" are read from {@code motion}
     * (scripts/npu-video/make_video_template.py), the rest from the checkpoint.
     */
    public static File assemble(File templateDir, String name, File checkpoint, File motion, File outDir,
                                Progress progress) throws IOException {
        JSONObject map;
        try {
            map = new JSONObject(IoUtil.readString(new File(templateDir, name + ".map.json")));
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
             SafetensorsFile mo = motion != null ? new SafetensorsFile(motion) : null;
             FileOutputStream fos = new FileOutputStream(tmpWeights);
             FileChannel out = fos.getChannel()) {
            long written = 0;
            int lastPct = -1;
            for (int i = 0; i < tensors.length(); i++) {
                JSONObject t = tensors.getJSONObject(i);
                String key = t.getString("key");
                SafetensorsFile src = ck;
                if (key.startsWith(MOTION_PREFIX)) {
                    if (mo == null) throw new IOException("動きモジュールが必要です: " + key);
                    src = mo;
                    key = key.substring(MOTION_PREFIX.length());
                }
                JSONArray dims = t.getJSONArray("dims");
                long offset = t.getLong("offset"), length = t.getLong("length");
                if (offset != written) throw new IOException("map のオフセットが不連続です: " + t.getString("name"));
                short[] v = src.readFp16(key);
                if ((long) v.length * 2 != length) {
                    throw new IOException("テンソルの大きさが一致しません（SD1.5 系ではないモデル？）: " + key);
                }
                if (t.getBoolean("transpose")) {
                    // Source viewed as [rows, cols] with rows = first dim; result is its transpose.
                    SafetensorsFile.Entry e = src.entries.get(key);
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
        IoUtil.copy(new File(templateDir, name + ".template.onnx"), tmpModel);
        IoUtil.move(tmpWeights, weights);
        IoUtil.move(tmpModel, model);
        return model;
    }
}
