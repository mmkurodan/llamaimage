package com.micklab.llamaimage;

import android.graphics.Bitmap;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Silent H.264 MP4 from a sequence of equally sized bitmaps (MediaCodec + MediaMuxer, no audio
 * track). Frames are converted to YUV 4:2:0 and written through {@link MediaCodec#getInputImage},
 * which handles planar / semi-planar layouts of any encoder.
 */
public final class VideoEncoder implements Closeable {
    private static final String MIME = MediaFormat.MIMETYPE_VIDEO_AVC;
    private static final long TIMEOUT_US = 10_000;

    private final MediaCodec codec;
    private final MediaMuxer muxer;
    private final int width, height, fps;
    private final MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
    private int track = -1;
    private boolean muxerStarted = false;
    private long frameIndex = 0;

    public VideoEncoder(File out, int width, int height, int fps) throws IOException {
        this.width = width & ~1;
        this.height = height & ~1;
        this.fps = fps;
        MediaFormat f = MediaFormat.createVideoFormat(MIME, this.width, this.height);
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        f.setInteger(MediaFormat.KEY_BIT_RATE, Math.max(2_000_000, this.width * this.height * fps / 4));
        f.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
        f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        codec = MediaCodec.createEncoderByType(MIME);
        try {
            codec.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            muxer = new MediaMuxer(out.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        } catch (IOException | RuntimeException e) {
            codec.release();
            throw e;
        }
    }

    /** Appends one frame (scaled if its size differs). */
    public void addFrame(Bitmap bmp) throws IOException {
        if (bmp.getWidth() != width || bmp.getHeight() != height) {
            bmp = Bitmap.createScaledBitmap(bmp, width, height, true);
        }
        int[] argb = new int[width * height];
        bmp.getPixels(argb, 0, width, 0, 0, width, height);
        int idx;
        while ((idx = codec.dequeueInputBuffer(TIMEOUT_US)) < 0) drain(false);
        Image img = codec.getInputImage(idx);
        if (img == null) throw new IOException("encoder input image unavailable");
        writeYuv(argb, img);
        long pts = frameIndex * 1_000_000L / fps;
        codec.queueInputBuffer(idx, 0, width * height * 3 / 2, pts, 0);
        frameIndex++;
        drain(false);
    }

    /** Finishes the file. */
    public void finish() throws IOException {
        int idx;
        while ((idx = codec.dequeueInputBuffer(TIMEOUT_US)) < 0) drain(false);
        codec.queueInputBuffer(idx, 0, 0, frameIndex * 1_000_000L / fps, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
        drain(true);
    }

    private void drain(boolean endOfStream) throws IOException {
        long deadline = System.currentTimeMillis() + 10_000;
        while (true) {
            int st = codec.dequeueOutputBuffer(info, TIMEOUT_US);
            if (st == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) return;
                if (System.currentTimeMillis() > deadline) throw new IOException("encoder did not finish");
            } else if (st == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (muxerStarted) throw new IOException("format changed twice");
                track = muxer.addTrack(codec.getOutputFormat());
                muxer.start();
                muxerStarted = true;
            } else if (st >= 0) {
                ByteBuffer buf = codec.getOutputBuffer(st);
                if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) info.size = 0;
                if (info.size > 0 && buf != null && muxerStarted) {
                    buf.position(info.offset);
                    buf.limit(info.offset + info.size);
                    muxer.writeSampleData(track, buf, info);
                }
                codec.releaseOutputBuffer(st, false);
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) return;
            }
        }
    }

    /** BT.601 limited range RGB -> YUV 4:2:0 into the codec's image planes. */
    private void writeYuv(int[] argb, Image img) {
        Image.Plane[] p = img.getPlanes();
        ByteBuffer yb = p[0].getBuffer(), ub = p[1].getBuffer(), vb = p[2].getBuffer();
        int yRow = p[0].getRowStride(), yPix = p[0].getPixelStride();
        int uRow = p[1].getRowStride(), uPix = p[1].getPixelStride();
        int vRow = p[2].getRowStride(), vPix = p[2].getPixelStride();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int c = argb[y * width + x];
                int r = (c >> 16) & 0xFF, g = (c >> 8) & 0xFF, b = c & 0xFF;
                int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                yb.put(y * yRow + x * yPix, (byte) Y);
            }
        }
        for (int y = 0; y < height / 2; y++) {
            for (int x = 0; x < width / 2; x++) {
                int r = 0, g = 0, b = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int c = argb[(2 * y + dy) * width + 2 * x + dx];
                        r += (c >> 16) & 0xFF;
                        g += (c >> 8) & 0xFF;
                        b += c & 0xFF;
                    }
                }
                r >>= 2;
                g >>= 2;
                b >>= 2;
                int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int V = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                ub.put(y * uRow + x * uPix, (byte) U);
                vb.put(y * vRow + x * vPix, (byte) V);
            }
        }
    }

    @Override
    public void close() {
        try {
            codec.stop();
        } catch (RuntimeException ignore) {
        }
        codec.release();
        try {
            if (muxerStarted) muxer.stop();
        } catch (RuntimeException ignore) {
        }
        muxer.release();
    }
}
