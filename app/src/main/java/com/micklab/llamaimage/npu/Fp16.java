package com.micklab.llamaimage.npu;

/** float32 -> IEEE 754 half, round-to-nearest-even (same result as numpy's astype(float16)). */
final class Fp16 {
    private Fp16() {}

    static short fromFloat(float f) {
        int x = Float.floatToRawIntBits(f);
        int sign = (x >>> 16) & 0x8000;
        int exp = (x >>> 23) & 0xFF;
        int mant = x & 0x7FFFFF;
        if (exp == 0xFF) {                                    // Inf / NaN
            return (short) (sign | 0x7C00 | (mant != 0 ? 0x200 | (mant >>> 13) : 0));
        }
        int e = exp - 127 + 15;
        if (e >= 0x1F) return (short) (sign | 0x7C00);        // overflow -> Inf
        if (e <= 0) {                                         // subnormal / zero
            if (e < -10) return (short) sign;
            mant |= 0x800000;
            int shift = 14 - e;
            int half = mant >>> shift;
            int rem = mant & ((1 << shift) - 1);
            int mid = 1 << (shift - 1);
            if (rem > mid || (rem == mid && (half & 1) != 0)) half++;
            return (short) (sign | half);
        }
        int half = (e << 10) | (mant >>> 13);
        int rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0)) half++;   // may carry into exponent: correct
        return (short) (sign | half);
    }
}
