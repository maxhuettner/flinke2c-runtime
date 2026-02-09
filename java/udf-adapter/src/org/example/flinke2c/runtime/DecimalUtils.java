package org.example.flinke2c.runtime;

import java.math.BigDecimal;
import java.math.BigInteger;

final class DecimalUtils {
    static final int DECIMAL_BYTES = 16;

    private DecimalUtils() {}

    static void writeDecimalBytes(byte[] target, int row, BigDecimal value) {
        BigInteger unscaled = value.unscaledValue();
        int offset = row * DECIMAL_BYTES;
        if (unscaled.bitLength() <= 63) {
            long v = unscaled.longValue();
            writeLongTo128(target, offset, v);
            return;
        }
        byte[] bytes = unscaled.toByteArray();
        if (bytes.length > DECIMAL_BYTES) {
            throw new IllegalArgumentException(
                    "Decimal value does not fit into 128 bits: " + value);
        }
        byte pad = (byte) (value.signum() < 0 ? 0xFF : 0x00);
        for (int i = 0; i < DECIMAL_BYTES; i++) {
            target[offset + i] = pad;
        }
        int copyStart = Math.max(0, bytes.length - DECIMAL_BYTES);
        int copyLen = Math.min(bytes.length, DECIMAL_BYTES);
        System.arraycopy(
                bytes,
                copyStart,
                target,
                offset + (DECIMAL_BYTES - copyLen),
                copyLen);
    }

    static BigDecimal decimalFromBytes(byte[] values, int row, int scale) {
        int offset = row * DECIMAL_BYTES;
        if (offset + DECIMAL_BYTES > values.length) {
            throw new IllegalArgumentException("Decimal byte array index out of range");
        }
        long high = readLong(values, offset);
        long low = readLong(values, offset + 8);
        if ((high == 0 && low >= 0) || (high == -1 && low < 0)) {
            return BigDecimal.valueOf(low, scale);
        }
        byte[] slice = new byte[DECIMAL_BYTES];
        System.arraycopy(values, offset, slice, 0, DECIMAL_BYTES);
        return new BigDecimal(new BigInteger(slice), scale);
    }

    private static void writeLongTo128(byte[] target, int offset, long value) {
        byte pad = (byte) (value < 0 ? 0xFF : 0x00);
        for (int i = 0; i < 8; i++) {
            target[offset + i] = pad;
        }
        for (int i = 0; i < 8; i++) {
            target[offset + 8 + i] = (byte) (value >>> (56 - (i * 8)));
        }
    }

    private static long readLong(byte[] values, int offset) {
        return ((long) (values[offset] & 0xFF) << 56)
                | ((long) (values[offset + 1] & 0xFF) << 48)
                | ((long) (values[offset + 2] & 0xFF) << 40)
                | ((long) (values[offset + 3] & 0xFF) << 32)
                | ((long) (values[offset + 4] & 0xFF) << 24)
                | ((long) (values[offset + 5] & 0xFF) << 16)
                | ((long) (values[offset + 6] & 0xFF) << 8)
                | ((long) (values[offset + 7] & 0xFF));
    }
}
