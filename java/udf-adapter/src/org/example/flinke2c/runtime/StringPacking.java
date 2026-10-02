package org.example.flinke2c.runtime;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Packs a String[] as {byte[] concatenated UTF-8, int[] offsets (length + 1)}. */
final class StringPacking {
    private StringPacking() {}

    /**
     * Null elements are written as empty and flagged in {@code nulls}, which the native side uses
     * to restore them.
     */
    static Object[] pack(String[] values, boolean[] nulls) {
        int rows = values.length;
        int[] offsets = new int[rows + 1];
        byte[] buf = new byte[Math.max(256, rows * 16)];
        int pos = 0;
        for (int row = 0; row < rows; row++) {
            String value = values[row];
            if (value == null) {
                nulls[row] = true;
            } else {
                int len = value.length();
                if (pos + len > buf.length) {
                    buf = Arrays.copyOf(buf, Math.max(buf.length * 2, pos + len));
                }
                int start = pos;
                boolean ascii = true;
                for (int i = 0; i < len; i++) {
                    char c = value.charAt(i);
                    if (c >= 0x80) {
                        ascii = false;
                        break;
                    }
                    buf[pos++] = (byte) c;
                }
                if (!ascii) {
                    byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
                    pos = start;
                    if (pos + encoded.length > buf.length) {
                        buf = Arrays.copyOf(buf, Math.max(buf.length * 2, pos + encoded.length));
                    }
                    System.arraycopy(encoded, 0, buf, pos, encoded.length);
                    pos += encoded.length;
                }
            }
            offsets[row + 1] = pos;
        }
        return new Object[] { Arrays.copyOf(buf, pos), offsets };
    }
}
