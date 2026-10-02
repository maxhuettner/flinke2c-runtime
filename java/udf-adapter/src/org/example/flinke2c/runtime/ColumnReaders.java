package org.example.flinke2c.runtime;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

final class ColumnReaders {
    private ColumnReaders() {}

    static ColumnReader[] buildReaders(
            Object[] columns,
            boolean[][] nulls,
            Class<?>[] paramTypes,
            ValueParser[] parsers,
            int[] decimalScales) {
        int argCount = paramTypes.length;
        ColumnReader[] readers = new ColumnReader[argCount];
        for (int i = 0; i < argCount; i++) {
            Object column = columns[i];
            boolean[] isNull = nulls != null && i < nulls.length ? nulls[i] : null;
            readers[i] = readerFor(column, isNull, paramTypes[i], parsers[i], decimalScales[i]);
        }
        return readers;
    }

    static int validateTypedColumns(Object[] columns, boolean[][] nulls, int argCount) {
        if (columns.length != argCount) {
            throw new IllegalArgumentException(
                    "Expected " + argCount + " argument columns but got " + columns.length);
        }

        int rowCount = columnLength(columns[0]);
        for (int arg = 1; arg < columns.length; arg++) {
            int len = columnLength(columns[arg]);
            if (len != rowCount) {
                throw new IllegalArgumentException("Argument columns have mismatched lengths");
            }
        }
        if (nulls != null && nulls.length != argCount) {
            throw new IllegalArgumentException("Null bitmap column count mismatch");
        }
        if (nulls != null) {
            for (int arg = 0; arg < nulls.length; arg++) {
                boolean[] colNulls = nulls[arg];
                if (colNulls != null && colNulls.length != rowCount) {
                    throw new IllegalArgumentException("Null bitmap length mismatch");
                }
            }
        }
        return rowCount;
    }

    private static ColumnReader readerFor(
            Object column,
            boolean[] nulls,
            Class<?> paramType,
            ValueParser parser,
            int decimalScale) {
        if (column instanceof byte[]) {
            byte[] values = (byte[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                BigDecimal decimal = DecimalUtils.decimalFromBytes(values, row, decimalScale);
                if (paramType == BigDecimal.class) {
                    return decimal;
                }
                if (paramType == String.class) {
                    return decimal.toString();
                }
                return decimal;
            };
        }
        if (column instanceof long[]) {
            long[] values = (long[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                long value = values[row];
                if (paramType == Timestamp.class) {
                    return new Timestamp(value);
                }
                if (paramType == LocalDateTime.class) {
                    return LocalDateTime.ofInstant(Instant.ofEpochMilli(value), ZoneOffset.UTC);
                }
                if (paramType == Date.class) {
                    return new Date(value);
                }
                if (paramType == Time.class) {
                    return new Time(value);
                }
                if (paramType == Integer.class || paramType == int.class) {
                    return (int) value;
                }
                return value;
            };
        }
        if (column instanceof int[]) {
            int[] values = (int[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                int value = values[row];
                if (paramType == Long.class || paramType == long.class) {
                    return (long) value;
                }
                return value;
            };
        }
        if (column instanceof double[]) {
            double[] values = (double[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                double value = values[row];
                if (paramType == Float.class || paramType == float.class) {
                    return (float) value;
                }
                return value;
            };
        }
        if (column instanceof float[]) {
            float[] values = (float[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                float value = values[row];
                if (paramType == Double.class || paramType == double.class) {
                    return (double) value;
                }
                return value;
            };
        }
        if (column instanceof boolean[]) {
            boolean[] values = (boolean[]) column;
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                return values[row];
            };
        }
        if (column instanceof String[]) {
            String[] values = (String[]) column;
            return row -> {
                String value = values[row];
                if (value == null || isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                return parser.parse(value);
            };
        }
        if (column instanceof Object[]) {
            // Packed strings from the native runtime: {byte[] utf8, int[] offsets}.
            Object[] packed = (Object[]) column;
            byte[] utf8 = (byte[]) packed[0];
            int[] offsets = (int[]) packed[1];
            return row -> {
                if (isNull(nulls, row)) {
                    return defaultValue(paramType);
                }
                int start = offsets[row];
                String value = new String(utf8, start, offsets[row + 1] - start, StandardCharsets.UTF_8);
                return parser.parse(value);
            };
        }

        throw new IllegalArgumentException("Unsupported column type: " + column.getClass());
    }

    private static int columnLength(Object column) {
        if (column instanceof long[]) {
            return ((long[]) column).length;
        }
        if (column instanceof int[]) {
            return ((int[]) column).length;
        }
        if (column instanceof double[]) {
            return ((double[]) column).length;
        }
        if (column instanceof float[]) {
            return ((float[]) column).length;
        }
        if (column instanceof boolean[]) {
            return ((boolean[]) column).length;
        }
        if (column instanceof byte[]) {
            int len = ((byte[]) column).length;
            if (len % DecimalUtils.DECIMAL_BYTES != 0) {
                throw new IllegalArgumentException("Decimal byte column has invalid length " + len);
            }
            return len / DecimalUtils.DECIMAL_BYTES;
        }
        if (column instanceof String[]) {
            return ((String[]) column).length;
        }
        if (column instanceof Object[]) {
            Object[] packed = (Object[]) column;
            if (packed.length == 2 && packed[1] instanceof int[]) {
                return ((int[]) packed[1]).length - 1;
            }
        }
        throw new IllegalArgumentException("Unsupported column type: " + column.getClass());
    }

    private static boolean isNull(boolean[] nulls, int row) {
        return nulls != null && row < nulls.length && nulls[row];
    }

    private static Object defaultValue(Class<?> paramType) {
        if (paramType == boolean.class) {
            return false;
        }
        if (paramType == byte.class) {
            return (byte) 0;
        }
        if (paramType == short.class) {
            return (short) 0;
        }
        if (paramType == int.class) {
            return 0;
        }
        if (paramType == long.class) {
            return 0L;
        }
        if (paramType == float.class) {
            return 0.0f;
        }
        if (paramType == double.class) {
            return 0.0d;
        }
        return null;
    }
}
