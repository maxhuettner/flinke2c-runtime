package org.example.flinke2c.runtime;

import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

final class OutputWriters {
    private OutputWriters() {}

    static Object allocateOutputArray(Class<?> returnType, int rowCount) {
        if (returnType == null) {
            return new String[rowCount];
        }
        if (returnType == Boolean.class || returnType == boolean.class) {
            return new boolean[rowCount];
        }
        if (returnType == Float.class || returnType == float.class
                || returnType == Double.class || returnType == double.class) {
            return new double[rowCount];
        }
        if (returnType == java.math.BigDecimal.class) {
            return new byte[rowCount * DecimalUtils.DECIMAL_BYTES];
        }
        if (returnType == Timestamp.class
                || returnType == LocalDateTime.class
                || returnType == Date.class
                || returnType == Time.class) {
            return new long[rowCount];
        }
        if (Number.class.isAssignableFrom(returnType)
                || returnType == long.class
                || returnType == int.class
                || returnType == short.class
                || returnType == byte.class) {
            return new long[rowCount];
        }
        return new String[rowCount];
    }

    static void writeOutputValue(Object array, Object value, int row) {
        if (array instanceof byte[]) {
            DecimalUtils.writeDecimalBytes((byte[]) array, row, (java.math.BigDecimal) value);
            return;
        }
        if (array instanceof long[]) {
            ((long[]) array)[row] = toEpochMillis(value);
            return;
        }
        if (array instanceof double[]) {
            ((double[]) array)[row] = toDouble(value);
            return;
        }
        if (array instanceof boolean[]) {
            ((boolean[]) array)[row] = (Boolean) value;
            return;
        }
        if (array instanceof String[]) {
            ((String[]) array)[row] = value == null ? null : value.toString();
        }
    }

    private static long toEpochMillis(Object value) {
        if (value == null) {
            return 0L;
        }
        if (value instanceof Timestamp) {
            return ((Timestamp) value).getTime();
        }
        if (value instanceof Date) {
            return ((Date) value).getTime();
        }
        if (value instanceof Time) {
            return ((Time) value).getTime();
        }
        if (value instanceof LocalDateTime) {
            return ((LocalDateTime) value)
                    .atZone(ZoneOffset.UTC)
                    .toInstant()
                    .toEpochMilli();
        }
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        throw new IllegalArgumentException("Unsupported value for long output: " + value.getClass());
    }

    private static double toDouble(Object value) {
        if (value == null) {
            return 0.0d;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        throw new IllegalArgumentException("Unsupported value for double output: " + value.getClass());
    }
}
