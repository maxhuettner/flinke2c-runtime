package org.example.flinke2c.runtime;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.Locale;
import java.util.function.Function;

final class TypeUtils {
    private static final DateTimeFormatter TIMESTAMP_FORMATTER = new DateTimeFormatterBuilder()
            .appendPattern("yyyy-MM-dd HH:mm:ss")
            .optionalStart()
            .appendFraction(ChronoField.NANO_OF_SECOND, 0, 9, true)
            .optionalEnd()
            .toFormatter(Locale.ROOT);

    private TypeUtils() {}

    static ValueParser[] buildParsers(String[] argTypes, Class<?>[] paramTypes) {
        int count = paramTypes.length;
        ValueParser[] out = new ValueParser[count];
        for (int i = 0; i < count; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = parserFor(configType, paramTypes[i]);
        }
        return out;
    }

    static Class<?>[] mapFlinkTypesToClasses(String[] argTypes, int paramCount) {
        Class<?>[] out = new Class<?>[paramCount];
        for (int i = 0; i < paramCount; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = classFor(configType);
        }
        return out;
    }

    static int[] extractDecimalScales(String[] argTypes, int paramCount) {
        int[] out = new int[paramCount];
        for (int i = 0; i < paramCount; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = extractScale(configType);
        }
        return out;
    }

    static boolean isScalarReturnType(Class<?> type) {
        if (type == null) {
            return true;
        }
        if (type.isPrimitive()) {
            return true;
        }
        if (Number.class.isAssignableFrom(type)
                || type == String.class
                || type == Boolean.class
                || type == BigDecimal.class
                || type == Timestamp.class
                || type == LocalDateTime.class
                || type == Date.class
                || type == Time.class) {
            return true;
        }
        return false;
    }

    private static <T> ValueParser valueOrNull(Function<String, T> returnFn) {
        return value -> value == null ? null : returnFn.apply(value);
    }

    private static ValueParser parserFor(String configType, Class<?> paramType) {
        String baseType = baseType(configType);

        if (isDecimal(paramType, baseType)) {
            return valueOrNull(BigDecimal::new);
        }
        if (isBoolean(paramType, baseType)) {
            return valueOrNull(Boolean::valueOf);
        }
        if (isByte(paramType, baseType)) {
            return valueOrNull(Byte::valueOf);
        }
        if (isShort(paramType, baseType)) {
            return valueOrNull(Short::valueOf);
        }
        if (isInteger(paramType, baseType)) {
            return valueOrNull(Integer::valueOf);
        }
        if (isLong(paramType, baseType)) {
            return valueOrNull(Long::valueOf);
        }
        if (isFloat(paramType, baseType)) {
            return valueOrNull(Float::valueOf);
        }
        if (isDouble(paramType, baseType)) {
            return valueOrNull(Double::valueOf);
        }
        if (isDate(paramType, baseType)) {
            return valueOrNull(Date::valueOf);
        }
        if (isTime(paramType, baseType)) {
            return valueOrNull(Time::valueOf);
        }
        if (isLocalDateTime(paramType)) {
            return valueOrNull(TypeUtils::parseLocalDateTime);
        }
        if (isTimestamp(paramType, baseType)) {
            return valueOrNull(Timestamp::valueOf);
        }

        return value -> value;
    }

    private static Class<?> classFor(String configType) {
        String baseType = baseType(configType);
        switch (baseType) {
            case "DECIMAL":
            case "NUMERIC":
                return BigDecimal.class;
            case "BOOLEAN":
                return Boolean.class;
            case "TINYINT":
                return Byte.class;
            case "SMALLINT":
                return Short.class;
            case "INTEGER":
            case "INT":
                return Integer.class;
            case "BIGINT":
            case "LONG":
                return Long.class;
            case "FLOAT":
                return Float.class;
            case "DOUBLE":
                return Double.class;
            case "DATE":
                return Date.class;
            case "TIME":
                return Time.class;
            case "TIMESTAMP":
            case "TIMESTAMP_LTZ":
            case "TIMESTAMP_WITH_LOCAL_TIME_ZONE":
                return Timestamp.class;
            default:
                return String.class;
        }
    }

    private static int extractScale(String configType) {
        if (configType == null) {
            return 0;
        }
        String trimmed = configType.trim();
        int paren = trimmed.indexOf('(');
        if (paren < 0) {
            return 0;
        }
        int comma = trimmed.indexOf(',', paren + 1);
        if (comma < 0) {
            return 0;
        }
        int end = trimmed.indexOf(')', comma + 1);
        if (end < 0) {
            end = trimmed.length();
        }
        String scaleStr = trimmed.substring(comma + 1, end).trim();
        try {
            return Integer.parseInt(scaleStr);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private static String baseType(String configType) {
        if (configType == null) {
            return "";
        }
        String trimmed = configType.trim();
        int paren = trimmed.indexOf('(');
        if (paren >= 0) {
            trimmed = trimmed.substring(0, paren);
        }
        return trimmed.toUpperCase(Locale.ROOT);
    }

    private static boolean isDecimal(Class<?> type, String baseType) {
        return type == BigDecimal.class || baseType.equals("DECIMAL") || baseType.equals("NUMERIC");
    }

    private static boolean isBoolean(Class<?> type, String baseType) {
        return type == Boolean.class || type == boolean.class || baseType.equals("BOOLEAN");
    }

    private static boolean isByte(Class<?> type, String baseType) {
        return type == Byte.class || type == byte.class || baseType.equals("TINYINT");
    }

    private static boolean isShort(Class<?> type, String baseType) {
        return type == Short.class || type == short.class || baseType.equals("SMALLINT");
    }

    private static boolean isInteger(Class<?> type, String baseType) {
        return type == Integer.class || type == int.class
                || baseType.equals("INTEGER") || baseType.equals("INT");
    }

    private static boolean isLong(Class<?> type, String baseType) {
        return type == Long.class || type == long.class
                || baseType.equals("BIGINT") || baseType.equals("LONG");
    }

    private static boolean isFloat(Class<?> type, String baseType) {
        return type == Float.class || type == float.class || baseType.equals("FLOAT");
    }

    private static boolean isDouble(Class<?> type, String baseType) {
        return type == Double.class || type == double.class || baseType.equals("DOUBLE");
    }

    private static boolean isDate(Class<?> type, String baseType) {
        return type == Date.class || baseType.equals("DATE");
    }

    private static boolean isTime(Class<?> type, String baseType) {
        return type == Time.class || baseType.equals("TIME");
    }

    private static boolean isTimestamp(Class<?> type, String baseType) {
        return type == Timestamp.class
                || baseType.equals("TIMESTAMP")
                || baseType.equals("TIMESTAMP_LTZ")
                || baseType.equals("TIMESTAMP_WITH_LOCAL_TIME_ZONE");
    }

    private static boolean isLocalDateTime(Class<?> type) {
        return type == LocalDateTime.class;
    }

    private static LocalDateTime parseLocalDateTime(String value) {
        return LocalDateTime.parse(value, TIMESTAMP_FORMATTER);
    }
}
