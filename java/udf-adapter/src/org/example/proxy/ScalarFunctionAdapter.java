package org.example.proxy;

import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

public final class ScalarFunctionAdapter {
    private interface ValueParser {
        Object parse(String value);
    }

    private static final String ROW_CLASS_NAME = "org.apache.flink.types.Row";
    private static final DateTimeFormatter TIMESTAMP_FORMATTER = new DateTimeFormatterBuilder()
            .appendPattern("yyyy-MM-dd HH:mm:ss")
            .optionalStart()
            .appendFraction(ChronoField.NANO_OF_SECOND, 0, 9, true)
            .optionalEnd()
            .toFormatter(Locale.ROOT);

    private final Object udf;
    private final Method eval;
    private final ValueParser[] parsers;
    private final Class<?> rowClass;
    private final Method rowGetArity;
    private final Method rowGetField;

    public ScalarFunctionAdapter(String udfClassName) throws Exception {
        this(udfClassName, new String[] { "DECIMAL" });
    }

    public ScalarFunctionAdapter(String udfClassName, String[] argTypes) throws Exception {
        Class<?> udfClass = Class.forName(udfClassName);
        this.udf = udfClass.getDeclaredConstructor().newInstance();

        String[] effectiveTypes = argTypes == null ? new String[0] : argTypes.clone();
        int paramCount = effectiveTypes.length == 0 ? 1 : effectiveTypes.length;

        Method resolvedEval = resolveEvalMethod(udfClass, paramCount, effectiveTypes);
        resolvedEval.setAccessible(true);
        this.eval = resolvedEval;
        this.parsers = buildParsers(effectiveTypes, resolvedEval.getParameterTypes());

        this.rowClass = loadRowClass();
        this.rowGetArity = resolveRowMethod(rowClass, "getArity");
        this.rowGetField = resolveRowMethod(rowClass, "getField", int.class);
    }

    public void evalBatch(String[] values) throws Exception {
        evalBatch(new String[][] { values });
    }

    public void evalBatch(String[][] columns) throws Exception {
        if (columns == null || columns.length == 0) {
            return;
        }

        int rowCount = validateColumns(columns);
        int argCount = eval.getParameterCount();
        Object[] args = new Object[argCount];
        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            eval.invoke(udf, args);
        }
    }

    public String[] evalBatchToString(String[] values) throws Exception {
        return evalBatchToString(new String[][] { values });
    }

    public String[] evalBatchToString(String[][] columns) throws Exception {
        if (columns == null || columns.length == 0) {
            return new String[0];
        }

        int rowCount = validateColumns(columns);
        int argCount = eval.getParameterCount();
        String[] out = new String[rowCount];
        Object[] args = new Object[argCount];
        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            Object result = eval.invoke(udf, args);
            out[row] = result == null ? null : result.toString();
        }
        return out;
    }

    public String[][] evalBatchToColumns(String[] values) throws Exception {
        return evalBatchToColumns(new String[][] { values });
    }

    public String[][] evalBatchToColumns(String[][] columns) throws Exception {
        if (columns == null || columns.length == 0) {
            return new String[0][0];
        }

        int rowCount = validateColumns(columns);
        int argCount = eval.getParameterCount();
        Object[] args = new Object[argCount];
        List<String[]> rowResults = new ArrayList<>(rowCount);
        int outputArity = -1;

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            Object result = eval.invoke(udf, args);
            String[] rowValues = toRowValues(result);
            if (rowValues.length > 0) {
                if (outputArity < 0) {
                    outputArity = rowValues.length;
                } else if (rowValues.length != outputArity) {
                    throw new IllegalStateException(
                            "Inconsistent output arity: expected " + outputArity + " but got " + rowValues.length);
                }
            }
            rowResults.add(rowValues);
        }

        if (outputArity < 0) {
            outputArity = isRowReturn() ? argCount : 1;
        }

        String[][] out = new String[outputArity][rowCount];
        for (int row = 0; row < rowCount; row++) {
            String[] rowValues = rowResults.get(row);
            if (rowValues.length == 0 && outputArity > 0) {
                continue;
            }
            if (rowValues.length != outputArity) {
                throw new IllegalStateException(
                        "Row " + row + " has arity " + rowValues.length + " but expected " + outputArity);
            }
            for (int col = 0; col < outputArity; col++) {
                out[col][row] = rowValues[col];
            }
        }
        return out;
    }

    private int validateColumns(String[][] columns) {
        int argCount = eval.getParameterCount();
        if (columns.length != argCount) {
            throw new IllegalArgumentException(
                    "Expected " + argCount + " argument columns but got " + columns.length);
        }

        int rowCount = columns[0] == null ? 0 : columns[0].length;
        for (int arg = 1; arg < columns.length; arg++) {
            int len = columns[arg] == null ? 0 : columns[arg].length;
            if (len != rowCount) {
                throw new IllegalArgumentException("Argument columns have mismatched lengths");
            }
        }
        return rowCount;
    }

    private boolean isRowReturn() {
        return rowClass != null && rowClass.isAssignableFrom(eval.getReturnType());
    }

    private String[] toRowValues(Object result) throws Exception {
        if (result == null) {
            return new String[0];
        }
        if (rowClass != null && rowClass.isInstance(result)) {
            int arity = (Integer) rowGetArity.invoke(result);
            String[] out = new String[arity];
            for (int i = 0; i < arity; i++) {
                Object value = rowGetField.invoke(result, i);
                out[i] = value == null ? null : value.toString();
            }
            return out;
        }
        return new String[] { result.toString() };
    }

    private static Class<?> loadRowClass() {
        try {
            return Class.forName(ROW_CLASS_NAME);
        } catch (ClassNotFoundException ignored) {
            return null;
        }
    }

    private static Method resolveRowMethod(Class<?> rowClass, String name, Class<?>... params)
            throws Exception {
        if (rowClass == null) {
            return null;
        }
        Method method = rowClass.getMethod(name, params);
        method.setAccessible(true);
        return method;
    }

    private static Method resolveEvalMethod(Class<?> udfClass, int paramCount, String[] argTypes)
            throws Exception {
        Class<?>[] mappedTypes = mapFlinkTypesToClasses(argTypes, paramCount);
        try {
            return udfClass.getMethod("eval", mappedTypes);
        } catch (NoSuchMethodException ignored) {
            // Fall through to a param-count based search.
        }

        for (Method method : udfClass.getMethods()) {
            if (!method.getName().equals("eval")) {
                continue;
            }
            if (method.getParameterCount() == paramCount) {
                return method;
            }
        }

        throw new NoSuchMethodException(
                "No eval method found on " + udfClass.getName() + " with " + paramCount
                        + " parameters");
    }

    private static ValueParser[] buildParsers(String[] argTypes, Class<?>[] paramTypes) {
        int count = paramTypes.length;
        ValueParser[] out = new ValueParser[count];
        for (int i = 0; i < count; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = parserFor(configType, paramTypes[i]);
        }
        return out;
    }

    private static Class<?>[] mapFlinkTypesToClasses(String[] argTypes, int paramCount) {
        Class<?>[] out = new Class<?>[paramCount];
        for (int i = 0; i < paramCount; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = classFor(configType);
        }
        return out;
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
            return valueOrNull(ScalarFunctionAdapter::parseLocalDateTime);
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
        return type == Integer.class || type == int.class || baseType.equals("INTEGER") || baseType.equals("INT");
    }

    private static boolean isLong(Class<?> type, String baseType) {
        return type == Long.class || type == long.class || baseType.equals("BIGINT") || baseType.equals("LONG");
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
