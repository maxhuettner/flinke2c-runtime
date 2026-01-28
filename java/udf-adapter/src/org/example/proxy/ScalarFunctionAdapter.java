package org.example.proxy;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

public final class ScalarFunctionAdapter {
    private static final int DECIMAL_BYTES = 16;
    public static final class ColumnarResult {
        private final Object[] columns;
        private final boolean[][] nulls;

        ColumnarResult(Object[] columns, boolean[][] nulls) {
            this.columns = columns;
            this.nulls = nulls;
        }

        public Object[] columns() {
            return columns;
        }

        public boolean[][] nulls() {
            return nulls;
        }
    }
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
    private final Method evalMethod;
    private final MethodHandle evalHandle;
    private final MethodHandle evalHandleBound;
    private final MethodHandle evalHandleSpreader;
    private final ValueParser[] parsers;
    private final int[] decimalScales;
    private final int argCount;
    private final Class<?> rowClass;
    private final MethodHandle rowGetArityHandle;
    private final MethodHandle rowGetFieldHandle;
    private final boolean rowReturn;

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
        this.evalMethod = resolvedEval;
        this.parsers = buildParsers(effectiveTypes, resolvedEval.getParameterTypes());
        this.decimalScales = extractDecimalScales(effectiveTypes, paramCount);
        this.argCount = resolvedEval.getParameterCount();

        MethodHandles.Lookup lookup = MethodHandles.lookup();
        this.evalHandle = lookup.unreflect(resolvedEval);
        this.evalHandleBound = evalHandle.bindTo(udf);
        this.evalHandleSpreader = evalHandleBound.asSpreader(Object[].class, argCount);

        this.rowClass = loadRowClass();
        Method rowArityMethod = resolveRowMethod(rowClass, "getArity");
        Method rowFieldMethod = resolveRowMethod(rowClass, "getField", int.class);
        this.rowGetArityHandle = rowArityMethod == null ? null : lookup.unreflect(rowArityMethod);
        this.rowGetFieldHandle = rowFieldMethod == null ? null : lookup.unreflect(rowFieldMethod);
        this.rowReturn = rowClass != null && rowClass.isAssignableFrom(resolvedEval.getReturnType());
    }

    public void evalBatch(String[] values) throws Exception {
        evalBatch(new String[][] { values });
    }

    public void evalBatch(String[][] columns) throws Exception {
        if (columns == null || columns.length == 0) {
            return;
        }

        int rowCount = validateColumns(columns);
        Object[] args = new Object[argCount];
        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            invokeEval(args);
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
        String[] out = new String[rowCount];
        Object[] args = new Object[argCount];
        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            Object result = invokeEval(args);
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
        Object[] args = new Object[argCount];
        int outputArity = -1;
        String[][] out = null;

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                String value = columns[arg][row];
                args[arg] = parsers[arg].parse(value);
            }
            Object result = invokeEval(args);
            if (rowReturn) {
                if (result == null) {
                    continue;
                }
                int arity = rowArity(result);
                if (outputArity < 0) {
                    outputArity = arity;
                    out = new String[outputArity][rowCount];
                } else if (arity != outputArity) {
                    throw new IllegalStateException(
                            "Inconsistent output arity: expected " + outputArity + " but got " + arity);
                }
                for (int col = 0; col < outputArity; col++) {
                    Object value = rowField(result, col);
                    out[col][row] = value == null ? null : value.toString();
                }
            } else {
                if (outputArity < 0) {
                    outputArity = 1;
                    out = new String[1][rowCount];
                }
                out[0][row] = result == null ? null : result.toString();
            }
        }

        if (out == null) {
            outputArity = rowReturn ? argCount : 1;
            out = new String[outputArity][rowCount];
        }
        return out;
    }

    public String[][] evalBatchToColumnsTyped(Object[] columns, boolean[][] nulls) throws Exception {
        if (columns == null || columns.length == 0) {
            return new String[0][0];
        }

        int rowCount = validateTypedColumns(columns, nulls);
        ColumnReader[] readers = buildReaders(columns, nulls);
        Object[] args = new Object[argCount];
        int outputArity = -1;
        String[][] out = null;

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                args[arg] = readers[arg].get(row);
            }
            Object result = invokeEval(args);
            if (rowReturn) {
                if (result == null) {
                    continue;
                }
                int arity = rowArity(result);
                if (outputArity < 0) {
                    outputArity = arity;
                    out = new String[outputArity][rowCount];
                } else if (arity != outputArity) {
                    throw new IllegalStateException(
                            "Inconsistent output arity: expected " + outputArity + " but got " + arity);
                }
                for (int col = 0; col < outputArity; col++) {
                    Object value = rowField(result, col);
                    out[col][row] = value == null ? null : value.toString();
                }
            } else {
                if (outputArity < 0) {
                    outputArity = 1;
                    out = new String[1][rowCount];
                }
                out[0][row] = result == null ? null : result.toString();
            }
        }

        if (out == null) {
            outputArity = rowReturn ? argCount : 1;
            out = new String[outputArity][rowCount];
        }
        return out;
    }

    public ColumnarResult evalBatchToColumnsTypedOut(Object[] columns, boolean[][] nulls)
            throws Exception {
        if (columns == null || columns.length == 0) {
            return new ColumnarResult(new Object[0], new boolean[0][0]);
        }

        int rowCount = validateTypedColumns(columns, nulls);
        ColumnReader[] readers = buildReaders(columns, nulls);
        Object[] args = new Object[argCount];
        int outputArity = -1;
        Object[] out = null;
        boolean[][] outNulls = null;
        List<Integer> pendingNullRows = new ArrayList<>();

        if (!rowReturn && argCount == 1) {
            Object outputArray = allocateOutputArray(evalMethod.getReturnType(), rowCount);
            Object[] outputColumns = new Object[] { outputArray };
            boolean[][] outputNulls = new boolean[1][rowCount];
            ColumnReader reader = readers[0];
            for (int row = 0; row < rowCount; row++) {
                Object arg = reader.get(row);
                Object result = invokeEvalSingle(arg);
                if (result == null) {
                    outputNulls[0][row] = true;
                    continue;
                }
                writeOutputValue(outputArray, result, row);
            }
            return new ColumnarResult(outputColumns, outputNulls);
        }

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                args[arg] = readers[arg].get(row);
            }
            Object result = invokeEval(args);
            if (rowReturn) {
                if (result == null) {
                    if (outputArity > 0) {
                        for (int col = 0; col < outputArity; col++) {
                            outNulls[col][row] = true;
                        }
                    } else {
                        pendingNullRows.add(row);
                    }
                    continue;
                }
                int arity = rowArity(result);
                if (outputArity < 0) {
                    outputArity = arity;
                    out = new Object[outputArity];
                    outNulls = new boolean[outputArity][rowCount];
                    for (int pendingRow : pendingNullRows) {
                        for (int col = 0; col < outputArity; col++) {
                            outNulls[col][pendingRow] = true;
                        }
                    }
                    pendingNullRows.clear();
                } else if (arity != outputArity) {
                    throw new IllegalStateException(
                            "Inconsistent output arity: expected " + outputArity + " but got " + arity);
                }
                for (int col = 0; col < outputArity; col++) {
                    Object value = rowField(result, col);
                    if (value == null) {
                        outNulls[col][row] = true;
                        continue;
                    }
                    out[col] = ensureOutputArray(out[col], value, rowCount);
                    writeOutputValue(out[col], value, row);
                }
            } else {
                if (outputArity < 0) {
                    outputArity = 1;
                    out = new Object[1];
                    outNulls = new boolean[1][rowCount];
                }
                if (result == null) {
                    outNulls[0][row] = true;
                    continue;
                }
                out[0] = ensureOutputArray(out[0], result, rowCount);
                writeOutputValue(out[0], result, row);
            }
        }

        if (out == null) {
            outputArity = rowReturn ? argCount : 1;
            out = new Object[outputArity];
            outNulls = new boolean[outputArity][rowCount];
            for (int row = 0; row < rowCount; row++) {
                for (int col = 0; col < outputArity; col++) {
                    outNulls[col][row] = true;
                }
            }
        }

        for (int col = 0; col < out.length; col++) {
            if (out[col] == null) {
                out[col] = new String[rowCount];
            }
        }

        return new ColumnarResult(out, outNulls);
    }

    private int validateColumns(String[][] columns) {
        int argCount = this.argCount;
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

    private interface ColumnReader {
        Object get(int row) throws Exception;
    }

    private ColumnReader[] buildReaders(Object[] columns, boolean[][] nulls) {
        ColumnReader[] readers = new ColumnReader[argCount];
        Class<?>[] paramTypes = evalMethod.getParameterTypes();
        for (int i = 0; i < argCount; i++) {
            Object column = columns[i];
            boolean[] isNull = nulls != null && i < nulls.length ? nulls[i] : null;
            readers[i] = readerFor(column, isNull, paramTypes[i], parsers[i], decimalScales[i]);
        }
        return readers;
    }

    private ColumnReader readerFor(
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
                BigDecimal decimal = decimalFromBytes(values, row, decimalScale);
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

        throw new IllegalArgumentException("Unsupported column type: " + column.getClass());
    }

    private int validateTypedColumns(Object[] columns, boolean[][] nulls) {
        int argCount = this.argCount;
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
            if (len % DECIMAL_BYTES != 0) {
                throw new IllegalArgumentException("Decimal byte column has invalid length " + len);
            }
            return len / DECIMAL_BYTES;
        }
        if (column instanceof String[]) {
            return ((String[]) column).length;
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

    private static Object ensureOutputArray(Object current, Object value, int rowCount) {
        if (current != null) {
            return current;
        }
        if (value == null) {
            return null;
        }
        if (value instanceof Boolean) {
            return new boolean[rowCount];
        }
        if (value instanceof Float || value instanceof Double) {
            return new double[rowCount];
        }
        if (value instanceof BigDecimal) {
            return new byte[rowCount * DECIMAL_BYTES];
        }
        if (value instanceof String) {
            return new String[rowCount];
        }
        if (value instanceof Timestamp
                || value instanceof LocalDateTime
                || value instanceof Date
                || value instanceof Time) {
            return new long[rowCount];
        }
        if (value instanceof Number) {
            return new long[rowCount];
        }
        return new String[rowCount];
    }

    private static Object allocateOutputArray(Class<?> returnType, int rowCount) {
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
        if (returnType == BigDecimal.class) {
            return new byte[rowCount * DECIMAL_BYTES];
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

    private static void writeOutputValue(Object array, Object value, int row) {
        if (array instanceof byte[]) {
            writeDecimalBytes((byte[]) array, row, (BigDecimal) value);
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

    private static void writeDecimalBytes(byte[] target, int row, BigDecimal value) {
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

    private static void writeLongTo128(byte[] target, int offset, long value) {
        byte pad = (byte) (value < 0 ? 0xFF : 0x00);
        for (int i = 0; i < 8; i++) {
            target[offset + i] = pad;
        }
        for (int i = 0; i < 8; i++) {
            target[offset + 8 + i] = (byte) (value >>> (56 - (i * 8)));
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

    private static BigDecimal decimalFromBytes(byte[] values, int row, int scale) {
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

    private Object invokeEval(Object[] args) throws Exception {
        try {
            return evalHandleSpreader.invoke(args);
        } catch (Throwable t) {
            if (t instanceof Exception) {
                throw (Exception) t;
            }
            throw new Exception("Failed to invoke eval method", t);
        }
    }

    private Object invokeEvalSingle(Object arg) throws Exception {
        try {
            return evalHandleBound.invoke(arg);
        } catch (Throwable t) {
            if (t instanceof Exception) {
                throw (Exception) t;
            }
            throw new Exception("Failed to invoke eval method", t);
        }
    }

    private int rowArity(Object row) throws Exception {
        if (rowGetArityHandle == null) {
            return 0;
        }
        try {
            return (int) rowGetArityHandle.invoke(row);
        } catch (Throwable t) {
            if (t instanceof Exception) {
                throw (Exception) t;
            }
            throw new Exception("Failed to read Row arity", t);
        }
    }

    private Object rowField(Object row, int index) throws Exception {
        if (rowGetFieldHandle == null) {
            return null;
        }
        try {
            return rowGetFieldHandle.invoke(row, index);
        } catch (Throwable t) {
            if (t instanceof Exception) {
                throw (Exception) t;
            }
            throw new Exception("Failed to read Row field", t);
        }
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

    private static int[] extractDecimalScales(String[] argTypes, int paramCount) {
        int[] out = new int[paramCount];
        for (int i = 0; i < paramCount; i++) {
            String configType = (argTypes != null && i < argTypes.length) ? argTypes[i] : null;
            out[i] = extractScale(configType);
        }
        return out;
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
