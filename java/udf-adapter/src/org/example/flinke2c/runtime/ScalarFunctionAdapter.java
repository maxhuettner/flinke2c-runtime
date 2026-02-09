package org.example.flinke2c.runtime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.math.BigDecimal;

public final class ScalarFunctionAdapter {
    private static final ColumnarResult EMPTY_COLUMNAR_RESULT =
            new ColumnarResult(new Object[0], new boolean[0][0]);

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

    private final Object udf;
    private final Method evalMethod;
    private final MethodHandle evalHandle;
    private final MethodHandle evalHandleBound;
    private final MethodHandle evalHandleSpreader;
    private final ValueParser[] parsers;
    private final int[] decimalScales;
    private final int argCount;

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
        this.parsers = TypeUtils.buildParsers(effectiveTypes, resolvedEval.getParameterTypes());
        this.decimalScales = TypeUtils.extractDecimalScales(effectiveTypes, paramCount);
        this.argCount = resolvedEval.getParameterCount();

        MethodHandles.Lookup lookup = MethodHandles.lookup();
        this.evalHandle = lookup.unreflect(resolvedEval);
        this.evalHandleBound = evalHandle.bindTo(udf);
        this.evalHandleSpreader = evalHandleBound.asSpreader(Object[].class, argCount);
    }

    private ColumnarResult evalBatchToColumnsTypedOut(Object[] columns, boolean[][] nulls)
            throws Exception {
        if (isEmptyColumns(columns)) {
            return EMPTY_COLUMNAR_RESULT;
        }

        int rowCount = ColumnReaders.validateTypedColumns(columns, nulls, argCount);
        ColumnReader[] readers = ColumnReaders.buildReaders(
                columns,
                nulls,
                evalMethod.getParameterTypes(),
                parsers,
                decimalScales);
        Object[] args = new Object[argCount];

        Class<?> returnType = evalMethod.getReturnType();
        if (!TypeUtils.isScalarReturnType(returnType)) {
            throw new IllegalArgumentException(
                    "POJO return requires named output mapping (use evalBatchFastNamed)");
        }

        if (argCount == 1) {
            Object outputArray = OutputWriters.allocateOutputArray(returnType, rowCount);
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
                OutputWriters.writeOutputValue(outputArray, result, row);
            }
            return new ColumnarResult(outputColumns, outputNulls);
        }

        Object[] out = new Object[] { OutputWriters.allocateOutputArray(returnType, rowCount) };
        boolean[][] outNulls = new boolean[1][rowCount];

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                args[arg] = readers[arg].get(row);
            }
            Object result = invokeEval(args);
            if (result == null) {
                outNulls[0][row] = true;
                continue;
            }
            OutputWriters.writeOutputValue(out[0], result, row);
        }

        return new ColumnarResult(out, outNulls);
    }

    public ColumnarResult evalBatchFast(Object[] columns, boolean[][] nulls) throws Exception {
        if (isEmptyColumns(columns)) {
            return EMPTY_COLUMNAR_RESULT;
        }

        if (argCount == 1 && columns[0] instanceof byte[]) {
            int rowCount = ColumnReaders.validateTypedColumns(columns, nulls, argCount);
            byte[] values = (byte[]) columns[0];
            boolean[] isNull = nulls != null && nulls.length > 0 ? nulls[0] : null;
            Object outputArray = OutputWriters.allocateOutputArray(evalMethod.getReturnType(), rowCount);
            boolean[][] outputNulls = new boolean[1][rowCount];
            int scale = decimalScales[0];

            for (int row = 0; row < rowCount; row++) {
                if (isNull != null && isNull[row]) {
                    outputNulls[0][row] = true;
                    continue;
                }
                BigDecimal input = DecimalUtils.decimalFromBytes(values, row, scale);
                Object result = invokeEvalSingle(input);
                if (result == null) {
                    outputNulls[0][row] = true;
                    continue;
                }
                OutputWriters.writeOutputValue(outputArray, result, row);
            }

            return new ColumnarResult(new Object[] { outputArray }, outputNulls);
        }

        return evalBatchToColumnsTypedOut(columns, nulls);
    }

    public ColumnarResult evalBatchFastNamed(Object[] columns, boolean[][] nulls, String[] outputNames)
            throws Exception {
        if (isEmptyColumns(columns)) {
            return EMPTY_COLUMNAR_RESULT;
        }
        if (outputNames == null || outputNames.length == 0) {
            return evalBatchFast(columns, nulls);
        }

        Class<?> returnType = evalMethod.getReturnType();
        if (TypeUtils.isScalarReturnType(returnType) && outputNames.length == 1) {
            return evalBatchFast(columns, nulls);
        }

        return evalBatchFastNamedPojo(columns, nulls, outputNames);
    }

    private ColumnarResult evalBatchFastNamedPojo(
            Object[] columns,
            boolean[][] nulls,
            String[] outputNames) throws Exception {
        int rowCount = ColumnReaders.validateTypedColumns(columns, nulls, argCount);
        ColumnReader[] readers = ColumnReaders.buildReaders(
                columns,
                nulls,
                evalMethod.getParameterTypes(),
                parsers,
                decimalScales);
        Object[] args = new Object[argCount];
        int outputArity = outputNames.length;
        Object[] out = new Object[outputArity];
        boolean[][] outNulls = new boolean[outputArity][rowCount];

        Class<?>[] fieldTypes = new Class<?>[outputArity];
        OutputAccessor[] accessors = PojoAccessors.buildPojoAccessors(
                evalMethod.getReturnType(),
                outputNames,
                fieldTypes);

        for (int col = 0; col < outputArity; col++) {
            out[col] = OutputWriters.allocateOutputArray(fieldTypes[col], rowCount);
        }

        for (int row = 0; row < rowCount; row++) {
            for (int arg = 0; arg < argCount; arg++) {
                args[arg] = readers[arg].get(row);
            }
            Object result = invokeEval(args);
            if (result == null) {
                for (int col = 0; col < outputArity; col++) {
                    outNulls[col][row] = true;
                }
                continue;
            }

            for (int col = 0; col < outputArity; col++) {
                Object value = accessors[col].get(result);
                if (value == null) {
                    outNulls[col][row] = true;
                    continue;
                }
                OutputWriters.writeOutputValue(out[col], value, row);
            }
        }

        return new ColumnarResult(out, outNulls);
    }

    private static boolean isEmptyColumns(Object[] columns) {
        return columns == null || columns.length == 0;
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

    private static Method resolveEvalMethod(Class<?> udfClass, int paramCount, String[] argTypes)
            throws Exception {
        Class<?>[] mappedTypes = TypeUtils.mapFlinkTypesToClasses(argTypes, paramCount);
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
}
