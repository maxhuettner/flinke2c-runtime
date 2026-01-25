package org.example.proxy;

import java.lang.reflect.Method;
import java.math.BigDecimal;

public final class ScalarFunctionAdapter {
    private final Object udf;
    private final Method eval;

    public ScalarFunctionAdapter(String udfClassName) throws Exception {
        Class<?> udfClass = Class.forName(udfClassName);
        this.udf = udfClass.getDeclaredConstructor().newInstance();
        this.eval = udfClass.getMethod("eval", BigDecimal.class);
        this.eval.setAccessible(true);
    }

    public void evalBatch(String[] values) throws Exception {
        for (String value : values) {
            BigDecimal input = value == null ? null : new BigDecimal(value);
            eval.invoke(udf, input);
        }
    }

    public String[] evalBatchToString(String[] values) throws Exception {
        String[] out = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            String value = values[i];
            BigDecimal input = value == null ? null : new BigDecimal(value);
            Object result = eval.invoke(udf, input);
            out[i] = result == null ? null : result.toString();
        }
        return out;
    }
}
