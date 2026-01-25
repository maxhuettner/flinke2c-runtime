package org.apache.flink.table.functions;

public abstract class ScalarFunction {
    public void open(FunctionContext context) throws Exception {}

    public void close() throws Exception {}

    public boolean isDeterministic() {
        return true;
    }
}
