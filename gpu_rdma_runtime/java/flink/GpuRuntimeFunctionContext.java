package org.apache.flink.table.runtime.functions.table.gpuruntime;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.table.types.logical.RowType;

import java.util.Map;

/** Immutable runtime information supplied to a dynamically loaded GPU function. */
@PublicEvolving
public interface GpuRuntimeFunctionContext {
    Map<String, String> getConf();
    RowType getInputRowType();
    RowType getResultRowType();
    int getBatchSize();
    ClassLoader getUserCodeClassLoader();
    int getSubtaskIndex();
    int getNumberOfParallelSubtasks();
}
