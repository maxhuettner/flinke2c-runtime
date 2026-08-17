package org.apache.flink.table.runtime.functions.table.gpuruntime;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.table.data.RowData;

import java.io.Serializable;

/** Dynamically loaded packed GPU function contract. */
@PublicEvolving
public interface GpuRuntimeFunction extends Serializable {

    interface Emitter {
        void collect(RowData row, boolean hasTimestamp, long timestamp) throws Exception;
    }

    /** Opens one function instance and supplies its output sink. */
    void open(GpuRuntimeFunctionContext context, Emitter emitter) throws Exception;

    /** Accepts one live input row; implementations buffer and submit it as appropriate. */
    void processElement(RowData row, boolean hasTimestamp, long timestamp) throws Exception;

    /**
     * Gives an implementation a chance to publish completed asynchronous work
     * on the Flink operator thread. The default is a no-op for synchronous
     * implementations.
     */
    default void poll() throws Exception {}

    /** Flushes all partial and in-flight batches in input order. */
    void flush() throws Exception;

    /** Releases all native and device resources. */
    void close() throws Exception;
}
