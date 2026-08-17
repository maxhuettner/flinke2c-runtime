/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.runtime.functions.table.gpuruntime;

import org.apache.flink.annotation.Internal;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.operators.TableStreamOperator;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.util.FlinkRuntimeException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Single in-process operator for GPU (or any other local-accelerator) scalar functions selected
 * via {@code table.exec.gpu-runtime.function-class}.
 *
 * <p><b>Why one operator, not a PRE/POST pair like the external runtime's RDMA/TCP operators:</b>
 * PRE and POST exist because that runtime lives on the far side of a network hop (a socket or an
 * RDMA endpoint) and may run on a different TaskManager; POST has to exist as a separate operator
 * so the pipeline can keep flowing while a batch is in flight. A GPU/accelerator function invoked
 * through {@link GpuRuntimeFunction} is an in-process call on the local machine, so one operator
 * can buffer input, dispatch, and emit output itself&mdash;there is no far side to place a second
 * operator on.
 *
 * <p><b>Why reflection instead of one operator subclass per function:</b> the RDMA/TCP PRE/POST
 * operators are pure relays&mdash;they move bytes over a fixed wire protocol and never interpret
 * them, so the same two classes serve every function and only need to exist once, built into
 * Flink. A GPU function is the opposite: the whole point is the in-process computation, and that
 * computation is different for every function. Hard-coding one operator subclass per function
 * would mean rebuilding Flink (or at least this module) every time a function's implementation
 * changes or a new one is added. Instead, this class is the fixed, function-agnostic half
 * (batching/flushing/lifecycle), and {@link GpuRuntimeFunction} is the swappable half, loaded by
 * class name from {@code impl=} at {@link #open()} time via the user code class loader&mdash;i.e.
 * from whatever jar was dropped into {@code lib/} (or shipped with the job), no Flink rebuild
 * required to add or change a function.
 *
 * <p><b>Batching:</b> the dynamically loaded implementation receives each live row immediately
 * and owns batching, native buffers, CUDA submission, and output ordering. This avoids the old
 * per-row serializer-copy path.
 */
@Internal
public final class GpuRuntimeOperator extends TableStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(GpuRuntimeOperator.class);

    private static final int DEFAULT_BATCH_SIZE = 256;

    private final String conf;
    private final RowType inputRowType;
    private final RowType resultRowType;

    private transient int batchSize;
    private transient GpuRuntimeFunction function;

    public GpuRuntimeOperator(String conf, RowType inputRowType, RowType resultRowType) {
        this.conf = conf == null ? "" : conf;
        this.inputRowType = Objects.requireNonNull(inputRowType, "inputRowType");
        this.resultRowType = resultRowType == null ? this.inputRowType : resultRowType;
    }

    @Override
    public void open() throws Exception {
        super.open();
        final Map<String, String> values = parseConf(conf);
        final String implClassName = values.get("impl");
        if (implClassName == null || implClassName.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "table.exec.gpu-runtime.conf.<functionClass> requires an 'impl=<class>' entry "
                            + "naming a class implementing "
                            + GpuRuntimeFunction.class.getName());
        }
        this.batchSize = intValue(values, "batchsize", DEFAULT_BATCH_SIZE);
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchsize must be positive, was " + batchSize);
        }

        final ClassLoader classLoader = getRuntimeContext().getUserCodeClassLoader();
        this.function = instantiate(implClassName.trim(), classLoader);
        final DefaultGpuRuntimeFunctionContext functionContext =
                new DefaultGpuRuntimeFunctionContext(
                        values,
                        inputRowType,
                        resultRowType,
                        batchSize,
                        classLoader,
                        getRuntimeContext().getTaskInfo().getIndexOfThisSubtask(),
                        getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks());
        function.open(
                functionContext,
                (row, hasTimestamp, timestamp) -> {
                    if (hasTimestamp) output.collect(new StreamRecord<>(row, timestamp));
                    else output.collect(new StreamRecord<>(row));
                });

        LOG.info(
                "GpuRuntimeOperator opened (impl={}, batchSize={})", implClassName.trim(), batchSize);
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        function.processElement(
                element.getValue(), element.hasTimestamp(),
                element.hasTimestamp() ? element.getTimestamp() : 0L);
        // Implementations may complete GPU work on a private worker, but all
        // output collection remains on this Flink operator thread.
        function.poll();
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        flush();
        super.processWatermark(mark);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        flush();
        super.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void endInput() throws Exception {
        flush();
    }

    @Override
    public void close() throws Exception {
        try {
            flush();
        } finally {
            try {
                if (function != null) {
                    function.close();
                }
            } finally {
                function = null;
                super.close();
            }
        }
    }

    private void flush() throws Exception {
        if (function != null) function.flush();
    }

    private static GpuRuntimeFunction instantiate(String className, ClassLoader classLoader) {
        final Class<?> rawClass;
        try {
            rawClass = Class.forName(className, true, classLoader);
        } catch (ClassNotFoundException e) {
            throw new FlinkRuntimeException(
                    "GPU runtime implementation class '"
                            + className
                            + "' was not found on the classpath. It must be available to the "
                            + "TaskManager, e.g. via a jar in lib/.",
                    e);
        }
        if (!GpuRuntimeFunction.class.isAssignableFrom(rawClass)) {
            throw new FlinkRuntimeException(
                    "GPU runtime implementation class '"
                            + className
                            + "' does not implement "
                            + GpuRuntimeFunction.class.getName());
        }
        try {
            return (GpuRuntimeFunction) rawClass.getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new FlinkRuntimeException(
                    "Failed to instantiate GPU runtime implementation class '"
                            + className
                            + "'. It must expose a public no-argument constructor.",
                    e);
        }
    }

    private static Map<String, String> parseConf(String conf) {
        if (conf == null || conf.trim().isEmpty()) {
            return Collections.emptyMap();
        }
        final Map<String, String> values = new HashMap<>();
        for (String item : conf.split(";")) {
            final int equals = item.indexOf('=');
            if (equals > 0 && equals < item.length() - 1) {
                values.put(
                        item.substring(0, equals).trim().toLowerCase(Locale.ROOT),
                        item.substring(equals + 1).trim());
            }
        }
        return values;
    }

    private static int intValue(Map<String, String> values, String key, int fallback) {
        final String value = values.get(key);
        return value == null || value.isEmpty() ? fallback : Integer.parseInt(value);
    }

    private static final class DefaultGpuRuntimeFunctionContext implements GpuRuntimeFunctionContext {
        private final Map<String, String> conf;
        private final RowType inputRowType;
        private final RowType resultRowType;
        private final int batchSize;
        private final ClassLoader classLoader;
        private final int subtaskIndex;
        private final int numberOfParallelSubtasks;

        private DefaultGpuRuntimeFunctionContext(
                Map<String, String> conf,
                RowType inputRowType,
                RowType resultRowType,
                int batchSize,
                ClassLoader classLoader,
                int subtaskIndex,
                int numberOfParallelSubtasks) {
            this.conf = Collections.unmodifiableMap(conf);
            this.inputRowType = inputRowType;
            this.resultRowType = resultRowType;
            this.batchSize = batchSize;
            this.classLoader = classLoader;
            this.subtaskIndex = subtaskIndex;
            this.numberOfParallelSubtasks = numberOfParallelSubtasks;
        }

        @Override
        public Map<String, String> getConf() {
            return conf;
        }

        @Override
        public RowType getInputRowType() {
            return inputRowType;
        }

        @Override
        public RowType getResultRowType() {
            return resultRowType;
        }

        @Override
        public int getBatchSize() {
            return batchSize;
        }

        @Override
        public ClassLoader getUserCodeClassLoader() {
            return classLoader;
        }

        @Override
        public int getSubtaskIndex() {
            return subtaskIndex;
        }

        @Override
        public int getNumberOfParallelSubtasks() {
            return numberOfParallelSubtasks;
        }
    }
}
