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

package org.apache.flink.table.runtime.functions.table.externalruntime;

import org.apache.flink.annotation.Internal;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Shared base for the one-sided RDMA PRE/POST operators.
 *
 * <p>The operator depends only on {@link RdmaRingBuffer}. The implementation of that abstraction
 * owns the registered host send and receive rings and publishes batches with RDMA
 * Write-with-Immediate. Returned slots become visible only after their receive CQE has been
 * observed. There is deliberately no application-level acknowledgement protocol: consuming a
 * completed batch returns ring credits locally.
 *
 * <p>PRE and POST may run on different TaskManagers. Each operator owns its own native session;
 * the accelerator runtime coordinates the independent bootstrap connections.
 */
@Internal
public abstract class RdmaOperator extends ExternalRuntimeOperator {

    private static final long serialVersionUID = 1L;

    static final int RING_BUFFER_ELEMENTS = 65536;
    static final int DEFAULT_MAX_ITEM_SIZE = 2048;
    private static final int DEFAULT_BATCH_SIZE = 64;

    /** Selects the JNI RDMA transport when {@code type=rdma} (or {@code transport=rdma}) is set. */
    public static boolean usesRdmaTransport(String conf) {
        if (conf == null) {
            return false;
        }
        for (String item : conf.split(";")) {
            int equals = item.indexOf('=');
            if (equals > 0
                    && ("type".equalsIgnoreCase(item.substring(0, equals).trim())
                            || "transport".equalsIgnoreCase(item.substring(0, equals).trim()))) {
                return "rdma".equalsIgnoreCase(item.substring(equals + 1).trim());
            }
        }
        return false;
    }

    private final @Nullable RdmaRingBufferFactory ringBufferFactory;
    protected transient RdmaConfig rdmaConfig;
    protected transient RdmaRingBuffer rdmaSession;

    protected RdmaOperator(String conf, RowType rowType) {
        this(conf, rowType, rowType, null);
    }

    protected RdmaOperator(String conf, RowType inputRowType, @Nullable RowType resultRowType) {
        this(conf, inputRowType, resultRowType, null);
    }

    protected RdmaOperator(
            String conf,
            RowType inputRowType,
            @Nullable RowType resultRowType,
            @Nullable RdmaRingBufferFactory ringBufferFactory) {
        super(conf, inputRowType, resultRowType);
        this.ringBufferFactory = ringBufferFactory;
    }

    /** Opens the native RDMA session owned by this operator. */
    protected final void openRdmaSession() throws IOException {
        this.rdmaConfig = RdmaConfig.from(conf, tcpConfig, getRuntimeContext().getTaskInfo()
                .getIndexOfThisSubtask());
        final RdmaRingBuffer ringBuffer = resolveFactory().open(rdmaConfig.endpoint());
        validateRingBuffer(ringBuffer);
        this.rdmaSession = ringBuffer;
    }

    protected final void closeRdmaSession() throws IOException {
        final RdmaRingBuffer session = rdmaSession;
        rdmaSession = null;
        rdmaConfig = null;
        if (session == null) {
            return;
        }
        session.close();
    }

    protected final void writeInputSlot(byte[] slot) throws IOException {
        if (slot == null || slot.length > rdmaConfig.maxItemSize) {
            throw new IOException(
                    "RDMA slot payload must be non-null and at most "
                            + rdmaConfig.maxItemSize
                            + " bytes");
        }
        requireSession().writeInputSlot(slot);
    }

    /**
     * Writes a whole batch of already-framed rows in one native call instead of one
     * per row. {@code batch} must be a direct buffer holding each row's frame at a
     * fixed {@code rdmaConfig.maxItemSize} stride starting at offset 0; {@code
     * frameLengths[i]} is row {@code i}'s real encoded length.
     */
    protected final void writeInputBatch(java.nio.ByteBuffer batch, int[] frameLengths, int count)
            throws IOException {
        if (count <= 0 || count > rdmaConfig.batchSize) {
            throw new IOException(
                    "RDMA batch size must be in 1.." + rdmaConfig.batchSize + ", but was " + count);
        }
        requireSession().writeInputBatch(batch, frameLengths, count);
    }

    protected final void publishInputBatch(int count) throws IOException {
        if (count <= 0 || count > rdmaConfig.batchSize) {
            throw new IOException(
                    "RDMA batch size must be in 1.." + rdmaConfig.batchSize + ", but was " + count);
        }
        requireSession().publishInputBatch(count);
    }

    protected final List<byte[]> receiveBatch() throws IOException {
        final List<byte[]> values = requireSession().receiveOutputBatch();
        if (values == null || values.isEmpty() || values.size() > rdmaConfig.batchSize) {
            throw new IOException("RDMA ring buffer returned an invalid completed batch");
        }
        for (byte[] value : values) {
            if (value == null || value.length > rdmaConfig.maxItemSize) {
                throw new IOException("RDMA ring buffer returned an invalid slot length");
            }
        }
        return values;
    }

    private RdmaRingBuffer requireSession() throws IOException {
        if (rdmaSession == null) {
            throw new IOException("RDMA session is not open");
        }
        return rdmaSession;
    }

    private RdmaRingBufferFactory resolveFactory() throws IOException {
        if (ringBufferFactory != null) {
            return ringBufferFactory;
        }
        return role() == Role.PRE
                ? RustRdmaRingBuffer.Factory.PRE
                : RustRdmaRingBuffer.Factory.POST;
    }

    private void validateRingBuffer(RdmaRingBuffer ringBuffer) throws IOException {
        if (ringBuffer == null) {
            throw new IOException("RdmaRingBufferFactory returned null");
        }
        if (ringBuffer.capacity() != RING_BUFFER_ELEMENTS) {
            throw new IOException(
                    "RDMA ring capacity mismatch: expected "
                            + RING_BUFFER_ELEMENTS
                            + " but was "
                            + ringBuffer.capacity());
        }
        if (ringBuffer.maxItemSize() != rdmaConfig.maxItemSize) {
            throw new IOException(
                    "RDMA slot size mismatch: expected "
                            + rdmaConfig.maxItemSize
                            + " but was "
                            + ringBuffer.maxItemSize());
        }
    }

    /**
     * Accelerator-independent communication abstraction for the two registered RDMA rings.
     * Implementations write slot bytes into registered send memory and use the batch count as the
     * Write-with-Immediate value. Receiving a batch must wait for its CQE before exposing slots and
     * must return local credits when those slots are consumed. PRE and POST can call the input and
     * output methods concurrently, so implementations must allow one producer and one consumer.
     */
    public interface RdmaRingBuffer extends AutoCloseable {
        int capacity();

        int maxItemSize();

        void writeInputSlot(byte[] value) throws IOException;

        /**
         * Writes {@code count} already-framed rows from a shared direct buffer in one
         * call. {@code batch} holds each row's frame at a fixed stride of the ring's
         * {@code maxItemSize()} starting at offset 0; {@code frameLengths[i]} is row
         * {@code i}'s real encoded length (at most {@code maxItemSize()}).
         */
        void writeInputBatch(java.nio.ByteBuffer batch, int[] frameLengths, int count) throws IOException;

        void publishInputBatch(int count) throws IOException;

        List<byte[]> receiveOutputBatch() throws IOException;

        @Override
        void close() throws IOException;
    }

    /** Creates the Java RDMA ring-buffer communication endpoint used by an operator pair. */
    public interface RdmaRingBufferFactory extends Serializable {
        RdmaRingBuffer open(RdmaEndpoint endpoint) throws IOException;
    }

    /** Immutable connection and shared-layout description passed to a ring-buffer factory. */
    public static final class RdmaEndpoint implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String host;
        public final int port;
        public final String ibDevice;
        public final int ibPort;
        public final int gidIndex;
        public final int batchSize;
        public final int ringElements;
        public final int maxItemSize;
        public final int connectTimeoutMs;
        public final String processingSpecJson;

        private RdmaEndpoint(RdmaConfig config) {
            this.host = config.host;
            this.port = config.port;
            this.ibDevice = config.ibDevice;
            this.ibPort = config.ibPort;
            this.gidIndex = config.gidIndex;
            this.batchSize = config.batchSize;
            this.ringElements = RING_BUFFER_ELEMENTS;
            this.maxItemSize = config.maxItemSize;
            this.connectTimeoutMs = config.connectTimeoutMs;
            this.processingSpecJson = config.processingSpecJson;
        }
    }

    static final class RdmaConfig implements Serializable {
        private static final long serialVersionUID = 1L;

        final String host;
        final int port;
        final String ibDevice;
        final int ibPort;
        final int gidIndex;
        final int batchSize;
        final int maxItemSize;
        final int connectTimeoutMs;
        final String processingSpecJson;

        private RdmaConfig(
                String host,
                int port,
                String ibDevice,
                int ibPort,
                int gidIndex,
                int batchSize,
                int maxItemSize,
                int connectTimeoutMs,
                String processingSpecJson) {
            this.host = host;
            this.port = port;
            this.ibDevice = ibDevice;
            this.ibPort = ibPort;
            this.gidIndex = gidIndex;
            this.batchSize = batchSize;
            this.maxItemSize = maxItemSize;
            this.connectTimeoutMs = connectTimeoutMs;
            this.processingSpecJson = processingSpecJson;
        }

        RdmaEndpoint endpoint() {
            return new RdmaEndpoint(this);
        }

        static RdmaConfig from(String conf, ExternalRuntimeTcpConfig tcpConfig, int subtaskIndex) {
            final Map<String, String> values = parse(conf);
            final ExternalRuntimeTcpConfig.ExternalRuntimeEndpoint endpoint =
                    tcpConfig.selectEndpoint(subtaskIndex);
            final int batchSize =
                    intValue(values, "rdmabatchsize", intValue(values, "batchsize", DEFAULT_BATCH_SIZE));
            if (batchSize <= 0 || batchSize >= RING_BUFFER_ELEMENTS) {
                throw new IllegalArgumentException(
                        "RDMA batchSize must be in 1.." + (RING_BUFFER_ELEMENTS - 1));
            }
            final int maxItemSize = intValue(values, "rdmamaxitemsize", DEFAULT_MAX_ITEM_SIZE);
            if (maxItemSize <= 0 || maxItemSize > DEFAULT_MAX_ITEM_SIZE) {
                throw new IllegalArgumentException(
                        "rdmaMaxItemSize must be in 1.." + DEFAULT_MAX_ITEM_SIZE);
            }
            return new RdmaConfig(
                    endpoint.getHost(),
                    endpoint.getSendPort(),
                    value(values, "rdmadevice", ""),
                    intValue(values, "rdmaibport", 1),
                    intValue(values, "rdmagidindex", 3),
                    batchSize,
                    maxItemSize,
                    tcpConfig.getConnectTimeoutMs(),
                    value(values, "rdmaprocessingspec", "{\"function\":\"INCREMENT\",\"field_index\":0,\"fields\":[\"INT32\"]}"));
        }

        private static Map<String, String> parse(String conf) {
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

        private static long longValue(Map<String, String> values, String key, long fallback) {
            final String value = values.get(key);
            return value == null || value.isEmpty() ? fallback : Long.decode(value);
        }

        private static String value(Map<String, String> values, String key, String fallback) {
            final String value = values.get(key);
            return value == null || value.isEmpty() ? fallback : value;
        }
    }

}
