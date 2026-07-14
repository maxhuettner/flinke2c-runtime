package org.apache.flink.table.runtime.functions.table.externalruntime;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.flink.table.runtime.functions.table.externalruntime.RdmaOperator.RdmaEndpoint;
import org.apache.flink.table.runtime.functions.table.externalruntime.RdmaOperator.RdmaRingBuffer;
import org.apache.flink.table.runtime.functions.table.externalruntime.RdmaOperator.RdmaRingBufferFactory;

/** Flink-facing adapter around the Rust raw-verbs JNI implementation. */
final class RustRdmaRingBuffer implements RdmaRingBuffer {
    private final RdmaEndpoint config;
    private final long handle;
    private boolean closed;

    private RustRdmaRingBuffer(RdmaEndpoint config, String role) throws IOException {
        this.config = config;
        System.err.println(
                "[RDMA-JNI] opening role=" + role + " to " + config.host + ":" + config.port
                        + " device=" + config.ibDevice + " ibPort=" + config.ibPort
                        + " gidIndex=" + config.gidIndex);
        this.handle = RustRdmaNative.open(
                config.host, config.port, config.ibDevice, config.ibPort, config.gidIndex, role,
                config.processingSpecJson);
        if (handle == 0) {
            throw new IOException("Rust RDMA JNI bridge returned a null session");
        }
    }

    @Override
    public int capacity() { return config.ringElements; }

    @Override
    public int maxItemSize() { return config.maxItemSize; }

    @Override
    public void writeInputSlot(byte[] value) throws IOException {
        if (value == null || value.length > maxItemSize()) {
            throw new IOException("Invalid RDMA input slot length");
        }
        RustRdmaNative.writeSlot(handle, value);
    }

    @Override
    public void publishInputBatch(int count) throws IOException {
        if (count <= 0 || count >= capacity()) {
            throw new IOException("Invalid RDMA batch count: " + count);
        }
        RustRdmaNative.publish(handle, count);
    }

    @Override
    public List<byte[]> receiveOutputBatch() throws IOException {
        return Arrays.asList(RustRdmaNative.receive(handle));
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RustRdmaNative.close(handle);
        }
    }

    static final class Factory implements RdmaRingBufferFactory {
        private static final long serialVersionUID = 1L;
        static final Factory INSTANCE = new Factory("pre");
        static final Factory PRE = new Factory("pre");
        static final Factory POST = new Factory("post");
        private final String role;

        private Factory(String role) { this.role = role; }

        @Override
        public RdmaRingBuffer open(RdmaEndpoint endpoint) throws IOException {
            return new RustRdmaRingBuffer(endpoint, role);
        }
    }
}
