package org.apache.flink.table.runtime.functions.table.externalruntime;

import java.io.IOException;

/** JNI entry point for the raw-verbs client implemented by gpu_rdma_runtime. */
final class RustRdmaNative {
    static {
        System.loadLibrary("gpu_rdma_runtime");
    }

    private RustRdmaNative() {}

    static native long open(
            String host, int port, String device, int ibPort, int gidIndex, String role,
            String processingSpecJson);

    static native void writeSlot(long handle, byte[] value) throws IOException;

    /**
     * Writes a whole batch in one JNI crossing. {@code batch} holds one frame per
     * row at a fixed {@code maxItemSize} stride starting at offset 0 (as produced by
     * {@link ExternalRuntimeBinaryCodec#encodeFramedRow(org.apache.flink.table.data.RowData,
     * int[], long, java.nio.ByteBuffer, int, int)}); {@code frameLengths[i]} is the real
     * encoded length of row {@code i}'s frame, which is at most {@code maxItemSize} but
     * usually smaller. {@code batch} must be a direct buffer.
     */
    static native void writeBatch(long handle, java.nio.ByteBuffer batch, int[] frameLengths, int count)
            throws IOException;

    static native void publish(long handle, int count) throws IOException;

    static native byte[][] receive(long handle) throws IOException;

    static native void close(long handle);
}
