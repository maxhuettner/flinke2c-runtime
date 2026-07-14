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

    static native void publish(long handle, int count) throws IOException;

    static native byte[][] receive(long handle) throws IOException;

    static native void close(long handle);
}
