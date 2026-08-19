package org.example.flinke2c;

import java.nio.ByteBuffer;

/** Native Black-Scholes ABI owned by the dynamically loaded implementation jar. */
public final class DirectCudaBlackScholesNative {
    static {
        System.loadLibrary("flinke2c_black_scholes_gpu");
    }

    private DirectCudaBlackScholesNative() {}

    public static native long create(int device, int capacity, int pipelineDepth, int threadsPerBlock);
    public static native ByteBuffer inputBuffer(long handle, int lane);
    public static native ByteBuffer outputBuffer(long handle, int lane);
    public static native void submitBatch(long handle, int lane, int count);
    public static native void waitBatch(long handle, int lane);
    public static native void destroy(long handle);
}
