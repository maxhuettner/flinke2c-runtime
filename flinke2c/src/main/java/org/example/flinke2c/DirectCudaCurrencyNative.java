package org.example.flinke2c;

import java.nio.ByteBuffer;

/** Native currency-conversion ABI owned by the dynamically loaded implementation jar. */
public final class DirectCudaCurrencyNative {
    static {
        System.loadLibrary("flinke2c_currency_conversion_gpu");
    }

    private DirectCudaCurrencyNative() {}

    public static native long create(int device, int capacity, int pipelineDepth, int threadsPerBlock);
    public static native ByteBuffer inputBuffer(long handle, int lane);
    public static native ByteBuffer outputBuffer(long handle, int lane);
    public static native void submitBatch(long handle, int lane, int count);
    public static native void waitBatch(long handle, int lane);
    public static native void destroy(long handle);
}
