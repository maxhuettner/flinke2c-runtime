package org.example.flinke2c;

import java.nio.ByteBuffer;

/** Native packed-imputation ABI owned by the dynamically loaded implementation jar. */
public final class DirectCudaImputationNative {
    public static final int MAX_ITEM_SIZE = 2048;
    public static final int SLOT_STRIDE = 16 + MAX_ITEM_SIZE;
    public static final int SLOT_VALUE_OFFSET = 16;

    static {
        // The packed bridge is compiled into the existing imputation library;
        // no second implementation-specific .so is required.
        System.loadLibrary("flinke2c_imputation_gpu");
    }

    private DirectCudaImputationNative() {}

    public static native long create(int device, int capacity, int pipelineDepth, int threadsPerBlock,
            boolean profiling);
    public static native ByteBuffer inputBuffer(long handle, int lane);
    public static native ByteBuffer outputBuffer(long handle, int lane);
    public static native void submitBatch(long handle, int lane, int count);
    public static native void waitBatch(long handle, int lane);
    /**
     * Same wait as {@link #waitBatch}, but for a handle created with
     * {@code profiling=true}: also returns the elapsed milliseconds of each
     * GPU-side stage (H2D copy, the three imputation kernels, D2H copy) as
     * {@code [h2dMs, prepareMs, processMs, commitMs, d2hMs]}. An empty array
     * means either profiling wasn't enabled at {@link #create} or this lane
     * had nothing pending.
     */
    public static native float[] waitBatchTimed(long handle, int lane);
    public static native void destroy(long handle);
}
