package org.example.flinke2c;

import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.annotation.FunctionHint;
import org.apache.flink.table.functions.AsyncScalarFunction;
import org.apache.flink.table.functions.FunctionContext;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Batched CUDA version of {@link CurrencyConversionFunction}.
 *
 * <p>Flink invokes this as an asynchronous scalar function. Calls are queued
 * in input order and submitted to one CUDA context in batches. Null prices
 * remain null; non-null prices are multiplied by 0.908 on the GPU.
 *
 * <p>The CUDA path uses double precision for the device multiply and rounds
 * the result to the DECIMAL(23,3) output scale in Java. As with the ordinary
 * BigDecimal implementation, this function is deterministic and stateless.
 */
public class CurrencyConversionFunctionGpu extends AsyncScalarFunction {
    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_BATCH_SIZE = 64;
    private static final long DEFAULT_MAX_BATCH_DELAY_MICROS = 1_000L;

    // Keep these layouts in sync with direct_currency_conversion_jni.cu.
    private static final int INPUT_STRIDE = 16;
    private static final int INPUT_PRICE_OFFSET = 0;
    private static final int INPUT_VALID_OFFSET = 8;
    private static final int OUTPUT_STRIDE = 16;
    private static final int OUTPUT_PRICE_OFFSET = 0;
    private static final int OUTPUT_VALID_OFFSET = 8;

    private final int requestedCudaDevice;
    private final int requestedBatchSize;
    private final long maxBatchDelayMicros;

    private transient Object queueLock;
    private transient ArrayDeque<PendingCall> pendingCalls;
    private transient ScheduledExecutorService batchExecutor;
    private transient long nativeHandle;
    private transient ByteBuffer nativeInput;
    private transient ByteBuffer nativeOutput;
    private transient int activeBatchSize;
    private transient long activeBatchDelayMicros;
    private transient boolean drainScheduled;
    private transient ScheduledFuture<?> partialFlush;
    private transient boolean closed;
    private transient Throwable terminalFailure;

    /** Uses CUDA device zero, batches of 64, and a 1 ms partial-batch timeout. */
    public CurrencyConversionFunctionGpu() {
        this(-1, -1, -1L);
    }

    public CurrencyConversionFunctionGpu(
            int cudaDevice, int batchSize, long maxBatchDelayMicros) {
        if (cudaDevice < -1) {
            throw new IllegalArgumentException("cudaDevice must be -1 or non-negative");
        }
        if (batchSize == 0 || batchSize < -1) {
            throw new IllegalArgumentException("batchSize must be -1 or positive");
        }
        if (maxBatchDelayMicros == 0L || maxBatchDelayMicros < -1L) {
            throw new IllegalArgumentException(
                    "maxBatchDelayMicros must be -1 or positive");
        }
        this.requestedCudaDevice = cudaDevice;
        this.requestedBatchSize = batchSize;
        this.maxBatchDelayMicros = maxBatchDelayMicros;
    }

    @Override
    public boolean isDeterministic() {
        return true;
    }

    @Override
    public void open(FunctionContext context) {
        final int device = requestedCudaDevice >= 0
                ? requestedCudaDevice
                : Integer.getInteger("flinke2c.currency.gpu.device", 0);
        this.activeBatchSize = resolveBatchSize();
        this.activeBatchDelayMicros = resolveBatchDelayMicros();

        this.queueLock = new Object();
        this.pendingCalls = new ArrayDeque<>(activeBatchSize);
        this.drainScheduled = false;
        this.partialFlush = null;
        this.closed = false;
        this.terminalFailure = null;
        this.nativeHandle = CurrencyConversionGpuNative.create(device, activeBatchSize);
        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA currency conversion context creation returned a null handle");
        }
        this.nativeInput = CurrencyConversionGpuNative.inputBuffer(nativeHandle)
                .order(ByteOrder.nativeOrder());
        this.nativeOutput = CurrencyConversionGpuNative.outputBuffer(nativeHandle)
                .order(ByteOrder.nativeOrder());

        final ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(
                1,
                runnable -> {
                    Thread thread = new Thread(runnable, "flinke2c-gpu-currency");
                    thread.setDaemon(true);
                    return thread;
                });
        executor.setRemoveOnCancelPolicy(true);
        this.batchExecutor = executor;
    }

    @FunctionHint(
            input = {@DataTypeHint("DECIMAL(23,3)")},
            output = @DataTypeHint("DECIMAL(23,3)"))
    public void eval(CompletableFuture<BigDecimal> future, BigDecimal price) {
        if (future == null) {
            throw new IllegalArgumentException("result future must not be null");
        }

        final PendingCall call = new PendingCall(future, price);
        boolean flushImmediately = false;
        synchronized (queueLock) {
            if (terminalFailure != null) {
                future.completeExceptionally(terminalFailure);
                return;
            }
            if (closed) {
                future.completeExceptionally(
                        new IllegalStateException("GPU currency conversion UDF is closed"));
                return;
            }
            pendingCalls.addLast(call);
            if (!drainScheduled) {
                drainScheduled = true;
                schedulePartialFlushLocked();
            }
            if (pendingCalls.size() >= activeBatchSize && partialFlush != null) {
                partialFlush.cancel(false);
                partialFlush = null;
                flushImmediately = true;
            }
        }
        if (flushImmediately) {
            batchExecutor.execute(() -> drainBatches(false));
        }
    }

    @Override
    public void close() throws Exception {
        final ScheduledExecutorService executor = batchExecutor;
        if (executor == null) {
            destroyNativeContext();
            return;
        }

        synchronized (queueLock) {
            closed = true;
            if (partialFlush != null) {
                partialFlush.cancel(false);
                partialFlush = null;
            }
        }
        executor.execute(() -> drainBatches(true));
        executor.shutdown();
        if (!executor.awaitTermination(30L, TimeUnit.SECONDS)) {
            executor.shutdownNow();
            if (!executor.awaitTermination(5L, TimeUnit.SECONDS)) {
                failPending(new IllegalStateException(
                        "Timed out while closing the GPU currency conversion batch executor"));
            }
        }
        batchExecutor = null;
        pendingCalls = null;
        queueLock = null;
        destroyNativeContext();
    }

    private void drainBatches(boolean flushPartialBatch) {
        while (true) {
            final List<PendingCall> batch;
            synchronized (queueLock) {
                final int available = pendingCalls.size();
                if (available == 0) {
                    drainScheduled = false;
                    return;
                }
                if (!flushPartialBatch && available < activeBatchSize) {
                    schedulePartialFlushLocked();
                    return;
                }
                final int count = Math.min(available, activeBatchSize);
                batch = new ArrayList<>(count);
                for (int i = 0; i < count; i++) {
                    batch.add(pendingCalls.removeFirst());
                }
            }

            try {
                processBatch(batch);
            } catch (Throwable error) {
                for (PendingCall call : batch) {
                    call.future.completeExceptionally(error);
                }
                failPending(error);
                return;
            }
        }
    }

    private void schedulePartialFlushLocked() {
        if (partialFlush != null || batchExecutor.isShutdown()) {
            return;
        }
        partialFlush = batchExecutor.schedule(
                () -> {
                    synchronized (queueLock) {
                        partialFlush = null;
                    }
                    drainBatches(true);
                },
                activeBatchDelayMicros,
                TimeUnit.MICROSECONDS);
    }

    private void processBatch(List<PendingCall> batch) {
        for (int i = 0; i < batch.size(); i++) {
            final PendingCall call = batch.get(i);
            final int base = i * INPUT_STRIDE;
            nativeInput.putDouble(base + INPUT_PRICE_OFFSET, call.price);
            nativeInput.putInt(base + INPUT_VALID_OFFSET, call.hasPrice ? 1 : 0);
        }

        CurrencyConversionGpuNative.processBatch(nativeHandle, batch.size());

        for (int i = 0; i < batch.size(); i++) {
            final int base = i * OUTPUT_STRIDE;
            final PendingCall call = batch.get(i);
            if (nativeOutput.getInt(base + OUTPUT_VALID_OFFSET) == 0) {
                call.future.complete(null);
                continue;
            }
            final double converted = nativeOutput.getDouble(base + OUTPUT_PRICE_OFFSET);
            call.future.complete(
                    BigDecimal.valueOf(converted).setScale(3, RoundingMode.HALF_UP));
        }
    }

    private void failPending(Throwable error) {
        final List<PendingCall> failed = new ArrayList<>();
        synchronized (queueLock) {
            if (terminalFailure == null) {
                terminalFailure = error;
            }
            if (partialFlush != null) {
                partialFlush.cancel(false);
                partialFlush = null;
            }
            drainScheduled = false;
            while (!pendingCalls.isEmpty()) {
                failed.add(pendingCalls.removeFirst());
            }
        }
        for (PendingCall call : failed) {
            call.future.completeExceptionally(error);
        }
    }

    private void destroyNativeContext() {
        if (nativeHandle != 0L) {
            CurrencyConversionGpuNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
        nativeInput = null;
        nativeOutput = null;
    }

    private int resolveBatchSize() {
        final int value = requestedBatchSize > 0
                ? requestedBatchSize
                : Integer.getInteger(
                        "flinke2c.currency.gpu.batch-size",
                        DEFAULT_BATCH_SIZE);
        if (value <= 0) {
            throw new IllegalArgumentException(
                    "flinke2c.currency.gpu.batch-size must be positive");
        }
        return value;
    }

    private long resolveBatchDelayMicros() {
        final long value = maxBatchDelayMicros > 0L
                ? maxBatchDelayMicros
                : Long.getLong(
                        "flinke2c.currency.gpu.batch-delay-micros",
                        DEFAULT_MAX_BATCH_DELAY_MICROS);
        if (value <= 0L) {
            throw new IllegalArgumentException(
                    "flinke2c.currency.gpu.batch-delay-micros must be positive");
        }
        return value;
    }

    private static final class PendingCall {
        final CompletableFuture<BigDecimal> future;
        final boolean hasPrice;
        final double price;

        PendingCall(CompletableFuture<BigDecimal> future, BigDecimal price) {
            this.future = future;
            this.hasPrice = price != null;
            this.price = price == null ? 0.0 : price.doubleValue();
        }
    }
}

/** Package-private JNI entry point for the direct currency conversion kernel. */
final class CurrencyConversionGpuNative {
    static {
        String absoluteLibrary =
                System.getProperty("flinke2c.currency.gpu.library", "").trim();
        if (absoluteLibrary.isEmpty()) {
            System.loadLibrary("flinke2c_currency_conversion_gpu");
        } else {
            File file = new File(absoluteLibrary);
            if (!file.isAbsolute()) {
                throw new IllegalArgumentException(
                        "flinke2c.currency.gpu.library must be an absolute path");
            }
            System.load(file.getAbsolutePath());
        }
    }

    private CurrencyConversionGpuNative() {}

    static native long create(int cudaDevice, int batchCapacity);

    static native ByteBuffer inputBuffer(long handle);

    static native ByteBuffer outputBuffer(long handle);

    static native void processBatch(long handle, int count);

    static native void destroy(long handle);
}
