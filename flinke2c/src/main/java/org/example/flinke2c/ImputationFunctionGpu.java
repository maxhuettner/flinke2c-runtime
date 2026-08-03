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
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Direct, batched CUDA/JNI version of {@link ImputationFunction}.
 *
 * <p>Flink invokes this as an asynchronous scalar function. Calls are queued in
 * input order and submitted to CUDA in batches. One native stream serializes
 * batches, so the GPU history has the same logical order as the Java UDF.
 *
 * <p>The history is local to this UDF instance and is not checkpointed.
 */
public class ImputationFunctionGpu extends AsyncScalarFunction {
    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_BATCH_SIZE = 64;
    private static final long DEFAULT_MAX_BATCH_DELAY_MICROS = 1_000L;

    // Native Input layout. Keep in sync with direct_imputation_jni.cu.
    private static final int INPUT_STRIDE = 48;
    private static final int PRICE_OFFSET = 0;
    private static final int BIDDER_OFFSET = 8;
    private static final int TIMESTAMP_OFFSET = 16;
    private static final int CHANNEL_HASH_OFFSET = 24;
    private static final int URL_HASH_OFFSET = 28;
    private static final int EXTRA_HASH_OFFSET = 32;
    private static final int HAS_PRICE_OFFSET = 40;

    private static final BigDecimal DEFAULT_PRICE = new BigDecimal("0.000");
    private static final long DEFAULT_LONG = 0L;
    private static final String DEFAULT_CHANNEL = "unknown";
    private static final String DEFAULT_STRING = "";
    private static final LocalDateTime DEFAULT_TIMESTAMP =
            LocalDateTime.of(1970, 1, 1, 0, 0);

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

    /**
     * Uses CUDA device zero, batches of 64, and a 1 ms partial-batch timeout.
     * System properties can override all three values on the TaskManager.
     */
    public ImputationFunctionGpu() {
        this(-1, -1, -1L);
    }

    public ImputationFunctionGpu(int cudaDevice, int batchSize, long maxBatchDelayMicros) {
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
        return false;
    }

    @Override
    public void open(FunctionContext context) {
        final int device = requestedCudaDevice >= 0
                ? requestedCudaDevice
                : Integer.getInteger("flinke2c.imputation.gpu.device", 0);
        this.activeBatchSize = resolveBatchSize();
        this.activeBatchDelayMicros = resolveBatchDelayMicros();

        this.queueLock = new Object();
        this.pendingCalls = new ArrayDeque<>(activeBatchSize);
        this.drainScheduled = false;
        this.partialFlush = null;
        this.closed = false;
        this.terminalFailure = null;
        this.nativeHandle = ImputationGpuNative.create(device, activeBatchSize);
        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA imputation context creation returned a null handle");
        }
        this.nativeInput = ImputationGpuNative.inputBuffer(nativeHandle)
                .order(ByteOrder.nativeOrder());
        this.nativeOutput = ImputationGpuNative.outputBuffer(nativeHandle)
                .order(ByteOrder.nativeOrder());

        final ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(
                1,
                runnable -> {
                    Thread thread = new Thread(runnable, "flinke2c-gpu-imputation");
                    thread.setDaemon(true);
                    return thread;
                });
        executor.setRemoveOnCancelPolicy(true);
        this.batchExecutor = executor;
    }

    @FunctionHint(
            input = {
                    @DataTypeHint("DECIMAL(23,3)"),
                    @DataTypeHint("BIGINT"),
                    @DataTypeHint("BIGINT"),
                    @DataTypeHint("STRING"),
                    @DataTypeHint("STRING"),
                    @DataTypeHint("TIMESTAMP(3)"),
                    @DataTypeHint("STRING")
            },
            output = @DataTypeHint(bridgedTo = ImputationFunction.ImputedBid.class)
    )
    public void eval(
            CompletableFuture<ImputationFunction.ImputedBid> future,
            BigDecimal price,
            Long auction,
            Long bidder,
            String channel,
            String url,
            LocalDateTime dateTime,
            String extra) {

        if (future == null) {
            throw new IllegalArgumentException("result future must not be null");
        }

        final long a = auction == null ? DEFAULT_LONG : auction;
        final long b = bidder == null ? DEFAULT_LONG : bidder;
        final String ch = isBlank(channel) ? DEFAULT_CHANNEL : channel;
        final String u = isBlank(url) ? DEFAULT_STRING : url;
        final LocalDateTime dt = dateTime == null ? DEFAULT_TIMESTAMP : dateTime;
        final String ex = isBlank(extra) ? DEFAULT_STRING : extra;
        final PendingCall call = new PendingCall(
                future,
                new ImputationFunction.ImputedBid(price, a, b, ch, u, dt, ex),
                price != null,
                price == null ? 0.0 : price.doubleValue(),
                b,
                dt.toInstant(ZoneOffset.UTC).toEpochMilli() / 1000.0,
                hashOrZero(ch),
                hashOrZero(u),
                hashOrZero(ex));

        boolean flushImmediately = false;
        synchronized (queueLock) {
            if (terminalFailure != null) {
                future.completeExceptionally(terminalFailure);
                return;
            }
            if (closed) {
                future.completeExceptionally(
                        new IllegalStateException("GPU imputation UDF is closed"));
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
                        "Timed out while closing the GPU imputation batch executor"));
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
            nativeInput.putDouble(base + PRICE_OFFSET, call.price);
            nativeInput.putLong(base + BIDDER_OFFSET, call.bidder);
            nativeInput.putDouble(base + TIMESTAMP_OFFSET, call.timestampSeconds);
            nativeInput.putInt(base + CHANNEL_HASH_OFFSET, call.channelHash);
            nativeInput.putInt(base + URL_HASH_OFFSET, call.urlHash);
            nativeInput.putInt(base + EXTRA_HASH_OFFSET, call.extraHash);
            nativeInput.putInt(base + HAS_PRICE_OFFSET, call.hasPrice ? 1 : 0);
        }

        ImputationGpuNative.processBatch(nativeHandle, batch.size());

        for (int i = 0; i < batch.size(); i++) {
            final PendingCall call = batch.get(i);
            if (!call.hasPrice) {
                final double gpuPrice = nativeOutput.getDouble(i * Double.BYTES);
                call.output.price = Double.isNaN(gpuPrice)
                        ? DEFAULT_PRICE
                        : BigDecimal.valueOf(gpuPrice)
                                .setScale(3, RoundingMode.HALF_UP);
            }
            call.future.complete(call.output);
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
            ImputationGpuNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
        nativeInput = null;
        nativeOutput = null;
    }

    private int resolveBatchSize() {
        final int value = requestedBatchSize > 0
                ? requestedBatchSize
                : Integer.getInteger(
                        "flinke2c.imputation.gpu.batch-size",
                        DEFAULT_BATCH_SIZE);
        if (value <= 0) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.batch-size must be positive");
        }
        return value;
    }

    private long resolveBatchDelayMicros() {
        final long value = maxBatchDelayMicros > 0L
                ? maxBatchDelayMicros
                : Long.getLong(
                        "flinke2c.imputation.gpu.batch-delay-micros",
                        DEFAULT_MAX_BATCH_DELAY_MICROS);
        if (value <= 0L) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.batch-delay-micros must be positive");
        }
        return value;
    }

    private static int hashOrZero(String value) {
        if (value == null || value.isBlank()) {
            return 0;
        }
        byte[] data = value.getBytes(StandardCharsets.UTF_8);
        int hash = 0x9747b28c;
        for (byte b : data) {
            hash ^= b;
            hash *= 0x5bd1e995;
            hash ^= (hash >>> 15);
        }
        return hash;
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private static final class PendingCall {
        final CompletableFuture<ImputationFunction.ImputedBid> future;
        final ImputationFunction.ImputedBid output;
        final boolean hasPrice;
        final double price;
        final long bidder;
        final double timestampSeconds;
        final int channelHash;
        final int urlHash;
        final int extraHash;

        PendingCall(
                CompletableFuture<ImputationFunction.ImputedBid> future,
                ImputationFunction.ImputedBid output,
                boolean hasPrice,
                double price,
                long bidder,
                double timestampSeconds,
                int channelHash,
                int urlHash,
                int extraHash) {
            this.future = future;
            this.output = output;
            this.hasPrice = hasPrice;
            this.price = price;
            this.bidder = bidder;
            this.timestampSeconds = timestampSeconds;
            this.channelHash = channelHash;
            this.urlHash = urlHash;
            this.extraHash = extraHash;
        }
    }
}

/** Package-private JNI entry point kept in this file with the GPU UDF. */
final class ImputationGpuNative {
    static {
        String absoluteLibrary =
                System.getProperty("flinke2c.imputation.gpu.library", "").trim();
        if (absoluteLibrary.isEmpty()) {
            System.loadLibrary("flinke2c_imputation_gpu");
        } else {
            File file = new File(absoluteLibrary);
            if (!file.isAbsolute()) {
                throw new IllegalArgumentException(
                        "flinke2c.imputation.gpu.library must be an absolute path");
            }
            System.load(file.getAbsolutePath());
        }
    }

    private ImputationGpuNative() {}

    static native long create(int cudaDevice, int batchCapacity);

    static native ByteBuffer inputBuffer(long handle);

    static native ByteBuffer outputBuffer(long handle);

    static native void processBatch(long handle, int count);

    static native void destroy(long handle);
}
