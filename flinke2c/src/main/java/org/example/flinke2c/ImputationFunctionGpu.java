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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Direct, batched CUDA/JNI version of {@link ImputationFunction}.
 *
 * <p>Flink invokes this as an asynchronous scalar function. Calls are queued in
 * input order and submitted to CUDA in batches.
 *
 * <p>The native context keeps {@code pipelineDepth} independent CUDA streams
 * ("lanes"), each with its own pinned host and device buffers, round-robined
 * the same way {@link CurrencyConversionFunctionGpu} pipelines its batches.
 * Unlike currency conversion, the KNN history is one piece of state shared by
 * every lane, so batches cannot complete in arbitrary order: the native side
 * makes each batch's kernels wait on a GPU event for the previous batch's
 * history commit (via {@code cudaStreamWaitEvent}), so history updates apply
 * in submission order even though lanes overlap. That wait is asynchronous
 * from the CPU's perspective, so this class still only blocks a lane when
 * reusing it or draining the queue, same as the currency conversion path.
 *
 * <p>The history is local to this UDF instance and is not checkpointed.
 *
 * <p>Tuning values (batch size, batch delay, pipeline depth, threads per
 * block, CUDA device) can be set per job from SQL instead of via JVM {@code
 * -D} flags, the same way {@code RdmaOperator} takes its {@code conf} string
 * from {@code table.exec.external-runtime.conf.<class>} — except here it
 * comes through Flink's job-parameter mechanism, since this class is a plain
 * registered UDF rather than a transparent external-runtime swap-in:
 *
 * <pre>{@code
 * SET 'pipeline.global-job-parameters' =
 *     'flinke2c.imputation.gpu.conf:batchsize=1024;pipelinedepth=8;threadsperblock=128';
 * }</pre>
 *
 * <p>See the README for the full key list and precedence order (constructor
 * argument, then this job parameter, then the JVM system property, then the
 * built-in default).
 */
public class ImputationFunctionGpu extends AsyncScalarFunction {
    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_BATCH_SIZE = 64;
    private static final long DEFAULT_MAX_BATCH_DELAY_MICROS = 1_000L;
    private static final int DEFAULT_PIPELINE_DEPTH = 4;
    private static final int MAX_PIPELINE_DEPTH = 64;
    // Imputation uses substantially more registers per tuple than currency
    // conversion, so its default block is smaller while still filling every
    // warp.
    private static final int DEFAULT_THREADS_PER_BLOCK = 128;
    // The CUDA hardware limit on every compute capability this library
    // targets (compute_80+), not a tuning recommendation.
    private static final int MAX_THREADS_PER_BLOCK = 1024;

    // Job-parameter key holding a semicolon-delimited "key=value;..." string,
    // parsed the same way RdmaOperator.RdmaConfig parses its conf string. Set
    // via SQL with SET 'pipeline.global-job-parameters' =
    // 'flinke2c.imputation.gpu.conf:batchsize=1024;pipelinedepth=8;...'. Flink
    // only forwards pipeline.global-job-parameters to
    // FunctionContext.getJobParameter; arbitrary SET keys are not visible to
    // UDFs.
    private static final String CONF_JOB_PARAMETER = "flinke2c.imputation.gpu.conf";

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
    private final int requestedPipelineDepth;
    private final int requestedThreadsPerBlock;

    private transient Object queueLock;
    private transient ArrayDeque<PendingCall> pendingCalls;
    // Dispatch: plain FIFO queue, used for every batch (the common case).
    private transient ExecutorService batchExecutor;
    // Timer: heap-backed scheduled queue, but only ever holds the single
    // pending partial-flush task, so its O(log n) cost is trivial. Kept
    // separate from batchExecutor so frequent dispatch doesn't pay for a
    // priority queue it doesn't need; see the README's profiling notes.
    private transient ScheduledExecutorService timerExecutor;
    private transient long nativeHandle;
    private transient ByteBuffer[] nativeInputs;
    private transient ByteBuffer[] nativeOutputs;
    private transient List<PendingCall>[] lanePending;
    private transient int nextLane;
    private transient int activeBatchSize;
    private transient long activeBatchDelayMicros;
    private transient int activePipelineDepth;
    private transient int activeThreadsPerBlock;
    private transient boolean drainScheduled;
    private transient ScheduledFuture<?> partialFlush;
    private transient boolean closed;
    private transient Throwable terminalFailure;

    /**
     * Uses CUDA device zero, batches of 64, a 1 ms partial-batch timeout, 4
     * pipeline lanes, and 128 threads per block. System properties can
     * override all five values on the TaskManager.
     */
    public ImputationFunctionGpu() {
        this(-1, -1, -1L, -1, -1);
    }

    public ImputationFunctionGpu(int cudaDevice, int batchSize, long maxBatchDelayMicros) {
        this(cudaDevice, batchSize, maxBatchDelayMicros, -1, -1);
    }

    public ImputationFunctionGpu(
            int cudaDevice, int batchSize, long maxBatchDelayMicros, int pipelineDepth) {
        this(cudaDevice, batchSize, maxBatchDelayMicros, pipelineDepth, -1);
    }

    /**
     * @param pipelineDepth number of independent CUDA streams (and buffer
     *     sets) batches are round-robined across, or -1 to use
     *     {@code flinke2c.imputation.gpu.pipeline-depth} (default 4). The
     *     native side still applies history updates in submission order
     *     regardless of this value; see the README for sweep guidance.
     * @param threadsPerBlock CUDA block size for the batch kernel, or -1 to
     *     use {@code flinke2c.imputation.gpu.threads-per-block} (default
     *     128). Total threads launched per batch is always {@code batchSize}
     *     (one thread per row); this only changes how those threads are
     *     grouped into blocks. Must be in 1..1024, the CUDA hardware limit.
     */
    public ImputationFunctionGpu(
            int cudaDevice, int batchSize, long maxBatchDelayMicros, int pipelineDepth,
            int threadsPerBlock) {
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
        if (pipelineDepth == 0 || pipelineDepth < -1) {
            throw new IllegalArgumentException("pipelineDepth must be -1 or positive");
        }
        if (threadsPerBlock == 0 || threadsPerBlock < -1) {
            throw new IllegalArgumentException("threadsPerBlock must be -1 or positive");
        }
        this.requestedCudaDevice = cudaDevice;
        this.requestedBatchSize = batchSize;
        this.maxBatchDelayMicros = maxBatchDelayMicros;
        this.requestedPipelineDepth = pipelineDepth;
        this.requestedThreadsPerBlock = threadsPerBlock;
    }

    @Override
    public boolean isDeterministic() {
        return false;
    }

    @Override
    public void open(FunctionContext context) {
        final Map<String, String> conf = parseConf(
                context.getJobParameter(CONF_JOB_PARAMETER, ""));
        final int device = requestedCudaDevice >= 0
                ? requestedCudaDevice
                : intValue(conf, "device",
                        Integer.getInteger("flinke2c.imputation.gpu.device", 0));
        this.activeBatchSize = resolveBatchSize(conf);
        this.activeBatchDelayMicros = resolveBatchDelayMicros(conf);
        this.activePipelineDepth = resolvePipelineDepth(conf);
        this.activeThreadsPerBlock = resolveThreadsPerBlock(conf);

        this.queueLock = new Object();
        this.pendingCalls = new ArrayDeque<>(activeBatchSize);
        this.nextLane = 0;
        this.drainScheduled = false;
        this.partialFlush = null;
        this.closed = false;
        this.terminalFailure = null;
        this.nativeHandle = ImputationGpuNative.create(
                device, activeBatchSize, activePipelineDepth, activeThreadsPerBlock);
        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA imputation context creation returned a null handle");
        }
        this.nativeInputs = new ByteBuffer[activePipelineDepth];
        this.nativeOutputs = new ByteBuffer[activePipelineDepth];
        @SuppressWarnings("unchecked")
        final List<PendingCall>[] lanes = new List[activePipelineDepth];
        this.lanePending = lanes;
        for (int lane = 0; lane < activePipelineDepth; lane++) {
            nativeInputs[lane] = ImputationGpuNative.inputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
            nativeOutputs[lane] = ImputationGpuNative.outputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
        }

        this.batchExecutor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "flinke2c-gpu-imputation");
            thread.setDaemon(true);
            return thread;
        });
        final ScheduledThreadPoolExecutor timer = new ScheduledThreadPoolExecutor(
                1,
                runnable -> {
                    Thread thread = new Thread(runnable, "flinke2c-gpu-imputation-timer");
                    thread.setDaemon(true);
                    return thread;
                });
        timer.setRemoveOnCancelPolicy(true);
        this.timerExecutor = timer;
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
        final ExecutorService executor = batchExecutor;
        final ScheduledExecutorService timer = timerExecutor;
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
        // No new timer can be scheduled past this point (eval() rejects new
        // calls once closed, and the final drainBatches(true) below never
        // re-arms the partial-flush timer), so the timer thread has nothing
        // left to do.
        if (timer != null) {
            timer.shutdownNow();
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
        timerExecutor = null;
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
                    // No more work is queued right now: wait out every lane
                    // still in flight so their futures resolve promptly
                    // instead of waiting on the next unrelated batch to reuse
                    // that lane. Lanes launched earlier in this drain already
                    // ran concurrently (subject to the native side's history
                    // ordering), so this only serializes the waiting, not the
                    // GPU work itself.
                    completeAllLanes();
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
        if (partialFlush != null || timerExecutor.isShutdown()) {
            return;
        }
        partialFlush = timerExecutor.schedule(
                () -> {
                    synchronized (queueLock) {
                        partialFlush = null;
                    }
                    // Hand off to the dispatch thread instead of running
                    // drainBatches on the timer thread, so lane/queue state
                    // (nextLane, lanePending, ...) is still only ever
                    // touched by the one dispatch thread.
                    batchExecutor.execute(() -> drainBatches(true));
                },
                activeBatchDelayMicros,
                TimeUnit.MICROSECONDS);
    }

    private void processBatch(List<PendingCall> batch) {
        final int lane = nextLane;
        nextLane = (nextLane + 1) % activePipelineDepth;

        // Reusing a lane requires its previous batch's GPU work to be done;
        // with pipelineDepth > 1 that work has typically overlapped with the
        // other lanes submitted since, so this is often a cheap/no-op wait.
        // This is unrelated to the history ordering the native side enforces
        // between lanes via cudaStreamWaitEvent.
        completeLane(lane);

        final ByteBuffer input = nativeInputs[lane];
        for (int i = 0; i < batch.size(); i++) {
            final PendingCall call = batch.get(i);
            final int base = i * INPUT_STRIDE;
            input.putDouble(base + PRICE_OFFSET, call.price);
            input.putLong(base + BIDDER_OFFSET, call.bidder);
            input.putDouble(base + TIMESTAMP_OFFSET, call.timestampSeconds);
            input.putInt(base + CHANNEL_HASH_OFFSET, call.channelHash);
            input.putInt(base + URL_HASH_OFFSET, call.urlHash);
            input.putInt(base + EXTRA_HASH_OFFSET, call.extraHash);
            input.putInt(base + HAS_PRICE_OFFSET, call.hasPrice ? 1 : 0);
        }

        ImputationGpuNative.submitBatch(nativeHandle, lane, batch.size());
        lanePending[lane] = batch;
    }

    /** Waits for a lane's outstanding batch, if any, and completes its futures. */
    private void completeLane(int lane) {
        final List<PendingCall> pending = lanePending[lane];
        if (pending == null) {
            return;
        }
        lanePending[lane] = null;
        ImputationGpuNative.waitBatch(nativeHandle, lane);
        final ByteBuffer output = nativeOutputs[lane];
        for (int i = 0; i < pending.size(); i++) {
            final PendingCall call = pending.get(i);
            if (!call.hasPrice) {
                final double gpuPrice = output.getDouble(i * Double.BYTES);
                call.output.price = Double.isNaN(gpuPrice) ? DEFAULT_PRICE : toScale3(gpuPrice);
            }
            call.future.complete(call.output);
        }
    }

    private void completeAllLanes() {
        for (int lane = 0; lane < activePipelineDepth; lane++) {
            completeLane(lane);
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
        // Batches already submitted to other lanes may still be in flight
        // after a CUDA failure; fail their futures directly rather than
        // risking another native call into a possibly-broken context.
        if (lanePending != null) {
            for (int lane = 0; lane < lanePending.length; lane++) {
                final List<PendingCall> inFlight = lanePending[lane];
                lanePending[lane] = null;
                if (inFlight != null) {
                    for (PendingCall call : inFlight) {
                        call.future.completeExceptionally(error);
                    }
                }
            }
        }
    }

    private void destroyNativeContext() {
        if (nativeHandle != 0L) {
            ImputationGpuNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
        nativeInputs = null;
        nativeOutputs = null;
        lanePending = null;
    }

    // Precedence, most to least specific: explicit constructor argument, the
    // flinke2c.imputation.gpu.conf job parameter (settable per job from
    // SQL), the JVM system property (cluster-wide, set once per
    // TaskManager), then the built-in default.

    private int resolveBatchSize(Map<String, String> conf) {
        final int fromProperty = Integer.getInteger(
                "flinke2c.imputation.gpu.batch-size", DEFAULT_BATCH_SIZE);
        final int value = requestedBatchSize > 0
                ? requestedBatchSize
                : intValue(conf, "batchsize", fromProperty);
        if (value <= 0) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.batch-size must be positive");
        }
        return value;
    }

    private long resolveBatchDelayMicros(Map<String, String> conf) {
        final long fromProperty = Long.getLong(
                "flinke2c.imputation.gpu.batch-delay-micros", DEFAULT_MAX_BATCH_DELAY_MICROS);
        final long value = maxBatchDelayMicros > 0L
                ? maxBatchDelayMicros
                : longValue(conf, "batchdelaymicros", fromProperty);
        if (value <= 0L) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.batch-delay-micros must be positive");
        }
        return value;
    }

    private int resolvePipelineDepth(Map<String, String> conf) {
        final int fromProperty = Integer.getInteger(
                "flinke2c.imputation.gpu.pipeline-depth", DEFAULT_PIPELINE_DEPTH);
        final int value = requestedPipelineDepth > 0
                ? requestedPipelineDepth
                : intValue(conf, "pipelinedepth", fromProperty);
        if (value <= 0 || value > MAX_PIPELINE_DEPTH) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.pipeline-depth must be in 1.." + MAX_PIPELINE_DEPTH);
        }
        return value;
    }

    private int resolveThreadsPerBlock(Map<String, String> conf) {
        final int fromProperty = Integer.getInteger(
                "flinke2c.imputation.gpu.threads-per-block", DEFAULT_THREADS_PER_BLOCK);
        final int value = requestedThreadsPerBlock > 0
                ? requestedThreadsPerBlock
                : intValue(conf, "threadsperblock", fromProperty);
        if (value <= 0 || value > MAX_THREADS_PER_BLOCK) {
            throw new IllegalArgumentException(
                    "flinke2c.imputation.gpu.threads-per-block must be in 1.."
                            + MAX_THREADS_PER_BLOCK);
        }
        return value;
    }

    /**
     * Parses a semicolon-delimited {@code key=value;...} string the same way
     * {@code RdmaOperator.RdmaConfig} parses its {@code conf} string: keys are
     * trimmed and lower-cased, values are trimmed, malformed entries (no
     * {@code =}, or an empty key/value) are ignored rather than rejected.
     */
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

    private static int intValue(Map<String, String> conf, String key, int fallback) {
        final String value = conf.get(key);
        return value == null || value.isEmpty() ? fallback : Integer.parseInt(value);
    }

    private static long longValue(Map<String, String> conf, String key, long fallback) {
        final String value = conf.get(key);
        return value == null || value.isEmpty() ? fallback : Long.parseLong(value);
    }

    // BigDecimal.valueOf(double) is `new BigDecimal(Double.toString(val))`
    // internally: it round-trips through decimal string formatting and
    // re-parsing on every call. That's expensive enough, called once per
    // imputed row on the single dispatch thread, to matter at high
    // throughput once GPU round trips are hidden by pipelining.
    // BigDecimal.valueOf(long, int) is a cheap wrap with no parsing, so
    // compute the scale-3 unscaled value ourselves and use that overload
    // instead. HALF_UP matches RoundingMode.HALF_UP: floor(x+0.5) for
    // non-negative, ceil(x-0.5) for negative (round half away from zero).
    // Values whose unscaled magnitude could approach Long.MAX_VALUE fall
    // back to the exact, slower path instead of risking silent overflow;
    // real prices are nowhere near this threshold.
    private static final double MAX_SAFE_FAST_PATH_MAGNITUDE = 1.0e15;

    private static BigDecimal toScale3(double value) {
        if (!Double.isFinite(value) || Math.abs(value) >= MAX_SAFE_FAST_PATH_MAGNITUDE) {
            return BigDecimal.valueOf(value).setScale(3, RoundingMode.HALF_UP);
        }
        final double scaled = value * 1000.0;
        final long unscaled = value >= 0.0
                ? (long) Math.floor(scaled + 0.5)
                : (long) Math.ceil(scaled - 0.5);
        return BigDecimal.valueOf(unscaled, 3);
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

    static native long create(
            int cudaDevice, int batchCapacity, int pipelineDepth, int threadsPerBlock);

    static native ByteBuffer inputBuffer(long handle, int lane);

    static native ByteBuffer outputBuffer(long handle, int lane);

    static native void submitBatch(long handle, int lane, int count);

    static native void waitBatch(long handle, int lane);

    static native void destroy(long handle);
}
