package org.example.flinke2c;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.functions.table.externalruntime.ExternalRuntimeBinaryCodec;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunctionContext;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunction;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.TimestampType;
import org.apache.flink.types.RowKind;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/** Dynamically loadable packed GPU implementation of bid-price imputation. */
public final class ImputationGpuFunction implements GpuRuntimeFunction {
    private static final long serialVersionUID = 1L;
    private static final int FIELDS = 7;
    private static final ExternalRuntimeBinaryCodec.WireType[] WIRES = {
        ExternalRuntimeBinaryCodec.WireType.DECIMAL_UNSCALED_BYTES,
        ExternalRuntimeBinaryCodec.WireType.INT64, ExternalRuntimeBinaryCodec.WireType.INT64,
        ExternalRuntimeBinaryCodec.WireType.STRING, ExternalRuntimeBinaryCodec.WireType.STRING,
        ExternalRuntimeBinaryCodec.WireType.TIMESTAMP_MILLIS, ExternalRuntimeBinaryCodec.WireType.STRING};

    private transient GpuRuntimeFunction.Emitter emitter;
    private transient ExternalRuntimeBinaryCodec codec;
    private transient ExternalRuntimeBinaryCodec decodeCodec;
    private transient long handle;
    private transient ByteBuffer[] inputs, outputs;
    private transient List<Pending> filling;
    private transient ArrayBlockingQueue<Work> workQueue;
    private transient ArrayBlockingQueue<Work> submittedQueue;
    private transient ArrayBlockingQueue<CompletedBatch> completedQueue;
    private transient Object laneMonitor;
    private transient boolean[] laneBusy;
    private transient Thread submitter;
    private transient Thread completer;
    private transient volatile Throwable workerFailure;
    private transient volatile boolean workerRunning;
    private transient volatile boolean submitterBusy;
    private transient volatile boolean completerBusy;
    private transient int[] fields;
    private transient int[] resultFieldWireIndexes;
    private transient int resultPriceField;
    private transient int passthroughResultField;
    private transient int passthroughInputIndex;
    private transient RowData.FieldGetter passthroughGetter;
    private transient RowType resultType;
    private transient int batchSize, depth, nextLane, fillCount, threadsPerBlock, cudaDevice;
    private transient long nextRowId;
    // decode() runs on the single completer thread, one row at a time, and
    // copies every field it needs out of this row into the row it hands off
    // before touching the next slot - so reusing one container here (instead
    // of allocating a fresh 7-field GenericRowData per row) is safe even
    // though the *result* rows it feeds must stay independently valid until
    // drainCompleted() later collects them.
    private transient GenericRowData wireScratch;
    // Null when perfcsv is unset (the default): every call site below is
    // guarded, so disabled logging costs nothing beyond the null checks.
    private transient PerfStats perf;
    private transient Path perfCsvPath;
    private transient long openNanos;
    private transient LongAdder totalRows;
    private transient LongAdder totalBatches;
    // Writes a growing snapshot every perfFlushMillis while the job is still
    // running, not just once at close(): a streaming query normally never
    // reaches close() on its own, and a killed (not gracefully cancelled)
    // job never reaches it at all, so waiting for close() alone can mean the
    // CSV never gets a single row.
    private transient Thread perfFlusher;
    private transient long perfFlushMillis;

    private static final class Pending {
        final long id, timestamp;
        final boolean hasTimestamp;
        final RowKind rowKind;
        final Object passthrough;
        // Values already pulled out of the live RowData on the Flink thread
        // (see ExternalRuntimeBinaryCodec#extractRowValues) - independent
        // Java objects, safe for the submitter thread to frame into the wire
        // format later without touching Flink's row abstraction at all.
        final Object[] wireValues;
        Pending(long id, boolean hasTimestamp, long timestamp, RowKind rowKind, Object passthrough,
                Object[] wireValues) {
            this.id = id; this.hasTimestamp = hasTimestamp; this.timestamp = timestamp;
            this.rowKind = rowKind; this.passthrough = passthrough; this.wireValues = wireValues;
        }
    }

    private static final class Work {
        final int lane;
        final int count;
        final List<Pending> rows;

        Work(int lane, int count, List<Pending> rows) {
            this.lane = lane;
            this.count = count;
            this.rows = rows;
        }
    }

    private static final class CompletedBatch {
        final List<CompletedRow> rows;

        CompletedBatch(List<CompletedRow> rows) {
            this.rows = rows;
        }
    }

    private static final class CompletedRow {
        final Pending metadata;
        final GenericRowData result;

        CompletedRow(Pending metadata, GenericRowData result) {
            this.metadata = metadata;
            this.result = result;
        }
    }

    @Override
    public void open(GpuRuntimeFunctionContext context, GpuRuntimeFunction.Emitter output) throws Exception {
        Map<String, String> c = context.getConf();
        batchSize = intValue(c, "batchsize", context.getBatchSize());
        depth = intValue(c, "pipelinedepth", 4);
        threadsPerBlock = intValue(c, "threadsperblock", 256);
        cudaDevice = intValue(c, "device", 0);
        // Per-stage timing CSV, disabled unless a path is given. Set via the
        // same conf string as the other keys, e.g.
        // impl=...ImputationGpuFunction;...;perfcsv=/tmp/imputation-perf.csv
        String perfCsv = c.get("perfcsv");
        perfCsvPath = perfCsv == null || perfCsv.isEmpty() ? null : Paths.get(perfCsv);
        // How often to write a running snapshot while still open, on top of
        // the one always written at close(). 0 (or perfcsv unset) disables
        // periodic flushing - only the close()-time row is written then.
        perfFlushMillis = 1000L * intValue(c, "perfflushseconds", 30);
        if (batchSize <= 0 || batchSize > 65536 || depth <= 0 || depth > 64 ||
                threadsPerBlock < 32 || threadsPerBlock > 1024 || threadsPerBlock % 32 != 0) {
            throw new IllegalArgumentException("invalid packed imputation configuration");
        }
        RowType rowType = context.getInputRowType();
        if (rowType.getFieldCount() != FIELDS && rowType.getFieldCount() != FIELDS + 1) {
            throw new IllegalArgumentException("imputation expects the seven bids fields with optional trailing latency_ts");
        }
        resultType = context.getResultRowType() == null ? rowType : context.getResultRowType();
        // Fixed NexMark bids layout. latency_ts, when present, is a trailing
        // field and is copied through unchanged; it is not part of the GPU
        // imputation wire row.
        fields = new int[] {2, 0, 1, 3, 4, 5, 6};
        LogicalType[] types = new LogicalType[FIELDS];
        for (int i = 0; i < FIELDS; i++) {
            if (fields[i] < 0 || fields[i] >= rowType.getFieldCount()) throw new IllegalArgumentException("field index out of range");
            types[i] = rowType.getTypeAt(fields[i]);
        }
        validate(types);
        LogicalType[] writeTargets = types.clone();
        writeTargets[0] = new DecimalType(23, 3);
        resultFieldWireIndexes = new int[resultType.getFieldCount()];
        resultPriceField = -1;
        passthroughResultField = -1;
        passthroughInputIndex = -1;
        List<String> inputNames = rowType.getFieldNames();
        List<String> resultNames = resultType.getFieldNames();
        for (int i = 0; i < resultNames.size(); i++) {
            int inputIndex = inputNames.indexOf(resultNames.get(i));
            if (inputIndex < 0) throw new IllegalArgumentException("result field is not present in input: " + resultNames.get(i));
            int wireIndex = -1;
            for (int w = 0; w < fields.length; w++) {
                if (fields[w] == inputIndex) {
                    wireIndex = w;
                    break;
                }
            }
            resultFieldWireIndexes[i] = wireIndex;
            if (wireIndex < 0) {
                // The only supported non-wire field is the optional trailing
                // latency_ts. Keep just its value instead of cloning the row.
                if (passthroughResultField >= 0) {
                    throw new IllegalArgumentException("only one non-imputation result field is supported");
                }
                passthroughResultField = i;
                passthroughInputIndex = inputIndex;
            }
            if (inputIndex == fields[0]) resultPriceField = i;
        }
        if (resultPriceField < 0) throw new IllegalArgumentException("result row does not contain the configured price field");
        passthroughGetter = passthroughInputIndex < 0
                ? null : RowData.createFieldGetter(rowType.getTypeAt(passthroughInputIndex), passthroughInputIndex);

        LogicalType[] readSources = types.clone();
        readSources[0] = new DecimalType(23, 3);
        LogicalType[] readTargets = new LogicalType[FIELDS];
        for (int i = 0; i < FIELDS; i++) {
            int resultIndex = resultNames.indexOf(inputNames.get(fields[i]));
            readTargets[i] = resultIndex >= 0 ? resultType.getTypeAt(resultIndex) : types[i];
        }
        LogicalTypeRoot[] roots = new LogicalTypeRoot[FIELDS];
        int[] precision = new int[FIELDS], scale = new int[FIELDS], timestampPrecision = new int[FIELDS];
        for (int i = 0; i < FIELDS; i++) {
            roots[i] = types[i].getTypeRoot();
            if (types[i] instanceof DecimalType) {
                precision[i] = ((DecimalType) types[i]).getPrecision();
                scale[i] = ((DecimalType) types[i]).getScale();
            }
            if (types[i] instanceof TimestampType) timestampPrecision[i] = ((TimestampType) types[i]).getPrecision();
        }
        codec = new ExternalRuntimeBinaryCodec(true, WIRES, writeTargets, roots, precision, scale,
                timestampPrecision, null, null, null, false);
        decodeCodec = new ExternalRuntimeBinaryCodec(true, null, null, null, null, null, null,
                WIRES, readSources, readTargets, false);
        wireScratch = new GenericRowData(FIELDS);
        emitter = output;
        handle = DirectCudaImputationNative.create(cudaDevice, batchSize, depth, threadsPerBlock,
                perfCsvPath != null);
        if (handle == 0L) throw new IllegalStateException("packed CUDA imputation context creation failed");
        inputs = new ByteBuffer[depth]; outputs = new ByteBuffer[depth];
        workQueue = new ArrayBlockingQueue<>(depth);
        submittedQueue = new ArrayBlockingQueue<>(depth);
        completedQueue = new ArrayBlockingQueue<>(depth);
        laneMonitor = new Object();
        laneBusy = new boolean[depth];
        for (int i = 0; i < depth; i++) {
            inputs[i] = DirectCudaImputationNative.inputBuffer(handle, i).order(ByteOrder.nativeOrder());
            outputs[i] = DirectCudaImputationNative.outputBuffer(handle, i).order(ByteOrder.nativeOrder());
        }
        filling = new ArrayList<>(batchSize); nextLane = 0; fillCount = 0; nextRowId = 0;
        workerFailure = null;
        workerRunning = true;
        submitterBusy = false;
        completerBusy = false;
        perf = perfCsvPath == null ? null
                : new PerfStats("extract", "frame", "submit", "wait",
                        "gpuH2D", "gpuPrepare", "gpuProcess", "gpuCommit", "gpuD2H",
                        "decode", "collect");
        totalRows = new LongAdder();
        totalBatches = new LongAdder();
        openNanos = System.nanoTime();
        submitter = new Thread(this::runSubmitter, "gpu-imputation-submit");
        completer = new Thread(this::runCompleter, "gpu-imputation-completion");
        submitter.setDaemon(true);
        completer.setDaemon(true);
        submitter.start();
        completer.start();
        if (perf != null && perfFlushMillis > 0L) {
            perfFlusher = new Thread(this::runPerfFlusher, "gpu-imputation-perf-flush");
            perfFlusher.setDaemon(true);
            perfFlusher.start();
        }
    }

    @Override
    public void processElement(RowData row, boolean hasTimestamp, long timestamp) throws Exception {
        drainCompleted();
        checkWorkerFailure();
        if (fillCount == 0) awaitLaneFree(nextLane);
        long id = nextRowId++;
        // Only pull values out of the row here - reading RowData is the one
        // part that can't be deferred, since pipeline.object-reuse means its
        // backing memory isn't guaranteed to outlive this call. Framing
        // those values into the wire byte layout touches no Flink state and
        // is deferred to the submitter thread (see runSubmitter), so this
        // method stays to a handful of getter calls and a list append.
        Object[] wireValues;
        if (perf == null) {
            wireValues = codec.extractRowValues(row, fields);
        } else {
            long t0 = System.nanoTime();
            wireValues = codec.extractRowValues(row, fields);
            perf.record("extract", System.nanoTime() - t0);
        }
        Object passthrough = passthroughGetter == null ? null : passthroughGetter.getFieldOrNull(row);
        filling.add(new Pending(id, hasTimestamp, timestamp, row.getRowKind(), passthrough, wireValues));
        if (perf != null) totalRows.increment();
        if (++fillCount == batchSize) submitFilling();
    }

    @Override
    public void flush() throws Exception {
        if (workQueue == null) return;
        submitFilling();
        for (;;) {
            checkWorkerFailure();
            drainCompleted();
            synchronized (laneMonitor) {
                boolean busy = false;
                for (boolean value : laneBusy) busy |= value;
                if (!busy && !submitterBusy && !completerBusy
                        && workQueue.isEmpty() && submittedQueue.isEmpty()) break;
                laneMonitor.wait(1L);
            }
        }
        drainCompleted();
        checkWorkerFailure();
    }

    private void submitFilling() throws Exception {
        if (fillCount == 0) return;
        int lane = nextLane; nextLane = (nextLane + 1) % depth;
        List<Pending> rows = filling;
        int count = fillCount;
        synchronized (laneMonitor) {
            laneBusy[lane] = true;
        }
        Work work = new Work(lane, count, rows);
        while (!workQueue.offer(work, 100L, TimeUnit.MILLISECONDS)) {
            checkWorkerFailure();
        }
        filling = new ArrayList<>(batchSize); fillCount = 0;
    }

    @Override
    public void poll() throws Exception {
        drainCompleted();
        checkWorkerFailure();
    }

    private void awaitLaneFree(int lane) throws Exception {
        synchronized (laneMonitor) {
            while (laneBusy[lane]) {
                checkWorkerFailure();
                laneMonitor.wait(1L);
            }
        }
    }

    private void runSubmitter() {
        try {
            while (workerRunning || !workQueue.isEmpty()) {
                Work work = workQueue.poll(100L, TimeUnit.MILLISECONDS);
                if (work == null) continue;
                submitterBusy = true;
                try {
                    // The Flink thread only extracted field values (see
                    // processElement); framing them into the pinned input
                    // buffer's wire layout happens here, off the Flink
                    // thread, in parallel with it already filling the next
                    // batch. Safe to write into inputs[work.lane] now: the
                    // Flink thread already confirmed this lane free (via
                    // awaitLaneFree) before it started accumulating the rows
                    // in this Work, and nothing else touches this lane's
                    // input buffer between then and here.
                    ByteBuffer input = inputs[work.lane];
                    long frameStart = perf == null ? 0L : System.nanoTime();
                    for (int i = 0; i < work.count; i++) {
                        Pending row = work.rows.get(i);
                        int base = i * DirectCudaImputationNative.SLOT_STRIDE;
                        int length = codec.writeExtractedRow(row.wireValues, row.rowKind, row.id, input,
                                base + DirectCudaImputationNative.SLOT_VALUE_OFFSET,
                                DirectCudaImputationNative.MAX_ITEM_SIZE);
                        input.putInt(base, length);
                    }
                    if (perf != null) perf.record("frame", System.nanoTime() - frameStart);
                    long submitStart = perf == null ? 0L : System.nanoTime();
                    DirectCudaImputationNative.submitBatch(handle, work.lane, work.count);
                    if (perf != null) {
                        perf.record("submit", System.nanoTime() - submitStart);
                        totalBatches.increment();
                    }
                    while (!submittedQueue.offer(work, 100L, TimeUnit.MILLISECONDS)) {
                        checkWorkerFailure();
                    }
                } finally {
                    submitterBusy = false;
                }
            }
        } catch (Throwable failure) {
            failWorker(failure);
        }
    }

    private void runCompleter() {
        try {
            while (workerRunning || !submittedQueue.isEmpty()) {
                Work work = submittedQueue.poll(100L, TimeUnit.MILLISECONDS);
                if (work == null) continue;
                completerBusy = true;
                try {
                    if (perf == null) {
                        DirectCudaImputationNative.waitBatch(handle, work.lane);
                    } else {
                        long waitStart = System.nanoTime();
                        // waitBatchTimed does the same synchronize as
                        // waitBatch, plus (since this handle was created
                        // with profiling=true) returns how long the GPU
                        // itself spent in each stage between submit and
                        // now - combines with the surrounding host-side
                        // stages in the same CSV row, so "wait" (CPU wall
                        // time blocked here) and the gpu* stages together
                        // show whether time is going into the GPU work
                        // itself or into host-side launch/scheduling gaps
                        // around it.
                        float[] gpuMillis = DirectCudaImputationNative.waitBatchTimed(handle, work.lane);
                        perf.record("wait", System.nanoTime() - waitStart);
                        if (gpuMillis != null && gpuMillis.length == 5) {
                            perf.record("gpuH2D", (long) (gpuMillis[0] * 1_000_000.0));
                            perf.record("gpuPrepare", (long) (gpuMillis[1] * 1_000_000.0));
                            perf.record("gpuProcess", (long) (gpuMillis[2] * 1_000_000.0));
                            perf.record("gpuCommit", (long) (gpuMillis[3] * 1_000_000.0));
                            perf.record("gpuD2H", (long) (gpuMillis[4] * 1_000_000.0));
                        }
                    }
                    long decodeStart = perf == null ? 0L : System.nanoTime();
                    CompletedBatch completed = decode(work);
                    if (perf != null) perf.record("decode", System.nanoTime() - decodeStart);
                    // decode() has copied all results out of the native output
                    // buffer, so the lane can be reused while completed rows
                    // wait for collection on the Flink thread.
                    synchronized (laneMonitor) {
                        laneBusy[work.lane] = false;
                        laneMonitor.notifyAll();
                    }
                    while (!completedQueue.offer(completed, 100L, TimeUnit.MILLISECONDS)) {
                        checkWorkerFailure();
                    }
                } finally {
                    completerBusy = false;
                    synchronized (laneMonitor) {
                        // Also release the lane on wait/decode failure.
                        laneBusy[work.lane] = false;
                        laneMonitor.notifyAll();
                    }
                }
            }
        } catch (Throwable failure) {
            failWorker(failure);
        }
    }

    private void failWorker(Throwable failure) {
        if (workerFailure == null) workerFailure = failure;
        workerRunning = false;
        submitterBusy = false;
        completerBusy = false;
        synchronized (laneMonitor) {
            for (int i = 0; i < laneBusy.length; i++) laneBusy[i] = false;
            laneMonitor.notifyAll();
        }
    }

    private CompletedBatch decode(Work work) throws Exception {
        ByteBuffer output = outputs[work.lane];
        List<CompletedRow> results = new ArrayList<>(work.count);
        for (int i = 0; i < work.count; i++) {
            int base = i * DirectCudaImputationNative.SLOT_STRIDE;
            int length = output.getInt(base);
            if (length < 4 || length > DirectCudaImputationNative.MAX_ITEM_SIZE) {
                throw new IOException("invalid output slot");
            }
            Pending meta = work.rows.get(i);
            ExternalRuntimeBinaryCodec.RowWithId decoded = decodeCodec.readFramedRow(
                    output, base + DirectCudaImputationNative.SLOT_VALUE_OFFSET,
                    RowKind.INSERT, wireScratch);
            if (decoded.rowId != meta.id) {
                throw new IOException(
                        "packed imputation row order violation: lane=" + work.lane
                                + ", slot=" + i
                                + ", expectedRowId=" + meta.id
                                + ", actualRowId=" + decoded.rowId
                                + ", frameLength=" + length);
            }
            GenericRowData result = new GenericRowData(resultType.getFieldCount());
            result.setRowKind(meta.rowKind);
            GenericRowData wireRow = (GenericRowData) decoded.row;
            for (int f = 0; f < resultFieldWireIndexes.length; f++) {
                int wireIndex = resultFieldWireIndexes[f];
                result.setField(f, wireIndex >= 0 ? wireRow.getField(wireIndex) : meta.passthrough);
            }
            results.add(new CompletedRow(meta, result));
        }
        return new CompletedBatch(results);
    }

    private void drainCompleted() throws Exception {
        CompletedBatch batch;
        while ((batch = completedQueue.poll()) != null) {
            long collectStart = perf == null ? 0L : System.nanoTime();
            for (CompletedRow row : batch.rows) {
                Pending meta = row.metadata;
                emitter.collect(row.result, meta.hasTimestamp, meta.timestamp);
            }
            if (perf != null) perf.record("collect", System.nanoTime() - collectStart);
        }
    }

    private void checkWorkerFailure() throws Exception {
        Throwable failure = workerFailure;
        if (failure == null) return;
        if (failure instanceof Exception) throw (Exception) failure;
        if (failure instanceof Error) throw (Error) failure;
        throw new IOException("asynchronous GPU imputation worker failed", failure);
    }

    @Override
    public void close() throws Exception {
        try {
            if (handle != 0L) flush();
        }
        finally {
            workerRunning = false;
            // Stop the periodic flusher before the final write below, so
            // the two never race on the same file (PerfStats.appendCsv is
            // synchronized as a defense in depth, but there's no reason to
            // rely on that when a clean handoff is this easy). Each join is
            // independently interruption-safe: Thread.join() throws
            // InterruptedException if *this* (closing) thread is
            // interrupted while waiting, and an uncaught one here would
            // abort the rest of this finally block, skipping the native
            // destroy() call and the final perf row entirely.
            interruptAndJoinQuietly(perfFlusher);
            interruptAndJoinQuietly(submitter);
            interruptAndJoinQuietly(completer);
            // flush() above drains every in-flight batch, so row/batch
            // counts are final by the time this runs - log before anything
            // is nulled out below.
            writePerfCsv("final");
            if (handle != 0L) DirectCudaImputationNative.destroy(handle);
            handle = 0L; inputs = null; outputs = null; workQueue = null;
            submittedQueue = null; completedQueue = null; laneMonitor = null; laneBusy = null;
            submitter = null; completer = null; perfFlusher = null; filling = null;
            codec = null; decodeCodec = null; wireScratch = null; emitter = null;
            submitterBusy = false; completerBusy = false;
            perf = null; totalRows = null; totalBatches = null;
        }
    }

    private static void interruptAndJoinQuietly(Thread thread) {
        if (thread == null) return;
        thread.interrupt();
        try {
            thread.join(5000L);
        } catch (InterruptedException e) {
            // Restore this (closing) thread's interrupted status for
            // whatever called close() to observe, but don't let it cut the
            // rest of close()'s cleanup short - the native handle still
            // needs destroying and the final perf row still needs writing
            // either way.
            Thread.currentThread().interrupt();
        }
    }

    /** Runs on its own daemon thread while perf logging is enabled, writing a growing
     * snapshot every perfFlushMillis so a still-running (or uncleanly killed) job still
     * leaves something in the CSV instead of only ever writing at close(). */
    private void runPerfFlusher() {
        try {
            while (workerRunning) {
                Thread.sleep(perfFlushMillis);
                if (workerRunning) writePerfCsv("periodic");
            }
        } catch (InterruptedException expected) {
            // close() is stopping this thread; the authoritative final row
            // is written there, after the pipeline has fully drained.
        }
    }

    private void writePerfCsv(String label) {
        // Captured once up front: close() can null the perf* fields out
        // from under this method if perfFlusher.join(5000L) ever times out
        // (it interrupts and joins the flusher before doing exactly that),
        // and re-reading a field after a null check is a real TOCTOU, not
        // just a style nit, once two threads are both allowed to touch it.
        PerfStats snapshot = perf;
        LongAdder rows = totalRows;
        LongAdder batches = totalBatches;
        Path csvPath = perfCsvPath;
        if (snapshot == null || rows == null || batches == null || csvPath == null) return;
        long wallMillis = (System.nanoTime() - openNanos) / 1_000_000L;
        Map<String, String> extra = PerfStats.columns();
        extra.put("rows", Long.toString(rows.sum()));
        extra.put("batches", Long.toString(batches.sum()));
        extra.put("batchSize", Integer.toString(batchSize));
        extra.put("pipelineDepth", Integer.toString(depth));
        extra.put("threadsPerBlock", Integer.toString(threadsPerBlock));
        extra.put("device", Integer.toString(cudaDevice));
        snapshot.appendCsv(csvPath, "ImputationGpuFunction-" + label, wallMillis, extra);
    }

    private static void validate(LogicalType[] t) {
        LogicalTypeRoot p = t[0].getTypeRoot();
        if (p != LogicalTypeRoot.DECIMAL && p != LogicalTypeRoot.BIGINT && p != LogicalTypeRoot.INTEGER && p != LogicalTypeRoot.SMALLINT && p != LogicalTypeRoot.TINYINT) throw new IllegalArgumentException("price must be DECIMAL or integral");
        if (t[1].getTypeRoot() != LogicalTypeRoot.BIGINT || t[2].getTypeRoot() != LogicalTypeRoot.BIGINT) throw new IllegalArgumentException("auction and bidder must be BIGINT");
        if (!ExternalRuntimeBinaryCodec.isStringRoot(t[3].getTypeRoot()) || !ExternalRuntimeBinaryCodec.isStringRoot(t[4].getTypeRoot()) || !ExternalRuntimeBinaryCodec.isStringRoot(t[6].getTypeRoot())) throw new IllegalArgumentException("channel, url and extra must be strings");
        if (!ExternalRuntimeBinaryCodec.isTimestampRoot(t[5].getTypeRoot())) throw new IllegalArgumentException("timestamp must be TIMESTAMP");
    }

    private static int intValue(Map<String, String> c, String key, int fallback) {
        String v = c.get(key); return v == null || v.isEmpty() ? fallback : Integer.parseInt(v);
    }
}
