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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.Map;

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
    private transient ArrayBlockingQueue<CompletedBatch> completedQueue;
    private transient Object laneMonitor;
    private transient boolean[] laneBusy;
    private transient Thread worker;
    private transient volatile Throwable workerFailure;
    private transient volatile boolean workerRunning;
    private transient volatile boolean workerBusy;
    private transient int[] fields;
    private transient int[] resultFieldWireIndexes;
    private transient int resultPriceField;
    private transient int passthroughResultField;
    private transient int passthroughInputIndex;
    private transient RowData.FieldGetter passthroughGetter;
    private transient RowType resultType;
    private transient int batchSize, depth, nextLane, fillCount;
    private transient long nextRowId;

    private static final class Pending {
        final long id, timestamp;
        final boolean hasTimestamp;
        final RowKind rowKind;
        final Object passthrough;
        Pending(long id, boolean hasTimestamp, long timestamp, RowKind rowKind, Object passthrough) {
            this.id = id; this.hasTimestamp = hasTimestamp; this.timestamp = timestamp;
            this.rowKind = rowKind; this.passthrough = passthrough;
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
        int threads = intValue(c, "threadsperblock", 256);
        int device = intValue(c, "device", 0);
        if (batchSize <= 0 || batchSize > 65536 || depth <= 0 || depth > 64 ||
                threads < 32 || threads > 1024 || threads % 32 != 0) {
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
        emitter = output;
        handle = DirectCudaImputationNative.create(device, batchSize, depth, threads);
        if (handle == 0L) throw new IllegalStateException("packed CUDA imputation context creation failed");
        inputs = new ByteBuffer[depth]; outputs = new ByteBuffer[depth];
        workQueue = new ArrayBlockingQueue<>(depth);
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
        workerBusy = false;
        worker = new Thread(this::runWorker, "gpu-imputation-completion");
        worker.setDaemon(true);
        worker.start();
    }

    @Override
    public void processElement(RowData row, boolean hasTimestamp, long timestamp) throws Exception {
        drainCompleted();
        checkWorkerFailure();
        if (fillCount == 0) awaitLaneFree(nextLane);
        long id = nextRowId++;
        ByteBuffer input = inputs[nextLane];
        int base = fillCount * DirectCudaImputationNative.SLOT_STRIDE;
        int length = codec.encodeFramedRow(row, fields, id, input,
                base + DirectCudaImputationNative.SLOT_VALUE_OFFSET,
                DirectCudaImputationNative.MAX_ITEM_SIZE);
        input.putInt(base, length);
        Object passthrough = passthroughGetter == null ? null : passthroughGetter.getFieldOrNull(row);
        filling.add(new Pending(id, hasTimestamp, timestamp, row.getRowKind(), passthrough));
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
                if (!busy && !workerBusy && workQueue.isEmpty()) break;
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

    private void runWorker() {
        try {
            while (workerRunning || !workQueue.isEmpty()) {
                Work work = workQueue.poll(100L, TimeUnit.MILLISECONDS);
                if (work == null) continue;
                workerBusy = true;
                try {
                    DirectCudaImputationNative.submitBatch(handle, work.lane, work.count);
                    DirectCudaImputationNative.waitBatch(handle, work.lane);
                    CompletedBatch completed = decode(work);
                    // decode() has copied all results out of the native output
                    // buffer, so the lane can be reused while the completed
                    // rows wait for collection on the Flink thread.
                    synchronized (laneMonitor) {
                        laneBusy[work.lane] = false;
                        laneMonitor.notifyAll();
                    }
                    completedQueue.put(completed);
                } finally {
                    workerBusy = false;
                    synchronized (laneMonitor) {
                        // Also release the lane on submit/wait/decode failure.
                        laneBusy[work.lane] = false;
                        laneMonitor.notifyAll();
                    }
                }
            }
        } catch (Throwable failure) {
            workerFailure = failure;
            workerRunning = false;
            workerBusy = false;
            synchronized (laneMonitor) {
                for (int i = 0; i < laneBusy.length; i++) laneBusy[i] = false;
                laneMonitor.notifyAll();
            }
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
                    RowKind.INSERT, null);
            if (decoded.rowId != meta.id) {
                throw new IOException("packed imputation row order violation");
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
            for (CompletedRow row : batch.rows) {
                Pending meta = row.metadata;
                emitter.collect(row.result, meta.hasTimestamp, meta.timestamp);
            }
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
            if (worker != null) {
                worker.interrupt();
                worker.join(5000L);
            }
            if (handle != 0L) DirectCudaImputationNative.destroy(handle);
            handle = 0L; inputs = null; outputs = null; workQueue = null;
            completedQueue = null; laneMonitor = null; laneBusy = null;
            worker = null; filling = null; codec = null; decodeCodec = null; emitter = null;
            workerBusy = false;
        }
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
