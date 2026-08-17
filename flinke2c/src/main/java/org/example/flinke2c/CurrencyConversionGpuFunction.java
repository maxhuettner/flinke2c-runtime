package org.example.flinke2c;

import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunctionContext;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunction;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Dynamically loadable packed GPU implementation of currency conversion. */
public final class CurrencyConversionGpuFunction implements GpuRuntimeFunction {
    private static final long serialVersionUID = 1L;
    private static final int INPUT_STRIDE = 16;
    private static final int OUTPUT_STRIDE = 16;

    private transient GpuRuntimeFunction.Emitter emitter;
    private transient long handle;
    private transient ByteBuffer[] inputs, outputs;
    private transient List<Pending>[] pending;
    private transient ArrayDeque<Integer> laneOrder;
    private transient List<Pending> filling;
    private transient RowData.FieldGetter[] getters;
    private transient int batchSize, depth, nextLane, fillCount, priceIndex, precision, scale;
    private transient boolean priceDecimal;
    private transient RowType rowType;

    private static final class Pending {
        final GenericRowData row; final boolean hasTimestamp; final long timestamp;
        Pending(GenericRowData row, boolean hasTimestamp, long timestamp) {
            this.row = row; this.hasTimestamp = hasTimestamp; this.timestamp = timestamp;
        }
    }

    @Override
    public void open(GpuRuntimeFunctionContext context, GpuRuntimeFunction.Emitter output) throws Exception {
        Map<String, String> c = context.getConf();
        batchSize = intValue(c, "batchsize", context.getBatchSize());
        depth = intValue(c, "pipelinedepth", 4);
        int threads = intValue(c, "threadsperblock", 256);
        int device = intValue(c, "device", 0);
        if (batchSize <= 0 || batchSize > 65536 || depth <= 0 || depth > 64 || threads < 1 || threads > 1024) throw new IllegalArgumentException("invalid currency GPU configuration");
        rowType = context.getInputRowType();
        priceIndex = intValue(c, "fieldindex", intValue(c, "pricefield", 2));
        if (priceIndex < 0 || priceIndex >= rowType.getFieldCount()) throw new IllegalArgumentException("invalid currency price field");
        LogicalType priceType = rowType.getTypeAt(priceIndex);
        if (priceType instanceof DecimalType) {
            priceDecimal = true; precision = ((DecimalType) priceType).getPrecision(); scale = ((DecimalType) priceType).getScale();
        } else if (priceType.getTypeRoot() == LogicalTypeRoot.BIGINT) {
            priceDecimal = false; precision = 19; scale = 0;
        } else throw new IllegalArgumentException("currency price must be DECIMAL or BIGINT");
        getters = new RowData.FieldGetter[rowType.getFieldCount()];
        for (int i = 0; i < getters.length; i++) getters[i] = RowData.createFieldGetter(rowType.getTypeAt(i), i);
        emitter = output;
        handle = DirectCudaCurrencyNative.create(device, batchSize, depth, threads);
        if (handle == 0L) throw new IllegalStateException("currency CUDA context creation failed");
        inputs = new ByteBuffer[depth]; outputs = new ByteBuffer[depth];
        @SuppressWarnings("unchecked") List<Pending>[] lanes = new List[depth];
        pending = lanes; laneOrder = new ArrayDeque<>(depth);
        for (int i = 0; i < depth; i++) {
            inputs[i] = DirectCudaCurrencyNative.inputBuffer(handle, i).order(ByteOrder.nativeOrder());
            outputs[i] = DirectCudaCurrencyNative.outputBuffer(handle, i).order(ByteOrder.nativeOrder());
        }
        filling = new ArrayList<>(batchSize); nextLane = 0; fillCount = 0;
    }

    @Override
    public void processElement(RowData row, boolean hasTimestamp, long timestamp) throws Exception {
        if (fillCount == 0) completeLane(nextLane);
        ByteBuffer input = inputs[nextLane];
        int base = fillCount * INPUT_STRIDE;
        if (row.isNullAt(priceIndex)) {
            input.putDouble(base, 0.0); input.putInt(base + 8, 0);
        } else {
            double value = priceDecimal ? row.getDecimal(priceIndex, precision, scale).toBigDecimal().doubleValue() : row.getLong(priceIndex);
            input.putDouble(base, value); input.putInt(base + 8, 1);
        }
        GenericRowData snapshot = new GenericRowData(row.getArity());
        snapshot.setRowKind(row.getRowKind());
        for (int f = 0; f < getters.length; f++) snapshot.setField(f, getters[f].getFieldOrNull(row));
        filling.add(new Pending(snapshot, hasTimestamp, timestamp));
        if (++fillCount == batchSize) submitFilling();
    }

    @Override
    public void flush() throws Exception {
        submitFilling();
        while (!laneOrder.isEmpty()) completeLane(laneOrder.peekFirst());
    }

    private void submitFilling() throws Exception {
        if (fillCount == 0) return;
        int lane = nextLane; nextLane = (nextLane + 1) % depth;
        completeLane(lane);
        DirectCudaCurrencyNative.submitBatch(handle, lane, fillCount);
        pending[lane] = filling; laneOrder.addLast(lane);
        filling = new ArrayList<>(batchSize); fillCount = 0;
    }

    private void completeLane(int lane) throws Exception {
        List<Pending> rows = pending[lane];
        if (rows == null) return;
        if (laneOrder.isEmpty() || laneOrder.peekFirst() != lane) throw new IllegalStateException("currency lane order violation");
        laneOrder.removeFirst(); pending[lane] = null;
        DirectCudaCurrencyNative.waitBatch(handle, lane);
        ByteBuffer output = outputs[lane];
        for (int i = 0; i < rows.size(); i++) {
            Pending meta = rows.get(i); GenericRowData source = meta.row;
            GenericRowData result = new GenericRowData(source.getArity()); result.setRowKind(source.getRowKind());
            for (int f = 0; f < getters.length; f++) result.setField(f, source.getField(f));
            int base = i * OUTPUT_STRIDE;
            if (output.getInt(base + 8) == 0) result.setField(priceIndex, null);
            else {
                double converted = output.getDouble(base);
                result.setField(priceIndex, priceDecimal ? toDecimal(converted) : Math.round(converted));
            }
            emitter.collect(result, meta.hasTimestamp, meta.timestamp);
        }
    }

    private DecimalData toDecimal(double value) {
        BigDecimal decimal = BigDecimal.valueOf(value).setScale(scale, RoundingMode.HALF_UP);
        return DecimalData.fromBigDecimal(decimal, precision, scale);
    }

    @Override
    public void close() throws Exception {
        try { if (handle != 0L) flush(); }
        finally {
            if (handle != 0L) DirectCudaCurrencyNative.destroy(handle);
            handle = 0L; inputs = null; outputs = null; pending = null; laneOrder = null; getters = null; emitter = null;
        }
    }

    private static int intValue(Map<String, String> c, String key, int fallback) {
        String v = c.get(key); return v == null || v.isEmpty() ? fallback : Integer.parseInt(v);
    }
}
