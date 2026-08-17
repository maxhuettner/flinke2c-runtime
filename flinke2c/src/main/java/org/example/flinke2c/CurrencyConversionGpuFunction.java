package org.example.flinke2c;

import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunction;
import org.apache.flink.table.runtime.functions.table.gpuruntime.GpuRuntimeFunctionContext;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

public final class CurrencyConversionGpuFunction
        implements GpuRuntimeFunction {

    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_PIPELINE_DEPTH = 4;
    private static final int DEFAULT_THREADS_PER_BLOCK = 256;
    private static final int DEFAULT_CUDA_DEVICE = 0;
    private static final int DEFAULT_FIELD_INDEX = 2;

    private static final int MAX_PIPELINE_DEPTH = 64;
    private static final int MAX_THREADS_PER_BLOCK = 1024;
    private static final int MAX_COMPACT_DECIMAL_PRECISION = 18;

    private static final int INPUT_STRIDE = 16;
    private static final int INPUT_PRICE_OFFSET = 0;
    private static final int INPUT_VALID_OFFSET = 8;

    private static final int OUTPUT_STRIDE = 16;
    private static final int OUTPUT_PRICE_OFFSET = 0;
    private static final int OUTPUT_VALID_OFFSET = 8;

    private static final double MAX_SAFE_FAST_PATH_MAGNITUDE = 1.0e15;

    private transient RowType inputRowType;
    private transient int cudaDevice;
    private transient int pipelineDepth;
    private transient int threadsPerBlock;
    private transient int priceFieldIndex;
    private transient int pricePrecision;
    private transient int priceScale;
    private transient int laneCapacity;
    private transient boolean priceIsDecimal;

    private transient long nativeHandle;
    private transient ByteBuffer[] nativeInputs;
    private transient ByteBuffer[] nativeOutputs;
    private transient RowData.FieldGetter[] fieldGetters;

    @Override
    public void open(GpuRuntimeFunctionContext context) throws Exception {
        final Map<String, String> values = context.getConf();

        this.cudaDevice =
                intValue(values, "device", DEFAULT_CUDA_DEVICE);

        this.pipelineDepth =
                intValue(values, "pipelinedepth", DEFAULT_PIPELINE_DEPTH);

        this.threadsPerBlock =
                intValue(values, "threadsperblock", DEFAULT_THREADS_PER_BLOCK);

        this.priceFieldIndex =
                intValue(values, "fieldindex", DEFAULT_FIELD_INDEX);

        if (pipelineDepth <= 0 || pipelineDepth > MAX_PIPELINE_DEPTH) {
            throw new IllegalArgumentException(
                    "pipelinedepth must be in 1.." + MAX_PIPELINE_DEPTH);
        }

        if (threadsPerBlock <= 0
                || threadsPerBlock > MAX_THREADS_PER_BLOCK) {
            throw new IllegalArgumentException(
                    "threadsperblock must be in 1.."
                            + MAX_THREADS_PER_BLOCK);
        }

        this.inputRowType = context.getInputRowType();

        if (priceFieldIndex < 0
                || priceFieldIndex >= inputRowType.getFieldCount()) {
            throw new IllegalArgumentException(
                    "Invalid fieldindex "
                            + priceFieldIndex
                            + " for input row "
                            + inputRowType.asSummaryString());
        }

        final LogicalType priceType =
                inputRowType.getTypeAt(priceFieldIndex);

        if (priceType instanceof DecimalType) {
            this.priceIsDecimal = true;

            final DecimalType decimalType =
                    (DecimalType) priceType;

            this.pricePrecision = decimalType.getPrecision();
            this.priceScale = decimalType.getScale();
        } else if (priceType.getTypeRoot()
                == LogicalTypeRoot.BIGINT) {
            this.priceIsDecimal = false;

            // BIGINT input is returned as DECIMAL(23,3).
            this.pricePrecision = 23;
            this.priceScale = 3;
        } else {
            throw new IllegalArgumentException(
                    "fieldindex must name a DECIMAL or BIGINT column, got "
                            + priceType);
        }

        this.fieldGetters =
                new RowData.FieldGetter[inputRowType.getFieldCount()];

        for (int i = 0; i < fieldGetters.length; i++) {
            fieldGetters[i] =
                    RowData.createFieldGetter(
                            inputRowType.getTypeAt(i),
                            i);
        }

        final int batchSize = context.getBatchSize();

        if (batchSize <= 0) {
            throw new IllegalArgumentException(
                    "GPU runtime batch size must be positive");
        }

        this.laneCapacity =
                (batchSize + pipelineDepth - 1)
                        / pipelineDepth;

        this.nativeHandle =
                DirectCudaCurrencyNative.create(
                        cudaDevice,
                        laneCapacity,
                        pipelineDepth,
                        threadsPerBlock);

        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA context creation returned a null handle");
        }

        this.nativeInputs =
                new ByteBuffer[pipelineDepth];

        this.nativeOutputs =
                new ByteBuffer[pipelineDepth];

        for (int lane = 0; lane < pipelineDepth; lane++) {
            nativeInputs[lane] =
                    DirectCudaCurrencyNative
                            .inputBuffer(nativeHandle, lane)
                            .order(ByteOrder.nativeOrder());

            nativeOutputs[lane] =
                    DirectCudaCurrencyNative
                            .outputBuffer(nativeHandle, lane)
                            .order(ByteOrder.nativeOrder());
        }
    }

    @Override
    public RowData[] processBatch(RowData[] batch)
            throws Exception {

        if (batch.length == 0) {
            return batch;
        }

        final int lanes =
                Math.min(pipelineDepth, batch.length);

        /*
         * Balanced partitioning:
         *
         * Example: 5 rows, 4 lanes => 2, 1, 1, 1
         *
         * This guarantees that every lane has a positive row count.
         */
        final int rowsPerLane =
                batch.length / lanes;

        final int remainder =
                batch.length % lanes;

        for (int lane = 0; lane < lanes; lane++) {
            final int start =
                    lane * rowsPerLane
                            + Math.min(lane, remainder);

            final int end =
                    start
                            + rowsPerLane
                            + (lane < remainder ? 1 : 0);

            final ByteBuffer input =
                    nativeInputs[lane];

            for (int i = start; i < end; i++) {
                final RowData row = batch[i];

                final int base =
                        (i - start) * INPUT_STRIDE;

                if (row.isNullAt(priceFieldIndex)) {
                    input.putDouble(
                            base + INPUT_PRICE_OFFSET,
                            0.0);

                    input.putInt(
                            base + INPUT_VALID_OFFSET,
                            0);
                } else {
                    final double price;

                    if (priceIsDecimal) {
                        price =
                                row.getDecimal(
                                                priceFieldIndex,
                                                pricePrecision,
                                                priceScale)
                                        .toBigDecimal()
                                        .doubleValue();
                    } else {
                        price =
                                row.getLong(priceFieldIndex);
                    }

                    input.putDouble(
                            base + INPUT_PRICE_OFFSET,
                            price);

                    input.putInt(
                            base + INPUT_VALID_OFFSET,
                            1);
                }
            }

            DirectCudaCurrencyNative.submitBatch(
                    nativeHandle,
                    lane,
                    end - start);
        }

        final RowData[] results =
                new RowData[batch.length];

        for (int lane = 0; lane < lanes; lane++) {
            final int start =
                    lane * rowsPerLane
                            + Math.min(lane, remainder);

            final int end =
                    start
                            + rowsPerLane
                            + (lane < remainder ? 1 : 0);

            DirectCudaCurrencyNative.waitBatch(
                    nativeHandle,
                    lane);

            final ByteBuffer nativeOutput =
                    nativeOutputs[lane];

            for (int i = start; i < end; i++) {
                final int base =
                        (i - start) * OUTPUT_STRIDE;

                final RowData source =
                        batch[i];

                final GenericRowData converted =
                        new GenericRowData(source.getArity());

                converted.setRowKind(
                        source.getRowKind());

                for (int field = 0;
                        field < fieldGetters.length;
                        field++) {
                    converted.setField(
                            field,
                            fieldGetters[field]
                                    .getFieldOrNull(source));
                }

                if (nativeOutput.getInt(
                        base + OUTPUT_VALID_OFFSET) == 0) {
                    converted.setField(
                            priceFieldIndex,
                            null);
                } else {
                    final double convertedPrice =
                            nativeOutput.getDouble(
                                    base + OUTPUT_PRICE_OFFSET);

                    converted.setField(
                            priceFieldIndex,
                            toDecimal(convertedPrice));
                }

                results[i] = converted;
            }
        }

        return results;
    }

    @Override
    public void close() {
        if (nativeHandle != 0L) {
            DirectCudaCurrencyNative.destroy(
                    nativeHandle);

            nativeHandle = 0L;
        }

        nativeInputs = null;
        nativeOutputs = null;
        fieldGetters = null;
    }

    private DecimalData toDecimal(double value) {
        final BigDecimal decimal =
                BigDecimal.valueOf(value)
                        .setScale(
                                priceScale,
                                RoundingMode.HALF_UP);

        /*
         * Flink's compact long representation supports precision <= 18.
         * DECIMAL(23,3) and other wider decimals must use BigDecimal.
         */
        if (pricePrecision > MAX_COMPACT_DECIMAL_PRECISION
                || !Double.isFinite(value)
                || Math.abs(value)
                        >= MAX_SAFE_FAST_PATH_MAGNITUDE) {
            return DecimalData.fromBigDecimal(
                    decimal,
                    pricePrecision,
                    priceScale);
        }

        final long unscaled =
                decimal.unscaledValue().longValueExact();

        return DecimalData.fromUnscaledLong(
                unscaled,
                pricePrecision,
                priceScale);
    }

    private static int intValue(
            Map<String, String> values,
            String key,
            int fallback) {

        final String value =
                values.get(key);

        return value == null || value.isEmpty()
                ? fallback
                : Integer.parseInt(value);
    }
}