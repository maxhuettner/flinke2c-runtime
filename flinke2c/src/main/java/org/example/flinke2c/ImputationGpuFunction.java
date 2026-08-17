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
import org.apache.flink.table.types.logical.TimestampType;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Batched GPU KNN price imputation for the GpuRuntimeFunction API.
 *
 * <p>The default input layout matches {@link ImputationFunctionGpu}:
 * price DECIMAL(23,3) or BIGINT, auction BIGINT, bidder BIGINT, channel STRING,
 * url STRING, dateTime TIMESTAMP(3), and extra STRING. The auction field is
 * copied through unchanged; the CUDA imputation kernel uses bidder, timestamp,
 * and the three string hashes as its similarity features.
 *
 * <p>Only rows whose price is null are replaced. Real prices remain the exact
 * input value in the output row, while only real prices are added to the
 * native history. A missing value with no available neighbors becomes zero in
 * the input price type, matching the scalar imputation function.
 */
public final class ImputationGpuFunction implements GpuRuntimeFunction {

    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_PIPELINE_DEPTH = 4;
    private static final int DEFAULT_THREADS_PER_BLOCK = 128;
    private static final int DEFAULT_CUDA_DEVICE = 0;

    private static final int DEFAULT_PRICE_FIELD_INDEX = 0;
    private static final int DEFAULT_BIDDER_FIELD_INDEX = 2;
    private static final int DEFAULT_TIMESTAMP_FIELD_INDEX = 5;
    private static final int DEFAULT_CHANNEL_FIELD_INDEX = 3;
    private static final int DEFAULT_URL_FIELD_INDEX = 4;
    private static final int DEFAULT_EXTRA_FIELD_INDEX = 6;

    private static final int MAX_PIPELINE_DEPTH = 64;
    private static final int MAX_THREADS_PER_BLOCK = 1024;
    private static final int MAX_COMPACT_DECIMAL_PRECISION = 18;

    // Keep these offsets in sync with direct_imputation_jni.cu.
    private static final int INPUT_STRIDE = 48;
    private static final int PRICE_OFFSET = 0;
    private static final int BIDDER_OFFSET = 8;
    private static final int TIMESTAMP_OFFSET = 16;
    private static final int CHANNEL_HASH_OFFSET = 24;
    private static final int URL_HASH_OFFSET = 28;
    private static final int EXTRA_HASH_OFFSET = 32;
    private static final int HAS_PRICE_OFFSET = 40;

    private static final int OUTPUT_STRIDE = Double.BYTES;
    private static final double MAX_SAFE_FAST_PATH_MAGNITUDE = 1.0e15;

    private static final BigDecimal DEFAULT_PRICE =
            new BigDecimal("0.000");

    private transient RowType inputRowType;
    private transient RowType resultRowType;
    private transient int pipelineDepth;
    private transient int threadsPerBlock;
    private transient int cudaDevice;
    private transient int laneCapacity;

    private transient int priceFieldIndex;
    private transient int bidderFieldIndex;
    private transient int timestampFieldIndex;
    private transient int channelFieldIndex;
    private transient int urlFieldIndex;
    private transient int extraFieldIndex;

    private transient int pricePrecision;
    private transient int priceScale;
    private transient int timestampPrecision;
    private transient boolean priceIsDecimal;
    private transient boolean resultPriceIsDecimal;
    private transient int resultPriceFieldIndex;
    private transient int resultPricePrecision;
    private transient int resultPriceScale;
    private transient int[] resultFieldInputIndexes;

    private transient long nativeHandle;
    private transient ByteBuffer[] nativeInputs;
    private transient ByteBuffer[] nativeOutputs;
    private transient RowData.FieldGetter[] fieldGetters;

    @Override
    public void open(GpuRuntimeFunctionContext context) throws Exception {
        final Map<String, String> values = context.getConf();

        this.cudaDevice = intValue(
                values, "device", DEFAULT_CUDA_DEVICE);
        this.pipelineDepth = intValue(
                values, "pipelinedepth", DEFAULT_PIPELINE_DEPTH);
        this.threadsPerBlock = intValue(
                values, "threadsperblock", DEFAULT_THREADS_PER_BLOCK);

        // fieldindex is accepted as an alias for pricefield, matching the
        // currency conversion function's configuration convention.
        this.priceFieldIndex = intValue(
                values,
                "pricefield",
                intValue(values, "fieldindex", DEFAULT_PRICE_FIELD_INDEX));
        this.bidderFieldIndex = intValue(
                values, "bidderfield", DEFAULT_BIDDER_FIELD_INDEX);
        this.timestampFieldIndex = intValue(
                values, "timestampfield", DEFAULT_TIMESTAMP_FIELD_INDEX);
        this.channelFieldIndex = intValue(
                values, "channelfield", DEFAULT_CHANNEL_FIELD_INDEX);
        this.urlFieldIndex = intValue(
                values, "urlfield", DEFAULT_URL_FIELD_INDEX);
        this.extraFieldIndex = intValue(
                values, "extrafield", DEFAULT_EXTRA_FIELD_INDEX);

        if (cudaDevice < 0) {
            throw new IllegalArgumentException(
                    "device must be non-negative");
        }
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
        this.resultRowType = context.getResultRowType();
        if (resultRowType == null) {
            resultRowType = inputRowType;
        }
        final int fieldCount = inputRowType.getFieldCount();

        validateIndex("pricefield", priceFieldIndex, fieldCount);
        validateIndex("bidderfield", bidderFieldIndex, fieldCount);
        validateIndex("timestampfield", timestampFieldIndex, fieldCount);
        validateIndex("channelfield", channelFieldIndex, fieldCount);
        validateIndex("urlfield", urlFieldIndex, fieldCount);
        validateIndex("extrafield", extraFieldIndex, fieldCount);

        final LogicalType priceType =
                inputRowType.getTypeAt(priceFieldIndex);
        if (priceType instanceof DecimalType) {
            this.priceIsDecimal = true;
            final DecimalType decimalType = (DecimalType) priceType;
            this.pricePrecision = decimalType.getPrecision();
            this.priceScale = decimalType.getScale();
        } else if (priceType.getTypeRoot() == LogicalTypeRoot.BIGINT) {
            this.priceIsDecimal = false;
            this.pricePrecision = 19;
            this.priceScale = 0;
        } else {
            throw new IllegalArgumentException(
                    "pricefield must reference DECIMAL or BIGINT, got "
                            + priceType);
        }

        this.resultFieldInputIndexes = new int[resultRowType.getFieldCount()];
        this.resultPriceFieldIndex = -1;
        final java.util.List<String> inputFieldNames =
                inputRowType.getFieldNames();
        final java.util.List<String> resultFieldNames =
                resultRowType.getFieldNames();
        for (int resultField = 0;
                resultField < resultFieldNames.size();
                resultField++) {
            final String fieldName = resultFieldNames.get(resultField);
            final int inputField = inputFieldNames.indexOf(fieldName);
            if (inputField < 0) {
                throw new IllegalArgumentException(
                        "result field '" + fieldName
                                + "' is not present in the input row");
            }
            resultFieldInputIndexes[resultField] = inputField;

            if (fieldName.equals(
                    inputRowType.getFieldNames().get(priceFieldIndex))) {
                resultPriceFieldIndex = resultField;
            }
        }
        if (resultPriceFieldIndex < 0) {
            throw new IllegalArgumentException(
                    "result row must contain the price field '"
                            + inputRowType.getFieldNames().get(priceFieldIndex)
                            + "'");
        }

        final LogicalType resultPriceType =
                resultRowType.getTypeAt(resultPriceFieldIndex);
        if (resultPriceType instanceof DecimalType) {
            this.resultPriceIsDecimal = true;
            final DecimalType decimalType = (DecimalType) resultPriceType;
            this.resultPricePrecision = decimalType.getPrecision();
            this.resultPriceScale = decimalType.getScale();
        } else if (resultPriceType.getTypeRoot() == LogicalTypeRoot.BIGINT) {
            this.resultPriceIsDecimal = false;
            this.resultPricePrecision = 19;
            this.resultPriceScale = 0;
        } else {
            throw new IllegalArgumentException(
                    "result price field must be DECIMAL or BIGINT, got "
                            + resultPriceType);
        }

        requireRoot(
                "bidderfield",
                inputRowType.getTypeAt(bidderFieldIndex),
                LogicalTypeRoot.BIGINT);

        final LogicalType timestampType =
                inputRowType.getTypeAt(timestampFieldIndex);
        if (timestampType.getTypeRoot()
                        != LogicalTypeRoot.TIMESTAMP_WITHOUT_TIME_ZONE
                && timestampType.getTypeRoot()
                        != LogicalTypeRoot.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
            throw new IllegalArgumentException(
                    "timestampfield must reference TIMESTAMP, got "
                            + timestampType);
        }
        this.timestampPrecision = timestampType instanceof TimestampType
                ? ((TimestampType) timestampType).getPrecision()
                : 3;

        requireString("channelfield", inputRowType.getTypeAt(channelFieldIndex));
        requireString("urlfield", inputRowType.getTypeAt(urlFieldIndex));
        requireString("extrafield", inputRowType.getTypeAt(extraFieldIndex));

        this.fieldGetters =
                new RowData.FieldGetter[fieldCount];
        for (int field = 0; field < fieldCount; field++) {
            fieldGetters[field] = RowData.createFieldGetter(
                    inputRowType.getTypeAt(field), field);
        }

        final int batchSize = context.getBatchSize();
        if (batchSize <= 0) {
            throw new IllegalArgumentException(
                    "GPU runtime batch size must be positive");
        }
        this.laneCapacity =
                (batchSize + pipelineDepth - 1) / pipelineDepth;

        this.nativeHandle = ImputationGpuNative.create(
                cudaDevice,
                laneCapacity,
                pipelineDepth,
                threadsPerBlock);
        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA imputation context creation returned a null handle");
        }

        this.nativeInputs = new ByteBuffer[pipelineDepth];
        this.nativeOutputs = new ByteBuffer[pipelineDepth];
        for (int lane = 0; lane < pipelineDepth; lane++) {
            nativeInputs[lane] = ImputationGpuNative
                    .inputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
            nativeOutputs[lane] = ImputationGpuNative
                    .outputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
        }
    }

    @Override
    public RowData[] processBatch(RowData[] batch) throws Exception {
        if (batch.length == 0) {
            return batch;
        }

        final int lanes = Math.min(pipelineDepth, batch.length);
        final int rowsPerLane = batch.length / lanes;
        final int remainder = batch.length % lanes;

        // Balanced partitioning guarantees every native submission has a
        // positive count, including partial batches such as 5 rows / 4 lanes.
        for (int lane = 0; lane < lanes; lane++) {
            final int start = lane * rowsPerLane
                    + Math.min(lane, remainder);
            final int end = start + rowsPerLane
                    + (lane < remainder ? 1 : 0);

            final ByteBuffer input = nativeInputs[lane];
            for (int index = start; index < end; index++) {
                final RowData row = batch[index];
                final int base = (index - start) * INPUT_STRIDE;
                final boolean hasPrice = !row.isNullAt(priceFieldIndex);

                input.putDouble(
                        base + PRICE_OFFSET,
                        hasPrice ? inputPrice(row) : 0.0);
                input.putLong(
                        base + BIDDER_OFFSET,
                        row.isNullAt(bidderFieldIndex)
                                ? 0L
                                : row.getLong(bidderFieldIndex));
                input.putDouble(
                        base + TIMESTAMP_OFFSET,
                        timestampSeconds(row));
                input.putInt(
                        base + CHANNEL_HASH_OFFSET,
                        stringHash(row, channelFieldIndex, "unknown"));
                input.putInt(
                        base + URL_HASH_OFFSET,
                        stringHash(row, urlFieldIndex, ""));
                input.putInt(
                        base + EXTRA_HASH_OFFSET,
                        stringHash(row, extraFieldIndex, ""));
                input.putInt(
                        base + HAS_PRICE_OFFSET,
                        hasPrice ? 1 : 0);
            }

            ImputationGpuNative.submitBatch(
                    nativeHandle, lane, end - start);
        }

        final RowData[] results = new RowData[batch.length];

        for (int lane = 0; lane < lanes; lane++) {
            final int start = lane * rowsPerLane
                    + Math.min(lane, remainder);
            final int end = start + rowsPerLane
                    + (lane < remainder ? 1 : 0);

            ImputationGpuNative.waitBatch(nativeHandle, lane);
            final ByteBuffer output = nativeOutputs[lane];

            for (int index = start; index < end; index++) {
                final int base = (index - start) * OUTPUT_STRIDE;
                final RowData source = batch[index];
                final GenericRowData converted =
                        new GenericRowData(resultRowType.getFieldCount());
                converted.setRowKind(source.getRowKind());

                for (int field = 0;
                        field < resultFieldInputIndexes.length;
                        field++) {
                    final int inputField = resultFieldInputIndexes[field];
                    if (field == resultPriceFieldIndex) {
                        continue;
                    }
                    converted.setField(
                            field,
                            fieldGetters[inputField].getFieldOrNull(source));
                }

                // Preserve real input prices exactly. The native output is a
                // double and is used only for rows whose price was missing.
                if (source.isNullAt(priceFieldIndex)) {
                    final double imputed =
                            output.getDouble(base);
                    if (resultPriceIsDecimal) {
                        converted.setField(
                                resultPriceFieldIndex,
                                toResultDecimal(imputed));
                    } else {
                        converted.setField(
                                resultPriceFieldIndex,
                                toResultBigInt(imputed));
                    }
                } else if (resultPriceIsDecimal) {
                    converted.setField(
                            resultPriceFieldIndex,
                            inputPriceAsDecimal(source));
                } else {
                    converted.setField(
                            resultPriceFieldIndex,
                            inputPriceAsBigInt(source));
                }

                results[index] = converted;
            }
        }

        return results;
    }

    @Override
    public void close() {
        if (nativeHandle != 0L) {
            ImputationGpuNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
        nativeInputs = null;
        nativeOutputs = null;
        fieldGetters = null;
    }

    private DecimalData toResultDecimal(double value) {
        if (!Double.isFinite(value)) {
            return DecimalData.fromBigDecimal(
                    DEFAULT_PRICE.setScale(resultPriceScale, RoundingMode.HALF_UP),
                    resultPricePrecision,
                    resultPriceScale);
        }

        final BigDecimal decimal = BigDecimal.valueOf(value)
                .setScale(resultPriceScale, RoundingMode.HALF_UP);

        if (resultPricePrecision > MAX_COMPACT_DECIMAL_PRECISION
                || Math.abs(value) >= MAX_SAFE_FAST_PATH_MAGNITUDE) {
            return DecimalData.fromBigDecimal(
                    decimal, resultPricePrecision, resultPriceScale);
        }

        return DecimalData.fromUnscaledLong(
                decimal.unscaledValue().longValueExact(),
                resultPricePrecision,
                resultPriceScale);
    }

    private long toResultBigInt(double value) {
        if (!Double.isFinite(value)) {
            return 0L;
        }

        try {
            return BigDecimal.valueOf(value)
                    .setScale(0, RoundingMode.HALF_UP)
                    .longValueExact();
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException(
                    "GPU-imputed BIGINT price is outside the long range: "
                            + value,
                    error);
        }
    }

    private double inputPrice(RowData row) {
        if (priceIsDecimal) {
            return row.getDecimal(
                            priceFieldIndex,
                            pricePrecision,
                            priceScale)
                    .toBigDecimal()
                    .doubleValue();
        }
        return row.getLong(priceFieldIndex);
    }

    private DecimalData inputPriceAsDecimal(RowData row) {
        if (priceIsDecimal) {
            final BigDecimal value = row.getDecimal(
                            priceFieldIndex,
                            pricePrecision,
                            priceScale)
                    .toBigDecimal()
                    .setScale(resultPriceScale, RoundingMode.HALF_UP);
            return DecimalData.fromBigDecimal(
                    value, resultPricePrecision, resultPriceScale);
        }
        return toResultDecimal(row.getLong(priceFieldIndex));
    }

    private long inputPriceAsBigInt(RowData row) {
        if (!priceIsDecimal) {
            return row.getLong(priceFieldIndex);
        }
        return row.getDecimal(
                        priceFieldIndex,
                        pricePrecision,
                        priceScale)
                .toBigDecimal()
                .setScale(0, RoundingMode.HALF_UP)
                .longValueExact();
    }

    private double timestampSeconds(RowData row) {
        if (row.isNullAt(timestampFieldIndex)) {
            return 0.0;
        }
        return row.getTimestamp(timestampFieldIndex, timestampPrecision)
                .getMillisecond() / 1000.0;
    }

    private int stringHash(RowData row, int field, String defaultValue) {
        if (row.isNullAt(field)) {
            return hashString(defaultValue);
        }
        return hashString(row.getString(field).toString());
    }

    private static int hashString(String value) {
        if (value == null || value.trim().isEmpty()) {
            return 0;
        }
        final byte[] data = value.getBytes(StandardCharsets.UTF_8);
        int hash = 0x9747b28c;
        for (byte item : data) {
            hash ^= item;
            hash *= 0x5bd1e995;
            hash ^= hash >>> 15;
        }
        return hash;
    }

    private static void validateIndex(
            String name, int index, int fieldCount) {
        if (index < 0 || index >= fieldCount) {
            throw new IllegalArgumentException(
                    name + " must be in 0.." + (fieldCount - 1));
        }
    }

    private static void requireRoot(
            String name, LogicalType type, LogicalTypeRoot expected) {
        if (type.getTypeRoot() != expected) {
            throw new IllegalArgumentException(
                    name + " must be " + expected + ", got " + type);
        }
    }

    private static void requireString(String name, LogicalType type) {
        final LogicalTypeRoot root = type.getTypeRoot();
        if (root != LogicalTypeRoot.CHAR
                && root != LogicalTypeRoot.VARCHAR) {
            throw new IllegalArgumentException(
                    name + " must be CHAR or VARCHAR, got " + type);
        }
    }

    private static int intValue(
            Map<String, String> values,
            String key,
            int fallback) {
        final String value = values.get(key);
        return value == null || value.isEmpty()
                ? fallback
                : Integer.parseInt(value);
    }
}
