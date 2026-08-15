/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.runtime.functions.table.externalruntime;

import org.apache.flink.annotation.Internal;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.RowType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Single-operator external runtime for the direct-CUDA currency conversion
 * kernel: a batch-native replacement for {@code CurrencyConversionFunctionGpu}
 * (the {@code AsyncScalarFunction} version) that avoids everything an async
 * scalar function costs per row.
 *
 * <p><b>Why one operator, not a PRE/POST pair like {@link RdmaPreOperator}/
 * {@link RdmaPostOperator}:</b> RDMA needs two operators because PRE and
 * POST can run on different TaskManagers, connected by a network round trip.
 * This is an in-process JNI call to a local GPU — no network boundary — so
 * one operator can buffer input, dispatch, and emit output itself.
 *
 * <p><b>Why this instead of {@code AsyncScalarFunction}:</b> profiling
 * CurrencyConversionFunctionGpu under load traced a large share of CPU time
 * to machinery that is unavoidable for any {@code AsyncScalarFunction},
 * regardless of how efficiently the GPU side is batched: one
 * {@code CompletableFuture} per row, {@code AsyncWaitOperator}'s ordered
 * result queue (to preserve input order across futures that may complete
 * out of order), and a mailbox repost per row for the single-threaded task
 * runtime to pick up each completion. Batching the GPU dispatch doesn't
 * reduce any of that, because the number of futures completed is still one
 * per row. This operator has none of it: {@code processElement} is plain
 * synchronous per-record processing (Flink guarantees it is never called
 * concurrently with itself on one operator instance, so — unlike the async
 * UDF — none of the batching state below needs locking), and results are
 * emitted with a direct {@code output.collect()} call, in a loop, like any
 * ordinary operator.
 *
 * <p><b>Batching/pipelining design:</b> rows are buffered into the batch
 * currently being filled. Once it reaches {@code batchSize}, it is submitted
 * to the next of {@code pipelineDepth} CUDA streams ("lanes") — the same
 * lane design {@code CurrencyConversionFunctionGpu} uses — non-blockingly,
 * so lane N+1 can be filled and launched while lane N's H2D copy/kernel/D2H
 * copy are still running on the GPU. A lane's previously-submitted batch is
 * only waited on and collected right before that lane is reused, or when a
 * watermark/checkpoint barrier/end-of-input forces a full flush. Because
 * lanes are always collected in the same round-robin order they were
 * submitted in, and a lane is always collected before being reused, rows are
 * always emitted in the same relative order they arrived — no explicit
 * sequence check is needed the way {@link RdmaPostOperator} needs one
 * (that one crosses a network boundary; this one is single-threaded and
 * fully deterministic by construction).
 *
 * <p><b>Selection:</b> intended to be selected as an external runtime for
 * calls to {@code org.example.flinke2c.CurrencyConversionFunction} the same
 * way {@link RdmaOperator} is, e.g. via a {@code type=direct-cuda} marker in
 * the {@code table.exec.external-runtime.conf.<class>} string — see
 * {@link #usesDirectCudaTransport}. The actual planner-side dispatch that
 * picks this class for a given conf string is not part of this repository
 * (see {@link RdmaOperator}'s Javadoc for the same caveat); wiring that up is
 * intentionally left to whoever owns that planner rule.
 *
 * <p><b>Conf keys</b> (semicolon-delimited {@code key=value}, same style as
 * {@link RdmaOperator.RdmaConfig}): {@code batchsize} (default 64),
 * {@code pipelinedepth} (default 4, max 64), {@code threadsperblock}
 * (default 256, max 1024), {@code device} (default 0), {@code fieldindex}
 * (default 0 — the row position of the {@code DECIMAL} price column to
 * convert).
 *
 * <p><b>NOTE ON PROVENANCE:</b> written by close analogy to {@link RdmaOperator}
 * / {@link RdmaPreOperator} / {@link RdmaPostOperator}, since
 * {@link ExternalRuntimeOperator}'s source is not available in this
 * repository. The constructor signature ({@code (String conf, RowType
 * inputRowType, RowType resultRowType)}), the inherited {@code conf} field,
 * the inherited {@code output} field, and the
 * {@code openInternal}/{@code closeInternal}/{@code processElementInternal}/
 * {@code processRow} template methods are all inferred from how the RDMA
 * operators use them, not verified against the actual base class. Check
 * these against the real {@code ExternalRuntimeOperator} when compiling this
 * in; the batching/pipelining/row-conversion logic below does not depend on
 * getting those exactly right, but the class will not compile until they
 * match.
 */
@Internal
public final class CudaCurrencyConversionOperator extends ExternalRuntimeOperator
        implements BoundedOneInput {

    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(CudaCurrencyConversionOperator.class);

    private static final int DEFAULT_BATCH_SIZE = 64;
    private static final int DEFAULT_PIPELINE_DEPTH = 4;
    private static final int DEFAULT_THREADS_PER_BLOCK = 256;
    private static final int DEFAULT_CUDA_DEVICE = 0;
    private static final int DEFAULT_FIELD_INDEX = 0;
    private static final int MAX_PIPELINE_DEPTH = 64;
    private static final int MAX_THREADS_PER_BLOCK = 1024;

    // Keep in sync with direct_currency_conversion_jni.cu's Input/Output
    // structs (shared by both JNI bridges) and CurrencyConversionFunctionGpu.
    private static final int INPUT_STRIDE = 16;
    private static final int INPUT_PRICE_OFFSET = 0;
    private static final int INPUT_VALID_OFFSET = 8;
    private static final int OUTPUT_STRIDE = 16;
    private static final int OUTPUT_PRICE_OFFSET = 0;
    private static final int OUTPUT_VALID_OFFSET = 8;

    // BigDecimal.valueOf(double) round-trips through decimal string
    // formatting (new BigDecimal(Double.toString(val))); DecimalData
    // .fromUnscaledLong is a cheap wrap with no parsing at all — cheaper
    // even than CurrencyConversionFunctionGpu's toScale3, which still has to
    // hand Flink a BigDecimal at the AsyncScalarFunction codegen boundary.
    // Here we construct Flink's internal decimal representation directly.
    // Magnitudes whose unscaled value could overflow a long fall back to the
    // exact BigDecimal path; real prices are nowhere near this threshold.
    private static final double MAX_SAFE_FAST_PATH_MAGNITUDE = 1.0e15;
    // DecimalData.fromUnscaledLong stores a compact decimal in a long and
    // accepts only Flink's compact precision range (1..18). DECIMAL(23,3),
    // which is the normal currency output type, must use fromBigDecimal.
    private static final int MAX_COMPACT_DECIMAL_PRECISION = 18;

    private final RowType rowType;

    private transient int cudaDevice;
    private transient int batchSize;
    private transient int pipelineDepth;
    private transient int threadsPerBlock;
    private transient int priceFieldIndex;
    private transient int pricePrecision;
    private transient int priceScale;

    private transient long nativeHandle;
    private transient ByteBuffer[] nativeInputs;
    private transient ByteBuffer[] nativeOutputs;
    private transient RowData.FieldGetter[] fieldGetters;

    // Lane state: rows already submitted to a lane, awaiting collection.
    private transient List<StreamRecord<RowData>>[] lanePending;
    private transient int nextLane;
    // The batch currently being filled, not yet submitted to any lane.
    private transient List<StreamRecord<RowData>> filling;
    private transient int fillCount;

    public CudaCurrencyConversionOperator(String conf, RowType rowType) {
        super(conf, rowType, rowType);
        this.rowType = rowType;
    }

    /**
     * Selects this operator when {@code type=direct-cuda} is set, mirroring
     * {@link RdmaOperator#usesRdmaTransport}. Whether/how a predicate like
     * this is actually consulted by the planner is outside this repository;
     * provided in case the dispatch convention expects one per candidate
     * operator class, matching RdmaOperator's shape.
     */
    public static boolean usesDirectCudaTransport(String conf) {
        if (conf == null) {
            return false;
        }
        for (String item : conf.split(";")) {
            int equals = item.indexOf('=');
            if (equals > 0 && "type".equalsIgnoreCase(item.substring(0, equals).trim())) {
                return "direct-cuda".equalsIgnoreCase(item.substring(equals + 1).trim());
            }
        }
        return false;
    }

    @Override
    protected void openInternal() throws Exception {
        final Map<String, String> values = parseConf(conf);
        this.cudaDevice = intValue(values, "device", DEFAULT_CUDA_DEVICE);
        this.batchSize = intValue(values, "batchsize", DEFAULT_BATCH_SIZE);
        this.pipelineDepth = intValue(values, "pipelinedepth", DEFAULT_PIPELINE_DEPTH);
        this.threadsPerBlock = intValue(values, "threadsperblock", DEFAULT_THREADS_PER_BLOCK);
        this.priceFieldIndex = intValue(values, "fieldindex", DEFAULT_FIELD_INDEX);

        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchsize must be positive");
        }
        if (pipelineDepth <= 0 || pipelineDepth > MAX_PIPELINE_DEPTH) {
            throw new IllegalArgumentException("pipelinedepth must be in 1.." + MAX_PIPELINE_DEPTH);
        }
        if (threadsPerBlock <= 0 || threadsPerBlock > MAX_THREADS_PER_BLOCK) {
            throw new IllegalArgumentException("threadsperblock must be in 1.." + MAX_THREADS_PER_BLOCK);
        }
        if (priceFieldIndex < 0
                || priceFieldIndex >= rowType.getFieldCount()
                || !(rowType.getTypeAt(priceFieldIndex) instanceof DecimalType)) {
            throw new IllegalArgumentException(
                    "fieldindex must name a DECIMAL column in the row, got index " + priceFieldIndex);
        }
        final DecimalType priceType = (DecimalType) rowType.getTypeAt(priceFieldIndex);
        this.pricePrecision = priceType.getPrecision();
        this.priceScale = priceType.getScale();

        this.fieldGetters = new RowData.FieldGetter[rowType.getFieldCount()];
        for (int i = 0; i < fieldGetters.length; i++) {
            fieldGetters[i] = RowData.createFieldGetter(rowType.getTypeAt(i), i);
        }

        this.nativeHandle = DirectCudaCurrencyNative.create(
                cudaDevice, batchSize, pipelineDepth, threadsPerBlock);
        if (nativeHandle == 0L) {
            throw new IllegalStateException(
                    "CUDA currency conversion context creation returned a null handle");
        }
        this.nativeInputs = new ByteBuffer[pipelineDepth];
        this.nativeOutputs = new ByteBuffer[pipelineDepth];
        @SuppressWarnings("unchecked")
        final List<StreamRecord<RowData>>[] lanes = new List[pipelineDepth];
        this.lanePending = lanes;
        for (int lane = 0; lane < pipelineDepth; lane++) {
            nativeInputs[lane] = DirectCudaCurrencyNative.inputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
            nativeOutputs[lane] = DirectCudaCurrencyNative.outputBuffer(nativeHandle, lane)
                    .order(ByteOrder.nativeOrder());
        }
        this.nextLane = 0;
        this.filling = new ArrayList<>(batchSize);
        this.fillCount = 0;

        LOG.info(
                "CudaCurrencyConversionOperator opened (batchSize={}, pipelineDepth={}, "
                        + "threadsPerBlock={}, device={}, priceFieldIndex={})",
                batchSize, pipelineDepth, threadsPerBlock, cudaDevice, priceFieldIndex);
    }

    @Override
    protected RowData processRow(RowData inRow) throws Exception {
        throw new UnsupportedOperationException(
                "CudaCurrencyConversionOperator batches rows; see processElementInternal");
    }

    @Override
    protected void processElementInternal(StreamRecord<RowData> element) throws Exception {
        final RowData row = element.getValue();
        final ByteBuffer input = nativeInputs[nextLane];
        final int base = fillCount * INPUT_STRIDE;
        if (row.isNullAt(priceFieldIndex)) {
            input.putDouble(base + INPUT_PRICE_OFFSET, 0.0);
            input.putInt(base + INPUT_VALID_OFFSET, 0);
        } else {
            final double price = row.getDecimal(priceFieldIndex, pricePrecision, priceScale)
                    .toBigDecimal()
                    .doubleValue();
            input.putDouble(base + INPUT_PRICE_OFFSET, price);
            input.putInt(base + INPUT_VALID_OFFSET, 1);
        }
        filling.add(element);
        fillCount++;

        if (fillCount == batchSize) {
            submitFilling();
        }
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        flushAll();
        super.processWatermark(mark);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        flushAll();
        super.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void endInput() throws Exception {
        flushAll();
    }

    @Override
    protected void closeInternal() throws Exception {
        try {
            flushAll();
        } finally {
            destroyNativeContext();
        }
    }

    /** Submits any partial batch, then waits out and emits every lane's pending rows, in lane order. */
    private void flushAll() throws Exception {
        submitFilling();
        for (int lane = 0; lane < pipelineDepth; lane++) {
            completeLane(lane);
        }
    }

    private void submitFilling() throws Exception {
        if (fillCount == 0) {
            return;
        }
        final int lane = nextLane;
        nextLane = (nextLane + 1) % pipelineDepth;

        // Reusing a lane requires its previous batch's GPU work (and our own
        // collection of it) to be done first; with pipelineDepth > 1 that
        // work has typically overlapped with the other lanes submitted
        // since, so this is often a cheap/no-op wait.
        completeLane(lane);

        DirectCudaCurrencyNative.submitBatch(nativeHandle, lane, fillCount);
        lanePending[lane] = filling;
        filling = new ArrayList<>(batchSize);
        fillCount = 0;
    }

    private void completeLane(int lane) throws Exception {
        final List<StreamRecord<RowData>> pending = lanePending[lane];
        if (pending == null) {
            return;
        }
        lanePending[lane] = null;
        DirectCudaCurrencyNative.waitBatch(nativeHandle, lane);
        final ByteBuffer nativeOutput = nativeOutputs[lane];
        for (int i = 0; i < pending.size(); i++) {
            final int base = i * OUTPUT_STRIDE;
            final StreamRecord<RowData> element = pending.get(i);
            final RowData source = element.getValue();

            final GenericRowData converted = new GenericRowData(source.getArity());
            converted.setRowKind(source.getRowKind());
            for (int f = 0; f < fieldGetters.length; f++) {
                converted.setField(f, fieldGetters[f].getFieldOrNull(source));
            }
            if (nativeOutput.getInt(base + OUTPUT_VALID_OFFSET) == 0) {
                converted.setField(priceFieldIndex, null);
            } else {
                final double price = nativeOutput.getDouble(base + OUTPUT_PRICE_OFFSET);
                converted.setField(priceFieldIndex, toDecimal(price));
            }
            output.collect(element.replace(converted));
        }
    }

    private void destroyNativeContext() {
        if (nativeHandle != 0L) {
            DirectCudaCurrencyNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
        nativeInputs = null;
        nativeOutputs = null;
        lanePending = null;
    }

    private DecimalData toDecimal(double value) {
        if (pricePrecision > MAX_COMPACT_DECIMAL_PRECISION
                || !Double.isFinite(value)
                || Math.abs(value) >= MAX_SAFE_FAST_PATH_MAGNITUDE) {
            return DecimalData.fromBigDecimal(
                    BigDecimal.valueOf(value).setScale(priceScale, RoundingMode.HALF_UP),
                    pricePrecision, priceScale);
        }
        final double scale = Math.pow(10, priceScale);
        final double scaled = value * scale;
        final long unscaled = value >= 0.0
                ? (long) Math.floor(scaled + 0.5)
                : (long) Math.ceil(scaled - 0.5);
        return DecimalData.fromUnscaledLong(unscaled, pricePrecision, priceScale);
    }

    /**
     * Parses a semicolon-delimited {@code key=value;...} string the same way
     * {@link RdmaOperator.RdmaConfig} parses its {@code conf} string: keys
     * are trimmed and lower-cased, values are trimmed, malformed entries (no
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

    private static int intValue(Map<String, String> values, String key, int fallback) {
        final String value = values.get(key);
        return value == null || value.isEmpty() ? fallback : Integer.parseInt(value);
    }
}
