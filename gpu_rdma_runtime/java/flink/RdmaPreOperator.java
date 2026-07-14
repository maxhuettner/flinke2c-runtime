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
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.RowKind;

import java.io.IOException;

/** PRE: encodes rows into RDMA ring slots, publishes batches, and emits placeholders. */
@Internal
public final class RdmaPreOperator extends RdmaOperator implements BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private transient int pendingSlotCount;
    private transient long nextRowId;

    public RdmaPreOperator(String conf, RowType rowType) {
        super(conf, rowType, null, RustRdmaRingBuffer.Factory.PRE);
    }

    public RdmaPreOperator(
            String conf, RowType rowType, RdmaRingBufferFactory ringBufferFactory) {
        super(conf, rowType, null, ringBufferFactory);
    }

    @Override
    protected Role role() {
        return Role.PRE;
    }

    @Override
    protected void openInternal() throws Exception {
        openRdmaSession();
        this.codec =
                new ExternalRuntimeBinaryCodec(
                        true,
                        payloadWireTypes,
                        payloadWriteTypes.toArray(new LogicalType[0]),
                        payloadSourceRoots,
                        payloadSourcePrecision,
                        payloadSourceScale,
                        payloadTimestampPrecision,
                        null,
                        null,
                        null,
                        false);
        this.pendingSlotCount = 0;
        this.nextRowId = 0L;
        LOG.info(
                "RdmaPreOperator opened {}:{} (batchSize={}, ringElements={}, maxItemSize={})",
                rdmaConfig.host,
                rdmaConfig.port,
                rdmaConfig.batchSize,
                RING_BUFFER_ELEMENTS,
                rdmaConfig.maxItemSize);
    }

    @Override
    protected RowData processRow(RowData inRow) throws Exception {
        final byte[] slot = codec.encodeFramedRow(inRow, payloadFieldIndicesArray, nextRowId++);
        if (slot.length > rdmaConfig.maxItemSize) {
            throw new IOException(
                    "Encoded row exceeds RDMA Slot.value capacity: "
                            + slot.length
                            + " > "
                            + rdmaConfig.maxItemSize);
        }
        writeInputSlot(slot);
        pendingSlotCount++;
        if (pendingSlotCount == rdmaConfig.batchSize) {
            flushBatch();
        }
        return placeholder(inRow.getRowKind());
    }

    @Override
    public void endInput() throws Exception {
        flushBatch();
    }

    private void flushBatch() throws IOException {
        if (pendingSlotCount == 0) {
            return;
        }
        publishInputBatch(pendingSlotCount);
        pendingSlotCount = 0;
    }

    private RowData placeholder(RowKind kind) {
        switch (kind) {
            case INSERT:
                if (insertPlaceholder == null) {
                    insertPlaceholder = new GenericRowData(inputFieldCount);
                    insertPlaceholder.setRowKind(kind);
                }
                return insertPlaceholder;
            case UPDATE_AFTER:
                if (updateAfterPlaceholder == null) {
                    updateAfterPlaceholder = new GenericRowData(inputFieldCount);
                    updateAfterPlaceholder.setRowKind(kind);
                }
                return updateAfterPlaceholder;
            case UPDATE_BEFORE:
                if (updateBeforePlaceholder == null) {
                    updateBeforePlaceholder = new GenericRowData(inputFieldCount);
                    updateBeforePlaceholder.setRowKind(kind);
                }
                return updateBeforePlaceholder;
            case DELETE:
                if (deletePlaceholder == null) {
                    deletePlaceholder = new GenericRowData(inputFieldCount);
                    deletePlaceholder.setRowKind(kind);
                }
                return deletePlaceholder;
            default:
                final GenericRowData row = new GenericRowData(inputFieldCount);
                row.setRowKind(kind);
                return row;
        }
    }

    @Override
    protected void closeInternal() throws Exception {
        IOException error = null;
        try {
            flushBatch();
        } catch (IOException e) {
            error = e;
        }
        try {
            closeRdmaSession();
        } catch (IOException e) {
            error = suppress(error, e);
        }
        pendingSlotCount = 0;
        codec = null;
        if (error != null) {
            throw error;
        }
    }
}
