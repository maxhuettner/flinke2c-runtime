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
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.RowKind;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.List;

/** POST: consumes completed RDMA output slots in CQ publication order. */
@Internal
public final class RdmaPostOperator extends RdmaOperator {

    private static final long serialVersionUID = 1L;

    private transient ArrayDeque<byte[]> completedSlots;
    private transient long expectedRowId;
    private transient boolean reuseObjects;
    private transient GenericRowData reuseRow;

    public RdmaPostOperator(String conf, RowType rowType) {
        super(conf, rowType, rowType, RustRdmaRingBuffer.Factory.POST);
    }

    public RdmaPostOperator(
            String conf, RowType rowType, RdmaRingBufferFactory ringBufferFactory) {
        this(conf, rowType, rowType, ringBufferFactory);
    }

    public RdmaPostOperator(
            String conf, RowType inputRowType, @Nullable RowType resultRowType) {
        super(conf, inputRowType, resultRowType, RustRdmaRingBuffer.Factory.POST);
    }

    public RdmaPostOperator(
            String conf,
            RowType inputRowType,
            @Nullable RowType resultRowType,
            RdmaRingBufferFactory ringBufferFactory) {
        super(conf, inputRowType, resultRowType, ringBufferFactory);
    }

    @Override
    protected Role role() {
        return Role.POST;
    }

    @Override
    protected void openInternal() throws Exception {
        openRdmaSession();
        this.reuseObjects = getRuntimeContext().isObjectReuseEnabled();
        this.codec =
                new ExternalRuntimeBinaryCodec(
                        true,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        resultWireTypes,
                        resultReadTypes.toArray(new LogicalType[0]),
                        resultFieldTypes.toArray(new LogicalType[0]),
                        reuseObjects);
        this.completedSlots = new ArrayDeque<>(rdmaConfig.batchSize);
        this.expectedRowId = 0L;
        this.reuseRow = reuseObjects ? new GenericRowData(resultFieldTypes.size()) : null;
        LOG.info(
                "RdmaPostOperator joined {}:{} (batchSize={}, no application ACKs)",
                rdmaConfig.host,
                rdmaConfig.port,
                rdmaConfig.batchSize);
    }

    @Override
    protected RowData processRow(RowData inRow) throws Exception {
        throw new UnsupportedOperationException(
                "RdmaPostOperator emits returned slots via processElementInternal");
    }

    @Override
    protected void processElementInternal(StreamRecord<RowData> element) throws Exception {
        if (completedSlots.isEmpty()) {
            final List<byte[]> batch = receiveBatch();
            completedSlots.addAll(batch);
        }

        final byte[] slot = completedSlots.removeFirst();
        final RowKind fallbackKind = element.getValue().getRowKind();
        final ExternalRuntimeBinaryCodec.RowWithId decoded =
                codec.readFramedRow(slot, fallbackKind, reuseRow);
        if (decoded.rowId != expectedRowId) {
            throw new IOException(
                    "RDMA output order violation: expected rowId "
                            + expectedRowId
                            + " but received "
                            + decoded.rowId);
        }
        expectedRowId++;
        output.collect(element.replace(decoded.row));
    }

    @Override
    protected void closeInternal() throws Exception {
        completedSlots = null;
        reuseRow = null;
        codec = null;
        closeRdmaSession();
    }
}
