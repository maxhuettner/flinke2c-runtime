package org.apache.flink.table.runtime.functions.table.externalruntime;

import org.apache.flink.table.api.TableException;
import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.DecimalDataUtils;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LocalZonedTimestampType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.TimestampType;
import org.apache.flink.types.RowKind;

import javax.annotation.Nullable;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Custom binary row codec implementing:
 * [int32_be frameLen][payload]
 * payload := int32 __op, [int64 __rowId if enabled], nullBitmap, values...
 */
public final class ExternalRuntimeBinaryCodec {

    public enum WireType {
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
        BOOL,
        STRING, // [int32_be len][utf8]
        BYTES, // [int32_be len][raw]
        TIMESTAMP_MILLIS, // int64 epoch millis
        DECIMAL_UNSCALED_I64, // int64 unscaled (precision <= 18)
        DECIMAL_UNSCALED_BYTES // [int32_be len][two's complement bytes] (precision > 18)
    }

    private static final int DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024; // 64 MB

    private final boolean includeRowId;

    // PRE write schema
    private final WireType[] writeWireTypes;
    private final LogicalType[] writeTargetTypes;
    private final LogicalTypeRoot[] writeSourceRoots;
    private final int[] writeSourcePrecision;
    private final int[] writeSourceScale;
    private final int[] writeTimestampPrecision;

    // POST read schema
    private final WireType[] readWireTypes;
    private final LogicalType[] readSourceTypes;
    private final LogicalType[] readTargetTypes;
    private final boolean reuseObjects;
    private final byte[][] reuseBytes;

    // reusable buffers
    private final GrowableBuffer outBuf = new GrowableBuffer(8 * 1024);
    private byte[] frameReadBuf = new byte[8 * 1024];

    private static final long[] POW10 = initPow10();

    public ExternalRuntimeBinaryCodec(
            boolean includeRowId,
            @Nullable WireType[] writeWireTypes,
            @Nullable LogicalType[] writeTargetTypes,
            @Nullable LogicalTypeRoot[] writeSourceRoots,
            @Nullable int[] writeSourcePrecision,
            @Nullable int[] writeSourceScale,
            @Nullable int[] writeTimestampPrecision,
            @Nullable WireType[] readWireTypes,
            @Nullable LogicalType[] readSourceTypes,
            @Nullable LogicalType[] readTargetTypes,
            boolean reuseObjects) {

        this.includeRowId = includeRowId;

        this.writeWireTypes = writeWireTypes;
        this.writeTargetTypes = writeTargetTypes;
        this.writeSourceRoots = writeSourceRoots;

        this.writeSourcePrecision = writeSourcePrecision;
        this.writeSourceScale = writeSourceScale;
        this.writeTimestampPrecision = writeTimestampPrecision;

        this.readWireTypes = readWireTypes;
        this.readSourceTypes = readSourceTypes;
        this.readTargetTypes = readTargetTypes;
        this.reuseObjects = reuseObjects;
        this.reuseBytes =
                reuseObjects && readWireTypes != null ? new byte[readWireTypes.length][] : null;
    }


    // ---------------------------------------------------------------------
    // PRE: write one framed row
    // ---------------------------------------------------------------------

    void writeFramedRow(
            OutputStream out,
            RowData row,
            int[] payloadFieldIndices,
            long rowId) throws IOException {

        if (writeWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for writing");
        }

        final int nFields = writeWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;

        outBuf.reset();

        // __op
        outBuf.putIntBE(rowKindToOp(row.getRowKind()));

        // optional __rowId
        if (includeRowId) {
            outBuf.putLongBE(rowId);
        }

        // null bitmap placeholder
        final int nullBitmapPos = outBuf.position();
        outBuf.ensureCapacity(nullBytes);
        for (int i = 0; i < nullBytes; i++) {
            outBuf.putByte((byte) 0);
        }

        // values
        for (int i = 0; i < nFields; i++) {
            final int sourceIndex = payloadFieldIndices[i];
            if (row.isNullAt(sourceIndex)) {
                setNullBit(outBuf.buf(), nullBitmapPos, i);
                continue;
            }
            writeValue(i, row, sourceIndex);
        }

        final int payloadLen = outBuf.position();

        // frame: [len][payload]
        writeIntBE(out, payloadLen);
        out.write(outBuf.buf(), 0, payloadLen);
    }

    /** Encodes one frame without constructing a temporary ByteArrayOutputStream. */
    public byte[] encodeFramedRow(RowData row, int[] payloadFieldIndices, long rowId) throws IOException {
        if (writeWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for writing");
        }
        final int nFields = writeWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        outBuf.reset();
        outBuf.putIntBE(rowKindToOp(row.getRowKind()));
        if (includeRowId) {
            outBuf.putLongBE(rowId);
        }
        final int nullBitmapPos = outBuf.position();
        outBuf.ensureCapacity(nullBytes);
        for (int i = 0; i < nullBytes; i++) {
            outBuf.putByte((byte) 0);
        }
        for (int i = 0; i < nFields; i++) {
            final int sourceIndex = payloadFieldIndices[i];
            if (row.isNullAt(sourceIndex)) {
                setNullBit(outBuf.buf(), nullBitmapPos, i);
            } else {
                writeValue(i, row, sourceIndex);
            }
        }
        final int payloadLen = outBuf.position();
        final byte[] frame = new byte[payloadLen + Integer.BYTES];
        frame[0] = (byte) (payloadLen >>> 24);
        frame[1] = (byte) (payloadLen >>> 16);
        frame[2] = (byte) (payloadLen >>> 8);
        frame[3] = (byte) payloadLen;
        System.arraycopy(outBuf.buf(), 0, frame, Integer.BYTES, payloadLen);
        return frame;
    }

    /**
     * Encodes directly into a caller-owned buffer. This is used by the local packed CUDA
     * operator so the hot path does not allocate one temporary byte array per input row.
     *
     * @return complete frame length, including the four-byte big-endian length prefix
     */
    public int encodeFramedRow(
            RowData row, int[] payloadFieldIndices, long rowId,
            java.nio.ByteBuffer destination, int offset) throws IOException {
        return encodeFramedRow(row, payloadFieldIndices, rowId, destination, offset,
                Integer.MAX_VALUE);
    }

    /** Encodes directly into a buffer while enforcing a per-record capacity. */
    public int encodeFramedRow(
            RowData row, int[] payloadFieldIndices, long rowId,
            java.nio.ByteBuffer destination, int offset, int maxFrameLength) throws IOException {
        if (writeWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for writing");
        }
        final int nFields = writeWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        outBuf.reset();
        outBuf.putIntBE(rowKindToOp(row.getRowKind()));
        if (includeRowId) outBuf.putLongBE(rowId);
        final int nullBitmapPos = outBuf.position();
        outBuf.ensureCapacity(nullBytes);
        for (int i = 0; i < nullBytes; i++) outBuf.putByte((byte) 0);
        for (int i = 0; i < nFields; i++) {
            final int sourceIndex = payloadFieldIndices[i];
            if (row.isNullAt(sourceIndex)) setNullBit(outBuf.buf(), nullBitmapPos, i);
            else writeValue(i, row, sourceIndex);
        }
        final int payloadLen = outBuf.position();
        final int frameLen = payloadLen + Integer.BYTES;
        if (frameLen > maxFrameLength || offset < 0 || offset > destination.limit() - frameLen) {
            throw new IOException("destination buffer is too small for encoded row");
        }
        destination.put(offset, (byte) (payloadLen >>> 24));
        destination.put(offset + 1, (byte) (payloadLen >>> 16));
        destination.put(offset + 2, (byte) (payloadLen >>> 8));
        destination.put(offset + 3, (byte) payloadLen);
        // Bulk relative put is JIT-intrinsified for a direct destination buffer;
        // the byte-at-a-time loop this replaced ran on the Flink task thread for
        // every input row, on the same critical path that has to keep up with
        // the source rate - the same class of bug as copyBytes() on the decode
        // side, just on the hot ingest path instead of the completer thread.
        // Absolute bulk put avoids allocating a duplicate ByteBuffer for every
        // encoded row. The caller owns the destination position, and this
        // method already uses absolute header writes.
        destination.put(offset + Integer.BYTES, outBuf.buf(), 0, payloadLen);
        return frameLen;
    }

    /**
     * Extracts one row's payload values without framing them, so the framing
     * step (null-bitmap layout, wire-format packing, buffer writes) can run
     * on a different thread than the one that read the row. Reading the row
     * itself cannot be deferred - {@code pipeline.object-reuse} means its
     * backing memory is not guaranteed to survive past this call - but once
     * a value is out as an independent boxed Java object, nothing about
     * packing it into bytes touches Flink's row abstraction or its
     * reuse/lifecycle rules, so that part is safe to hand off. See
     * {@link #writeExtractedRow} for the other half; used by
     * {@code ImputationGpuFunction} to keep its Flink-thread-side
     * {@code processElement} to just these getter calls, moving the actual
     * byte-packing work onto its submitter thread.
     *
     * @return one entry per configured write field, null for a null source
     *     field, otherwise a boxed value in the representation
     *     {@link #writeValueBytes} expects for that field's {@link WireType}
     */
    public Object[] extractRowValues(RowData row, int[] payloadFieldIndices) {
        if (writeWireTypes == null) {
            throw new IllegalStateException("ExternalRuntimeBinaryCodec not configured for writing");
        }
        final int nFields = writeWireTypes.length;
        final Object[] values = new Object[nFields];
        for (int i = 0; i < nFields; i++) {
            final int sourceIndex = payloadFieldIndices[i];
            values[i] = row.isNullAt(sourceIndex) ? null : extractValue(i, row, sourceIndex);
        }
        return values;
    }

    private Object extractValue(int fieldPos, RowData row, int sourceIndex) {
        final WireType wt = writeWireTypes[fieldPos];
        final LogicalType targetType = writeTargetTypes[fieldPos];
        final LogicalTypeRoot sourceRoot = writeSourceRoots[fieldPos];

        switch (wt) {
            case BOOL:
                return row.getBoolean(sourceIndex);
            case INT32:
                switch (sourceRoot) {
                    case TINYINT:
                        return (int) row.getByte(sourceIndex);
                    case SMALLINT:
                        return (int) row.getShort(sourceIndex);
                    default:
                        return row.getInt(sourceIndex);
                }
            case INT64:
                return row.getLong(sourceIndex);
            case FLOAT32:
                return row.getFloat(sourceIndex);
            case FLOAT64:
                return row.getDouble(sourceIndex);
            case STRING:
                return row.getString(sourceIndex).toBytes();
            case BYTES:
                return row.getBinary(sourceIndex);
            case TIMESTAMP_MILLIS: {
                final int precision = writeTimestampPrecision[fieldPos];
                return row.getTimestamp(sourceIndex, precision).getMillisecond();
            }
            case DECIMAL_UNSCALED_I64: {
                final DecimalType dt = (DecimalType) targetType;
                final int srcPrecision = writeSourcePrecision[fieldPos];
                final int srcScale = writeSourceScale[fieldPos];
                return toUnscaledLong(row, sourceIndex, sourceRoot, dt.getPrecision(), dt.getScale(),
                        srcPrecision, srcScale);
            }
            case DECIMAL_UNSCALED_BYTES: {
                final DecimalType dt = (DecimalType) targetType;
                final int srcPrecision = writeSourcePrecision[fieldPos];
                final int srcScale = writeSourceScale[fieldPos];
                if (sourceRoot == LogicalTypeRoot.DECIMAL && srcScale == dt.getScale()) {
                    final DecimalData dec = row.getDecimal(sourceIndex, srcPrecision, srcScale);
                    byte[] bytes = dec.toUnscaledBytes();
                    // DECIMAL_UNSCALED_BYTES must never use a zero-length
                    // representation. BigInteger zero is encoded as 00.
                    if (bytes.length == 0) {
                        bytes = new byte[] {0};
                    }
                    return bytes;
                }
                final BigInteger unscaled = toUnscaledBigInt(row, sourceIndex, sourceRoot, dt.getPrecision(),
                        dt.getScale(), srcPrecision, srcScale);
                return unscaled.toByteArray(); // two's complement big-endian
            }
            default:
                throw new IllegalStateException("Unsupported wire type: " + wt);
        }
    }

    /**
     * Frames values previously returned by {@link #extractRowValues} into
     * the wire format and writes them directly into a caller-owned buffer -
     * the deferred half of {@link #encodeFramedRow}'s work, safe to call
     * from a thread other than the one that read the row, since every value
     * here is already an independent Java object with no tie back to the
     * source {@code RowData}.
     *
     * @return complete frame length, including the four-byte length prefix
     */
    public int writeExtractedRow(
            Object[] values, RowKind rowKind, long rowId,
            java.nio.ByteBuffer destination, int offset, int maxFrameLength) throws IOException {
        if (writeWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for writing");
        }
        final int nFields = writeWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        outBuf.reset();
        outBuf.putIntBE(rowKindToOp(rowKind));
        if (includeRowId) outBuf.putLongBE(rowId);
        final int nullBitmapPos = outBuf.position();
        outBuf.ensureCapacity(nullBytes);
        for (int i = 0; i < nullBytes; i++) outBuf.putByte((byte) 0);
        for (int i = 0; i < nFields; i++) {
            final Object value = values[i];
            if (value == null) setNullBit(outBuf.buf(), nullBitmapPos, i);
            else writeValueBytes(i, value);
        }
        final int payloadLen = outBuf.position();
        final int frameLen = payloadLen + Integer.BYTES;
        if (frameLen > maxFrameLength || offset < 0 || offset > destination.limit() - frameLen) {
            throw new IOException("destination buffer is too small for encoded row");
        }
        destination.put(offset, (byte) (payloadLen >>> 24));
        destination.put(offset + 1, (byte) (payloadLen >>> 16));
        destination.put(offset + 2, (byte) (payloadLen >>> 8));
        destination.put(offset + 3, (byte) payloadLen);
        final java.nio.ByteBuffer view = destination.duplicate();
        view.position(offset + Integer.BYTES);
        view.put(outBuf.buf(), 0, payloadLen);
        return frameLen;
    }

    private void writeValueBytes(int fieldPos, Object value) {
        switch (writeWireTypes[fieldPos]) {
            case BOOL:
                outBuf.putByte((byte) (((Boolean) value) ? 1 : 0));
                return;
            case INT32:
                outBuf.putIntBE((Integer) value);
                return;
            case INT64:
            case TIMESTAMP_MILLIS:
            case DECIMAL_UNSCALED_I64:
                outBuf.putLongBE((Long) value);
                return;
            case FLOAT32:
                outBuf.putIntBE(Float.floatToIntBits((Float) value));
                return;
            case FLOAT64:
                outBuf.putLongBE(Double.doubleToLongBits((Double) value));
                return;
            case STRING:
            case BYTES:
            case DECIMAL_UNSCALED_BYTES: {
                final byte[] bytes = (byte[]) value;
                outBuf.putIntBE(bytes.length);
                outBuf.putBytes(bytes);
                return;
            }
            default:
                throw new IllegalStateException("Unsupported wire type: " + writeWireTypes[fieldPos]);
        }
    }

    private void writeValue(int fieldPos, RowData row, int sourceIndex) throws IOException {
        final WireType wt = writeWireTypes[fieldPos];
        final LogicalType targetType = writeTargetTypes[fieldPos];
        final LogicalTypeRoot sourceRoot = writeSourceRoots[fieldPos];

        switch (wt) {
            case BOOL:
                outBuf.putByte((byte) (row.getBoolean(sourceIndex) ? 1 : 0));
                return;
            case INT32:
                switch (sourceRoot) {
                    case TINYINT:
                        outBuf.putIntBE(row.getByte(sourceIndex));
                        return;
                    case SMALLINT:
                        outBuf.putIntBE(row.getShort(sourceIndex));
                        return;
                    default:
                        outBuf.putIntBE(row.getInt(sourceIndex));
                        return;
                }
            case INT64:
                outBuf.putLongBE(row.getLong(sourceIndex));
                return;
            case FLOAT32:
                outBuf.putIntBE(Float.floatToIntBits(row.getFloat(sourceIndex)));
                return;
            case FLOAT64:
                outBuf.putLongBE(Double.doubleToLongBits(row.getDouble(sourceIndex)));
                return;
            case STRING: {
                final StringData sd = row.getString(sourceIndex);
                final byte[] bytes = sd.toBytes();
                outBuf.putIntBE(bytes.length);
                outBuf.putBytes(bytes);
                return;
            }
            case BYTES: {
                final byte[] bytes = row.getBinary(sourceIndex);
                outBuf.putIntBE(bytes.length);
                outBuf.putBytes(bytes);
                return;
            }
            case TIMESTAMP_MILLIS: {
                final int precision = writeTimestampPrecision[fieldPos];
                final long ms = row.getTimestamp(sourceIndex, precision).getMillisecond();
                outBuf.putLongBE(ms);
                return;
            }
            case DECIMAL_UNSCALED_I64: {
                final DecimalType dt = (DecimalType) targetType;
                final int srcPrecision = writeSourcePrecision[fieldPos];
                final int srcScale = writeSourceScale[fieldPos];
                final long unscaled = toUnscaledLong(row, sourceIndex, sourceRoot, dt.getPrecision(), dt.getScale(),
                        srcPrecision, srcScale);
                outBuf.putLongBE(unscaled);
                return;
            }
            case DECIMAL_UNSCALED_BYTES: {
                final DecimalType dt = (DecimalType) targetType;
                final int srcPrecision = writeSourcePrecision[fieldPos];
                final int srcScale = writeSourceScale[fieldPos];
                if (sourceRoot == LogicalTypeRoot.DECIMAL && srcScale == dt.getScale()) {
                    final DecimalData dec = row.getDecimal(sourceIndex, srcPrecision, srcScale);
                    byte[] bytes = dec.toUnscaledBytes();
                    // DECIMAL_UNSCALED_BYTES must never use a zero-length
                    // representation. BigInteger zero is encoded as 00.
                    if (bytes.length == 0) {
                        bytes = new byte[] {0};
                    }
                    outBuf.putIntBE(bytes.length);
                    outBuf.putBytes(bytes);
                } else {
                    final BigInteger unscaled = toUnscaledBigInt(row, sourceIndex, sourceRoot, dt.getPrecision(),
                            dt.getScale(),
                            srcPrecision, srcScale);
                    final byte[] bytes = unscaled.toByteArray(); // two's complement big-endian
                    outBuf.putIntBE(bytes.length);
                    outBuf.putBytes(bytes);
                }
                return;
            }
            default:
                throw new IOException("Unsupported wire type: " + wt);
        }
    }

    RowWithId readFramedRow(
            InputStream in, RowKind fallbackKind, @Nullable GenericRowData reuseRow) throws IOException {
        if (readWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for reading");
        }

        final int frameLen = readIntBE(in);
        if (frameLen < 0 || frameLen > DEFAULT_MAX_FRAME_SIZE) {
            throw new IOException("Invalid frame length: " + frameLen);
        }

        ensureReadBuf(frameLen);
        readFully(in, frameReadBuf, 0, frameLen);

        return decodeFrame(frameReadBuf, 0, frameLen, fallbackKind, reuseRow);
    }

    /** Decodes a complete RDMA slot directly, avoiding a ByteArrayInputStream allocation. */
    public RowWithId readFramedRow(byte[] frame, RowKind fallbackKind, @Nullable GenericRowData reuseRow)
            throws IOException {
        if (frame.length < Integer.BYTES) {
            throw new IOException("Truncated frame header");
        }
        final int frameLen = readIntBE(frame, 0);
        if (frameLen < 0 || frameLen > DEFAULT_MAX_FRAME_SIZE || frameLen + 4 > frame.length) {
            throw new IOException(
                    "Invalid frame length: "
                            + frameLen
                            + " for RDMA slot payload length "
                            + frame.length);
        }
        return decodeFrame(frame, Integer.BYTES, frameLen, fallbackKind, reuseRow);
    }

    /**
     * Decodes a frame in a native direct buffer without copying the slot into a
     * temporary byte array. The offset points at the four-byte frame length.
     */
    public RowWithId readFramedRow(
            java.nio.ByteBuffer frame, int offset, RowKind fallbackKind,
            @Nullable GenericRowData reuseRow) throws IOException {
        if (offset < 0 || offset > frame.limit() - Integer.BYTES) {
            throw new IOException("Truncated frame header");
        }
        final int frameLen = readIntBE(frame, offset);
        if (frameLen < 0 || frameLen > DEFAULT_MAX_FRAME_SIZE
                || frameLen > frame.limit() - offset - Integer.BYTES) {
            throw new IOException("Invalid frame length: " + frameLen);
        }
        return decodeFrame(frame, offset + Integer.BYTES, frameLen, fallbackKind, reuseRow);
    }

    private RowWithId decodeFrame(
            byte[] frame, int frameOffset, int frameLen, RowKind fallbackKind,
            @Nullable GenericRowData reuseRow) throws IOException {
        int p = frameOffset;

        final int op = readIntBE(frame, p);
        p += 4;

        final long rowId;
        if (includeRowId) {
            rowId = readLongBE(frame, p);
            p += 8;
        } else {
            rowId = -1L;
        }

        final int nFields = readWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;

        if (p + nullBytes > frameOffset + frameLen) {
            throw new IOException("Truncated payload: missing nullBitmap");
        }

        final int nullBitmapPos = p;
        p += nullBytes;

        final GenericRowData outRow =
                reuseRow != null && reuseRow.getArity() == nFields
                        ? reuseRow
                        : new GenericRowData(nFields);
        for (int i = 0; i < nFields; i++) {
            if (isNullBitSet(frame, nullBitmapPos, i)) {
                outRow.setField(i, null);
                continue;
            }
            p = readValueIntoRow(i, frame, p, frameOffset + frameLen, outRow);
        }

        final RowKind kind = opToRowKind(op, fallbackKind);
        outRow.setRowKind(kind);
        return new RowWithId(rowId, outRow);
    }

    private RowWithId decodeFrame(
            java.nio.ByteBuffer frame, int frameOffset, int frameLen, RowKind fallbackKind,
            @Nullable GenericRowData reuseRow) throws IOException {
        int p = frameOffset;
        final int limit = frameOffset + frameLen;
        if (p + 4 > limit) throw new IOException("Truncated payload: missing operation");
        final int op = readIntBE(frame, p);
        p += 4;
        final long rowId;
        if (includeRowId) {
            if (p + 8 > limit) throw new IOException("Truncated payload: missing row id");
            rowId = readLongBE(frame, p);
            p += 8;
        } else {
            rowId = -1L;
        }
        final int nFields = readWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        if (p + nullBytes > limit) throw new IOException("Truncated payload: missing nullBitmap");
        final int nullBitmapPos = p;
        p += nullBytes;
        final GenericRowData outRow = reuseRow != null && reuseRow.getArity() == nFields
                ? reuseRow : new GenericRowData(nFields);
        for (int i = 0; i < nFields; i++) {
            if (isNullBitSet(frame, nullBitmapPos, i)) {
                outRow.setField(i, null);
            } else {
                p = readValueIntoRow(i, frame, p, limit, outRow);
            }
        }
        outRow.setRowKind(opToRowKind(op, fallbackKind));
        return new RowWithId(rowId, outRow);
    }

    private int readValueIntoRow(int i, byte[] buf, int p, int limit, GenericRowData outRow)
            throws IOException {
        final WireType wt = readWireTypes[i];

        switch (wt) {
            case BOOL:
                if (p + 1 > limit)
                    throw new IOException("Truncated BOOL");
                outRow.setField(
                        i, castIfNeeded(buf[p] != 0, readSourceTypes[i], readTargetTypes[i]));
                return p + 1;
            case INT32:
                if (p + 4 > limit)
                    throw new IOException("Truncated INT32");
                outRow.setField(
                        i,
                        castIfNeeded(
                                readIntBE(buf, p), readSourceTypes[i], readTargetTypes[i]));
                return p + 4;
            case INT64:
            case TIMESTAMP_MILLIS:
                if (p + 8 > limit)
                    throw new IOException("Truncated INT64");
                outRow.setField(
                        i,
                        castIfNeeded(
                                readLongBE(buf, p), readSourceTypes[i], readTargetTypes[i]));
                return p + 8;
            case FLOAT32:
                if (p + 4 > limit)
                    throw new IOException("Truncated FLOAT32");
                outRow.setField(
                        i,
                        castIfNeeded(
                                Float.intBitsToFloat(readIntBE(buf, p)),
                                readSourceTypes[i],
                                readTargetTypes[i]));
                return p + 4;
            case FLOAT64:
                if (p + 8 > limit)
                    throw new IOException("Truncated FLOAT64");
                outRow.setField(
                        i,
                        castIfNeeded(
                                Double.longBitsToDouble(readLongBE(buf, p)),
                                readSourceTypes[i],
                                readTargetTypes[i]));
                return p + 8;
            case STRING: {
                if (p + 4 > limit)
                    throw new IOException("Truncated STRING len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit)
                    throw new IOException("Invalid STRING len: " + len);
                final StringData sd =
                        reuseObjects
                                ? StringData.fromBytes(buf, p, len)
                                : StringData.fromBytes(copyBytes(i, buf, p, len), 0, len);
                outRow.setField(i, castIfNeeded(sd, readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            case BYTES: {
                if (p + 4 > limit)
                    throw new IOException("Truncated BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit)
                    throw new IOException("Invalid BYTES len: " + len);
                final byte[] out = copyBytes(i, buf, p, len);
                outRow.setField(i, castIfNeeded(out, readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            case DECIMAL_UNSCALED_I64: {
                if (p + 8 > limit)
                    throw new IOException("Truncated DECIMAL_UNSCALED_I64");
                final long unscaled = readLongBE(buf, p);
                outRow.setField(i, castIfNeeded(unscaled, readSourceTypes[i], readTargetTypes[i]));
                return p + 8;
            }
            case DECIMAL_UNSCALED_BYTES: {
                if (p + 4 > limit)
                    throw new IOException("Truncated DECIMAL_UNSCALED_BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit)
                    throw new IOException("Invalid DECIMAL bytes len: " + len);
                final byte[] bi = copyBytes(i, buf, p, len);
                outRow.setField(i, castIfNeeded(bi, readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            default:
                throw new IOException("Unsupported read wire type: " + wt);
        }
    }

    private int readValueIntoRow(
            int i, java.nio.ByteBuffer buf, int p, int limit, GenericRowData outRow)
            throws IOException {
        return readValueIntoRow(i, buf, p, limit, outRow, i);
    }

    /**
     * Same decode as the four-arg overload, but writes to {@code targetField} in
     * {@code outRow} instead of {@code i}. Lets a caller with its own wire-index ->
     * result-field-index mapping (e.g. one that reorders or drops fields) decode
     * straight into its final row instead of an intermediate one it would otherwise
     * have to copy out of field-by-field afterward.
     */
    private int readValueIntoRow(
            int i, java.nio.ByteBuffer buf, int p, int limit, GenericRowData outRow, int targetField)
            throws IOException {
        final WireType wt = readWireTypes[i];
        switch (wt) {
            case BOOL:
                if (p + 1 > limit) throw new IOException("Truncated BOOL");
                outRow.setField(targetField, castIfNeeded(buf.get(p) != 0, readSourceTypes[i], readTargetTypes[i]));
                return p + 1;
            case INT32:
                if (p + 4 > limit) throw new IOException("Truncated INT32");
                outRow.setField(targetField, castIfNeeded(readIntBE(buf, p), readSourceTypes[i], readTargetTypes[i]));
                return p + 4;
            case INT64:
            case TIMESTAMP_MILLIS:
                if (p + 8 > limit) throw new IOException("Truncated INT64");
                outRow.setField(targetField, castIfNeeded(readLongBE(buf, p), readSourceTypes[i], readTargetTypes[i]));
                return p + 8;
            case FLOAT32:
                if (p + 4 > limit) throw new IOException("Truncated FLOAT32");
                outRow.setField(targetField, castIfNeeded(Float.intBitsToFloat(readIntBE(buf, p)), readSourceTypes[i], readTargetTypes[i]));
                return p + 4;
            case FLOAT64:
                if (p + 8 > limit) throw new IOException("Truncated FLOAT64");
                outRow.setField(targetField, castIfNeeded(Double.longBitsToDouble(readLongBE(buf, p)), readSourceTypes[i], readTargetTypes[i]));
                return p + 8;
            case STRING: {
                if (p + 4 > limit) throw new IOException("Truncated STRING len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid STRING len: " + len);
                final byte[] bytes = copyBytes(i, buf, p, len);
                final StringData sd = StringData.fromBytes(bytes, 0, len);
                outRow.setField(targetField, castIfNeeded(sd, readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            case BYTES: {
                if (p + 4 > limit) throw new IOException("Truncated BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid BYTES len: " + len);
                outRow.setField(targetField, castIfNeeded(copyBytes(i, buf, p, len), readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            case DECIMAL_UNSCALED_I64:
                if (p + 8 > limit) throw new IOException("Truncated DECIMAL_UNSCALED_I64");
                outRow.setField(targetField, castIfNeeded(readLongBE(buf, p), readSourceTypes[i], readTargetTypes[i]));
                return p + 8;
            case DECIMAL_UNSCALED_BYTES: {
                if (p + 4 > limit) throw new IOException("Truncated DECIMAL_UNSCALED_BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid DECIMAL bytes len: " + len);
                outRow.setField(targetField, castIfNeeded(copyBytes(i, buf, p, len), readSourceTypes[i], readTargetTypes[i]));
                return p + len;
            }
            default:
                throw new IOException("Unsupported read wire type: " + wt);
        }
    }

    /** Advances past one wire field's bytes without materializing or casting a value. */
    private int skipValue(int i, java.nio.ByteBuffer buf, int p, int limit) throws IOException {
        switch (readWireTypes[i]) {
            case BOOL:
                if (p + 1 > limit) throw new IOException("Truncated BOOL");
                return p + 1;
            case INT32:
            case FLOAT32:
                if (p + 4 > limit) throw new IOException("Truncated INT32/FLOAT32");
                return p + 4;
            case INT64:
            case TIMESTAMP_MILLIS:
            case FLOAT64:
            case DECIMAL_UNSCALED_I64:
                if (p + 8 > limit) throw new IOException("Truncated INT64/FLOAT64/DECIMAL_UNSCALED_I64");
                return p + 8;
            case STRING:
            case BYTES:
            case DECIMAL_UNSCALED_BYTES: {
                if (p + 4 > limit) throw new IOException("Truncated length prefix");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid length: " + len);
                return p + len;
            }
            default:
                throw new IOException("Unsupported read wire type: " + readWireTypes[i]);
        }
    }

    /**
     * Decodes a frame directly into a caller-owned result row via
     * {@code wireIndexToTargetField}, instead of building the usual intermediate
     * row and making the caller copy every field out of it afterward. Wire field
     * {@code i} is written to {@code outRow} field {@code wireIndexToTargetField[i]},
     * or - if that's negative, meaning the wire field isn't part of the result row -
     * parsed just enough to stay correctly positioned for the fields after it,
     * without allocating or casting a value for it. Row kind is intentionally not
     * touched here: callers of the packed local-GPU path derive it from their own
     * per-row metadata rather than the wire's {@code __op}, since a reuse-safe
     * result row's kind is set once by the caller regardless of what's decoded.
     *
     * @return the frame's row id ({@code includeRowId} must be true)
     */
    public long readFramedRowInto(
            java.nio.ByteBuffer frame, int offset, GenericRowData outRow, int[] wireIndexToTargetField)
            throws IOException {
        if (readWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for reading");
        }
        if (offset < 0 || offset > frame.limit() - Integer.BYTES) {
            throw new IOException("Truncated frame header");
        }
        final int frameLen = readIntBE(frame, offset);
        if (frameLen < 0 || frameLen > DEFAULT_MAX_FRAME_SIZE
                || frameLen > frame.limit() - offset - Integer.BYTES) {
            throw new IOException("Invalid frame length: " + frameLen);
        }
        int p = offset + Integer.BYTES;
        final int limit = p + frameLen;
        if (p + 4 > limit) throw new IOException("Truncated payload: missing operation");
        p += 4; // __op: the caller sets row kind from its own metadata, not the wire.
        final long rowId;
        if (includeRowId) {
            if (p + 8 > limit) throw new IOException("Truncated payload: missing row id");
            rowId = readLongBE(frame, p);
            p += 8;
        } else {
            rowId = -1L;
        }
        final int nFields = readWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        if (p + nullBytes > limit) throw new IOException("Truncated payload: missing nullBitmap");
        final int nullBitmapPos = p;
        p += nullBytes;
        for (int i = 0; i < nFields; i++) {
            final int targetField = wireIndexToTargetField[i];
            if (isNullBitSet(frame, nullBitmapPos, i)) {
                if (targetField >= 0) outRow.setField(targetField, null);
                continue; // no bytes on the wire for a null field - nothing to skip.
            }
            p = targetField >= 0
                    ? readValueIntoRow(i, frame, p, limit, outRow, targetField)
                    : skipValue(i, frame, p, limit);
        }
        return rowId;
    }

    /**
     * Same decode as {@link #readFramedRowInto}, but writes straight into a
     * {@link BinaryRowWriter} instead of a {@code GenericRowData}. Every fixed-width
     * field ({@code writeInt}/{@code writeLong}/{@code writeDouble}/...) lands in the
     * row's packed byte layout with no boxed wrapper object retained in the row - unlike
     * {@code GenericRowData}, which stores every field as a boxed {@code Object} in an
     * {@code Object[]} for the row's whole lifetime. String/decimal/timestamp fields
     * still go through the same intermediate {@code StringData}/{@code DecimalData}/
     * {@code TimestampData} construction {@link #castIfNeeded} already did for the
     * {@code GenericRowData} path, but that object is now transient - consumed and
     * discarded by the writer call on the spot - rather than retained until the row
     * itself is collected, which can be several batches later on a queued decode path
     * like this codec's callers use.
     *
     * <p>Row kind is intentionally not touched here, matching {@link #readFramedRowInto}:
     * callers derive it from their own per-row metadata and should call {@link
     * BinaryRowWriter#writeRowKind} themselves before {@link BinaryRowWriter#complete}.
     *
     * @return the frame's row id ({@code includeRowId} must be true)
     */
    public long readFramedRowIntoBinary(
            java.nio.ByteBuffer frame, int offset, BinaryRowWriter writer, int[] wireIndexToTargetField)
            throws IOException {
        if (readWireTypes == null) {
            throw new IOException("ExternalRuntimeBinaryCodec not configured for reading");
        }
        if (offset < 0 || offset > frame.limit() - Integer.BYTES) {
            throw new IOException("Truncated frame header");
        }
        final int frameLen = readIntBE(frame, offset);
        if (frameLen < 0 || frameLen > DEFAULT_MAX_FRAME_SIZE
                || frameLen > frame.limit() - offset - Integer.BYTES) {
            throw new IOException("Invalid frame length: " + frameLen);
        }
        int p = offset + Integer.BYTES;
        final int limit = p + frameLen;
        if (p + 4 > limit) throw new IOException("Truncated payload: missing operation");
        p += 4; // __op: the caller sets row kind from its own metadata, not the wire.
        final long rowId;
        if (includeRowId) {
            if (p + 8 > limit) throw new IOException("Truncated payload: missing row id");
            rowId = readLongBE(frame, p);
            p += 8;
        } else {
            rowId = -1L;
        }
        final int nFields = readWireTypes.length;
        final int nullBytes = (nFields + 7) >>> 3;
        if (p + nullBytes > limit) throw new IOException("Truncated payload: missing nullBitmap");
        final int nullBitmapPos = p;
        p += nullBytes;
        for (int i = 0; i < nFields; i++) {
            final int targetField = wireIndexToTargetField[i];
            if (isNullBitSet(frame, nullBitmapPos, i)) {
                if (targetField >= 0) writer.setNullAt(targetField);
                continue; // no bytes on the wire for a null field - nothing to skip.
            }
            p = targetField >= 0
                    ? readValueIntoWriter(i, frame, p, limit, writer, targetField)
                    : skipValue(i, frame, p, limit);
        }
        return rowId;
    }

    private int readValueIntoWriter(
            int i, java.nio.ByteBuffer buf, int p, int limit, BinaryRowWriter writer, int targetField)
            throws IOException {
        final WireType wt = readWireTypes[i];
        final LogicalType targetType = readTargetTypes[i];
        if (targetType == null) {
            throw new IOException(
                    "readFramedRowIntoBinary requires a known read target type for wire field " + i);
        }
        switch (wt) {
            case BOOL:
                if (p + 1 > limit) throw new IOException("Truncated BOOL");
                writeCastedValue(writer, targetField,
                        castIfNeeded(buf.get(p) != 0, readSourceTypes[i], targetType), targetType);
                return p + 1;
            case INT32:
                if (p + 4 > limit) throw new IOException("Truncated INT32");
                writeCastedValue(writer, targetField,
                        castIfNeeded(readIntBE(buf, p), readSourceTypes[i], targetType), targetType);
                return p + 4;
            case INT64:
            case TIMESTAMP_MILLIS:
                if (p + 8 > limit) throw new IOException("Truncated INT64");
                writeCastedValue(writer, targetField,
                        castIfNeeded(readLongBE(buf, p), readSourceTypes[i], targetType), targetType);
                return p + 8;
            case FLOAT32:
                if (p + 4 > limit) throw new IOException("Truncated FLOAT32");
                writeCastedValue(writer, targetField,
                        castIfNeeded(Float.intBitsToFloat(readIntBE(buf, p)), readSourceTypes[i], targetType),
                        targetType);
                return p + 4;
            case FLOAT64:
                if (p + 8 > limit) throw new IOException("Truncated FLOAT64");
                writeCastedValue(writer, targetField,
                        castIfNeeded(Double.longBitsToDouble(readLongBE(buf, p)), readSourceTypes[i], targetType),
                        targetType);
                return p + 8;
            case STRING: {
                if (p + 4 > limit) throw new IOException("Truncated STRING len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid STRING len: " + len);
                // Written straight into the writer's own backing segment inside
                // writeCastedValue -> writer.writeString, so - unlike the GenericRowData
                // path - this scratch array's contents don't need to outlive this call.
                final byte[] bytes = copyBytes(i, buf, p, len);
                final StringData sd = StringData.fromBytes(bytes, 0, len);
                writeCastedValue(writer, targetField, castIfNeeded(sd, readSourceTypes[i], targetType), targetType);
                return p + len;
            }
            case BYTES: {
                if (p + 4 > limit) throw new IOException("Truncated BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid BYTES len: " + len);
                writeCastedValue(writer, targetField,
                        castIfNeeded(copyBytes(i, buf, p, len), readSourceTypes[i], targetType), targetType);
                return p + len;
            }
            case DECIMAL_UNSCALED_I64:
                if (p + 8 > limit) throw new IOException("Truncated DECIMAL_UNSCALED_I64");
                writeCastedValue(writer, targetField,
                        castIfNeeded(readLongBE(buf, p), readSourceTypes[i], targetType), targetType);
                return p + 8;
            case DECIMAL_UNSCALED_BYTES: {
                if (p + 4 > limit) throw new IOException("Truncated DECIMAL_UNSCALED_BYTES len");
                final int len = readIntBE(buf, p);
                p += 4;
                if (len < 0 || p + len > limit) throw new IOException("Invalid DECIMAL bytes len: " + len);
                writeCastedValue(writer, targetField,
                        castIfNeeded(copyBytes(i, buf, p, len), readSourceTypes[i], targetType), targetType);
                return p + len;
            }
            default:
                throw new IOException("Unsupported read wire type: " + wt);
        }
    }

    /**
     * Writes a value already cast to {@code targetType}'s runtime representation (i.e.
     * already run through {@link #castIfNeeded}) into a {@link BinaryRowWriter} field.
     * Public so callers decoding straight into a {@code BinaryRowWriter} (see {@link
     * #readFramedRowIntoBinary}) can also use it for fields that don't come off this
     * codec's wire at all - e.g. a value pulled via {@code RowData.createFieldGetter}
     * and passed through unchanged - since Flink's internal per-type representation
     * (the same one {@code RowData.createFieldGetter} itself produces) is standardized
     * regardless of where the value came from.
     */
    public static void writeCastedValue(BinaryRowWriter writer, int pos, Object value, LogicalType targetType) {
        switch (targetType.getTypeRoot()) {
            case BOOLEAN:
                writer.writeBoolean(pos, (Boolean) value);
                return;
            case TINYINT:
                writer.writeByte(pos, (Byte) value);
                return;
            case SMALLINT:
                writer.writeShort(pos, (Short) value);
                return;
            case INTEGER:
            case DATE:
            case TIME_WITHOUT_TIME_ZONE:
                writer.writeInt(pos, (Integer) value);
                return;
            case BIGINT:
                writer.writeLong(pos, (Long) value);
                return;
            case FLOAT:
                writer.writeFloat(pos, (Float) value);
                return;
            case DOUBLE:
                writer.writeDouble(pos, (Double) value);
                return;
            case CHAR:
            case VARCHAR:
                writer.writeString(pos, (StringData) value);
                return;
            case BINARY:
            case VARBINARY:
                writer.writeBinary(pos, (byte[]) value);
                return;
            case TIMESTAMP_WITHOUT_TIME_ZONE:
                writer.writeTimestamp(pos, (TimestampData) value, ((TimestampType) targetType).getPrecision());
                return;
            case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
                writer.writeTimestamp(
                        pos, (TimestampData) value, ((LocalZonedTimestampType) targetType).getPrecision());
                return;
            case DECIMAL: {
                final DecimalType dt = (DecimalType) targetType;
                writer.writeDecimal(pos, (DecimalData) value, dt.getPrecision());
                return;
            }
            default:
                throw new IllegalStateException("Unsupported target type for binary row write: " + targetType);
        }
    }

    private static Object castIfNeeded(Object value, LogicalType sourceType, LogicalType targetType) {
        if (value == null || targetType == null) {
            return value;
        }
        if (sourceType == null) {
            // if no source, interpret based on target (mostly for decimal/timestamp)
            return materializeFromWire(value, targetType);
        }

        final LogicalTypeRoot sr = sourceType.getTypeRoot();
        final LogicalTypeRoot tr = targetType.getTypeRoot();

        // normalize decimals if needed
        if (sr == tr
                || (isStringRoot(sr) && isStringRoot(tr))
                || (isTimestampRoot(sr) && isTimestampRoot(tr))) {

            if (tr == LogicalTypeRoot.DECIMAL) {
                final DecimalType dt = (DecimalType) targetType;
                // The wire bytes have the scale of the source schema; materializing with
                // targetType would silently interpret those bytes at the result scale
                // before the cast.
                final DecimalType st = (DecimalType) sourceType;
                // DecimalDataUtils.castFrom always round-trips through BigDecimal, even
                // when nothing actually needs rescaling - confirmed in Flink source: it's
                // unconditional, no same-precision/same-scale fast path. Worse, whenever
                // the target precision exceeds the 18-digit compact threshold (as with
                // this codec's packed-decimal wire types, wired narrow specifically to
                // stay compact, cast up to a wider declared result type), the resulting
                // DecimalData is non-compact for the rest of its life, not just for this
                // call. That widening is unavoidable when the target's declared precision
                // demands it - but going through materializeFromWire first builds a
                // compact DecimalData(source-scale) only to immediately discard it inside
                // castFrom's toBigDecimal() call. When value is already the raw unscaled
                // long (the DECIMAL_UNSCALED_I64 wire case) and the scale isn't changing,
                // build the BigDecimal directly and skip that throwaway allocation -
                // bit-identical result, one fewer object per row.
                if (value instanceof Long && dt.getScale() == st.getScale()) {
                    final BigDecimal bd = BigDecimal.valueOf((Long) value, st.getScale());
                    return DecimalData.fromBigDecimal(bd, dt.getPrecision(), dt.getScale());
                }
                final DecimalData d = (DecimalData) materializeFromWire(value, sourceType);
                return DecimalDataUtils.castFrom(d, dt.getPrecision(), dt.getScale());
            }
            return materializeFromWire(value, targetType);
        }

        return castValue(materializeFromWire(value, sourceType), sourceType, targetType);
    }

    private static Object castValue(Object value, LogicalType sourceType, LogicalType targetType) {
        final LogicalTypeRoot sourceRoot = sourceType.getTypeRoot();
        final LogicalTypeRoot targetRoot = targetType.getTypeRoot();

        if (targetRoot == LogicalTypeRoot.DECIMAL) {
            final DecimalType dt = (DecimalType) targetType;
            if (value instanceof DecimalData) {
                return DecimalDataUtils.castFrom((DecimalData) value, dt.getPrecision(), dt.getScale());
            }
            if (value instanceof StringData) {
                return DecimalDataUtils.castFrom(value.toString(), dt.getPrecision(), dt.getScale());
            }
            if (value instanceof Number) {
                if (sourceRoot == LogicalTypeRoot.FLOAT || sourceRoot == LogicalTypeRoot.DOUBLE) {
                    return DecimalDataUtils.castFrom(((Number) value).doubleValue(), dt.getPrecision(), dt.getScale());
                }
                return DecimalDataUtils.castFrom(((Number) value).longValue(), dt.getPrecision(), dt.getScale());
            }
        }

        if (sourceRoot == LogicalTypeRoot.DECIMAL) {
            final DecimalData dec = (DecimalData) value;
            final long integral = DecimalDataUtils.castToIntegral(dec);
            switch (targetRoot) {
                case BIGINT:
                    return integral;
                case INTEGER:
                    return (int) integral;
                case SMALLINT:
                    return (short) integral;
                case TINYINT:
                    return (byte) integral;
                case FLOAT:
                    return (float) DecimalDataUtils.doubleValue(dec);
                case DOUBLE:
                    return DecimalDataUtils.doubleValue(dec);
                default:
                    break;
            }
        }

        if (value instanceof Number) {
            final Number number = (Number) value;
            switch (targetRoot) {
                case BIGINT:
                    return number.longValue();
                case INTEGER:
                    return number.intValue();
                case SMALLINT:
                    return number.shortValue();
                case TINYINT:
                    return number.byteValue();
                case FLOAT:
                    return number.floatValue();
                case DOUBLE:
                    return number.doubleValue();
                default:
                    break;
            }
        }

        if (isStringRoot(targetRoot)) {
            if (value instanceof StringData) {
                return value;
            }
            return StringData.fromString(String.valueOf(value));
        }

        throw new TableException(
                "ExternalRuntimeOperator cannot cast "
                        + sourceType.asSerializableString()
                        + " to "
                        + targetType.asSerializableString());
    }

    private static Object materializeFromWire(Object value, LogicalType type) {
        final LogicalTypeRoot root = type.getTypeRoot();

        switch (root) {
            case TIMESTAMP_WITHOUT_TIME_ZONE:
            case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
                return TimestampData.fromEpochMillis((Long) value);
            case DECIMAL: {
                final DecimalType dt = (DecimalType) type;
                if (value instanceof DecimalData) {
                    return value;
                }
                if (value instanceof Long) {
                    return DecimalData.fromUnscaledLong((Long) value, dt.getPrecision(), dt.getScale());
                }
                if (value instanceof byte[]) {
                    final BigInteger bi = new BigInteger((byte[]) value);
                    final BigDecimal bd = new BigDecimal(bi, dt.getScale());
                    final DecimalData dd = DecimalData.fromBigDecimal(bd, dt.getPrecision(), dt.getScale());
                    if (dd == null) {
                        throw new TableException(
                                "Failed to deserialize DECIMAL(" + dt.getPrecision() + "," + dt.getScale() + ")");
                    }
                    return dd;
                }
                if (value instanceof StringData) {
                    return DecimalDataUtils.castFrom(value.toString(), dt.getPrecision(), dt.getScale());
                }
                break;
            }
            default:
                break;
        }
        return value;
    }

    static WireType wireTypeFor(LogicalType type) {
        if (type == null) {
            throw new TableException("Cannot determine wire type for null logical type");
        }
        switch (type.getTypeRoot()) {
            case BOOLEAN:
                return WireType.BOOL;
            case TINYINT:
            case SMALLINT:
            case INTEGER:
            case DATE:
            case TIME_WITHOUT_TIME_ZONE:
                return WireType.INT32;
            case BIGINT:
                return WireType.INT64;
            case FLOAT:
                return WireType.FLOAT32;
            case DOUBLE:
                return WireType.FLOAT64;
            case CHAR:
            case VARCHAR:
                return WireType.STRING;
            case TIMESTAMP_WITHOUT_TIME_ZONE:
            case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
                return WireType.TIMESTAMP_MILLIS;
            case DECIMAL: {
                final DecimalType dt = (DecimalType) type;
                return dt.getPrecision() <= 18 ? WireType.DECIMAL_UNSCALED_I64 : WireType.DECIMAL_UNSCALED_BYTES;
            }
            case BINARY:
            case VARBINARY:
                return WireType.BYTES;
            default:
                throw new TableException(
                        "Unsupported logical type for external runtime wire format: "
                                + type.asSerializableString());
        }
    }

    public static boolean isStringRoot(LogicalTypeRoot root) {
        return root == LogicalTypeRoot.CHAR || root == LogicalTypeRoot.VARCHAR;
    }

    public static boolean isTimestampRoot(LogicalTypeRoot root) {
        return root == LogicalTypeRoot.TIMESTAMP_WITHOUT_TIME_ZONE
                || root == LogicalTypeRoot.TIMESTAMP_WITH_LOCAL_TIME_ZONE;
    }

    static int rowKindToOp(RowKind kind) {
        switch (kind) {
            case INSERT:
                return 0;
            case UPDATE_AFTER:
                return 1;
            case UPDATE_BEFORE:
                return 2;
            case DELETE:
                return 3;
            default:
                return 127;
        }
    }

    static RowKind opToRowKind(int op, RowKind fallback) {
        switch (op) {
            case 0:
                return RowKind.INSERT;
            case 1:
                return RowKind.UPDATE_AFTER;
            case 2:
                return RowKind.UPDATE_BEFORE;
            case 3:
                return RowKind.DELETE;
            default:
                return fallback;
        }
    }

    private long toUnscaledLong(
            RowData row,
            int pos,
            LogicalTypeRoot sourceRoot,
            int targetPrecision,
            int targetScale,
            int sourcePrecision,
            int sourceScale) {

        if (sourceRoot == LogicalTypeRoot.DECIMAL) {
            final DecimalData dec = row.getDecimal(pos, sourcePrecision, sourceScale);
            if (sourceScale == targetScale) {
                return dec.toUnscaledLong();
            }
            try {
                final long unscaled = dec.toUnscaledLong();
                return rescaleLong(unscaled, targetScale - sourceScale);
            } catch (ArithmeticException e) {
                final BigDecimal bd = dec.toBigDecimal();
                final BigDecimal scaled = bd.scale() == targetScale
                        ? bd
                        : bd.setScale(targetScale, RoundingMode.HALF_UP);
                return scaled.unscaledValue().longValueExact();
            }
        }

        switch (sourceRoot) {
            case BIGINT:
                return rescaleLongExact(row.getLong(pos), targetScale);
            case INTEGER:
            case DATE:
            case TIME_WITHOUT_TIME_ZONE:
                return rescaleLongExact(row.getInt(pos), targetScale);
            case SMALLINT:
                return rescaleLongExact(row.getShort(pos), targetScale);
            case TINYINT:
                return rescaleLongExact(row.getByte(pos), targetScale);
            case FLOAT:
                return bigDecimalToUnscaledLong(BigDecimal.valueOf(row.getFloat(pos)), targetScale);
            case DOUBLE:
                return bigDecimalToUnscaledLong(BigDecimal.valueOf(row.getDouble(pos)), targetScale);
            default:
                throw new TableException(
                        "Cannot cast " + sourceRoot + " to DECIMAL(" + targetPrecision + "," + targetScale + ")");
        }
    }

    private BigInteger toUnscaledBigInt(
            RowData row,
            int pos,
            LogicalTypeRoot sourceRoot,
            int targetPrecision,
            int targetScale,
            int sourcePrecision,
            int sourceScale) {

        if (sourceRoot == LogicalTypeRoot.DECIMAL) {
            final DecimalData dec = row.getDecimal(pos, sourcePrecision, sourceScale);
            if (sourceScale == targetScale) {
                return new BigInteger(dec.toUnscaledBytes());
            }
            final BigInteger unscaled = new BigInteger(dec.toUnscaledBytes());
            return rescaleBigInt(unscaled, targetScale - sourceScale);
        }

        switch (sourceRoot) {
            case BIGINT:
                return rescaleBigInt(BigInteger.valueOf(row.getLong(pos)), targetScale);
            case INTEGER:
            case DATE:
            case TIME_WITHOUT_TIME_ZONE:
                return rescaleBigInt(BigInteger.valueOf(row.getInt(pos)), targetScale);
            case SMALLINT:
                return rescaleBigInt(BigInteger.valueOf(row.getShort(pos)), targetScale);
            case TINYINT:
                return rescaleBigInt(BigInteger.valueOf(row.getByte(pos)), targetScale);
            case FLOAT:
                return bigDecimalToUnscaledBigInt(BigDecimal.valueOf(row.getFloat(pos)), targetScale);
            case DOUBLE:
                return bigDecimalToUnscaledBigInt(BigDecimal.valueOf(row.getDouble(pos)), targetScale);
            default:
                throw new TableException(
                        "Cannot cast " + sourceRoot + " to DECIMAL(" + targetPrecision + "," + targetScale + ")");
        }
    }

    public static final class RowWithId {
        public final long rowId;
        public final RowData row;

        RowWithId(long rowId, RowData row) {
            this.rowId = rowId;
            this.row = row;
        }
    }

    private byte[] copyBytes(int fieldIndex, byte[] buf, int p, int len) {
        final byte[] out;
        if (reuseObjects) {
            byte[] existing = reuseBytes[fieldIndex];
            if (existing == null || existing.length != len) {
                existing = new byte[len];
                reuseBytes[fieldIndex] = existing;
            }
            out = existing;
        } else {
            out = new byte[len];
        }
        System.arraycopy(buf, p, out, 0, len);
        return out;
    }

    private byte[] copyBytes(int fieldIndex, java.nio.ByteBuffer buf, int p, int len) {
        final byte[] out;
        if (reuseObjects) {
            byte[] existing = reuseBytes[fieldIndex];
            if (existing == null || existing.length != len) {
                existing = new byte[len];
                reuseBytes[fieldIndex] = existing;
            }
            out = existing;
        } else {
            out = new byte[len];
        }
        // Absolute bulk get avoids allocating a duplicate ByteBuffer for every
        // variable-length field while preserving the caller's position.
        buf.get(p, out, 0, len);
        return out;
    }

    private void ensureReadBuf(int len) {
        if (frameReadBuf.length >= len)
            return;
        int n = frameReadBuf.length;
        while (n < len)
            n <<= 1;
        frameReadBuf = new byte[n];
    }

    private static void setNullBit(byte[] bitmapHolder, int bitmapPos, int fieldIndex) {
        final int byteIndex = bitmapPos + (fieldIndex >>> 3);
        final int bit = fieldIndex & 7;
        bitmapHolder[byteIndex] |= (byte) (1 << bit);
    }

    private static boolean isNullBitSet(byte[] payload, int bitmapPos, int fieldIndex) {
        final int byteIndex = bitmapPos + (fieldIndex >>> 3);
        final int bit = fieldIndex & 7;
        final int b = payload[byteIndex] & 0xFF;
        return (b & (1 << bit)) != 0;
    }

    private static boolean isNullBitSet(java.nio.ByteBuffer payload, int bitmapPos, int fieldIndex) {
        final int byteIndex = bitmapPos + (fieldIndex >>> 3);
        final int bit = fieldIndex & 7;
        return (payload.get(byteIndex) & (1 << bit)) != 0;
    }

    private static void readFully(InputStream in, byte[] b, int off, int len) throws IOException {
        int n = 0;
        while (n < len) {
            final int r = in.read(b, off + n, len - n);
            if (r < 0) {
                throw new EOFException("Truncated frame");
            }
            n += r;
        }
    }

    private static int readIntBE(InputStream in) throws IOException {
        final int b1 = in.read();
        final int b2 = in.read();
        final int b3 = in.read();
        final int b4 = in.read();
        if ((b1 | b2 | b3 | b4) < 0) {
            throw new EOFException("EOF while reading int32");
        }
        return (b1 << 24) | (b2 << 16) | (b3 << 8) | (b4);
    }

    private static int readIntBE(byte[] buf, int p) {
        return ((buf[p] & 0xff) << 24)
                | ((buf[p + 1] & 0xff) << 16)
                | ((buf[p + 2] & 0xff) << 8)
                | (buf[p + 3] & 0xff);
    }

    private static int readIntBE(java.nio.ByteBuffer buf, int p) {
        return ((buf.get(p) & 0xff) << 24)
                | ((buf.get(p + 1) & 0xff) << 16)
                | ((buf.get(p + 2) & 0xff) << 8)
                | (buf.get(p + 3) & 0xff);
    }

    private static long readLongBE(byte[] buf, int p) {
        return ((long) (buf[p] & 0xff) << 56)
                | ((long) (buf[p + 1] & 0xff) << 48)
                | ((long) (buf[p + 2] & 0xff) << 40)
                | ((long) (buf[p + 3] & 0xff) << 32)
                | ((long) (buf[p + 4] & 0xff) << 24)
                | ((long) (buf[p + 5] & 0xff) << 16)
                | ((long) (buf[p + 6] & 0xff) << 8)
                | (buf[p + 7] & 0xff);
    }

    private static long readLongBE(java.nio.ByteBuffer buf, int p) {
        return ((long) (buf.get(p) & 0xff) << 56)
                | ((long) (buf.get(p + 1) & 0xff) << 48)
                | ((long) (buf.get(p + 2) & 0xff) << 40)
                | ((long) (buf.get(p + 3) & 0xff) << 32)
                | ((long) (buf.get(p + 4) & 0xff) << 24)
                | ((long) (buf.get(p + 5) & 0xff) << 16)
                | ((long) (buf.get(p + 6) & 0xff) << 8)
                | (buf.get(p + 7) & 0xff);
    }

    private static void writeIntBE(OutputStream out, int v) throws IOException {
        out.write((v >>> 24) & 0xff);
        out.write((v >>> 16) & 0xff);
        out.write((v >>> 8) & 0xff);
        out.write(v & 0xff);
    }

    private static long[] initPow10() {
        final long[] pow = new long[19];
        pow[0] = 1L;
        for (int i = 1; i < pow.length; i++) {
            pow[i] = pow[i - 1] * 10L;
        }
        return pow;
    }

    private static long rescaleLongExact(long value, int targetScale) {
        if (targetScale == 0) {
            return value;
        }
        if (targetScale > 0 && targetScale < POW10.length) {
            return Math.multiplyExact(value, POW10[targetScale]);
        }
        return bigDecimalToUnscaledLong(BigDecimal.valueOf(value), targetScale);
    }

    private static long rescaleLong(long unscaled, int scaleDiff) {
        if (scaleDiff == 0) {
            return unscaled;
        }
        if (scaleDiff > 0) {
            if (scaleDiff >= POW10.length) {
                throw new ArithmeticException("scaleDiff too large for long: " + scaleDiff);
            }
            return Math.multiplyExact(unscaled, POW10[scaleDiff]);
        }
        final int diff = -scaleDiff;
        if (diff >= POW10.length) {
            throw new ArithmeticException("scaleDiff too large for long: " + scaleDiff);
        }
        final long divisor = POW10[diff];
        final long abs = Math.abs(unscaled);
        long quotient = abs / divisor;
        final long remainder = abs - quotient * divisor;
        final long half = divisor >>> 1;
        final boolean increment = remainder > half || (remainder == half && (divisor & 1L) == 0);
        if (increment) {
            quotient++;
        }
        return unscaled < 0 ? -quotient : quotient;
    }

    private static BigInteger rescaleBigInt(BigInteger unscaled, int scaleDiff) {
        if (scaleDiff == 0) {
            return unscaled;
        }
        if (scaleDiff > 0) {
            return unscaled.multiply(BigInteger.TEN.pow(scaleDiff));
        }
        final int diff = -scaleDiff;
        final BigInteger divisor = BigInteger.TEN.pow(diff);
        final BigInteger[] qr = unscaled.divideAndRemainder(divisor);
        if (qr[1].signum() == 0) {
            return qr[0];
        }
        final BigInteger twiceRem = qr[1].abs().shiftLeft(1);
        if (twiceRem.compareTo(divisor) >= 0) {
            return qr[0].add(BigInteger.valueOf(unscaled.signum()));
        }
        return qr[0];
    }

    private static long bigDecimalToUnscaledLong(BigDecimal bd, int scale) {
        final BigDecimal scaled = bd.scale() == scale ? bd : bd.setScale(scale, RoundingMode.HALF_UP);
        return scaled.unscaledValue().longValueExact();
    }

    private static BigInteger bigDecimalToUnscaledBigInt(BigDecimal bd, int scale) {
        final BigDecimal scaled = bd.scale() == scale ? bd : bd.setScale(scale, RoundingMode.HALF_UP);
        return scaled.unscaledValue();
    }

    // ---------------------------------------------------------------------
    // Small reusable output buffer
    // ---------------------------------------------------------------------

    private static final class GrowableBuffer {
        private byte[] buf;
        private int pos;

        GrowableBuffer(int initial) {
            this.buf = new byte[Math.max(256, initial)];
            this.pos = 0;
        }

        void reset() {
            pos = 0;
        }

        int position() {
            return pos;
        }

        byte[] buf() {
            return buf;
        }

        void ensureCapacity(int additional) {
            final int need = pos + additional;
            if (need <= buf.length)
                return;
            int n = buf.length;
            while (n < need)
                n <<= 1;
            final byte[] nb = new byte[n];
            System.arraycopy(buf, 0, nb, 0, pos);
            buf = nb;
        }

        void putByte(byte v) {
            ensureCapacity(1);
            buf[pos++] = v;
        }

        void putBytes(byte[] bytes) {
            ensureCapacity(bytes.length);
            System.arraycopy(bytes, 0, buf, pos, bytes.length);
            pos += bytes.length;
        }

        void putIntBE(int v) {
            ensureCapacity(4);
            buf[pos++] = (byte) (v >>> 24);
            buf[pos++] = (byte) (v >>> 16);
            buf[pos++] = (byte) (v >>> 8);
            buf[pos++] = (byte) (v);
        }

        void putLongBE(long v) {
            ensureCapacity(8);
            buf[pos++] = (byte) (v >>> 56);
            buf[pos++] = (byte) (v >>> 48);
            buf[pos++] = (byte) (v >>> 40);
            buf[pos++] = (byte) (v >>> 32);
            buf[pos++] = (byte) (v >>> 24);
            buf[pos++] = (byte) (v >>> 16);
            buf[pos++] = (byte) (v >>> 8);
            buf[pos++] = (byte) (v);
        }
    }
}
