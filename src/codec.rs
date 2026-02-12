use anyhow::{bail, Context, Result};
use std::io::{ErrorKind, Read, Write};

use crate::config::{FieldType, SessionConfig};
use crate::constants::DEFAULT_MAX_FRAME_SIZE;
use crate::udf::InputColumn;
use crate::values::{v_to_bool, v_to_decimal_i128, v_to_f64, v_to_i64, v_to_string, V};

#[derive(Debug)]
pub struct OutputBlock {
    pub op: i32,
    pub row_id: i64,
    pub row: Option<Vec<V>>,
}

pub fn read_framed_payload<R: Read>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let len = match read_i32_be_stream_opt(reader)? {
        Some(v) => v,
        None => return Ok(None),
    };
    if len < 0 {
        bail!("invalid negative frame length: {}", len);
    }

    let len: usize = len.try_into().context("frame length overflow")?;
    if len > DEFAULT_MAX_FRAME_SIZE {
        bail!("frame length {} exceeds max_frame_size {}", len, DEFAULT_MAX_FRAME_SIZE);
    }

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).context("read frame payload")?;
    Ok(Some(payload))
}

fn skip_len_bytes(buf: &[u8], p: &mut usize, label: &str) -> Result<()> {
    let len = read_i32_be(buf, p)?;
    if len < 0 {
        bail!("negative length for {}", label);
    }
    let len = len as usize;
    if *p + len > buf.len() {
        bail!(
            "truncated {} len-bytes: need {}, have {}",
            label,
            len,
            buf.len().saturating_sub(*p)
        );
    }
    *p += len;
    Ok(())
}

fn skip_field(buf: &[u8], p: &mut usize, t: &FieldType) -> Result<()> {
    match t {
        FieldType::Boolean => {
            read_u8(buf, p)?;
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            read_i64_be(buf, p)?;
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 | FieldType::Date => {
            read_i32_be(buf, p)?;
        }
        FieldType::Float32 => {
            read_u32_be(buf, p)?;
        }
        FieldType::Float64 => {
            read_u64_be(buf, p)?;
        }
        FieldType::String | FieldType::Bytes | FieldType::Unknown(_) => {
            skip_len_bytes(buf, p, "string/bytes")?;
        }
        FieldType::Decimal { precision, .. } => {
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                read_i64_be(buf, p)?;
            } else {
                skip_len_bytes(buf, p, "decimal bytes")?;
            }
        }
        FieldType::DecimalUnscaledI64 => {
            read_i64_be(buf, p)?;
        }
        FieldType::DecimalUnscaledBytes => {
            skip_len_bytes(buf, p, "decimal bytes")?;
        }
    }
    if *p > buf.len() {
        bail!("skip overran buffer");
    }
    Ok(())
}

pub fn write_output_blocks<W: Write>(
    writer: &mut W,
    blocks: &[OutputBlock],
    session: &SessionConfig,
    payload: &mut Vec<u8>,
) -> Result<()> {
    // Send one batched frame per input row (per OutputBlock)
    for block in blocks {
        payload.clear();

        // Count: 1 if this input row produced output, 0 if filtered
        let count = if block.row.is_some() { 1 } else { 0 };

        // header section, no frame length prefix
        write_i32_be_vec(payload, block.op);
        write_i64_be_vec(payload, block.row_id);
        payload.push(0u8); // null bitmap for count field (not null)
        write_i32_be_vec(payload, count);

        // row section if present, no frame length prefix
        if let Some(row) = block.row.as_ref() {
            // Row header: __op and __rowId
            write_i32_be_vec(payload, block.op);
            write_i64_be_vec(payload, block.row_id);

            // null bitmap for payload fields (post payload only)
            let n_fields = session.post_payload_positions.len();
            let null_bytes = (n_fields + 7) >> 3;
            let null_pos = payload.len();
            payload.resize(null_pos + null_bytes, 0);

            // encode values in postFields order (excluding __op/__rowId)
            for (i, (&row_pos, ftype)) in session
                .post_payload_positions
                .iter()
                .zip(session.post_payload_types.iter())
                .enumerate()
            {
                let v = row.get(row_pos).unwrap_or(&V::Null);
                if matches!(v, V::Null) {
                    set_null_bit(&mut payload[null_pos..null_pos + null_bytes], i);
                    continue;
                }
                encode_field(payload, v, ftype)
                    .with_context(|| format!("encode post field {} at row_pos {}", i, row_pos))?;
            }
        }

        if payload.len() > DEFAULT_MAX_FRAME_SIZE {
            bail!(
                "batch payload exceeds max_frame_size: {} > {}",
                payload.len(),
                DEFAULT_MAX_FRAME_SIZE
            );
        }

        // Write entire batch as one frame: [length][payload]
        write_i32_be_stream(writer, payload.len() as i32)?;
        writer.write_all(payload.as_slice()).context("write batch payload")?;
    }
    Ok(())
}


pub fn encode_field(out: &mut Vec<u8>, v: &V, t: &FieldType) -> Result<()> {
    match t {
        FieldType::Boolean => {
            let b = v_to_bool(v)?;
            out.push(if b { 1 } else { 0 });
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            let x = v_to_i64(v)?;
            write_i64_be_vec(out, x);
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let x = v_to_i64(v)?;
            write_i32_be_vec(out, x as i32);
        }
        FieldType::Float32 => {
            let x = v_to_f64(v)? as f32;
            write_u32_be_vec(out, x.to_bits());
        }
        FieldType::Float64 => {
            let x = v_to_f64(v)?;
            write_u64_be_vec(out, x.to_bits());
        }
        FieldType::String | FieldType::Unknown(_) => {
            let s = v_to_string(v)?;
            let bytes = s.as_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Bytes => {
            let bytes = match v {
                V::Bytes(b) => b.as_slice(),
                V::String(s) => s.as_bytes(),
                _ => bail!("cannot encode BYTES from {:?}", v),
            };
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Date => {
            // wire expects int32 days since epoch.
            let millis = v_to_i64(v)?;
            let days = (millis / 86_400_000) as i32;
            write_i32_be_vec(out, days);
        }
        FieldType::Decimal { precision, .. } => {
            let unscaled = v_to_decimal_i128(v)?;
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                let as_i64 =
                    i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
                write_i64_be_vec(out, as_i64);
            } else {
                let bytes = i128_to_twos_complement_be_minimal(unscaled);
                write_i32_be_vec(out, bytes.len() as i32);
                out.extend_from_slice(&bytes);
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = v_to_decimal_i128(v)?;
            let as_i64 =
                i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
            write_i64_be_vec(out, as_i64);
        }
        FieldType::DecimalUnscaledBytes => {
            let unscaled = v_to_decimal_i128(v)?;
            let bytes = i128_to_twos_complement_be_minimal(unscaled);
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
    }
    Ok(())
}

// bitmap helpers (LSB-first)
pub fn set_null_bit(bitmap: &mut [u8], field_index: usize) {
    let byte = field_index >> 3;
    let bit = field_index & 7;
    bitmap[byte] |= 1u8 << bit;
}

fn is_null_bit_set(payload: &[u8], bitmap_pos: usize, field_index: usize) -> bool {
    let byte = bitmap_pos + (field_index >> 3);
    let bit = field_index & 7;
    (payload[byte] & (1u8 << bit)) != 0
}

// endian + bounds helpers
fn read_u8(buf: &[u8], p: &mut usize) -> Result<u8> {
    if *p + 1 > buf.len() {
        bail!("truncated u8");
    }
    let v = buf[*p];
    *p += 1;
    Ok(v)
}

fn read_u32_be(buf: &[u8], p: &mut usize) -> Result<u32> {
    if *p + 4 > buf.len() {
        bail!("truncated u32");
    }
    let b = &buf[*p..*p + 4];
    *p += 4;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64_be(buf: &[u8], p: &mut usize) -> Result<u64> {
    if *p + 8 > buf.len() {
        bail!("truncated u64");
    }
    let b = &buf[*p..*p + 8];
    *p += 8;
    Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

fn read_i32_be(buf: &[u8], p: &mut usize) -> Result<i32> {
    Ok(read_u32_be(buf, p)? as i32)
}

fn read_i64_be(buf: &[u8], p: &mut usize) -> Result<i64> {
    Ok(read_u64_be(buf, p)? as i64)
}

fn read_len_bytes(buf: &[u8], p: &mut usize) -> Result<Vec<u8>> {
    let len = read_i32_be(buf, p)?;
    if len < 0 {
        bail!("negative length");
    }
    let len = len as usize;
    if *p + len > buf.len() {
        bail!(
            "truncated len-bytes: need {}, have {}",
            len,
            buf.len().saturating_sub(*p)
        );
    }
    let out = buf[*p..*p + len].to_vec();
    *p += len;
    Ok(out)
}

pub fn write_u32_be_vec(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub fn write_u64_be_vec(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub fn write_i32_be_vec(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub fn write_i64_be_vec(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub fn write_i32_be_stream<W: Write>(w: &mut W, v: i32) -> Result<()> {
    w.write_all(&v.to_be_bytes()).context("write i32 be")
}

pub fn read_i32_be_stream<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read i32 be")?;
    Ok(i32::from_be_bytes(b))
}

fn read_i32_be_stream_opt<R: Read>(r: &mut R) -> Result<Option<i32>> {
    let mut b = [0u8; 4];
    match r.read_exact(&mut b) {
        Ok(()) => Ok(Some(i32::from_be_bytes(b))),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e).context("read i32 be opt"),
    }
}

// DECIMAL helpers: Java BigInteger.toByteArray() compatible representation
pub fn i128_to_twos_complement_be_minimal(v: i128) -> Vec<u8> {
    let bytes = v.to_be_bytes(); // 16 bytes
    let mut start = 0usize;
    while start < 15 {
        let b0 = bytes[start];
        let b1 = bytes[start + 1];
        // drop leading 0x00 if next byte sign bit is 0, or leading 0xFF if next sign bit is 1
        if (b0 == 0x00 && (b1 & 0x80) == 0x00) || (b0 == 0xFF && (b1 & 0x80) == 0x80) {
            start += 1;
            continue;
        }
        break;
    }
    bytes[start..].to_vec()
}

fn twos_complement_be_to_i128(bytes: &[u8]) -> Result<i128> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 16 {
        bail!("too many bytes for i128: {}", bytes.len());
    }
    let sign_extend = if (bytes[0] & 0x80) != 0 { 0xFFu8 } else { 0x00u8 };
    let mut out = [sign_extend; 16];
    let start = 16 - bytes.len();
    out[start..].copy_from_slice(bytes);
    Ok(i128::from_be_bytes(out))
}

// columnar batch: decode wire format straight into columns

pub struct ColumnarBatch {
    pub ops: Vec<i32>,
    pub row_ids: Vec<i64>,
    pub columns: Vec<InputColumn>,
}

impl ColumnarBatch {
    pub fn new(types: &[FieldType], capacity: usize) -> Self {
        let columns = types.iter().map(|t| new_empty_column(t, capacity)).collect();
        ColumnarBatch {
            ops: Vec::with_capacity(capacity),
            row_ids: Vec::with_capacity(capacity),
            columns,
        }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.row_ids.clear();
        for col in &mut self.columns {
            clear_column(col);
        }
    }
}

pub fn read_framed_payload_into<R: Read>(
    reader: &mut R,
    scratch: &mut Vec<u8>,
) -> Result<bool> {
    let len = match read_i32_be_stream_opt(reader)? {
        Some(v) => v,
        None => return Ok(false),
    };
    if len < 0 {
        bail!("invalid negative frame length: {}", len);
    }
    let len: usize = len.try_into().context("frame length overflow")?;
    if len > DEFAULT_MAX_FRAME_SIZE {
        bail!("frame length {} exceeds max_frame_size {}", len, DEFAULT_MAX_FRAME_SIZE);
    }
    scratch.resize(len, 0);
    reader.read_exact(scratch).context("read frame payload")?;
    Ok(true)
}

pub fn decode_payload_into_batch(
    payload: &[u8],
    batch: &mut ColumnarBatch,
    types: &[FieldType],
    needed: &[bool],
) -> Result<()> {
    let mut p = 0usize;
    if payload.len() < 12 {
        bail!("truncated payload: need at least 12 bytes for op+rowId");
    }
    let op = read_i32_be(payload, &mut p)?;
    let row_id = read_i64_be(payload, &mut p)?;
    batch.ops.push(op);
    batch.row_ids.push(row_id);

    let n_fields = types.len();
    let null_bytes = (n_fields + 7) >> 3;
    if p + null_bytes > payload.len() {
        bail!("truncated payload: missing nullBitmap");
    }
    let null_pos = p;
    p += null_bytes;

    for (i, ftype) in types.iter().enumerate() {
        let needed_flag = needed.get(i).copied().unwrap_or(true);
        if is_null_bit_set(payload, null_pos, i) {
            if needed_flag {
                push_null_to_column(&mut batch.columns[i]);
            }
            continue;
        }
        if needed_flag {
            decode_field_into_column(payload, &mut p, ftype, &mut batch.columns[i])?;
        } else {
            skip_field(payload, &mut p, ftype)?;
        }
    }

    if p > payload.len() {
        bail!("payload decode overran buffer");
    }
    Ok(())
}

fn new_empty_column(ftype: &FieldType, capacity: usize) -> InputColumn {
    match ftype {
        FieldType::String | FieldType::Bytes | FieldType::Unknown(_) => {
            InputColumn::String(Vec::with_capacity(capacity))
        }
        FieldType::Boolean => InputColumn::Bool {
            values: Vec::with_capacity(capacity),
            is_null: None,
        },
        FieldType::Int64 | FieldType::TimestampMillis | FieldType::Date => InputColumn::I64 {
            values: Vec::with_capacity(capacity),
            is_null: None,
        },
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => InputColumn::I32 {
            values: Vec::with_capacity(capacity),
            is_null: None,
        },
        FieldType::Float64 => InputColumn::F64 {
            values: Vec::with_capacity(capacity),
            is_null: None,
        },
        FieldType::Float32 => InputColumn::F32 {
            values: Vec::with_capacity(capacity),
            is_null: None,
        },
        FieldType::Decimal { .. } | FieldType::DecimalUnscaledI64 | FieldType::DecimalUnscaledBytes => {
            InputColumn::Decimal128 {
                values: Vec::with_capacity(capacity),
                is_null: None,
            }
        }
    }
}

fn clear_column(col: &mut InputColumn) {
    match col {
        InputColumn::String(v) => v.clear(),
        InputColumn::I64 { values, is_null } => {
            values.clear();
            *is_null = None;
        }
        InputColumn::I32 { values, is_null } => {
            values.clear();
            *is_null = None;
        }
        InputColumn::F64 { values, is_null } => {
            values.clear();
            *is_null = None;
        }
        InputColumn::F32 { values, is_null } => {
            values.clear();
            *is_null = None;
        }
        InputColumn::Bool { values, is_null } => {
            values.clear();
            *is_null = None;
        }
        InputColumn::Decimal128 { values, is_null } => {
            values.clear();
            *is_null = None;
        }
    }
}

fn push_null_to_column(col: &mut InputColumn) {
    match col {
        InputColumn::String(v) => v.push(None),
        InputColumn::I64 { values, is_null } => push_null_typed(values, is_null, 0),
        InputColumn::I32 { values, is_null } => push_null_typed(values, is_null, 0),
        InputColumn::F64 { values, is_null } => push_null_typed(values, is_null, 0.0),
        InputColumn::F32 { values, is_null } => push_null_typed(values, is_null, 0.0f32),
        InputColumn::Bool { values, is_null } => push_null_typed(values, is_null, false),
        InputColumn::Decimal128 { values, is_null } => push_null_typed(values, is_null, 0i128),
    }
}

fn push_null_typed<T>(values: &mut Vec<T>, is_null: &mut Option<Vec<bool>>, default: T) {
    let prev_len = values.len();
    values.push(default);
    if let Some(n) = is_null.as_mut() {
        n.push(true);
    } else {
        let mut v = vec![false; prev_len];
        v.push(true);
        *is_null = Some(v);
    }
}

fn decode_field_into_column(
    buf: &[u8],
    p: &mut usize,
    t: &FieldType,
    col: &mut InputColumn,
) -> Result<()> {
    match t {
        FieldType::Boolean => {
            let b = read_u8(buf, p)?;
            if let InputColumn::Bool { values, is_null } = col {
                values.push(b != 0);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            let v = read_i64_be(buf, p)?;
            if let InputColumn::I64 { values, is_null } = col {
                values.push(v);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let v = read_i32_be(buf, p)?;
            if let InputColumn::I32 { values, is_null } = col {
                values.push(v);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::Float32 => {
            let bits = read_u32_be(buf, p)?;
            if let InputColumn::F32 { values, is_null } = col {
                values.push(f32::from_bits(bits));
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::Float64 => {
            let bits = read_u64_be(buf, p)?;
            if let InputColumn::F64 { values, is_null } = col {
                values.push(f64::from_bits(bits));
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::String | FieldType::Unknown(_) => {
            let bytes = read_len_bytes(buf, p)?;
            let s = String::from_utf8(bytes).context("invalid UTF-8")?;
            if let InputColumn::String(values) = col {
                values.push(Some(s));
            }
        }
        FieldType::Bytes => {
            let bytes = read_len_bytes(buf, p)?;
            if let InputColumn::String(values) = col {
                values.push(Some(String::from_utf8_lossy(&bytes).to_string()));
            }
        }
        FieldType::Date => {
            let days = read_i32_be(buf, p)? as i64;
            let millis = days.checked_mul(86_400_000).context("DATE days->millis overflow")?;
            if let InputColumn::I64 { values, is_null } = col {
                values.push(millis);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::Decimal { precision, .. } => {
            let prec = precision.unwrap_or(38);
            let unscaled = if prec <= 18 {
                read_i64_be(buf, p)? as i128
            } else {
                let bytes = read_len_bytes(buf, p)?;
                twos_complement_be_to_i128(&bytes)?
            };
            if let InputColumn::Decimal128 { values, is_null } = col {
                values.push(unscaled);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = read_i64_be(buf, p)? as i128;
            if let InputColumn::Decimal128 { values, is_null } = col {
                values.push(unscaled);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
        FieldType::DecimalUnscaledBytes => {
            let bytes = read_len_bytes(buf, p)?;
            let unscaled = twos_complement_be_to_i128(&bytes)?;
            if let InputColumn::Decimal128 { values, is_null } = col {
                values.push(unscaled);
                if let Some(n) = is_null.as_mut() { n.push(false); }
            }
        }
    }
    Ok(())
}
