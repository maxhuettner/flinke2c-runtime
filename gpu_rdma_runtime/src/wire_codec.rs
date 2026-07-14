use anyhow::{bail, Context, Result};
use std::io::{ErrorKind, Read, Write};

pub const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct RowF32 {
    pub op: i32,
    pub row_id: i64,
    pub value: Option<f32>,
}

/// Benchmark row matching the external row wire layout with one INT32 field:
/// op(i32 BE), row_id(i64 BE), null bitmap, value(i32 BE).
pub fn encode_row_i32_payload(op: i32, row_id: i64, value: i32, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&op.to_be_bytes());
    out.extend_from_slice(&row_id.to_be_bytes());
    out.push(0);
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends a Flink `DECIMAL_UNSCALED_BYTES` value:
/// `[int32_be byte_length][minimal two's-complement big-endian bytes]`.
///
/// DECIMAL(23, 3) uses this representation. In particular, zero is encoded
/// as one byte (`00`), never as a zero-length value.
pub fn append_decimal_unscaled_bytes(unscaled: i128, out: &mut Vec<u8>) {
    let bytes = unscaled.to_be_bytes();
    let negative = unscaled < 0;
    let mut first = 0;
    while first < bytes.len() - 1 {
        let current = bytes[first];
        let next = bytes[first + 1];
        let redundant_positive = !negative && current == 0 && next & 0x80 == 0;
        let redundant_negative = negative && current == 0xff && next & 0x80 != 0;
        if redundant_positive || redundant_negative {
            first += 1;
        } else {
            break;
        }
    }
    let encoded = &bytes[first..];
    out.extend_from_slice(&(encoded.len() as i32).to_be_bytes());
    out.extend_from_slice(encoded);
}

pub fn decode_row_i32_payload(payload: &[u8]) -> Result<(i32, i64, i32)> {
    if payload.len() < 17 {
        bail!("truncated INT32 row payload");
    }
    let op = i32::from_be_bytes(payload[0..4].try_into()?);
    let row_id = i64::from_be_bytes(payload[4..12].try_into()?);
    anyhow::ensure!(payload[12] & 1 == 0, "benchmark INT32 field is null");
    let value = i32::from_be_bytes(payload[13..17].try_into()?);
    Ok((op, row_id, value))
}

pub fn write_i32_be_stream<W: Write>(w: &mut W, v: i32) -> Result<()> {
    w.write_all(&v.to_be_bytes()).context("write i32 be")
}

pub fn read_i32_be_stream_opt<R: Read>(r: &mut R) -> Result<Option<i32>> {
    let mut b = [0u8; 4];
    match r.read_exact(&mut b) {
        Ok(()) => Ok(Some(i32::from_be_bytes(b))),
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err).context("read i32 be opt"),
    }
}

pub fn write_framed_payload<W: Write>(writer: &mut W, payload: &[u8]) -> Result<()> {
    if payload.len() > MAX_FRAME_SIZE {
        bail!(
            "frame length {} exceeds max_frame_size {}",
            payload.len(),
            MAX_FRAME_SIZE
        );
    }
    write_i32_be_stream(writer, payload.len() as i32)?;
    writer.write_all(payload).context("write frame payload")?;
    Ok(())
}

pub fn read_framed_payload<R: Read>(reader: &mut R, scratch: &mut Vec<u8>) -> Result<bool> {
    let len = match read_i32_be_stream_opt(reader)? {
        Some(v) => v,
        None => return Ok(false),
    };
    if len < 0 {
        bail!("invalid negative frame length: {}", len);
    }
    let len = len as usize;
    if len > MAX_FRAME_SIZE {
        bail!("frame length {} exceeds max_frame_size {}", len, MAX_FRAME_SIZE);
    }
    scratch.resize(len, 0);
    reader.read_exact(scratch).context("read frame payload")?;
    Ok(true)
}

pub fn encode_row_f32_payload(row: RowF32, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&row.op.to_be_bytes());
    out.extend_from_slice(&row.row_id.to_be_bytes());

    // Same null-bitmap convention as src/codec.rs (LSB-first).
    let mut null_bitmap = 0u8;
    if row.value.is_none() {
        null_bitmap |= 1u8;
    }
    out.push(null_bitmap);

    if let Some(v) = row.value {
        out.extend_from_slice(&v.to_bits().to_be_bytes());
    }
}

pub fn decode_row_f32_payload(payload: &[u8]) -> Result<RowF32> {
    if payload.len() < 13 {
        bail!("truncated payload: need at least 13 bytes");
    }

    let op = i32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let row_id = i64::from_be_bytes([
        payload[4],
        payload[5],
        payload[6],
        payload[7],
        payload[8],
        payload[9],
        payload[10],
        payload[11],
    ]);
    let null_bitmap = payload[12];
    let is_null = (null_bitmap & 0x01) != 0;

    let value = if is_null {
        None
    } else {
        if payload.len() < 17 {
            bail!("truncated payload: float32 field missing");
        }
        let bits = u32::from_be_bytes([payload[13], payload[14], payload[15], payload[16]]);
        Some(f32::from_bits(bits))
    };

    Ok(RowF32 { op, row_id, value })
}

#[cfg(test)]
mod tests {
    use super::append_decimal_unscaled_bytes;

    #[test]
    fn decimal_zero_has_one_zero_byte() {
        let mut encoded = Vec::new();
        append_decimal_unscaled_bytes(0, &mut encoded);
        assert_eq!(encoded, [0, 0, 0, 1, 0]);
    }

    #[test]
    fn decimal_12345_matches_flink_bytes() {
        let mut encoded = Vec::new();
        append_decimal_unscaled_bytes(12_345, &mut encoded);
        assert_eq!(encoded, [0, 0, 0, 2, 0x30, 0x39]);
    }

    #[test]
    fn decimal_negative_matches_flink_bytes() {
        let mut encoded = Vec::new();
        append_decimal_unscaled_bytes(-12_345, &mut encoded);
        assert_eq!(encoded, [0, 0, 0, 2, 0xcf, 0xc7]);
    }
}
