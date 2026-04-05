use anyhow::{bail, Context, Result};
use std::io::{ErrorKind, Read, Write};

pub const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct RowF32 {
    pub op: i32,
    pub row_id: i64,
    pub value: Option<f32>,
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
