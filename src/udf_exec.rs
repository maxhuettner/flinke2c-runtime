use anyhow::{Context, Result, bail};
use std::io::Write;

use crate::codec::{
    ColumnarBatch, OutputBlock, i128_to_twos_complement_be_minimal, set_null_bit, write_i32_be_stream,
    write_i32_be_vec, write_i64_be_vec, write_u32_be_vec, write_u64_be_vec,
};
use crate::config::{FieldType, FunctionKind, PayloadSource, PostFieldSourceKind, SessionConfig};
use crate::constants::DEFAULT_MAX_FRAME_SIZE;
use crate::udf::{InputColumn, UdfHandle};
use crate::values::{
    V, decimal_to_f64, decimal_to_i64, decimal_to_string, parse_bool_string, parse_decimal_to_i128, parse_with,
};

pub struct ColumnarUdfConfig<'a, W: Write> {
    pub writer: &'a mut W,
    pub batch: &'a ColumnarBatch,
    pub session: &'a SessionConfig,
    pub udf: &'a mut UdfHandle,
    pub method: &'a str,
}

pub fn apply_udf_to_batch_stream<W: Write>(config: &mut ColumnarUdfConfig<W>) -> Result<()> {
    let batch = config.batch;
    let session = config.session;

    if batch.len() == 0 {
        return Ok(());
    }

    let input_columns = extract_arg_columns(batch, session);
    let output_columns = call_udf_to_columns(&mut *config.udf, config.method, &input_columns, &session.output_names)?;

    if output_columns.len() < session.output_positions.len() {
        bail!(
            "UDF returned {} columns but functionResults resolved {} targets",
            output_columns.len(),
            session.output_positions.len()
        );
    }

    let pred_col = if session.function_kind == FunctionKind::Filter {
        Some(
            output_columns
                .get(0)
                .with_context(|| "filter UDF returned no columns")?,
        )
    } else {
        None
    };

    let n_fields = session.post_payload_positions.len();
    let null_bytes = (n_fields + 7) >> 3;
    let mut batch_payload = Vec::with_capacity(256);
    let writer = &mut *config.writer;

    // Send one batched frame per input row
    for row_idx in 0..batch.len() {
        batch_payload.clear();

        let op = batch.ops[row_idx];
        let row_id = batch.row_ids[row_idx];
        let passes = filter_passes(pred_col, row_idx)?;
        let count = if passes { 1 } else { 0 };

        // header section, no frame length prefix
        write_i32_be_vec(&mut batch_payload, op);
        write_i64_be_vec(&mut batch_payload, row_id);
        batch_payload.push(0u8); // null bitmap for count field (not null)
        write_i32_be_vec(&mut batch_payload, count);

        // row section if the filter passes, no frame length prefix
        if passes {
            // Row header: __op and __rowId
            write_i32_be_vec(&mut batch_payload, op);
            write_i64_be_vec(&mut batch_payload, row_id);

            // null bitmap for payload fields
            let null_pos = batch_payload.len();
            batch_payload.resize(null_pos + null_bytes, 0);

            for (i, source) in session.post_payload_sources.iter().enumerate() {
                let ftype = &session.post_payload_types[i];
                let is_null = match source {
                    PayloadSource::InputAt(pre_pos) => {
                        if let Some(slot) = session.pre_pos_to_payload_slot.get(*pre_pos).and_then(|s| *s) {
                            encode_input_column_at(&mut batch_payload, &batch.columns[slot], row_idx, ftype)?
                        } else {
                            true
                        }
                    }
                    PayloadSource::OutputAt(col_idx) => {
                        encode_input_column_at(&mut batch_payload, &output_columns[*col_idx], row_idx, ftype)?
                    }
                };
                if is_null {
                    set_null_bit(&mut batch_payload[null_pos..null_pos + null_bytes], i);
                }
            }
        }

        if batch_payload.len() > DEFAULT_MAX_FRAME_SIZE {
            bail!(
                "batch payload exceeds max_frame_size: {} > {}",
                batch_payload.len(),
                DEFAULT_MAX_FRAME_SIZE
            );
        }

        // Write entire batched frame: [length][payload]
        write_i32_be_stream(writer, batch_payload.len() as i32)?;
        writer
            .write_all(batch_payload.as_slice())
            .context("write batch payload")?;
    }

    Ok(())
}

pub fn apply_udf_to_batch(
    batch: &ColumnarBatch,
    session: &SessionConfig,
    udf: &mut UdfHandle,
    method: &str,
) -> Result<Vec<OutputBlock>> {
    if batch.len() == 0 {
        return Ok(Vec::new());
    }

    let input_columns = extract_arg_columns(batch, session);
    let output_columns = call_udf_to_columns(udf, method, &input_columns, &session.output_names)?;

    if output_columns.len() < session.output_positions.len() {
        bail!(
            "UDF returned {} columns but functionResults resolved {} targets",
            output_columns.len(),
            session.output_positions.len()
        );
    }

    let pred_col = if session.function_kind == FunctionKind::Filter {
        Some(
            output_columns
                .get(0)
                .with_context(|| "filter UDF returned no columns")?,
        )
    } else {
        None
    };

    let mut out = Vec::with_capacity(batch.len());
    for row_idx in 0..batch.len() {
        let op = batch.ops[row_idx];
        let row_id = batch.row_ids[row_idx];
        let passes = filter_passes(pred_col, row_idx)?;

        if !passes {
            out.push(OutputBlock { op, row_id, row: None });
            continue;
        }

        let mut row = vec![V::Null; session.output_row_len];
        row[0] = V::I32(op);
        if session.output_row_len > 1 {
            row[1] = V::I64(row_id);
        }

        for source in &session.post_field_sources {
            match &source.kind {
                PostFieldSourceKind::Op => {
                    if source.pos < row.len() {
                        row[source.pos] = V::I32(op);
                    }
                }
                PostFieldSourceKind::RowId => {
                    if source.pos < row.len() {
                        row[source.pos] = V::I64(row_id);
                    }
                }
                PostFieldSourceKind::InputPos(pre_pos) => {
                    if source.pos < row.len() {
                        if let Some(slot) = session.pre_pos_to_payload_slot.get(*pre_pos).and_then(|s| *s) {
                            row[source.pos] = column_to_v_at(&batch.columns[slot], row_idx);
                        }
                    }
                }
                PostFieldSourceKind::Output => {}
            }
        }

        for (idx, pos) in session.output_positions.iter().enumerate() {
            if *pos >= row.len() {
                bail!("functionResult outputIndex {} is out of range", pos);
            }
            let v = output_column_to_v(
                &output_columns[idx],
                row_idx,
                session.output_types.get(idx).unwrap_or(&FieldType::String),
            )?;
            row[*pos] = v;
        }

        out.push(OutputBlock {
            op,
            row_id,
            row: Some(row),
        });
    }

    Ok(out)
}

fn extract_arg_columns(batch: &ColumnarBatch, session: &SessionConfig) -> Vec<InputColumn> {
    session
        .arg_positions
        .iter()
        .map(|&pos| {
            if let Some(slot) = session.pre_pos_to_payload_slot.get(pos).and_then(|s| *s) {
                batch.columns[slot].clone()
            } else {
                InputColumn::I64 {
                    values: vec![0; batch.len()],
                    is_null: Some(vec![true; batch.len()]),
                }
            }
        })
        .collect()
}

fn column_to_v_at(col: &InputColumn, row: usize) -> V {
    match col {
        InputColumn::String(values) => values
            .get(row)
            .and_then(|v| v.clone())
            .map(V::String)
            .unwrap_or(V::Null),
        InputColumn::I64 { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::I64(values.get(row).copied().unwrap_or(0))
            }
        }
        InputColumn::I32 { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::I32(values.get(row).copied().unwrap_or(0))
            }
        }
        InputColumn::F64 { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::F64(values.get(row).copied().unwrap_or(0.0))
            }
        }
        InputColumn::F32 { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::F32(values.get(row).copied().unwrap_or(0.0))
            }
        }
        InputColumn::Bool { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::Bool(values.get(row).copied().unwrap_or(false))
            }
        }
        InputColumn::Decimal128 { values, is_null } => {
            if is_null_at(is_null.as_deref(), row) {
                V::Null
            } else {
                V::DecimalI128(values.get(row).copied().unwrap_or(0))
            }
        }
    }
}

fn call_udf_to_columns(
    udf: &mut UdfHandle,
    method: &str,
    input_columns: &[InputColumn],
    output_names: &[String],
) -> Result<Vec<InputColumn>> {
    let columns_method = if method.ends_with("Fast") || method.ends_with("ToColumnsTypedOut") {
        method.to_string()
    } else {
        format!("{method}Fast")
    };

    if output_names.is_empty() {
        return udf.call_typed_columns_to_typed_results(&columns_method, input_columns);
    }

    let named_method = if columns_method.ends_with("Named") {
        columns_method.clone()
    } else {
        format!("{columns_method}Named")
    };
    udf.call_typed_columns_to_named_results(&named_method, input_columns, output_names)
}

fn column_is_null(column: &InputColumn, row: usize) -> bool {
    match column {
        InputColumn::String(values) => values.get(row).map(|v| v.is_none()).unwrap_or(true),
        InputColumn::I64 { is_null, .. }
        | InputColumn::I32 { is_null, .. }
        | InputColumn::F64 { is_null, .. }
        | InputColumn::F32 { is_null, .. }
        | InputColumn::Bool { is_null, .. }
        | InputColumn::Decimal128 { is_null, .. } => is_null_at(is_null.as_deref(), row),
    }
}

fn is_null_at(nulls: Option<&[bool]>, idx: usize) -> bool {
    nulls
        .map(|vals| vals.get(idx).copied().unwrap_or(false))
        .unwrap_or(false)
}

fn output_column_to_v(column: &InputColumn, row: usize, output_type: &FieldType) -> Result<V> {
    if column_is_null(column, row) {
        return Ok(V::Null);
    }
    match output_type {
        FieldType::String | FieldType::Unknown(_) => Ok(V::String(input_column_to_string(column, row, output_type)?)),
        FieldType::Bytes => Ok(V::Bytes(input_column_to_string(column, row, output_type)?.into_bytes())),
        FieldType::Boolean => Ok(V::Bool(input_column_to_bool(column, row, output_type)?)),
        FieldType::Int64 | FieldType::TimestampMillis | FieldType::Date => {
            Ok(V::I64(input_column_to_i64(column, row, output_type)?))
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            Ok(V::I32(input_column_to_i64(column, row, output_type)? as i32))
        }
        FieldType::Float64 => Ok(V::F64(input_column_to_f64(column, row, output_type)?)),
        FieldType::Float32 => Ok(V::F32(input_column_to_f64(column, row, output_type)? as f32)),
        FieldType::Decimal { .. } | FieldType::DecimalUnscaledI64 | FieldType::DecimalUnscaledBytes => {
            Ok(V::DecimalI128(input_column_to_decimal(column, row, output_type)?))
        }
    }
}

fn input_column_to_string(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<String> {
    match column {
        InputColumn::String(values) => values
            .get(row)
            .and_then(|v| v.clone())
            .ok_or_else(|| anyhow::anyhow!("null string value")),
        InputColumn::Bool { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::I64 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::I32 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::F64 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::F32 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_string(value, scale))
        }
    }
}

fn input_column_to_bool(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<bool> {
    match column {
        InputColumn::Bool { values, .. } => Ok(*values.get(row).unwrap_or(&false)),
        InputColumn::I64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) != 0),
        InputColumn::I32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) != 0),
        InputColumn::F64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) != 0.0),
        InputColumn::F32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) != 0.0),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_i64(value, scale)? != 0)
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_bool_string(value)
        }
    }
}

fn input_column_to_i64(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<i64> {
    match column {
        InputColumn::I64 { values, .. } => Ok(*values.get(row).unwrap_or(&0)),
        InputColumn::I32 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i64),
        InputColumn::F64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) as i64),
        InputColumn::F32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) as i64),
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1 } else { 0 }),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            decimal_to_i64(value, scale)
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_with(value, "i64")
        }
    }
}

fn input_column_to_f64(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<f64> {
    match column {
        InputColumn::F64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0)),
        InputColumn::F32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) as f64),
        InputColumn::I64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) as f64),
        InputColumn::I32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) as f64),
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1.0 } else { 0.0 }),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_f64(value, scale))
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_with(value, "f64")
        }
    }
}

fn input_column_to_decimal(column: &InputColumn, row: usize, _source_type: &FieldType) -> Result<i128> {
    match column {
        InputColumn::Decimal128 { values, .. } => Ok(values.get(row).copied().unwrap_or(0)),
        InputColumn::I64 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i128),
        InputColumn::I32 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i128),
        InputColumn::F64 { values, .. } => {
            parse_decimal_to_i128(&values.get(row).copied().unwrap_or(0.0).to_string(), 0)
        }
        InputColumn::F32 { values, .. } => {
            parse_decimal_to_i128(&values.get(row).copied().unwrap_or(0.0).to_string(), 0)
        }
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1 } else { 0 }),
        InputColumn::String(values) => {
            let s = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_decimal_to_i128(s, 0)
        }
    }
}

fn encode_input_column_at(out: &mut Vec<u8>, col: &InputColumn, row: usize, ftype: &FieldType) -> Result<bool> {
    if column_is_null(col, row) {
        return Ok(true);
    }
    match ftype {
        FieldType::Boolean => {
            let b = input_column_to_bool(col, row, ftype)?;
            out.push(if b { 1 } else { 0 });
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            let x = input_column_to_i64(col, row, ftype)?;
            write_i64_be_vec(out, x);
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let x = input_column_to_i64(col, row, ftype)?;
            write_i32_be_vec(out, x as i32);
        }
        FieldType::Float32 => {
            let x = input_column_to_f64(col, row, ftype)? as f32;
            write_u32_be_vec(out, x.to_bits());
        }
        FieldType::Float64 => {
            let x = input_column_to_f64(col, row, ftype)?;
            write_u64_be_vec(out, x.to_bits());
        }
        FieldType::String | FieldType::Unknown(_) => {
            let s = input_column_to_string(col, row, ftype)?;
            let bytes = s.as_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Bytes => {
            let s = input_column_to_string(col, row, ftype)?;
            let bytes = s.into_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
        FieldType::Date => {
            let millis = input_column_to_i64(col, row, ftype)?;
            let days = (millis / 86_400_000) as i32;
            write_i32_be_vec(out, days);
        }
        FieldType::Decimal { precision, .. } => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
                write_i64_be_vec(out, as_i64);
            } else {
                let bytes = i128_to_twos_complement_be_minimal(unscaled);
                write_i32_be_vec(out, bytes.len() as i32);
                out.extend_from_slice(&bytes);
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
            write_i64_be_vec(out, as_i64);
        }
        FieldType::DecimalUnscaledBytes => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let bytes = i128_to_twos_complement_be_minimal(unscaled);
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
    }
    Ok(false)
}

fn filter_passes(pred_col: Option<&InputColumn>, row_idx: usize) -> Result<bool> {
    match pred_col {
        Some(col) => input_column_to_bool(col, row_idx, &FieldType::Boolean),
        None => Ok(true),
    }
}
