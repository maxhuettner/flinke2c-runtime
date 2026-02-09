use anyhow::{bail, Context, Result};
use std::io::Write;

use crate::codec::{
    encode_field, i128_to_twos_complement_be_minimal, row_op, row_row_id, set_null_bit,
    write_header_frame, write_i32_be_stream, write_i32_be_vec, write_i64_be_vec, write_output_blocks,
    write_u32_be_vec, write_u64_be_vec, OutputBlock,
};
use crate::config::{FieldType, FunctionKind, PayloadSource, PostFieldSourceKind, SessionConfig};
use crate::constants::DEFAULT_MAX_FRAME_SIZE;
use crate::udf::{InputColumn, UdfHandle};
use crate::values::{
    decimal_to_f64, decimal_to_i64, decimal_to_string, parse_bool_string, parse_decimal_to_i128,
    parse_with, v_to_bool, v_to_decimal_i128, v_to_f64, v_to_i64, v_to_string, V,
};

pub struct UdfConfig<'a, W: Write> {
    pub writer: &'a mut W,
    pub rows: &'a [Vec<V>],
    pub session: &'a SessionConfig,
    pub udf: &'a mut UdfHandle,
    pub method: &'a str,
    pub debug_sample_rows: usize,
    pub debug_batches_remaining: &'a mut usize,
}

pub fn apply_udf_to_rows(
    rows: &[Vec<V>],
    session: &SessionConfig,
    udf: &mut UdfHandle,
    method: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<Vec<OutputBlock>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
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

    let mut out_rows = Vec::with_capacity(rows.len());
    for row_idx in 0..rows.len() {
        let source_row = rows.get(row_idx).context("missing source row")?;
        let op = row_op(source_row);
        let row_id = row_row_id(source_row)?;

        let passes = filter_passes(pred_col, row_idx)?;

        if !passes {
            out_rows.push(OutputBlock {
                op,
                row_id,
                row: None,
            });
            continue;
        }

        let mut out_row = if session.passthrough_identity && source_row.len() == session.output_row_len {
            source_row.clone()
        } else {
            let mut row = vec![V::Null; session.output_row_len];
            if let Some(op) = source_row.first() {
                row[0] = op.clone();
            }
            if let Some(row_id) = source_row.get(1) {
                if session.output_row_len > 1 {
                    row[1] = row_id.clone();
                }
            }

            for source in &session.post_field_sources {
                match source.kind {
                    PostFieldSourceKind::Op => {
                        if source.pos < row.len() {
                            row[source.pos] = row[0].clone();
                        }
                    }
                    PostFieldSourceKind::RowId => {
                        if source.pos < row.len() {
                            let row_id = row.get(1).cloned().unwrap_or(V::Null);
                            row[source.pos] = row_id;
                        }
                    }
                    PostFieldSourceKind::InputPos(pre_pos) => {
                        if source.pos < row.len() {
                            row[source.pos] = source_row.get(pre_pos).cloned().unwrap_or(V::Null);
                        }
                    }
                    PostFieldSourceKind::Output => {}
                }
            }

            row
        };

        for (idx, pos) in session.output_positions.iter().enumerate() {
            if *pos >= out_row.len() {
                bail!("functionResult outputIndex {} is out of range", pos);
            }
            let v = output_column_to_v(
                &output_columns[idx],
                row_idx,
                session.output_types.get(idx).unwrap_or(&FieldType::String),
            )?;
            out_row[*pos] = v;
        }

        out_rows.push(OutputBlock {
            op,
            row_id,
            row: Some(out_row),
        });
    }

    maybe_print_debug_rows(rows, session, &out_rows, debug_sample_rows, debug_batches_remaining);
    Ok(out_rows)
}

pub fn apply_udf_to_rows_stream<W: Write>(config: &mut UdfConfig<W>) -> Result<()> {
    let rows = config.rows;
    let session = config.session;
    let method = config.method;
    let debug_sample_rows = config.debug_sample_rows;

    if rows.is_empty() {
        return Ok(());
    }

    if debug_sample_rows > 0 && *config.debug_batches_remaining > 0 {
        let out_rows = apply_udf_to_rows(
            rows,
            session,
            &mut *config.udf,
            method,
            debug_sample_rows,
            &mut *config.debug_batches_remaining,
        )?;
        let mut payload_buf = Vec::with_capacity(256);
        write_output_blocks(&mut *config.writer, &out_rows, session, &mut payload_buf)?;
        return Ok(());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
    let output_columns = call_udf_to_columns(&mut *config.udf, method, &input_columns, &session.output_names)?;

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
    let mut payload = Vec::with_capacity(256);
    let writer = &mut *config.writer;

    for (row_idx, source_row) in rows.iter().enumerate() {
        let op = row_op(source_row);
        let row_id = row_row_id(source_row)?;
        let passes = filter_passes(pred_col, row_idx)?;
        let count = if passes { 1 } else { 0 };
        write_header_frame(writer, op, row_id, count, &mut payload)?;
        if count == 0 {
            continue;
        }

        payload.clear();
        write_i32_be_vec(&mut payload, op);
        write_i64_be_vec(&mut payload, row_id);

        let null_pos = payload.len();
        payload.resize(null_pos + null_bytes, 0);

        for (i, source) in session.post_payload_sources.iter().enumerate() {
            let ftype = &session.post_payload_types[i];
            let is_null = match source {
                PayloadSource::InputAt(pre_pos) => {
                    let v = source_row.get(*pre_pos).unwrap_or(&V::Null);
                    if matches!(v, V::Null) {
                        true
                    } else {
                        encode_field(&mut payload, v, ftype)?;
                        false
                    }
                }
                PayloadSource::OutputAt(col_idx) => {
                    encode_input_column_at(&mut payload, &output_columns[*col_idx], row_idx, ftype)?
                }
            };
            if is_null {
                set_null_bit(&mut payload[null_pos..null_pos + null_bytes], i);
            }
        }

        if payload.len() > DEFAULT_MAX_FRAME_SIZE {
            bail!(
                "row payload exceeds max_frame_size: {} > {}",
                payload.len(),
                DEFAULT_MAX_FRAME_SIZE
            );
        }

        write_i32_be_stream(writer, payload.len() as i32)?;
        writer.write_all(payload.as_slice()).context("write payload")?;
    }

    Ok(())
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

fn build_input_columns(
    rows: &[Vec<V>],
    arg_positions: &[usize],
    arg_types: &[FieldType],
) -> Result<Vec<InputColumn>> {
    let mut columns = Vec::with_capacity(arg_positions.len());
    for (idx, arg_index) in arg_positions.iter().enumerate() {
        let target_type = arg_types
            .get(idx)
            .with_context(|| format!("missing arg type at {}", idx))?;
        let column = build_input_column(rows, *arg_index, target_type)?;
        columns.push(column);
    }
    Ok(columns)
}

fn build_input_column(
    rows: &[Vec<V>],
    index: usize,
    target_type: &FieldType,
) -> Result<InputColumn> {
    match target_type {
        FieldType::String | FieldType::Unknown(_) => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                if matches!(v, V::Null) {
                    values.push(None);
                } else {
                    values.push(Some(v_to_string(v)?));
                }
            }
            Ok(InputColumn::String(values))
        }
        FieldType::Bytes => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                if matches!(v, V::Null) {
                    values.push(None);
                } else {
                    match v {
                        V::Bytes(b) => values.push(Some(String::from_utf8_lossy(b).to_string())),
                        _ => values.push(Some(v_to_string(v)?)),
                    }
                }
            }
            Ok(InputColumn::String(values))
        }
        FieldType::Boolean => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(false);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_bool(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::Bool { values, is_null: nulls })
        }
        FieldType::Int64 | FieldType::TimestampMillis | FieldType::Date => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_i64(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::I64 { values, is_null: nulls })
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    let x = v_to_i64(v)?;
                    let casted = i32::try_from(x)
                        .map_err(|_| anyhow::anyhow!("value {} overflows i32", x))?;
                    values.push(casted);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::I32 { values, is_null: nulls })
        }
        FieldType::Float64 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_f64(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::F64 { values, is_null: nulls })
        }
        FieldType::Float32 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_f64(v)? as f32);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::F32 { values, is_null: nulls })
        }
        FieldType::Decimal { .. } | FieldType::DecimalUnscaledI64 | FieldType::DecimalUnscaledBytes => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_decimal_i128(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::Decimal128 { values, is_null: nulls })
        }
    }
}

fn push_null(nulls: &mut Option<Vec<bool>>, previous_len: usize, is_null: bool) {
    if let Some(nulls) = nulls.as_mut() {
        nulls.push(is_null);
        return;
    }
    if is_null {
        let mut vec = vec![false; previous_len];
        vec.push(true);
        *nulls = Some(vec);
    }
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
        FieldType::String | FieldType::Unknown(_) => {
            Ok(V::String(input_column_to_string(column, row, output_type)?))
        }
        FieldType::Bytes => Ok(V::Bytes(
            input_column_to_string(column, row, output_type)?.into_bytes(),
        )),
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

fn encode_input_column_at(
    out: &mut Vec<u8>,
    col: &InputColumn,
    row: usize,
    ftype: &FieldType,
) -> Result<bool> {
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
                let as_i64 = i64::try_from(unscaled)
                    .map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
                write_i64_be_vec(out, as_i64);
            } else {
                let bytes = i128_to_twos_complement_be_minimal(unscaled);
                write_i32_be_vec(out, bytes.len() as i32);
                out.extend_from_slice(&bytes);
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let as_i64 =
                i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
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

fn maybe_print_debug_rows(
    input_rows: &[Vec<V>],
    session: &SessionConfig,
    output_rows: &[OutputBlock],
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) {
    if debug_sample_rows == 0 || *debug_batches_remaining == 0 || input_rows.is_empty() {
        return;
    }

    let sample_rows = debug_sample_rows.min(input_rows.len());
    println!("Debug sample ({} of {} rows):", sample_rows, input_rows.len());
    for (row_idx, row) in input_rows.iter().take(sample_rows).enumerate() {
        let mut input_parts = Vec::with_capacity(session.arg_names.len());
        for (arg_pos, name) in session.arg_names.iter().enumerate() {
            let idx = session.arg_positions.get(arg_pos).copied().unwrap_or(0);
            let value = row.get(idx).unwrap_or(&V::Null);
            input_parts.push(format!("{}={}", name, debug_v(value)));
        }

        let mut output_parts = Vec::new();
        let output_row = output_rows
            .get(row_idx)
            .and_then(|block| block.row.as_ref());
        if !session.output_positions.is_empty() {
            if let Some(row) = output_row {
                for (idx, name) in session.output_names.iter().enumerate() {
                    let pos = session.output_positions.get(idx).copied().unwrap_or(0);
                    let value = row.get(pos).unwrap_or(&V::Null);
                    output_parts.push(format!("{}={}", name, debug_v(value)));
                }
            } else {
                output_parts.push("<filtered>".to_string());
            }
        } else if output_row.is_none() && session.function_kind == FunctionKind::Filter {
            output_parts.push("<filtered>".to_string());
        }

        println!(
            "  row {}: {} -> {}",
            row_idx,
            input_parts.join(", "),
            output_parts.join(", ")
        );
    }

    *debug_batches_remaining = debug_batches_remaining.saturating_sub(1);
}

fn debug_v(v: &V) -> String {
    match v {
        V::Null => "<null>".to_string(),
        V::Bool(b) => b.to_string(),
        V::I32(x) => x.to_string(),
        V::I64(x) => x.to_string(),
        V::F32(x) => x.to_string(),
        V::F64(x) => x.to_string(),
        V::String(s) => s.clone(),
        V::Bytes(b) => format!("{:?}", b),
        V::DecimalI128(x) => x.to_string(),
    }
}
