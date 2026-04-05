use anyhow::{bail, Context, Result};

use crate::{UdfColumnType, UdfInputColumn, UdfStrView};

pub fn validate_function_class(function_class: UdfStrView, expected_class: &str) -> Result<()> {
    if function_class.ptr.is_null() {
        return Ok(());
    }
    let class = view_to_string(function_class)?;
    if class != expected_class {
        bail!("unsupported rust udf class: {class}");
    }
    Ok(())
}

pub fn ensure_column_lengths(columns: &[UdfInputColumn]) -> Result<usize> {
    let Some(first) = columns.first() else {
        return Ok(0);
    };
    let len = first.len;
    for col in columns.iter().skip(1) {
        if col.len != len {
            bail!("typed columns have mismatched lengths");
        }
    }
    Ok(len)
}

pub fn expect_i128<'a>(column: &'a UdfInputColumn, label: &str) -> Result<(&'a [i128], Option<&'a [u8]>)> {
    if column.kind != UdfColumnType::Decimal128 {
        bail!("expected DECIMAL column for {label}");
    }
    if column.values.is_null() {
        bail!("decimal column {label} missing values");
    }
    let values = unsafe { std::slice::from_raw_parts(column.values as *const i128, column.len) };
    let nulls = if column.nulls.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(column.nulls, column.len) })
    };
    Ok((values, nulls))
}

pub fn expect_i64<'a>(column: &'a UdfInputColumn, label: &str) -> Result<(&'a [i64], Option<&'a [u8]>)> {
    if column.kind != UdfColumnType::I64 {
        bail!("expected BIGINT column for {label}");
    }
    if column.values.is_null() {
        bail!("i64 column {label} missing values");
    }
    let values = unsafe { std::slice::from_raw_parts(column.values as *const i64, column.len) };
    let nulls = if column.nulls.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(column.nulls, column.len) })
    };
    Ok((values, nulls))
}

pub fn expect_strings<'a>(column: &'a UdfInputColumn, label: &str) -> Result<(&'a [UdfStrView], Option<&'a [u8]>)> {
    if column.kind != UdfColumnType::String {
        bail!("expected STRING column for {label}");
    }
    if column.strings.is_null() {
        bail!("string column {label} missing string views");
    }
    let values = unsafe { std::slice::from_raw_parts(column.strings, column.len) };
    let nulls = if column.nulls.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(column.nulls, column.len) })
    };
    Ok((values, nulls))
}

pub fn view_to_string(view: UdfStrView) -> Result<String> {
    if view.ptr.is_null() {
        bail!("null string pointer");
    }
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    let s = std::str::from_utf8(bytes).context("invalid utf-8")?;
    Ok(s.to_string())
}

pub fn is_null_at(nulls: Option<&[u8]>, row: usize) -> bool {
    nulls.and_then(|vals| vals.get(row)).map(|v| *v != 0).unwrap_or(false)
}
