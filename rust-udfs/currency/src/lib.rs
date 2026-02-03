use anyhow::{bail, Context, Result};
use std::ffi::c_void;
use std::ptr;

use udf_abi::{UdfColumnType, UdfInputColumn, UdfOutputColumn, UdfResult, UdfStrView, UDF_ABI_VERSION};

const CLASS_NAME: &str = "org.example.flinke2c.CurrencyConversionFunction";
const CONVERSION_FACTOR_UNSCALED: i128 = 908; // BigDecimal("0.908") unscaled

struct UdfState;

struct ResultStorage {
    columns: Vec<UdfOutputColumn>,
    buffers: Vec<Buffer>,
}

enum Buffer {
    Bytes { _buf: Vec<u8> },
    I128 { _buf: Vec<i128> },
}

#[no_mangle]
pub extern "C" fn proxy_udf_abi_version() -> u32 {
    UDF_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn proxy_udf_create() -> *mut c_void {
    Box::into_raw(Box::new(UdfState)) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn proxy_udf_drop(state: *mut c_void) {
    if state.is_null() {
        return;
    }
    drop(Box::from_raw(state as *mut UdfState));
}

#[no_mangle]
pub unsafe extern "C" fn proxy_udf_free_result(result: *mut UdfResult) {
    if result.is_null() {
        return;
    }
    let result = Box::from_raw(result);
    if !result.private.is_null() {
        drop(Box::from_raw(result.private as *mut ResultStorage));
    }
}

#[no_mangle]
pub unsafe extern "C" fn proxy_udf_eval(
    state: *mut c_void,
    function_class: UdfStrView,
    columns: *const UdfInputColumn,
    num_columns: usize,
    _output_names: *const UdfStrView,
    _num_output_names: usize,
) -> *mut UdfResult {
    if state.is_null() {
        return ptr::null_mut();
    }

    if let Err(err) = validate_class(function_class) {
        eprintln!("rust udf: {err:#}");
        return ptr::null_mut();
    }

    let input_columns = if columns.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(columns, num_columns)
    };

    let output = match eval_currency(input_columns) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("rust udf eval error: {err:#}");
            return ptr::null_mut();
        }
    };

    let storage = Box::new(output);
    let columns_ptr = storage.columns.as_ptr();
    let len = storage.columns.len();
    let storage_ptr = Box::into_raw(storage) as *mut c_void;

    let result = UdfResult {
        columns: columns_ptr,
        len,
        private: storage_ptr,
    };
    Box::into_raw(Box::new(result))
}

fn validate_class(function_class: UdfStrView) -> Result<()> {
    if function_class.ptr.is_null() {
        return Ok(());
    }
    let class = view_to_string(function_class)?;
    if class != CLASS_NAME {
        bail!("unsupported rust udf class: {class}");
    }
    Ok(())
}

fn eval_currency(columns: &[UdfInputColumn]) -> Result<ResultStorage> {
    if columns.len() != 1 {
        bail!("CurrencyConversionFunction expects 1 input column, got {}", columns.len());
    }
    let row_count = ensure_column_lengths(columns)?;
    let (values, nulls) = expect_i128(&columns[0], "price")?;

    let mut out = Vec::with_capacity(row_count);
    let mut out_nulls = nulls.map(|v| v.to_vec());

    for row in 0..row_count {
        let is_null = out_nulls
            .as_ref()
            .map(|v| v.get(row).copied().unwrap_or(0) != 0)
            .unwrap_or(false);
        if is_null {
            out.push(0);
            continue;
        }
        let value = *values.get(row).unwrap_or(&0);
        let scaled = value
            .checked_mul(CONVERSION_FACTOR_UNSCALED)
            .ok_or_else(|| anyhow::anyhow!("decimal overflow in currency conversion"))?;
        out.push(scaled);
    }

    let mut storage = ResultStorage {
        columns: Vec::new(),
        buffers: Vec::new(),
    };

    let values_ptr = out.as_ptr();
    storage.buffers.push(Buffer::I128 { _buf: out });

    let nulls_ptr = if let Some(nulls) = out_nulls.take() {
        let ptr = nulls.as_ptr();
        storage.buffers.push(Buffer::Bytes { _buf: nulls });
        ptr
    } else {
        ptr::null()
    };

    storage.columns.push(UdfOutputColumn {
        kind: UdfColumnType::Decimal128,
        len: row_count,
        values: values_ptr as *const u8,
        nulls: nulls_ptr,
        strings: ptr::null(),
    });

    Ok(storage)
}

fn ensure_column_lengths(columns: &[UdfInputColumn]) -> Result<usize> {
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

fn expect_i128<'a>(
    column: &'a UdfInputColumn,
    label: &str,
) -> Result<(&'a [i128], Option<&'a [u8]>)> {
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

fn view_to_string(view: UdfStrView) -> Result<String> {
    if view.ptr.is_null() {
        bail!("null string pointer");
    }
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    let s = std::str::from_utf8(bytes).context("invalid utf-8")?;
    Ok(s.to_string())
}
