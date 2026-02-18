use anyhow::{bail, Result};
use std::ffi::c_void;
use std::ptr;

use udf_abi::helpers::{ensure_column_lengths, expect_i128, validate_function_class};
use udf_abi::{UDF_ABI_VERSION, UdfColumnType, UdfInputColumn, UdfOutputColumn, UdfResult, UdfStrView};

const CLASS_NAME: &str = "org.example.flinke2c.PriceGreaterThan";
const DECIMAL_SCALE_FACTOR: i128 = 1_000; // DECIMAL(23,3)

#[cfg(feature = "less_sensitive")]
const THRESHOLD_UNSCALED: i128 = 100_000 * DECIMAL_SCALE_FACTOR; // 100000.000

#[cfg(not(feature = "less_sensitive"))]
const THRESHOLD_UNSCALED: i128 = 1_000 * DECIMAL_SCALE_FACTOR; // 1000.000

struct UdfState;

struct ResultStorage {
    columns: Vec<UdfOutputColumn>,
    buffers: Vec<Buffer>,
}

enum Buffer {
    Bytes { _buf: Vec<u8> },
}

#[no_mangle]
pub extern "C" fn flinke2c_runtime_udf_abi_version() -> u32 {
    UDF_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn flinke2c_runtime_udf_create() -> *mut c_void {
    Box::into_raw(Box::new(UdfState)) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn flinke2c_runtime_udf_drop(state: *mut c_void) {
    if state.is_null() {
        return;
    }
    drop(Box::from_raw(state as *mut UdfState));
}

#[no_mangle]
pub unsafe extern "C" fn flinke2c_runtime_udf_free_result(result: *mut UdfResult) {
    if result.is_null() {
        return;
    }
    let result = Box::from_raw(result);
    if !result.private.is_null() {
        drop(Box::from_raw(result.private as *mut ResultStorage));
    }
}

#[no_mangle]
pub unsafe extern "C" fn flinke2c_runtime_udf_eval(
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

    let output = match eval_price_greater_than(input_columns) {
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
    validate_function_class(function_class, CLASS_NAME)
}

fn eval_price_greater_than(columns: &[UdfInputColumn]) -> Result<ResultStorage> {
    if columns.len() != 1 {
        bail!("PriceGreaterThan expects 1 input column, got {}", columns.len());
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
            out.push(0u8);
            continue;
        }

        let price_unscaled = *values.get(row).unwrap_or(&0);
        out.push((price_unscaled > THRESHOLD_UNSCALED) as u8);
    }

    let mut storage = ResultStorage {
        columns: Vec::new(),
        buffers: Vec::new(),
    };

    let values_ptr = out.as_ptr();
    storage.buffers.push(Buffer::Bytes { _buf: out });

    let nulls_ptr = if let Some(nulls) = out_nulls.take() {
        let ptr = nulls.as_ptr();
        storage.buffers.push(Buffer::Bytes { _buf: nulls });
        ptr
    } else {
        ptr::null()
    };

    storage.columns.push(UdfOutputColumn {
        kind: UdfColumnType::Bool,
        len: row_count,
        values: values_ptr,
        nulls: nulls_ptr,
        strings: ptr::null(),
    });

    Ok(storage)
}
