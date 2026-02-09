use anyhow::{bail, Context, Result};
use libloading::Library;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use udf_abi::{UdfColumnType, UdfInputColumn, UdfResult, UdfStrView, UDF_ABI_VERSION};

use crate::java_udf::InputColumn;

const SYMBOL_ABI_VERSION: &[u8] = b"flinke2c_runtime_udf_abi_version\0";
const SYMBOL_CREATE: &[u8] = b"flinke2c_runtime_udf_create\0";
const SYMBOL_DROP: &[u8] = b"flinke2c_runtime_udf_drop\0";
const SYMBOL_EVAL: &[u8] = b"flinke2c_runtime_udf_eval\0";
const SYMBOL_FREE_RESULT: &[u8] = b"flinke2c_runtime_udf_free_result\0";

#[derive(Debug)]
pub struct RustUdfHandle {
    lib_path: PathBuf,
    lib_state: LibState,
    lib: Library,
    api: UdfApi,
    state: *mut c_void,
    function_class: String,
}

impl RustUdfHandle {
    pub fn new(lib_path: &Path, function_class: &str) -> Result<Self> {
        let lib_state = lib_state(lib_path)?;
        let lib = unsafe { Library::new(lib_path) }
            .with_context(|| format!("load rust udf library {}", lib_path.display()))?;
        let api = unsafe { UdfApi::load(&lib)? };
        let state = unsafe { (api.create)() };
        if state.is_null() {
            bail!("rust udf create returned null state");
        }
        Ok(Self {
            lib_path: lib_path.to_path_buf(),
            lib_state,
            lib,
            api,
            state,
            function_class: function_class.to_string(),
        })
    }

    pub fn reload_if_changed(&mut self) -> Result<bool> {
        let new_state = lib_state(&self.lib_path)?;
        if new_state == self.lib_state {
            return Ok(false);
        }

        unsafe {
            (self.api.drop)(self.state);
        }
        let lib = unsafe { Library::new(&self.lib_path) }
            .with_context(|| format!("reload rust udf library {}", self.lib_path.display()))?;
        let api = unsafe { UdfApi::load(&lib)? };
        let state = unsafe { (api.create)() };
        if state.is_null() {
            bail!("rust udf create returned null state after reload");
        }

        self.lib_state = new_state;
        self.lib = lib;
        self.api = api;
        self.state = state;
        Ok(true)
    }

    pub fn call_typed_columns_to_typed_results(
        &mut self,
        _method: &str,
        columns: &[InputColumn],
    ) -> Result<Vec<InputColumn>> {
        self.call_eval(columns, &[])
    }

    pub fn call_typed_columns_to_named_results(
        &mut self,
        _method: &str,
        columns: &[InputColumn],
        output_names: &[String],
    ) -> Result<Vec<InputColumn>> {
        self.call_eval(columns, output_names)
    }

    fn call_eval(&mut self, columns: &[InputColumn], output_names: &[String]) -> Result<Vec<InputColumn>> {
        let batch = RustInputBatch::new(columns)?;
        let output_views = build_str_views(output_names);
        let function_view = str_to_view(&self.function_class);

        let result_ptr = unsafe {
            (self.api.eval)(
                self.state,
                function_view,
                batch.columns.as_ptr(),
                batch.columns.len(),
                output_views.as_ptr(),
                output_views.len(),
            )
        };
        if result_ptr.is_null() {
            bail!("rust udf eval returned null result");
        }

        let output = unsafe { decode_result(result_ptr)? };
        unsafe { (self.api.free_result)(result_ptr) };
        Ok(output)
    }
}

impl Drop for RustUdfHandle {
    fn drop(&mut self) {
        unsafe {
            (self.api.drop)(self.state);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LibState {
    modified: Option<SystemTime>,
    len: u64,
}

fn lib_state(path: &Path) -> Result<LibState> {
    let meta = std::fs::metadata(path).with_context(|| format!("metadata {}", path.display()))?;
    Ok(LibState {
        modified: meta.modified().ok(),
        len: meta.len(),
    })
}

#[derive(Clone, Copy, Debug)]
struct UdfApi {
    create: unsafe extern "C" fn() -> *mut c_void,
    drop: unsafe extern "C" fn(*mut c_void),
    eval: unsafe extern "C" fn(
        *mut c_void,
        UdfStrView,
        *const UdfInputColumn,
        usize,
        *const UdfStrView,
        usize,
    ) -> *mut UdfResult,
    free_result: unsafe extern "C" fn(*mut UdfResult),
}

impl UdfApi {
    unsafe fn load(lib: &Library) -> Result<Self> {
        let abi_version: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib.get(SYMBOL_ABI_VERSION)?;
        let version = abi_version();
        if version != UDF_ABI_VERSION {
            bail!("rust udf ABI mismatch: host {} vs plugin {}", UDF_ABI_VERSION, version);
        }

        let create: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> = lib.get(SYMBOL_CREATE)?;
        let drop_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib.get(SYMBOL_DROP)?;
        let eval: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                UdfStrView,
                *const UdfInputColumn,
                usize,
                *const UdfStrView,
                usize,
            ) -> *mut UdfResult,
        > = lib.get(SYMBOL_EVAL)?;
        let free_result: libloading::Symbol<unsafe extern "C" fn(*mut UdfResult)> = lib.get(SYMBOL_FREE_RESULT)?;

        Ok(Self {
            create: *create,
            drop: *drop_fn,
            eval: *eval,
            free_result: *free_result,
        })
    }
}

struct RustInputBatch {
    columns: Vec<UdfInputColumn>,
    _buffers: Vec<Buffer>,
}

enum Buffer {
    Bytes { _buf: Vec<u8> },
    StrViews { _buf: Vec<UdfStrView> },
}

impl RustInputBatch {
    fn new(columns: &[InputColumn]) -> Result<Self> {
        let mut out = Vec::with_capacity(columns.len());
        let mut buffers = Vec::new();

        for column in columns {
            match column {
                InputColumn::String(values) => {
                    let (views, nulls) = build_string_views(values);
                    let views_ptr = views.as_ptr();
                    let nulls_ptr = nulls.as_ref().map(|v| v.as_ptr()).unwrap_or(std::ptr::null());
                    let len = values.len();

                    buffers.push(Buffer::StrViews { _buf: views });
                    if let Some(nulls) = nulls {
                        buffers.push(Buffer::Bytes { _buf: nulls });
                    }

                    out.push(UdfInputColumn {
                        kind: UdfColumnType::String,
                        len,
                        values: std::ptr::null(),
                        nulls: nulls_ptr,
                        strings: views_ptr,
                    });
                }
                InputColumn::I64 { values, is_null } => {
                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::I64,
                        len: values.len(),
                        values: values.as_ptr() as *const u8,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
                InputColumn::I32 { values, is_null } => {
                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::I32,
                        len: values.len(),
                        values: values.as_ptr() as *const u8,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
                InputColumn::F64 { values, is_null } => {
                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::F64,
                        len: values.len(),
                        values: values.as_ptr() as *const u8,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
                InputColumn::F32 { values, is_null } => {
                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::F32,
                        len: values.len(),
                        values: values.as_ptr() as *const u8,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
                InputColumn::Bool { values, is_null } => {
                    let mut raw = Vec::with_capacity(values.len());
                    for v in values {
                        raw.push(if *v { 1u8 } else { 0u8 });
                    }
                    let values_ptr = raw.as_ptr();
                    buffers.push(Buffer::Bytes { _buf: raw });

                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::Bool,
                        len: values.len(),
                        values: values_ptr,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
                InputColumn::Decimal128 { values, is_null } => {
                    let (nulls_ptr, nulls_buf) = build_nulls(is_null.as_deref());
                    if let Some(buf) = nulls_buf {
                        buffers.push(Buffer::Bytes { _buf: buf });
                    }
                    out.push(UdfInputColumn {
                        kind: UdfColumnType::Decimal128,
                        len: values.len(),
                        values: values.as_ptr() as *const u8,
                        nulls: nulls_ptr,
                        strings: std::ptr::null(),
                    });
                }
            }
        }

        Ok(Self {
            columns: out,
            _buffers: buffers,
        })
    }
}

fn build_nulls(nulls: Option<&[bool]>) -> (*const u8, Option<Vec<u8>>) {
    let Some(nulls) = nulls else {
        return (std::ptr::null(), None);
    };
    let mut raw = Vec::with_capacity(nulls.len());
    for v in nulls {
        raw.push(if *v { 1u8 } else { 0u8 });
    }
    let ptr = raw.as_ptr();
    (ptr, Some(raw))
}

fn build_string_views(values: &[Option<String>]) -> (Vec<UdfStrView>, Option<Vec<u8>>) {
    let mut views = Vec::with_capacity(values.len());
    let mut nulls: Option<Vec<u8>> = None;

    for (idx, value) in values.iter().enumerate() {
        match value {
            Some(v) => {
                let bytes = v.as_bytes();
                views.push(UdfStrView {
                    ptr: bytes.as_ptr(),
                    len: bytes.len(),
                });
                if let Some(nulls) = nulls.as_mut() {
                    nulls.push(0);
                }
            }
            None => {
                views.push(UdfStrView::empty());
                if let Some(nulls) = nulls.as_mut() {
                    nulls.push(1);
                } else {
                    let mut new = vec![0u8; idx];
                    new.push(1);
                    nulls = Some(new);
                }
            }
        }
    }

    (views, nulls)
}

fn build_str_views(values: &[String]) -> Vec<UdfStrView> {
    values
        .iter()
        .map(|v| UdfStrView {
            ptr: v.as_bytes().as_ptr(),
            len: v.len(),
        })
        .collect()
}

fn str_to_view(value: &str) -> UdfStrView {
    UdfStrView {
        ptr: value.as_bytes().as_ptr(),
        len: value.len(),
    }
}

unsafe fn decode_result(result_ptr: *mut UdfResult) -> Result<Vec<InputColumn>> {
    let result = &*result_ptr;
    if result.len == 0 {
        return Ok(Vec::new());
    }
    if result.columns.is_null() {
        bail!("rust udf returned null columns");
    }

    let columns = std::slice::from_raw_parts(result.columns, result.len);
    let mut out = Vec::with_capacity(columns.len());

    for column in columns {
        let len = column.len;
        let nulls = if column.nulls.is_null() {
            None
        } else {
            let raw = std::slice::from_raw_parts(column.nulls, len);
            Some(raw.iter().map(|v| *v != 0).collect::<Vec<bool>>())
        };

        match column.kind {
            UdfColumnType::String => {
                if column.strings.is_null() {
                    bail!("rust udf string column missing strings pointer");
                }
                let views = std::slice::from_raw_parts(column.strings, len);
                let mut values = Vec::with_capacity(len);
                for (idx, view) in views.iter().enumerate() {
                    if nulls
                        .as_ref()
                        .map(|v| v.get(idx).copied().unwrap_or(false))
                        .unwrap_or(false)
                    {
                        values.push(None);
                        continue;
                    }
                    if view.ptr.is_null() {
                        values.push(None);
                        continue;
                    }
                    let bytes = std::slice::from_raw_parts(view.ptr, view.len);
                    let s = std::str::from_utf8(bytes).with_context(|| "invalid utf-8 in rust udf output")?;
                    values.push(Some(s.to_string()));
                }
                out.push(InputColumn::String(values));
            }
            UdfColumnType::I64 => {
                let values = std::slice::from_raw_parts(column.values as *const i64, len).to_vec();
                out.push(InputColumn::I64 { values, is_null: nulls });
            }
            UdfColumnType::I32 => {
                let values = std::slice::from_raw_parts(column.values as *const i32, len).to_vec();
                out.push(InputColumn::I32 { values, is_null: nulls });
            }
            UdfColumnType::F64 => {
                let values = std::slice::from_raw_parts(column.values as *const f64, len).to_vec();
                out.push(InputColumn::F64 { values, is_null: nulls });
            }
            UdfColumnType::F32 => {
                let values = std::slice::from_raw_parts(column.values as *const f32, len).to_vec();
                out.push(InputColumn::F32 { values, is_null: nulls });
            }
            UdfColumnType::Bool => {
                let raw = std::slice::from_raw_parts(column.values, len);
                let values = raw.iter().map(|v| *v != 0).collect::<Vec<bool>>();
                out.push(InputColumn::Bool { values, is_null: nulls });
            }
            UdfColumnType::Decimal128 => {
                let values = std::slice::from_raw_parts(column.values as *const i128, len).to_vec();
                out.push(InputColumn::Decimal128 { values, is_null: nulls });
            }
        }
    }

    Ok(out)
}
