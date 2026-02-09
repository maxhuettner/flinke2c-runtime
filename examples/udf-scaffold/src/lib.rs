use std::ffi::c_void;
use std::ptr;

use udf_abi::{UdfInputColumn, UdfResult, UdfStrView, UDF_ABI_VERSION};

struct UdfState;

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
    drop(Box::from_raw(result));
}

#[no_mangle]
pub unsafe extern "C" fn flinke2c_runtime_udf_eval(
    state: *mut c_void,
    _function_class: UdfStrView,
    _columns: *const UdfInputColumn,
    _num_columns: usize,
    _output_names: *const UdfStrView,
    _num_output_names: usize,
) -> *mut UdfResult {
    if state.is_null() {
        return ptr::null_mut();
    }

    // TODO: parse function_class and implement columnar UDF logic.
    let result = UdfResult {
        columns: ptr::null(),
        len: 0,
        private: ptr::null_mut(),
    };
    Box::into_raw(Box::new(result))
}
