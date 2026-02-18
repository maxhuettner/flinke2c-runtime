use std::ffi::c_void;

pub const UDF_ABI_VERSION: u32 = 1;
pub mod helpers;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UdfColumnType {
    String = 0,
    I64 = 1,
    I32 = 2,
    F64 = 3,
    F32 = 4,
    Bool = 5,
    Decimal128 = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UdfStrView {
    pub ptr: *const u8,
    pub len: usize,
}

impl UdfStrView {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UdfInputColumn {
    pub kind: UdfColumnType,
    pub len: usize,
    pub values: *const u8,
    pub nulls: *const u8,
    pub strings: *const UdfStrView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UdfOutputColumn {
    pub kind: UdfColumnType,
    pub len: usize,
    pub values: *const u8,
    pub nulls: *const u8,
    pub strings: *const UdfStrView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UdfResult {
    pub columns: *const UdfOutputColumn,
    pub len: usize,
    pub private: *mut c_void,
}
