use anyhow::{bail, Context, Result};
use std::ffi::c_void;
use std::ptr;

use udf_abi::{UdfColumnType, UdfInputColumn, UdfOutputColumn, UdfResult, UdfStrView, UDF_ABI_VERSION};

const CLASS_NAME: &str = "org.example.flinke2c.ImputationFunction";

const HISTORY_SIZE: usize = 5_000;
const SEARCH_LIMIT: usize = 512;
const K: usize = 10;

const EPS: f64 = 1e-6;
const W_BIDDER: f64 = 0.25;
const W_TIME: f64 = 1.0;
const W_STR: f64 = 0.25;

const DEFAULT_PRICE_UNSCALED: i128 = 0; // 0.000 with scale 3
const DEFAULT_LONG: i64 = 0;
const DEFAULT_CHANNEL: &str = "unknown";
const DEFAULT_STRING: &str = "";
const DEFAULT_TIMESTAMP_MILLIS: i64 = 0;

const PRICE_SCALE_FACTOR: f64 = 1000.0; // 10^3

#[derive(Clone, Copy, Debug, Default)]
struct Obs {
    has_price: bool,
    price_double: f64,
    bidder_id: i64,
    ts_seconds: f64,
    channel_hash: i32,
    url_hash: i32,
    extra_hash: i32,
}

impl Obs {
    fn from_parts(bidder_id: i64, ts_seconds: f64, channel: &str, url: &str, extra: &str) -> Self {
        Self {
            has_price: false,
            price_double: 0.0,
            bidder_id,
            ts_seconds,
            channel_hash: hash_or_zero(channel),
            url_hash: hash_or_zero(url),
            extra_hash: hash_or_zero(extra),
        }
    }
}

#[derive(Debug)]
struct BoundedRing {
    capacity: usize,
    buffer: Vec<Obs>,
    start: usize,
    size: usize,
}

impl BoundedRing {
    fn new() -> Self {
        Self {
            capacity: HISTORY_SIZE,
            buffer: vec![Obs::default(); HISTORY_SIZE],
            start: 0,
            size: 0,
        }
    }

    fn add(&mut self, obs: Obs) {
        if self.size < self.capacity {
            let idx = (self.start + self.size) % self.capacity;
            self.buffer[idx] = obs;
            self.size += 1;
        } else {
            self.buffer[self.start] = obs;
            self.start = (self.start + 1) % self.capacity;
        }
    }

    fn snapshot_last(&self, limit: usize) -> Vec<Obs> {
        let n = self.size.min(limit);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let idx = (self.start + self.size - 1 - i + self.capacity) % self.capacity;
            out.push(self.buffer[idx]);
        }
        out
    }
}

struct UdfState {
    history: BoundedRing,
}

struct ResultStorage {
    columns: Vec<UdfOutputColumn>,
    buffers: Vec<Buffer>,
}

enum Buffer {
    I64 { _buf: Vec<i64> },
    I128 { _buf: Vec<i128> },
    StrViews { _buf: Vec<UdfStrView> },
    Strings { _buf: Vec<String> },
}

#[no_mangle]
pub extern "C" fn proxy_udf_abi_version() -> u32 {
    UDF_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn proxy_udf_create() -> *mut c_void {
    let state = UdfState {
        history: BoundedRing::new(),
    };
    Box::into_raw(Box::new(state)) as *mut c_void
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
    output_names: *const UdfStrView,
    num_output_names: usize,
) -> *mut UdfResult {
    if state.is_null() {
        return ptr::null_mut();
    }

    if let Err(err) = validate_class(function_class) {
        eprintln!("rust udf: {err:#}");
        return ptr::null_mut();
    }

    let state = &mut *(state as *mut UdfState);
    let input_columns = if columns.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(columns, num_columns)
    };

    let output_views = if output_names.is_null() || num_output_names == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(output_names, num_output_names)
    };

    let output = match eval_imputation(state, input_columns, output_views) {
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

fn eval_imputation(
    state: &mut UdfState,
    columns: &[UdfInputColumn],
    output_names: &[UdfStrView],
) -> Result<ResultStorage> {
    if output_names.is_empty() {
        bail!("ImputationFunction requires named output columns");
    }
    if columns.len() != 7 {
        bail!("ImputationFunction expects 7 input columns, got {}", columns.len());
    }

    let row_count = ensure_column_lengths(columns)?;

    let (price_values, price_nulls) = expect_i128(&columns[0], "price")?;
    let (auction_values, auction_nulls) = expect_i64(&columns[1], "auction")?;
    let (bidder_values, bidder_nulls) = expect_i64(&columns[2], "bidder")?;
    let (channel_values, channel_nulls) = expect_strings(&columns[3], "channel")?;
    let (url_values, url_nulls) = expect_strings(&columns[4], "url")?;
    let (dt_values, dt_nulls) = expect_i64(&columns[5], "dateTime")?;
    let (extra_values, extra_nulls) = expect_strings(&columns[6], "extra")?;

    let mut out_price = Vec::with_capacity(row_count);
    let mut out_auction = Vec::with_capacity(row_count);
    let mut out_bidder = Vec::with_capacity(row_count);
    let mut out_channel = Vec::with_capacity(row_count);
    let mut out_url = Vec::with_capacity(row_count);
    let mut out_dt = Vec::with_capacity(row_count);
    let mut out_extra = Vec::with_capacity(row_count);

    for row in 0..row_count {
        let price_unscaled = if is_null_at(price_nulls, row) {
            None
        } else {
            Some(*price_values.get(row).unwrap_or(&0))
        };

        let auction = if is_null_at(auction_nulls, row) {
            DEFAULT_LONG
        } else {
            *auction_values.get(row).unwrap_or(&DEFAULT_LONG)
        };
        let bidder = if is_null_at(bidder_nulls, row) {
            DEFAULT_LONG
        } else {
            *bidder_values.get(row).unwrap_or(&DEFAULT_LONG)
        };

        let channel = string_or_default(channel_values, channel_nulls, row, DEFAULT_CHANNEL);
        let url = string_or_default(url_values, url_nulls, row, DEFAULT_STRING);
        let extra = string_or_default(extra_values, extra_nulls, row, DEFAULT_STRING);

        let dt_millis = if is_null_at(dt_nulls, row) {
            DEFAULT_TIMESTAMP_MILLIS
        } else {
            *dt_values.get(row).unwrap_or(&DEFAULT_TIMESTAMP_MILLIS)
        };

        let ts_seconds = (dt_millis as f64) / 1000.0;
        let mut obs = Obs::from_parts(bidder, ts_seconds, channel, url, extra);

        let price_out = if let Some(unscaled) = price_unscaled {
            obs.has_price = true;
            obs.price_double = (unscaled as f64) / PRICE_SCALE_FACTOR;
            state.history.add(obs);
            unscaled
        } else {
            let imputed = knn_impute_price(&state.history, &obs);
            if imputed.is_nan() {
                DEFAULT_PRICE_UNSCALED
            } else {
                round_half_up(imputed, 3)?
            }
        };

        out_price.push(price_out);
        out_auction.push(auction);
        out_bidder.push(bidder);
        out_channel.push(channel.to_string());
        out_url.push(url.to_string());
        out_dt.push(dt_millis);
        out_extra.push(extra.to_string());
    }

    let mut storage = ResultStorage {
        columns: Vec::new(),
        buffers: Vec::new(),
    };

    let output_names = output_names
        .iter()
        .map(|v| view_to_string(*v))
        .collect::<Result<Vec<_>>>()?;

    let mut used = std::collections::HashSet::new();
    for name in output_names.iter() {
        if !used.insert(name.as_str()) {
            bail!("duplicate output field {name}");
        }
        match name.as_str() {
            "price" => push_i128_column(&mut storage, &out_price),
            "auction" => push_i64_column(&mut storage, &out_auction),
            "bidder" => push_i64_column(&mut storage, &out_bidder),
            "channel" => push_string_column(&mut storage, &out_channel),
            "url" => push_string_column(&mut storage, &out_url),
            "dateTime" => push_i64_column(&mut storage, &out_dt),
            "extra" => push_string_column(&mut storage, &out_extra),
            other => bail!("unknown output field {other}"),
        }
    }

    Ok(storage)
}

fn push_i64_column(storage: &mut ResultStorage, values: &[i64]) {
    let owned = values.to_vec();
    let ptr = owned.as_ptr();
    storage.buffers.push(Buffer::I64 { _buf: owned });
    storage.columns.push(UdfOutputColumn {
        kind: UdfColumnType::I64,
        len: values.len(),
        values: ptr as *const u8,
        nulls: ptr::null(),
        strings: ptr::null(),
    });
}

fn push_i128_column(storage: &mut ResultStorage, values: &[i128]) {
    let owned = values.to_vec();
    let ptr = owned.as_ptr();
    storage.buffers.push(Buffer::I128 { _buf: owned });
    storage.columns.push(UdfOutputColumn {
        kind: UdfColumnType::Decimal128,
        len: values.len(),
        values: ptr as *const u8,
        nulls: ptr::null(),
        strings: ptr::null(),
    });
}

fn push_string_column(storage: &mut ResultStorage, values: &[String]) {
    let strings = values.to_vec();
    let mut views = Vec::with_capacity(strings.len());
    for s in &strings {
        let bytes = s.as_bytes();
        views.push(UdfStrView {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        });
    }
    let views_ptr = views.as_ptr();
    storage.buffers.push(Buffer::Strings { _buf: strings });
    storage.buffers.push(Buffer::StrViews { _buf: views });
    storage.columns.push(UdfOutputColumn {
        kind: UdfColumnType::String,
        len: values.len(),
        values: ptr::null(),
        nulls: ptr::null(),
        strings: views_ptr,
    });
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

fn expect_i64<'a>(
    column: &'a UdfInputColumn,
    label: &str,
) -> Result<(&'a [i64], Option<&'a [u8]>)> {
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

fn expect_strings<'a>(
    column: &'a UdfInputColumn,
    label: &str,
) -> Result<(&'a [UdfStrView], Option<&'a [u8]>)> {
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

fn view_to_string(view: UdfStrView) -> Result<String> {
    if view.ptr.is_null() {
        bail!("null string pointer");
    }
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    let s = std::str::from_utf8(bytes).context("invalid utf-8")?;
    Ok(s.to_string())
}

fn is_null_at(nulls: Option<&[u8]>, row: usize) -> bool {
    nulls
        .and_then(|vals| vals.get(row))
        .map(|v| *v != 0)
        .unwrap_or(false)
}

fn string_or_default<'a>(
    values: &'a [UdfStrView],
    nulls: Option<&[u8]>,
    row: usize,
    default: &'a str,
) -> &'a str {
    if is_null_at(nulls, row) {
        return default;
    }
    let view = values.get(row).copied().unwrap_or(UdfStrView::empty());
    if view.ptr.is_null() {
        return default;
    }
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    std::str::from_utf8(bytes).unwrap_or(default)
}

fn knn_impute_price(history: &BoundedRing, target: &Obs) -> f64 {
    let snap = history.snapshot_last(SEARCH_LIMIT.min(HISTORY_SIZE));
    if snap.is_empty() {
        return f64::NAN;
    }

    let mut best_dist = [0.0_f64; K];
    let mut best_price = [0.0_f64; K];
    let mut found = 0usize;

    for obs in snap.iter() {
        if !obs.has_price {
            continue;
        }

        let dist = distance(target, obs);
        if found < K {
            best_dist[found] = dist;
            best_price[found] = obs.price_double;
            found += 1;
        } else {
            let mut worst_idx = 0usize;
            let mut worst = best_dist[0];
            for j in 1..K {
                if best_dist[j] > worst {
                    worst = best_dist[j];
                    worst_idx = j;
                }
            }
            if dist < worst {
                best_dist[worst_idx] = dist;
                best_price[worst_idx] = obs.price_double;
            }
        }
    }

    if found == 0 {
        return f64::NAN;
    }

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..found {
        let w = 1.0 / (best_dist[i] + EPS);
        num += best_price[i] * w;
        den += w;
    }
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

fn distance(t: &Obs, o: &Obs) -> f64 {
    let mut s = 0.0;

    s += W_BIDDER * if t.bidder_id == o.bidder_id { 0.0 } else { 1.0 };

    let dt = t.ts_seconds - o.ts_seconds;
    s += W_TIME * (dt * dt) * 1e-8;

    s += W_STR * if t.channel_hash == o.channel_hash { 0.0 } else { 1.0 };
    s += W_STR * if t.url_hash == o.url_hash { 0.0 } else { 1.0 };
    s += W_STR * if t.extra_hash == o.extra_hash { 0.0 } else { 1.0 };

    s
}

fn is_blank(value: &str) -> bool {
    value.trim_matches(|c| c <= ' ').is_empty()
}

fn hash_or_zero(value: &str) -> i32 {
    if is_blank(value) {
        0
    } else {
        murmur_like_hash(value)
    }
}

fn murmur_like_hash(value: &str) -> i32 {
    let mut h: i32 = 0x9747b28c_u32 as i32;
    for &byte in value.as_bytes() {
        let b = byte as i8 as i32;
        h ^= b;
        h = h.wrapping_mul(0x5bd1e995_u32 as i32);
        let shifted = ((h as u32) >> 15) as i32;
        h ^= shifted;
    }
    h
}

fn round_half_up(value: f64, scale: i32) -> Result<i128> {
    let factor = 10f64.powi(scale);
    let scaled = value * factor;
    if !scaled.is_finite() {
        return Ok(DEFAULT_PRICE_UNSCALED);
    }
    let rounded = scaled.round();
    if rounded > i128::MAX as f64 || rounded < i128::MIN as f64 {
        bail!("decimal overflow while rounding imputed price");
    }
    Ok(rounded as i128)
}
