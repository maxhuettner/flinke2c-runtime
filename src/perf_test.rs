//! Local perf comparison of the Rust and Java CurrencyConversionFunction.
//!
//! Run via `scripts/perf_currency.sh` (builds the Rust UDF for the host first).
//! Both paths go through `UdfHandle`, so the numbers include the same
//! marshalling the runtime does per batch.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::java_udf::{InputColumn, set_jvm_opts};
use crate::udf::{UdfHandle, UdfLanguage};

const UDF_CLASS: &str = "org.example.flinke2c.CurrencyConversionFunction";
const ADAPTER_CLASS: &str = "org.example.flinke2c.runtime.ScalarFunctionAdapter";
const METHOD: &str = "evalBatchFast"; // runtime appends "Fast" to the configured udf_method

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn make_input(rows: usize) -> InputColumn {
    // DECIMAL(23,3) unscaled values, ~5% nulls, deterministic LCG.
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut values = Vec::with_capacity(rows);
    let mut nulls = Vec::with_capacity(rows);
    for _ in 0..rows {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        values.push(((state >> 24) % 10_000_000_000) as i128);
        nulls.push((state >> 60) == 0);
    }
    InputColumn::Decimal128 {
        values,
        is_null: Some(nulls),
    }
}

fn decimals(cols: &[InputColumn]) -> (&[i128], Option<&[bool]>) {
    match &cols[0] {
        InputColumn::Decimal128 { values, is_null } => (values, is_null.as_deref()),
        other => panic!("unexpected output column {other:?}"),
    }
}

fn bench(handle: &mut UdfHandle, input: &[InputColumn], warmup: usize, iters: usize) -> (Duration, Vec<Duration>) {
    for _ in 0..warmup {
        handle.call_typed_columns_to_typed_results(METHOD, input).unwrap();
    }
    let mut samples = Vec::with_capacity(iters);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let out = handle.call_typed_columns_to_typed_results(METHOD, input).unwrap();
        samples.push(t.elapsed());
        std::hint::black_box(out);
    }
    (total.elapsed(), samples)
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore = "perf test; run with scripts/perf_currency.sh"]
fn currency_conversion_rust_vs_java() {
    let rust_lib = PathBuf::from(std::env::var("RUST_UDF_LIB").expect("RUST_UDF_LIB must point at the host-built dylib"));
    let jars: Vec<PathBuf> = ["jar/flinke2c.jar", "jar/udf-adapter.jar", "jar/flink-stubs.jar"]
        .iter()
        .map(PathBuf::from)
        .collect();
    let arg_types = vec!["DECIMAL(23,3)".to_string()];

    set_jvm_opts(
        std::env::var("JVM_OPTS")
            .unwrap_or_else(|_| "-Xms256m -Xmx512m -XX:+UseG1GC".to_string())
            .split(' ')
            .map(str::to_string)
            .collect(),
    );

    let mut java = UdfHandle::new(UdfLanguage::Java, &jars, ADAPTER_CLASS, UDF_CLASS, &arg_types, &PathBuf::new()).unwrap();
    let mut rust = UdfHandle::new(UdfLanguage::Rust, &[], ADAPTER_CLASS, UDF_CLASS, &arg_types, &rust_lib).unwrap();

    let warmup = env_usize("PERF_WARMUP", 200);
    let iters = env_usize("PERF_ITERS", 500);
    let batch_sizes: Vec<usize> = std::env::var("PERF_BATCH_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![100, 1_000, 10_000, 100_000]);

    println!("warmup={warmup} iters={iters}");
    println!(
        "{:>9} {:>6} {:>10} {:>10} {:>10} {:>12} {:>9}",
        "batch", "impl", "p50 us", "p99 us", "ns/row", "rows/s", "speedup"
    );

    for rows in batch_sizes {
        let input = vec![make_input(rows)];

        // Correctness: both implementations must agree before timing means anything.
        let j_out = java.call_typed_columns_to_typed_results(METHOD, &input).unwrap();
        let r_out = rust.call_typed_columns_to_typed_results(METHOD, &input).unwrap();
        let (jv, jn) = decimals(&j_out);
        let (rv, rn) = decimals(&r_out);
        assert_eq!(jv.len(), rv.len());
        for i in 0..rows {
            let j_null = jn.map_or(false, |n| n[i]);
            let r_null = rn.map_or(false, |n| n[i]);
            assert_eq!(j_null, r_null, "null mismatch at row {i}");
            if !j_null {
                assert_eq!(jv[i], rv[i], "value mismatch at row {i}");
            }
        }

        let mut baseline_ns = 0.0;
        for (name, handle) in [("java", &mut java), ("rust", &mut rust)] {
            let (total, mut samples) = bench(handle, &input, warmup, iters);
            samples.sort();
            let ns_per_row = total.as_nanos() as f64 / (iters * rows) as f64;
            let rows_per_s = 1e9 / ns_per_row;
            let speedup = if name == "java" {
                baseline_ns = ns_per_row;
                "1.00x".to_string()
            } else {
                format!("{:.2}x", baseline_ns / ns_per_row)
            };
            println!(
                "{:>9} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>9}",
                rows,
                name,
                pct(&samples, 0.50).as_secs_f64() * 1e6,
                pct(&samples, 0.99).as_secs_f64() * 1e6,
                ns_per_row,
                rows_per_s,
                speedup
            );
        }
    }
}
