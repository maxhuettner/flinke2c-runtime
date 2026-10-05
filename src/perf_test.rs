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

/// The UDF handle API takes borrowed columns; adapt an owned column vector.
fn refs(columns: &[InputColumn]) -> Vec<&InputColumn> {
    columns.iter().collect()
}

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
        handle.call_typed_columns_to_typed_results(METHOD, &refs(input)).unwrap();
    }
    let mut samples = Vec::with_capacity(iters);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let out = handle.call_typed_columns_to_typed_results(METHOD, &refs(input)).unwrap();
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
        let j_out = java.call_typed_columns_to_typed_results(METHOD, &refs(&input)).unwrap();
        let r_out = rust.call_typed_columns_to_typed_results(METHOD, &refs(&input)).unwrap();
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

// ImputationFunction: 7 inputs (3 strings, 1 timestamp), POJO output, KNN over a history ring.

const IMPUTATION_CLASS: &str = "org.example.flinke2c.ImputationFunction";
const IMPUTATION_ARG_TYPES: [&str; 7] = [
    "DECIMAL(23,3)",
    "BIGINT",
    "BIGINT",
    "STRING",
    "STRING",
    "TIMESTAMP(3)",
    "STRING",
];
const IMPUTATION_OUTPUTS: [&str; 7] = ["price", "auction", "bidder", "channel", "url", "dateTime", "extra"];
const NAMED_METHOD: &str = "evalBatchFastNamed";

fn make_imputation_input(rows: usize, null_pct: u64) -> Vec<InputColumn> {
    let channels = ["Google", "Facebook", "Baidu", "Apple"];
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 17
    };

    let mut price = Vec::with_capacity(rows);
    let mut price_null = Vec::with_capacity(rows);
    let mut auction = Vec::with_capacity(rows);
    let mut bidder = Vec::with_capacity(rows);
    let mut channel = Vec::with_capacity(rows);
    let mut url = Vec::with_capacity(rows);
    let mut dt = Vec::with_capacity(rows);
    let mut extra = Vec::with_capacity(rows);
    for i in 0..rows {
        let r = next();
        price.push(((r % 1_000_000_000) as i128) + 1);
        price_null.push(next() % 100 < null_pct);
        auction.push((next() % 10_000) as i64);
        bidder.push((next() % 5_000) as i64);
        // Mix in non-ASCII, empty, blank and null strings to exercise the packed string path.
        let ch = match next() % 25 {
            0 => None,
            1 => Some(String::new()),
            2 => Some(String::new()),
            3 => Some("Zürich € 東京 \u{1F600}".to_string()),
            _ => Some(channels[(next() % 4) as usize].to_string()),
        };
        channel.push(ch);
        url.push(Some(format!("https://www.nexmark.com/{}/item.htm?query=1&id={}", next() % 1000, next() % 100000)));
        dt.push(1_700_000_000_000i64 + (i as i64) * 3);
        extra.push(Some("x".repeat(90) + &format!("{:010}", next() % 10_000_000_000)));
    }
    vec![
        InputColumn::Decimal128 { values: price, is_null: Some(price_null) },
        InputColumn::I64 { values: auction, is_null: None },
        InputColumn::I64 { values: bidder, is_null: None },
        InputColumn::String(channel),
        InputColumn::String(url),
        InputColumn::I64 { values: dt, is_null: None },
        InputColumn::String(extra),
    ]
}

fn bench_named(handle: &mut UdfHandle, input: &[InputColumn], names: &[String], warmup: usize, iters: usize) -> (Duration, Vec<Duration>) {
    for _ in 0..warmup {
        handle.call_typed_columns_to_named_results(NAMED_METHOD, &refs(input), names).unwrap();
    }
    let mut samples = Vec::with_capacity(iters);
    let total = Instant::now();
    for _ in 0..iters {
        let t = Instant::now();
        let out = handle.call_typed_columns_to_named_results(NAMED_METHOD, &refs(input), names).unwrap();
        samples.push(t.elapsed());
        std::hint::black_box(out);
    }
    (total.elapsed(), samples)
}

#[test]
#[ignore = "perf test; run with scripts/perf_imputation.sh"]
fn imputation_rust_vs_java() {
    let rust_lib = PathBuf::from(std::env::var("RUST_UDF_LIB").expect("RUST_UDF_LIB must point at the host-built dylib"));
    let jars: Vec<PathBuf> = ["jar/flinke2c.jar", "jar/udf-adapter.jar", "jar/flink-stubs.jar"]
        .iter()
        .map(PathBuf::from)
        .collect();
    // PERF_SERVER_CONFIG=1 mimics the config the Flink PRE operator sends (verbose type names,
    // a single named output).
    let server_cfg = std::env::var("PERF_SERVER_CONFIG").is_ok();
    let arg_types: Vec<String> = if server_cfg {
        ["DECIMAL(23, 3)", "BIGINT", "BIGINT", "VARCHAR(2147483647)", "VARCHAR(2147483647)", "TIMESTAMP(3)", "VARCHAR(2147483647)"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        IMPUTATION_ARG_TYPES.iter().map(|s| s.to_string()).collect()
    };
    let names: Vec<String> = if server_cfg {
        vec!["price".to_string()]
    } else {
        IMPUTATION_OUTPUTS.iter().map(|s| s.to_string()).collect()
    };

    set_jvm_opts(
        std::env::var("JVM_OPTS")
            .unwrap_or_else(|_| "-Xms256m -Xmx512m -XX:+UseG1GC".to_string())
            .split(' ')
            .map(str::to_string)
            .collect(),
    );

    let mut java = UdfHandle::new(UdfLanguage::Java, &jars, ADAPTER_CLASS, IMPUTATION_CLASS, &arg_types, &PathBuf::new()).unwrap();
    let mut rust = UdfHandle::new(UdfLanguage::Rust, &[], ADAPTER_CLASS, IMPUTATION_CLASS, &arg_types, &rust_lib).unwrap();

    let warmup = env_usize("PERF_WARMUP", 100);
    let iters = env_usize("PERF_ITERS", 200);
    let null_pct = env_usize("PERF_NULL_PCT", 10) as u64;
    let batch_sizes: Vec<usize> = std::env::var("PERF_BATCH_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1_000, 10_000]);

    println!("warmup={warmup} iters={iters} null_pct={null_pct}");
    println!(
        "{:>9} {:>6} {:>10} {:>10} {:>10} {:>12} {:>9}",
        "batch", "impl", "p50 us", "p99 us", "ns/row", "rows/s", "speedup"
    );

    for rows in batch_sizes {
        let input = make_imputation_input(rows, null_pct);

        // Both keep their own history, so compare the very first batch after fresh construction.
        if rows == *batch_sizes_first(&std::env::var("PERF_BATCH_SIZES").ok()).get_or_insert(rows) {
            let j_out = java.call_typed_columns_to_named_results(NAMED_METHOD, &refs(&input), &names).unwrap();
            let r_out = rust.call_typed_columns_to_named_results(NAMED_METHOD, &refs(&input), &names).unwrap();
            if let (InputColumn::Decimal128 { values: jv, .. }, InputColumn::Decimal128 { values: rv, .. }) = (&j_out[0], &r_out[0]) {
                let mismatches = jv.iter().zip(rv).filter(|(a, b)| (**a - **b).abs() > 1).count();
                println!("  first-batch price mismatches (>0.001): {mismatches} / {rows}");
            }
            for col in 1..7 {
                let same = match (&j_out[col], &r_out[col]) {
                    (InputColumn::String(a), InputColumn::String(b)) => a == b,
                    (InputColumn::I64 { values: a, .. }, InputColumn::I64 { values: b, .. }) => a == b,
                    _ => false,
                };
                println!("  first-batch output column {} ({}) identical: {same}", col, IMPUTATION_OUTPUTS[col]);
                if !same {
                    if let (InputColumn::String(a), InputColumn::String(b), InputColumn::String(inp)) = (&j_out[col], &r_out[col], &input[col + 0]) {
                        for i in 0..a.len() {
                            if a[i] != b[i] {
                                println!("    row {i}: java={:?} rust={:?}", a[i], b[i]);
                                break;
                            }
                        }
                        let _ = inp;
                    }
                }
            }
        }

        // Optional: compare against a reference (e.g. original) flinke2c jar, exact equality.
        if let Ok(ref_jar) = std::env::var("PERF_REF_JAR") {
            let ref_jars: Vec<PathBuf> = [ref_jar.as_str(), "jar/udf-adapter.jar", "jar/flink-stubs.jar"]
                .iter()
                .map(PathBuf::from)
                .collect();
            let mut reference =
                UdfHandle::new(UdfLanguage::Java, &ref_jars, ADAPTER_CLASS, IMPUTATION_CLASS, &arg_types, &PathBuf::new()).unwrap();
            let mut fresh =
                UdfHandle::new(UdfLanguage::Java, &jars, ADAPTER_CLASS, IMPUTATION_CLASS, &arg_types, &PathBuf::new()).unwrap();
            for round in 0..3 {
                let a = reference.call_typed_columns_to_named_results(NAMED_METHOD, &refs(&input), &names).unwrap();
                let b = fresh.call_typed_columns_to_named_results(NAMED_METHOD, &refs(&input), &names).unwrap();
                let same = a.len() == b.len()
                    && a.iter().zip(&b).all(|(x, y)| match (x, y) {
                        (InputColumn::String(p), InputColumn::String(q)) => p == q,
                        (InputColumn::I64 { values: p, is_null: pn }, InputColumn::I64 { values: q, is_null: qn }) => p == q && pn == qn,
                        (InputColumn::Decimal128 { values: p, is_null: pn }, InputColumn::Decimal128 { values: q, is_null: qn }) => p == q && pn == qn,
                        _ => false,
                    });
                println!("  reference jar vs new jar, round {round}: all 7 output columns identical: {same}");
                assert!(same, "output differs from reference jar");
            }
        }

        let mut baseline_ns = 0.0;
        for (name, handle) in [("java", &mut java), ("rust", &mut rust)] {
            let (total, mut samples) = bench_named(handle, &input, &names, warmup, iters);
            samples.sort();
            let ns_per_row = total.as_nanos() as f64 / (iters * rows) as f64;
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
                1e9 / ns_per_row,
                speedup
            );
        }
    }
}

fn batch_sizes_first(v: &Option<String>) -> Option<usize> {
    v.as_ref()
        .and_then(|s| s.split(',').next().and_then(|x| x.trim().parse().ok()))
        .or(Some(1_000))
}
