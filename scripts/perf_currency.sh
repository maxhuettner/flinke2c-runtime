#!/usr/bin/env bash
# Local perf comparison: Rust vs Java CurrencyConversionFunction.
# Env knobs: PERF_WARMUP, PERF_ITERS, PERF_BATCH_SIZES (comma list), JVM_OPTS.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo build --release --manifest-path rust-udfs/currency/Cargo.toml

case "$(uname -s)" in
  Darwin) EXT=dylib ;;
  *) EXT=so ;;
esac
export RUST_UDF_LIB="$ROOT_DIR/rust-udfs/currency/target/release/liborg_example_flinke2c_currency_conversion_function.$EXT"

if [[ -z "${JAVA_HOME:-}" ]] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home)"
fi

# Test binary is linked against libjvm via the jni crate's invocation feature; make it findable.
JVM_LIB_DIR="$(dirname "$(find "$JAVA_HOME" -name 'libjvm.*' | head -1)")"
export DYLD_LIBRARY_PATH="$JVM_LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$JVM_LIB_DIR:${LD_LIBRARY_PATH:-}"

cargo test --release currency_conversion_rust_vs_java -- --ignored --nocapture
