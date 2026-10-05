#!/usr/bin/env bash
# End-to-end throughput of the runtime's TCP path (PRE -> UDF -> POST) on loopback,
# using the Rust PriceGreaterThan filter. Env: ROWS (default 5000000), WORKERS (runtime --workers), BATCH (batchSize sent in config, default 2048),
# MODE (filter = Rust PriceGreaterThan, imputation = Java ImputationFunction).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PORT="${PORT:-19001}"
ROWS="${ROWS:-5000000}"

MODE="${MODE:-filter}"
if [[ "$MODE" == "filter" ]]; then
  cargo build --release --manifest-path rust-udfs/price-greater-than/Cargo.toml
fi
cargo build --release --bin flinke2c-runtime --example load_client

if [[ -z "${JAVA_HOME:-}" ]] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home)"
fi
JVM_LIB_DIR="$(dirname "$(find "$JAVA_HOME" -name 'libjvm.*' | head -1)")"
export DYLD_LIBRARY_PATH="$JVM_LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$JVM_LIB_DIR:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
[[ -n "${WORKERS:-}" ]] && EXTRA_ARGS+=(--workers "$WORKERS")

if [[ "$MODE" == "imputation" ]]; then
  # Java ImputationFunction from jar/flinke2c.jar, as the Flink query configures it.
  RT_ARGS=(--udf-lang java)
else
  RT_ARGS=(--udf-lang rust --rust-udf-lib rust-udfs/price-greater-than/target/release)
fi
target/release/flinke2c-runtime --listen-host 127.0.0.1 --in-port "$PORT" "${RT_ARGS[@]}" "${EXTRA_ARGS[@]}" ${RT_EXTRA:-} >/dev/null 2>&1 &
RT_PID=$!
trap 'kill $RT_PID 2>/dev/null || true' EXIT
sleep 1

target/release/examples/load_client "$PORT" "$ROWS" 100 "${BATCH:-2048}" "$MODE"
