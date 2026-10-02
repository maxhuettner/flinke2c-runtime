#!/usr/bin/env bash
# End-to-end throughput of the runtime's TCP path (PRE -> UDF -> POST) on loopback,
# using the Rust PriceGreaterThan filter. Env: ROWS (default 5000000), WORKERS (runtime --workers), BATCH (batchSize sent in config, default 2048).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PORT="${PORT:-19001}"
ROWS="${ROWS:-5000000}"

cargo build --release --manifest-path rust-udfs/price-greater-than/Cargo.toml
cargo build --release --bin flinke2c-runtime --example load_client

if [[ -z "${JAVA_HOME:-}" ]] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home)"
fi
JVM_LIB_DIR="$(dirname "$(find "$JAVA_HOME" -name 'libjvm.*' | head -1)")"
export DYLD_LIBRARY_PATH="$JVM_LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$JVM_LIB_DIR:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
[[ -n "${WORKERS:-}" ]] && EXTRA_ARGS+=(--workers "$WORKERS")

target/release/flinke2c-runtime --listen-host 127.0.0.1 --in-port "$PORT" --udf-lang rust \
  --rust-udf-lib rust-udfs/price-greater-than/target/release "${EXTRA_ARGS[@]}" >/dev/null 2>&1 &
RT_PID=$!
trap 'kill $RT_PID 2>/dev/null || true' EXIT
sleep 1

target/release/examples/load_client "$PORT" "$ROWS" 100 "${BATCH:-2048}"
