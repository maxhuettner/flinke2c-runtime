#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/flink-stubs/src"
BUILD_DIR="$ROOT_DIR/flink-stubs/build"
JAR_PATH="$ROOT_DIR/flink-stubs/flink-stubs.jar"
OUT_JAR="$ROOT_DIR/../jar/flink-stubs.jar"
JAVA_RELEASE="${JAVA_RELEASE:-8}"

mkdir -p "$BUILD_DIR"

SOURCES="$(find "$SRC_DIR" -name "*.java")"
if [[ -z "$SOURCES" ]]; then
  echo "No Java sources found in $SRC_DIR" >&2
  exit 1
fi

javac --release "$JAVA_RELEASE" -d "$BUILD_DIR" $SOURCES
jar cf "$JAR_PATH" -C "$BUILD_DIR" .
cp "$JAR_PATH" "$OUT_JAR"

echo "Wrote $OUT_JAR"
