#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/udf-adapter/src"
BUILD_DIR="$ROOT_DIR/udf-adapter/build"
JAR_PATH="$ROOT_DIR/udf-adapter/udf-adapter.jar"
OUT_JAR="$ROOT_DIR/../jar/udf-adapter.jar"
STUBS_JAR="$ROOT_DIR/../jar/flink-stubs.jar"
JAVA_RELEASE="${JAVA_RELEASE:-8}"

mkdir -p "$BUILD_DIR"

SOURCES="$(find "$SRC_DIR" -name "*.java")"
if [[ -z "$SOURCES" ]]; then
  echo "No Java sources found in $SRC_DIR" >&2
  exit 1
fi

JAVAC_CLASSPATH=()
if [[ -f "$STUBS_JAR" ]]; then
  JAVAC_CLASSPATH=(-cp "$STUBS_JAR")
fi

javac --release "$JAVA_RELEASE" "${JAVAC_CLASSPATH[@]}" -d "$BUILD_DIR" $SOURCES
jar cf "$JAR_PATH" -C "$BUILD_DIR" .
cp "$JAR_PATH" "$OUT_JAR"

echo "Wrote $OUT_JAR"
