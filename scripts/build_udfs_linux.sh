#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-x86_64-unknown-linux-gnu}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="${ROOT_DIR}/lib"
UPDATED_LIB_DIR="${LIB_DIR}/updated"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found" >&2
  exit 1
fi

if ! cargo --list | grep -q "zigbuild"; then
  echo "error: cargo-zigbuild not installed (cargo zigbuild)" >&2
  exit 1
fi

mkdir -p "${LIB_DIR}"
mkdir -p "${UPDATED_LIB_DIR}"

for crate_dir in "${ROOT_DIR}"/rust-udfs/*; do
  if [[ -f "${crate_dir}/Cargo.toml" ]]; then
    echo "Building ${crate_dir##*/} for ${TARGET}..."
    cargo zigbuild --release --target "${TARGET}" --manifest-path "${crate_dir}/Cargo.toml"

    lib_name="$(awk -F'=' '
      $0 ~ /^\[lib\]/ { inlib=1; next }
      $0 ~ /^\[/ { if ($0 != "[lib]") inlib=0 }
      inlib && $1 ~ /^[[:space:]]*name[[:space:]]*$/ { gsub(/[[:space:]]|"/, "", $2); print $2; exit }
    ' "${crate_dir}/Cargo.toml")"
    if [[ -z "${lib_name}" ]]; then
      lib_name="$(awk -F'=' '
        $0 ~ /^\[package\]/ { inpkg=1; next }
        $0 ~ /^\[/ { if ($0 != "[package]") inpkg=0 }
        inpkg && $1 ~ /^[[:space:]]*name[[:space:]]*$/ { gsub(/[[:space:]]|"/, "", $2); print $2; exit }
      ' "${crate_dir}/Cargo.toml")"
    fi

    if [[ -z "${lib_name}" ]]; then
      echo "error: failed to resolve lib name from ${crate_dir}/Cargo.toml" >&2
      exit 1
    fi

    lib_name="${lib_name//-/_}"
    so_path="${crate_dir}/target/${TARGET}/release/lib${lib_name}.so"
    if [[ ! -f "${so_path}" ]]; then
      echo "error: expected output not found: ${so_path}" >&2
      exit 1
    fi

    echo "Copying $(basename "${so_path}") -> ${LIB_DIR}"
    cp -f "${so_path}" "${LIB_DIR}/"

    if [[ "${crate_dir##*/}" == "price-greater-than" ]]; then
      echo "Building ${crate_dir##*/} (less_sensitive) for ${TARGET}..."
      cargo zigbuild --release --target "${TARGET}" --features less_sensitive --manifest-path "${crate_dir}/Cargo.toml"

      so_path_less_sensitive="${crate_dir}/target/${TARGET}/release/lib${lib_name}.so"
      if [[ ! -f "${so_path_less_sensitive}" ]]; then
        echo "error: expected output not found: ${so_path_less_sensitive}" >&2
        exit 1
      fi

      echo "Copying $(basename "${so_path_less_sensitive}") -> ${UPDATED_LIB_DIR}"
      cp -f "${so_path_less_sensitive}" "${UPDATED_LIB_DIR}/"
    fi
  fi
done

echo "Done. Linux UDFs are in ${LIB_DIR} (plus updated variants in ${UPDATED_LIB_DIR})"
