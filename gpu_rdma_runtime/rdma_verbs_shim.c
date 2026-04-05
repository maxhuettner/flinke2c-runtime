/*
 * Minimal RDMA shim translation unit.
 *
 * The Rust build script compiles this file into `flinke2c_rdma_shim`.
 * Keep at least one symbol here so the static archive is never empty.
 */
void flinke2c_rdma_shim_keepalive(void) {}
