pub mod cli;
mod cuda;
mod endpoint;
mod memory_region;
mod server;

// Stateless in-place kernels map one thread to one row. Kernels that materialize
// a separate output map one 32-thread warp to each row, so a block handles eight
// rows while preserving coalesced row copies.
const THREADS_PER_BLOCK: u32 = 256;
const ROWS_PER_COPY_BLOCK: u32 = THREADS_PER_BLOCK / 32;
