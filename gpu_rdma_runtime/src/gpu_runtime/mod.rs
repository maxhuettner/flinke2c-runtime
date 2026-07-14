pub mod cli;
mod cuda;
mod endpoint;
mod memory_region;
mod server;

// Typical framed rows are tens of bytes. 64 threads keeps enough parallelism
// for larger rows while avoiding 192 idle threads for normal Nexmark rows.
const THREADS_PER_BLOCK: u32 = 64;
