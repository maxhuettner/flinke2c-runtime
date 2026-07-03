pub mod cli;
mod cuda;
mod endpoint;
mod memory_region;
mod server;

const THREADS_PER_BLOCK: u32 = 256;
