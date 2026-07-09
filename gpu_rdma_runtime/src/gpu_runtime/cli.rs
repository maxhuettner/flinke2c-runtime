use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::constants::RING_BUFFER_ELEMENTS;

use super::server::{self, ServerConfig};

#[derive(Debug, Parser)]
#[command(name = "rdma-gpu-server")]
#[command(about = "Runs CUDA kernels and GPUDirect RDMA control directly from Rust")]
struct Args {
    #[arg(long, default_value_os_t = default_kernel_path())]
    kernel: PathBuf,
    #[arg(long, short = 'p', default_value_t = 50001)]
    port: u16,
    #[arg(long)]
    ib_device: Option<String>,
    #[arg(long, default_value_t = 1)]
    ib_port: u8,
    #[arg(long, short = 'g', default_value_t = 0)]
    gid_index: u8,
    #[arg(long, default_value_t = RING_BUFFER_ELEMENTS as u64 - 1)]
    iterations: u64,
    #[arg(long, default_value_t = 0)]
    warmup_iterations: u64,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 4)]
    pipeline_depth: usize,
    #[arg(long)]
    profile_stages: bool,
    #[arg(long, default_value_t = 0)]
    cuda_device: u32,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    server::run(ServerConfig {
        port: args.port,
        ib_device: args.ib_device.as_deref(),
        ib_port: args.ib_port,
        gid_index: args.gid_index,
        iterations: args.iterations,
        warmup_iterations: args.warmup_iterations,
        batch_size: args.batch_size,
        pipeline_depth: args.pipeline_depth,
        profile_stages: args.profile_stages,
        cuda_device: args.cuda_device,
        kernel_path: &args.kernel,
    })
}

fn default_kernel_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda/process_function.ptx")
}
