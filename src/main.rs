use anyhow::Result;
use clap::Parser;

mod codec;
mod config;
mod constants;
mod java_udf;
mod reload;
mod rust_udf;
mod session;
mod udf;
mod udf_exec;
mod values;

fn main() -> Result<()> {
    let args = config::Args::parse();
    session::run_server(args)
}
