use std::net::TcpListener;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use sideway::ibverbs::device_context::Mtu;
use sideway::ibverbs::queue_pair::QueuePair;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::control_helpers::{recv_json, send_json};
use crate::control_protocol::{EndpointBootstrap, RdmaDestination};

use super::endpoint::GpuRdmaEndpoint;

pub struct ServerConfig<'a> {
    pub port: u16,
    pub ib_device: Option<&'a str>,
    pub ib_port: u8,
    pub gid_index: u8,
    pub iterations: u64,
    pub warmup_iterations: u64,
    pub batch_size: usize,
    pub cuda_device: u32,
    pub kernel_path: &'a Path,
}

pub fn run(config: ServerConfig<'_>) -> Result<()> {
    ensure!(config.iterations > 0, "--iterations must be greater than zero");
    ensure!(
        (1..RING_BUFFER_ELEMENTS).contains(&config.batch_size),
        "--batch-size must be in 1..{RING_BUFFER_ELEMENTS}"
    );

    let mut endpoint =
        GpuRdmaEndpoint::build(config.ib_device, config.ib_port, config.cuda_device, config.kernel_path)?;
    let active_mtu = endpoint.ctx.query_port(config.ib_port)?.active_mtu();
    let gid = endpoint.ctx.query_gid(config.ib_port, config.gid_index.into())?;
    let packet_seq_num = rand::random::<u32>() & 0x00ff_ffff;
    let local = EndpointBootstrap {
        dest: RdmaDestination {
            gid,
            qp_number: endpoint.qp.qp_number(),
            packet_seq_num,
        },
        writable: endpoint.input_region_info(),
        path_mtu: active_mtu as u32,
    };

    let listener =
        TcpListener::bind(("0.0.0.0", config.port)).with_context(|| format!("listen on TCP port {}", config.port))?;
    println!("waiting for RDMA peer on port {}", config.port);
    let (mut stream, peer) = listener.accept().context("accept RDMA peer")?;
    send_json(&mut stream, &local).context("send server bootstrap")?;
    let remote: EndpointBootstrap = recv_json(&mut stream).context("receive client bootstrap")?;
    let path_mtu = active_mtu.min(parse_mtu(remote.path_mtu)?);
    endpoint.connect(&remote.dest, config.ib_port, packet_seq_num, path_mtu, config.gid_index)?;
    println!(
        "QP connected to {peer}; warm-up {} slots, then process {} measured slots with {:?}, batch size {}",
        config.warmup_iterations, config.iterations, path_mtu, config.batch_size
    );

    let mut state = ProcessingState::default();
    if config.warmup_iterations > 0 {
        process_slots(
            &mut endpoint,
            &remote,
            config.warmup_iterations,
            config.batch_size,
            &mut state,
        )?;
        println!("server warm-up complete");
    }
    let started = Instant::now();
    process_slots(&mut endpoint, &remote, config.iterations, config.batch_size, &mut state)?;
    let elapsed = started.elapsed();
    println!(
        "processed {} ordered slots in {:.2?} ({:.2} slots/s)",
        config.iterations,
        elapsed,
        config.iterations as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

#[derive(Default)]
struct ProcessingState {
    input_tail: u64,
    output_head: u64,
}

fn process_slots(
    endpoint: &mut GpuRdmaEndpoint,
    remote: &EndpointBootstrap,
    iterations: u64,
    batch_size: usize,
    state: &mut ProcessingState,
) -> Result<()> {
    let mask = RING_BUFFER_ELEMENTS as u64 - 1;
    let mut processed = 0u64;

    while processed < iterations {
        let input_head = endpoint.read_input_head()?;
        let available = input_head.wrapping_sub(state.input_tail) & mask;
        if available == 0 {
            std::hint::spin_loop();
            continue;
        }

        let count = available.min(batch_size as u64).min(iterations - processed) as u32;
        endpoint.process(state.input_tail, state.output_head, count)?;
        endpoint.write_output_batch(&remote.writable, state.output_head, count)?;

        state.input_tail = (state.input_tail + count as u64) & mask;
        state.output_head = (state.output_head + count as u64) & mask;
        processed += count as u64;
    }
    Ok(())
}

fn parse_mtu(raw: u32) -> Result<Mtu> {
    ensure!((1..=5).contains(&raw), "peer sent invalid path_mtu {raw}");
    Ok(Mtu::from(raw))
}
