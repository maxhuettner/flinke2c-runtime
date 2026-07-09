use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use sideway::ibverbs::device_context::Mtu;
use sideway::ibverbs::queue_pair::QueuePair;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::control_helpers::{recv_json, send_json};
use crate::control_protocol::{EndpointBootstrap, RdmaDestination};

use super::cuda::CudaBatch;
use super::endpoint::GpuRdmaEndpoint;

pub struct ServerConfig<'a> {
    pub port: u16,
    pub ib_device: Option<&'a str>,
    pub ib_port: u8,
    pub gid_index: u8,
    pub iterations: u64,
    pub warmup_iterations: u64,
    pub batch_size: usize,
    pub pipeline_depth: usize,
    pub profile_stages: bool,
    pub cuda_device: u32,
    pub kernel_path: &'a Path,
}

pub fn run(config: ServerConfig<'_>) -> Result<()> {
    ensure!(config.iterations > 0, "--iterations must be greater than zero");
    ensure!(
        (1..RING_BUFFER_ELEMENTS).contains(&config.batch_size),
        "--batch-size must be in 1..{RING_BUFFER_ELEMENTS}"
    );
    ensure!(
        (1..=64).contains(&config.pipeline_depth),
        "--pipeline-depth must be in 1..=64"
    );

    let mut endpoint = GpuRdmaEndpoint::build(
        config.ib_device,
        config.ib_port,
        config.cuda_device,
        config.kernel_path,
        config.pipeline_depth,
    )?;
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
        "QP connected to {peer}; warm-up {} slots, then process {} measured slots with {:?}, batch size {}, pipeline depth {}",
        config.warmup_iterations,
        config.iterations,
        path_mtu,
        config.batch_size,
        config.pipeline_depth
    );

    let mut state = ProcessingState::default();
    if config.warmup_iterations > 0 {
        process_slots(
            &mut endpoint,
            &remote,
            config.warmup_iterations,
            config.batch_size,
            &mut state,
            false,
        )?;
        println!("server warm-up complete");
    }
    let started = Instant::now();
    let timings = process_slots(
        &mut endpoint,
        &remote,
        config.iterations,
        config.batch_size,
        &mut state,
        config.profile_stages,
    )?;
    let elapsed = started.elapsed();
    println!(
        "processed {} ordered slots in {:.2?} ({:.2} slots/s)",
        config.iterations,
        elapsed,
        config.iterations as f64 / elapsed.as_secs_f64()
    );
    timings.print();
    Ok(())
}

#[derive(Default)]
struct ProcessingState {
    input_tail: u64,
    output_head: u64,
}

struct PendingBatch {
    output_head: u64,
    count: u32,
    cuda: CudaBatch,
}

#[derive(Default)]
struct StageTimings {
    enabled: bool,
    batches: u64,
    receive: Duration,
    flush: Duration,
    submit: Duration,
    cuda_wait: Duration,
    output: Duration,
    output_drain: Duration,
}

impl StageTimings {
    fn print(&self) {
        if !self.enabled || self.batches == 0 {
            return;
        }
        println!("stage timings (overlapped, average per batch):");
        println!(
            "  receive/collect:     {:.3} us",
            average_us(self.receive, self.batches)
        );
        println!("  GPUDirect flush:     {:.3} us", average_us(self.flush, self.batches));
        println!("  CUDA submit:         {:.3} us", average_us(self.submit, self.batches));
        println!(
            "  CUDA retirement wait:{:.3} us",
            average_us(self.cuda_wait, self.batches)
        );
        println!("  RDMA output submit:  {:.3} us", average_us(self.output, self.batches));
        println!(
            "  final output drain:  {:.3} us",
            self.output_drain.as_secs_f64() * 1_000_000.0
        );
    }
}

fn process_slots(
    endpoint: &mut GpuRdmaEndpoint,
    remote: &EndpointBootstrap,
    iterations: u64,
    batch_size: usize,
    state: &mut ProcessingState,
    profile_stages: bool,
) -> Result<StageTimings> {
    let mut submitted = 0u64;
    let mut completed = 0u64;
    let mut pending = VecDeque::with_capacity(endpoint.pipeline_depth());
    let mut timings = StageTimings {
        enabled: profile_stages,
        ..StageTimings::default()
    };

    while completed < iterations {
        let arrivals = timed(profile_stages, &mut timings.receive, || {
            let mut arrivals = Vec::with_capacity(endpoint.pipeline_depth() - pending.len());
            if pending.is_empty() && submitted < iterations {
                arrivals.push(endpoint.wait_for_input_batch()?);
            }
            while submitted + arrivals.iter().map(|count| u64::from(*count)).sum::<u64>() < iterations
                && pending.len() + arrivals.len() < endpoint.pipeline_depth()
            {
                let Some(count) = endpoint.try_input_batch()? else {
                    break;
                };
                arrivals.push(count);
            }
            Ok(arrivals)
        })?;

        if !arrivals.is_empty() {
            timed(profile_stages, &mut timings.flush, || endpoint.flush_input_writes())?;
            let arrival_count = arrivals.len() as u64;
            timed(profile_stages, &mut timings.submit, || {
                for count in arrivals {
                    validate_batch(count, batch_size, iterations - submitted)?;
                    let output_head = state.output_head;
                    let cuda = endpoint.submit_process(state.input_tail, output_head, count)?;
                    state.input_tail += u64::from(count);
                    state.output_head += u64::from(count);
                    submitted += u64::from(count);
                    pending.push_back(PendingBatch {
                        output_head,
                        count,
                        cuda,
                    });
                }
                Ok(())
            })?;
            timings.batches += arrival_count;
        }

        let oldest = pending.front().context("CUDA pipeline made no progress")?;
        timed(profile_stages, &mut timings.cuda_wait, || {
            if !endpoint.process_complete(oldest.cuda)? {
                endpoint.wait_for_process(oldest.cuda)?;
            }
            Ok(())
        })?;

        let mut first_ready = true;
        loop {
            let batch = pending.front().context("CUDA pipeline made no progress")?;
            if !first_ready
                && !timed(profile_stages, &mut timings.cuda_wait, || {
                    endpoint.process_complete(batch.cuda)
                })?
            {
                break;
            }
            let phase_end = completed + u64::from(batch.count) == iterations;
            timed(profile_stages, &mut timings.output, || {
                endpoint.write_output_batch(&remote.writable, batch.output_head, batch.count, phase_end)
            })?;
            completed += u64::from(batch.count);
            pending.pop_front();
            first_ready = false;
            if pending.is_empty() {
                break;
            }
        }
    }
    timed(profile_stages, &mut timings.output_drain, || endpoint.finish_output())?;
    Ok(timings)
}

fn validate_batch(count: u32, batch_size: usize, remaining: u64) -> Result<()> {
    ensure!(
        count as usize <= batch_size,
        "peer sent batch of {count} slots, exceeding configured batch size {batch_size}"
    );
    ensure!(
        u64::from(count) <= remaining,
        "peer batch crosses benchmark phase boundary"
    );
    Ok(())
}

fn timed<T>(enabled: bool, total: &mut Duration, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled {
        return operation();
    }
    let started = Instant::now();
    let result = operation();
    *total += started.elapsed();
    result
}

fn average_us(duration: Duration, count: u64) -> f64 {
    duration.as_secs_f64() * 1_000_000.0 / count as f64
}

fn parse_mtu(raw: u32) -> Result<Mtu> {
    ensure!((1..=5).contains(&raw), "peer sent invalid path_mtu {raw}");
    Ok(Mtu::from(raw))
}
