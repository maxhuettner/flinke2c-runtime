use std::collections::VecDeque;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use sideway::ibverbs::device_context::Mtu;
use sideway::ibverbs::queue_pair::QueuePair;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::control_helpers::{mtu_value, recv_json, send_json};
use crate::control_protocol::{BootstrapHello, ClientRole, EndpointBootstrap, InputDone, ProcessingSpec, RdmaDestination};

use super::cuda::CudaBatch;
use super::endpoint::GpuRdmaEndpoint;

pub struct ServerConfig<'a> {
    pub port: u16,
    pub ib_device: Option<&'a str>,
    pub ib_port: u8,
    pub gid_index: u8,
    pub iterations: Option<u64>,
    pub warmup_iterations: u64,
    pub batch_size: usize,
    pub profile_stages: bool,
    pub cuda_device: u32,
    pub cuda_pipeline_depth: usize,
    pub kernel_path: &'a Path,
}

pub fn run(config: ServerConfig<'_>) -> Result<()> {
    if let Some(iterations) = config.iterations {
        ensure!(iterations > 0, "--iterations must be greater than zero");
    }
    ensure!(
        (1..RING_BUFFER_ELEMENTS).contains(&config.batch_size),
        "--batch-size must be in 1..{RING_BUFFER_ELEMENTS}"
    );
    let mut endpoint =
        GpuRdmaEndpoint::build(config.ib_device, config.ib_port, config.cuda_device, config.cuda_pipeline_depth, config.kernel_path)?;
    let active_mtu = endpoint.ctx.query_port(config.ib_port)?.active_mtu();
    let gid = endpoint.ctx.query_gid(config.ib_port, config.gid_index.into())?;
    let listener = TcpListener::bind(("0.0.0.0", config.port))
        .with_context(|| format!("listen on TCP port {}", config.port))?;
    println!("waiting for pre and post clients on port {}", config.port);
    let (first_stream, first_peer, first_hello) = accept_role(&listener)?;
    println!("received first client role {:?} from {first_peer}", first_hello.role);
    let (second_stream, second_peer, second_hello) = accept_role(&listener)?;
    println!("received second client role {:?} from {second_peer}", second_hello.role);
    ensure!(first_hello.role != second_hello.role, "expected one pre and one post client");
    let (mut pre_stream, pre_peer, mut post_stream, post_peer) = if first_hello.role == ClientRole::Pre {
        (first_stream, first_peer, second_stream, second_peer)
    } else {
        (second_stream, second_peer, first_stream, first_peer)
    };
    println!("assigned pre client {pre_peer}; assigned post client {post_peer}");

    let input_psn = rand::random::<u32>() & 0x00ff_ffff;
    let output_psn = rand::random::<u32>() & 0x00ff_ffff;
    let pre_local = EndpointBootstrap {
        dest: RdmaDestination { gid: gid.clone(), qp_number: endpoint.input_qp.qp_number(), packet_seq_num: input_psn },
        writable: endpoint.input_region_info(),
        path_mtu: mtu_value(active_mtu),
        processing: ProcessingSpec { function: crate::control_protocol::ProcessingFunction::Increment, field_index: 0, fields: vec![crate::control_protocol::WireFieldType::Int32] },
    };
    let post_local = EndpointBootstrap {
        dest: RdmaDestination { gid, qp_number: endpoint.output_qp.qp_number(), packet_seq_num: output_psn },
        writable: endpoint.output_region_info(),
        path_mtu: mtu_value(active_mtu),
        processing: pre_local.processing.clone(),
    };
    send_json(&mut pre_stream, &pre_local).context("send pre client bootstrap")?;
    send_json(&mut post_stream, &post_local).context("send post client bootstrap")?;
    let pre_remote: EndpointBootstrap = recv_json(&mut pre_stream).context("receive pre client bootstrap")?;
    let post_remote: EndpointBootstrap = recv_json(&mut post_stream).context("receive post client bootstrap")?;
    ensure!(!pre_remote.processing.fields.is_empty(), "processing schema must contain at least one field");
    let processing = super::cuda::CudaProcessSpec::from_protocol(&pre_remote.processing)?;
    let path_mtu = active_mtu.min(parse_mtu(pre_remote.path_mtu)?).min(parse_mtu(post_remote.path_mtu)?);
    endpoint.connect_input(&pre_remote.dest, config.ib_port, input_psn, path_mtu, config.gid_index)?;
    endpoint.connect_output(&post_remote.dest, config.ib_port, output_psn, path_mtu, config.gid_index)?;
    let (done_tx, done_rx) = sync_channel(1);
    let mut control_reader = pre_stream.try_clone().context("clone pre control stream")?;
    thread::spawn(move || {
        if let Ok(message) = recv_json::<InputDone>(&mut control_reader) {
            if message.done {
                let _ = done_tx.send(message.slots);
            }
        }
    });
    let measured_iterations = config.iterations;
    println!(
        "pre QP connected to {pre_peer}, post QP connected to {post_peer}; warm-up {} slots, then process {} measured slots with {:?}, batch size {}, CUDA pipeline depth {}",
        config.warmup_iterations,
        measured_iterations.map_or_else(|| "until PRE closes".to_string(), |value| value.to_string()),
        path_mtu,
        config.batch_size,
        endpoint.pipeline_depth()
    );

    let mut state = ProcessingState::default();
    if config.warmup_iterations > 0 {
        process_slots(
            &mut endpoint,
            &post_remote,
            &mut pre_stream,
            &mut post_stream,
            None,
            Some(config.warmup_iterations),
            config.batch_size,
            &mut state,
            false,
            processing,
        )?;
        println!("server warm-up complete");
    }
    let started = Instant::now();
    let timings = process_slots(
        &mut endpoint,
        &post_remote,
        &mut pre_stream,
        &mut post_stream,
        Some(done_rx),
        measured_iterations,
        config.batch_size,
        &mut state,
        config.profile_stages,
        processing,
    )?;
    let elapsed = started.elapsed();
    println!(
        "processed {} ordered slots in {:.2?} ({:.2} slots/s)",
        timings.processed,
        elapsed,
        timings.processed as f64 / elapsed.as_secs_f64()
    );
    timings.print();
    Ok(())
}

fn accept_role(listener: &TcpListener) -> Result<(TcpStream, std::net::SocketAddr, BootstrapHello)> {
    let (mut stream, peer) = listener.accept().context("accept RDMA peer")?;
    println!("accepted TCP connection from {peer}; waiting for role hello");
    let hello: BootstrapHello = recv_json(&mut stream).context("receive client role")?;
    Ok((stream, peer, hello))
}

#[derive(Default)]
struct ProcessingState {
    input_tail: u64,
    output_head: u64,
    output_in_flight: usize,
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
    processed: u64,
}

impl StageTimings {
    fn print(&self) {
        if !self.enabled || self.batches == 0 {
            return;
        }
        println!("stage timings (overlapped, average per batch):");
        println!(
            "  input notify/collect:{:.3} us",
            average_us(self.receive, self.batches)
        );
        println!("  input visibility:    {:.3} us", average_us(self.flush, self.batches));
        println!("  CUDA kernel launch:  {:.3} us", average_us(self.submit, self.batches));
        println!(
            "  CUDA completion wait:{:.3} us",
            average_us(self.cuda_wait, self.batches)
        );
        println!("  output RDMA submit:  {:.3} us", average_us(self.output, self.batches));
        println!(
            "  output completion:   {:.3} us",
            self.output_drain.as_secs_f64() * 1_000_000.0
        );
    }
}

fn process_slots(
    endpoint: &mut GpuRdmaEndpoint,
    remote: &EndpointBootstrap,
    pre_stream: &mut TcpStream,
    post_stream: &mut TcpStream,
    done_rx: Option<Receiver<u64>>,
    iterations: Option<u64>,
    batch_size: usize,
    state: &mut ProcessingState,
    profile_stages: bool,
    processing: super::cuda::CudaProcessSpec,
) -> Result<StageTimings> {
    let mut submitted = 0u64;
    let mut completed = 0u64;
    let mut input_done = false;
    let mut expected_input_slots = None;
    let mut pending = VecDeque::with_capacity(endpoint.pipeline_depth());
    let mut timings = StageTimings {
        enabled: profile_stages,
        ..StageTimings::default()
    };

    loop {
        if iterations.is_some_and(|limit| completed >= limit) {
            break;
        }
        if !input_done {
            if let Some(done_rx) = done_rx.as_ref() {
                if let Ok(slots) = done_rx.try_recv() {
                    input_done = true;
                    expected_input_slots = Some(slots);
                }
            }
        }
        let input_complete = input_done
            && expected_input_slots.map_or(true, |expected| submitted >= expected);
        let arrivals = timed(profile_stages, &mut timings.receive, || {
            let mut arrivals = Vec::with_capacity(endpoint.pipeline_depth() - pending.len());
            if pending.is_empty() && !input_complete && iterations.is_some() {
                arrivals.push(endpoint.wait_for_input_batch()?);
            } else if pending.is_empty() && !input_complete {
                loop {
                    if let Some(done_rx) = done_rx.as_ref() {
                        if let Ok(slots) = done_rx.try_recv() {
                            input_done = true;
                            expected_input_slots = Some(slots);
                        }
                    }
                    let input_complete = input_done
                        && expected_input_slots.map_or(true, |expected| submitted >= expected);
                    if input_complete {
                        break;
                    }
                    if let Some(count) = endpoint.try_input_batch()? {
                        arrivals.push(count);
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
            while !(input_done
                && expected_input_slots.map_or(true, |expected| submitted >= expected))
                && iterations.map_or(true, |limit| {
                    submitted + arrivals.iter().map(|count| u64::from(*count)).sum::<u64>() < limit
                })
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
                    validate_batch(count, batch_size, iterations.map(|limit| limit - submitted))?;
                    let output_head = state.output_head;
                    let cuda = endpoint.submit_process(state.input_tail, output_head, count, processing)?;
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

        if pending.is_empty() {
            let input_complete = input_done
                && expected_input_slots.map_or(true, |expected| submitted >= expected);
            if input_complete || iterations.is_some() {
                break;
            }
            continue;
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
            // In open-ended mode, retain the last ready batch until PRE either
            // publishes another batch or announces completion. This ensures
            // the final RDMA write can be posted with a completion request;
            // the TCP done notification may arrive concurrently with the
            // final RDMA input notification.
            let input_complete = input_done
                && expected_input_slots.map_or(true, |expected| submitted >= expected);
            if iterations.is_none() && pending.len() == 1 && !input_complete {
                break;
            }
            let batch = pending.front().context("CUDA pipeline made no progress")?;
            if !first_ready
                && !timed(profile_stages, &mut timings.cuda_wait, || {
                    endpoint.process_complete(batch.cuda)
                })?
            {
                break;
            }
            let input_complete = input_done
                && expected_input_slots.map_or(true, |expected| submitted >= expected);
            let phase_end = iterations.map_or(input_complete && pending.len() == 1, |limit| {
                completed + u64::from(batch.count) == limit
            });
            while state.output_in_flight + batch.count as usize >= RING_BUFFER_ELEMENTS {
                let credit: u32 = recv_json(post_stream).context("receive post-client output credit")?;
                let credit = credit as usize;
                ensure!(credit <= state.output_in_flight, "post client returned too much output credit");
                state.output_in_flight -= credit;
            }
            timed(profile_stages, &mut timings.output, || {
                endpoint.write_output_batch(&remote.writable, batch.output_head, batch.count, phase_end)
            })?;
            completed += u64::from(batch.count);
            state.output_in_flight += batch.count as usize;
            send_json(pre_stream, &batch.count).context("send pre-client input credit")?;
            pending.pop_front();
            first_ready = false;
            if pending.is_empty() {
                break;
            }
        }
    }
    timed(profile_stages, &mut timings.output_drain, || endpoint.finish_output())?;
    while state.output_in_flight > 0 {
        let credit: u32 = recv_json(post_stream).context("receive final post-client output credit")?;
        let credit = credit as usize;
        ensure!(credit > 0 && credit <= state.output_in_flight,
            "post client returned too much final output credit {credit}");
        state.output_in_flight -= credit;
    }
    timings.processed = completed;
    Ok(timings)
}

fn validate_batch(count: u32, batch_size: usize, remaining: Option<u64>) -> Result<()> {
    ensure!(
        count as usize <= batch_size,
        "peer sent batch of {count} slots, exceeding configured batch size {batch_size}"
    );
    if let Some(remaining) = remaining {
        ensure!(u64::from(count) <= remaining, "peer batch crosses benchmark phase boundary");
    }
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
    Ok(match raw {
        256 => Mtu::Mtu256,
        512 => Mtu::Mtu512,
        1024 => Mtu::Mtu1024,
        2048 => Mtu::Mtu2048,
        4096 => Mtu::Mtu4096,
        other => anyhow::bail!("peer sent invalid path_mtu {other}"),
    })
}
