mod linux_impl {
    use std::collections::VecDeque;
    use std::net::TcpStream;
    use std::time::{Duration, Instant};

    use anyhow::{ensure, Context, Result};
    use clap::Parser;
    use gpu_rdma_runtime::constants::RING_BUFFER_ELEMENTS;
    use gpu_rdma_runtime::control_helpers::{recv_json, send_json, CliMtu};
    use gpu_rdma_runtime::control_protocol::{EndpointBootstrap, RdmaDestination, MAX_ITEM_SIZE};
    use gpu_rdma_runtime::rdma::RdmaEndpoint;
    use gpu_rdma_runtime::ring_buffer::slot::Slot;
    use sideway::ibverbs::device_context::Mtu;
    use sideway::ibverbs::queue_pair::QueuePair;

    const MAX_IN_FLIGHT_SENDS: usize = 256;

    #[derive(Parser, Debug)]
    #[command(name = "rdma-test-client")]
    #[command(about = "Benchmarks ordered Rust -> GPU -> Rust processing over GPUDirect RDMA")]
    struct Args {
        #[arg(long)]
        server: String,
        #[arg(long)]
        ib_device: Option<String>,
        #[arg(long, default_value_t = 1)]
        ib_port: u8,
        #[arg(long, short = 's', default_value_t = 1024)]
        size: u32,
        #[arg(long, short = 'g', default_value_t = 0)]
        gid_index: u8,
        #[arg(long, short = 'm', default_value_t = CliMtu(Mtu::Mtu1024))]
        mtu: CliMtu,
        #[arg(long, default_value_t = RING_BUFFER_ELEMENTS as u64 - 1)]
        iterations: u64,
        #[arg(long, default_value_t = 0)]
        warmup_iterations: u64,
        #[arg(long, default_value_t = 256)]
        in_flight: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long)]
        skip_validation: bool,
    }

    struct PendingRequest {
        sequence: u64,
        started: Option<Instant>,
    }

    struct PhaseResult {
        elapsed: Duration,
        latencies_ns: Vec<u64>,
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        ensure!(args.iterations > 0, "--iterations must be greater than zero");
        ensure!(
            args.size > 0 && args.size as usize <= MAX_ITEM_SIZE,
            "--size must be in 1..={MAX_ITEM_SIZE}"
        );
        ensure!(
            (1..RING_BUFFER_ELEMENTS).contains(&args.in_flight),
            "--in-flight must be in 1..{RING_BUFFER_ELEMENTS}"
        );
        ensure!(
            (1..=args.in_flight).contains(&args.batch_size),
            "--batch-size must be in 1..=--in-flight"
        );
        ensure!(
            args.iterations <= usize::MAX as u64,
            "--iterations is too large for latency collection"
        );

        let mut stream = TcpStream::connect(&args.server).with_context(|| format!("connect {}", args.server))?;
        let mut endpoint = RdmaEndpoint::build(args.ib_device.as_deref(), args.ib_port)?;
        let active_mtu = endpoint.ctx.query_port(args.ib_port)?.active_mtu();

        println!("client waiting for server bootstrap");
        let server: EndpointBootstrap = recv_json(&mut stream)?;
        println!("client received server bootstrap: {server:?}");
        let path_mtu = active_mtu.min(parse_mtu(server.path_mtu)?);

        let gid = endpoint.ctx.query_gid(args.ib_port, args.gid_index.into())?;
        let packet_seq_num = rand::random::<u32>() & 0x00ff_ffff;
        let local = EndpointBootstrap {
            dest: RdmaDestination {
                gid,
                qp_number: endpoint.qp.qp_number(),
                packet_seq_num,
            },
            writable: endpoint.memory_region_info(),
            path_mtu: active_mtu as u32,
        };
        println!("client sending bootstrap: {local:?}");
        send_json(&mut stream, &local)?;
        endpoint.connect(&server.dest, args.ib_port, packet_seq_num, path_mtu, 0, args.gid_index)?;
        println!("client QP connected with {path_mtu:?}");

        let payload_len = args.size as usize;
        let mut next_sequence = 0u64;
        if args.warmup_iterations > 0 {
            println!("warming up with {} round trips", args.warmup_iterations);
            run_phase(
                &mut endpoint,
                &server,
                &mut next_sequence,
                args.warmup_iterations,
                payload_len,
                args.in_flight,
                args.batch_size,
                !args.skip_validation,
                false,
            )?;
        }

        println!(
            "measuring {} round trips with at most {} in flight",
            args.iterations, args.in_flight
        );
        let result = run_phase(
            &mut endpoint,
            &server,
            &mut next_sequence,
            args.iterations,
            payload_len,
            args.in_flight,
            args.batch_size,
            !args.skip_validation,
            true,
        )?;
        print_result(&args, result);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_phase(
        endpoint: &mut RdmaEndpoint,
        server: &EndpointBootstrap,
        next_sequence: &mut u64,
        iterations: u64,
        payload_len: usize,
        request_window: usize,
        batch_size: usize,
        validate: bool,
        measure: bool,
    ) -> Result<PhaseResult> {
        let first_sequence = *next_sequence;
        let last_sequence = first_sequence
            .checked_add(iterations)
            .context("sequence number overflow")?;
        let mut next_to_send = first_sequence;
        let mut pending = VecDeque::with_capacity(request_window);
        let mut send_completions = VecDeque::with_capacity(MAX_IN_FLIGHT_SENDS);
        let mut latencies_ns = Vec::with_capacity(if measure { iterations as usize } else { 0 });
        let started = Instant::now();

        while *next_sequence < last_sequence {
            while next_to_send < last_sequence && pending.len() < request_window {
                if send_completions.len() >= MAX_IN_FLIGHT_SENDS {
                    wait_oldest_send(endpoint, &mut send_completions)?;
                }

                let capacity = batch_size
                    .min(request_window - pending.len())
                    .min((last_sequence - next_to_send) as usize);
                let mut batch = Vec::with_capacity(capacity);
                for _ in 0..capacity {
                    let request_started = measure.then(Instant::now);
                    let slot = input_slot(next_to_send, payload_len);
                    if endpoint.write_slot_local(slot).is_err() {
                        break;
                    }
                    batch.push(PendingRequest {
                        sequence: next_to_send,
                        started: request_started,
                    });
                    next_to_send += 1;
                }
                if batch.is_empty() {
                    break;
                }

                let wr_id = endpoint.write_slots_remote(&server.writable, batch.len())?;
                send_completions.push_back(wr_id);
                pending.extend(batch);
            }

            let drained = drain_responses(
                endpoint,
                &mut pending,
                next_sequence,
                payload_len,
                validate,
                &mut latencies_ns,
            )?;
            if drained == 0 {
                if !send_completions.is_empty() {
                    wait_oldest_send(endpoint, &mut send_completions)?;
                } else {
                    std::hint::spin_loop();
                }
            }
        }

        while !send_completions.is_empty() {
            wait_oldest_send(endpoint, &mut send_completions)?;
        }
        ensure!(pending.is_empty(), "responses completed with pending requests");
        Ok(PhaseResult {
            elapsed: started.elapsed(),
            latencies_ns,
        })
    }

    fn wait_oldest_send(endpoint: &mut RdmaEndpoint, completions: &mut VecDeque<u64>) -> Result<()> {
        let wr_id = completions.pop_front().context("no send completion available")?;
        endpoint.wait_for_completion(wr_id)
    }

    fn drain_responses(
        endpoint: &mut RdmaEndpoint,
        pending: &mut VecDeque<PendingRequest>,
        next_sequence: &mut u64,
        payload_len: usize,
        validate: bool,
        latencies_ns: &mut Vec<u64>,
    ) -> Result<usize> {
        let mut drained = 0;
        while let Some(slot) = endpoint.read_slot_local() {
            let request = pending
                .pop_front()
                .context("received a response without a pending request")?;
            ensure!(
                request.sequence == *next_sequence,
                "response order mismatch: expected {}, pending {}",
                *next_sequence,
                request.sequence
            );
            if validate {
                validate_response(&slot, request.sequence, payload_len)?;
            }
            if let Some(started) = request.started {
                latencies_ns.push(duration_ns(started.elapsed()));
            }
            endpoint.complete_round_trip_local();
            *next_sequence += 1;
            drained += 1;
        }
        Ok(drained)
    }

    fn print_result(args: &Args, mut result: PhaseResult) {
        result.latencies_ns.sort_unstable();
        let throughput = args.iterations as f64 / result.elapsed.as_secs_f64();
        let one_way_gbps = throughput * args.size as f64 * 8.0 / 1_000_000_000.0;
        let round_trip_gbps = one_way_gbps * 2.0;
        let mean_ns = result.latencies_ns.iter().map(|value| *value as u128).sum::<u128>() as f64
            / result.latencies_ns.len() as f64;

        println!("benchmark result:");
        println!("  tuples:             {}", args.iterations);
        println!("  elapsed:            {:.3?}", result.elapsed);
        println!("  throughput:         {:.2} tuples/s", throughput);
        println!("  input goodput:      {:.3} Gbit/s", one_way_gbps);
        println!("  bidirectional data: {:.3} Gbit/s", round_trip_gbps);
        println!("  latency mean:       {:.3} us", mean_ns / 1_000.0);
        println!(
            "  latency p50:        {:.3} us",
            percentile(&result.latencies_ns, 0.50) / 1_000.0
        );
        println!(
            "  latency p95:        {:.3} us",
            percentile(&result.latencies_ns, 0.95) / 1_000.0
        );
        println!(
            "  latency p99:        {:.3} us",
            percentile(&result.latencies_ns, 0.99) / 1_000.0
        );
        println!(
            "  latency p99.9:      {:.3} us",
            percentile(&result.latencies_ns, 0.999) / 1_000.0
        );
        println!(
            "  latency max:        {:.3} us",
            result.latencies_ns.last().copied().unwrap_or(0) as f64 / 1_000.0
        );
        println!("  validation:         {}", !args.skip_validation);
        println!("  batch size:         {}", args.batch_size);
    }

    fn percentile(sorted: &[u64], quantile: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
        sorted[index] as f64
    }

    fn duration_ns(duration: Duration) -> u64 {
        duration.as_nanos().min(u64::MAX as u128) as u64
    }

    fn input_slot(sequence: u64, payload_len: usize) -> Slot {
        let mut value = [0u8; MAX_ITEM_SIZE];
        value[..payload_len].fill(input_byte(sequence));
        Slot {
            len: payload_len as u32,
            value,
        }
    }

    fn validate_response(slot: &Slot, sequence: u64, payload_len: usize) -> Result<()> {
        ensure!(
            slot.len as usize == payload_len,
            "response {sequence} has an invalid length"
        );
        let expected = input_byte(sequence).wrapping_add(1);
        ensure!(
            slot.value[..payload_len].iter().all(|byte| *byte == expected),
            "response {sequence} is out of order or contains invalid mapped data"
        );
        Ok(())
    }

    fn input_byte(sequence: u64) -> u8 {
        (sequence % 251) as u8
    }

    fn parse_mtu(raw: u32) -> Result<Mtu> {
        ensure!((1..=5).contains(&raw), "server sent invalid path_mtu {raw}");
        Ok(Mtu::from(raw))
    }
}

fn main() -> anyhow::Result<()> {
    linux_impl::run()
}
