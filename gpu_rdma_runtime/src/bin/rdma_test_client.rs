mod linux_impl {
    use std::collections::VecDeque;
    use std::net::TcpStream;
    use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError};
    use std::thread;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, ensure, Context, Result};
    use clap::Parser;
    use gpu_rdma_runtime::constants::RING_BUFFER_ELEMENTS;
    use gpu_rdma_runtime::control_helpers::{recv_json, send_json, CliMtu};
    use gpu_rdma_runtime::control_protocol::{EndpointBootstrap, RdmaDestination, MAX_ITEM_SIZE};
    use gpu_rdma_runtime::rdma::{RdmaEndpoint, RdmaReceiver, RdmaSender};
    use gpu_rdma_runtime::ring_buffer::slot::Slot;
    use sideway::ibverbs::device_context::Mtu;
    use sideway::ibverbs::queue_pair::QueuePair;

    const MAX_IN_FLIGHT_SENDS: usize = 256;
    const INPUT_PATTERN_PERIOD: usize = 251;

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
        #[arg(long)]
        pre_generate_inputs: bool,
        #[arg(long)]
        throughput_only: bool,
        #[arg(long)]
        profile_stages: bool,
    }

    struct PendingRequest {
        sequence: u64,
        started: Option<Instant>,
    }

    struct PhaseResult {
        elapsed: Duration,
        latencies_ns: Vec<u64>,
        producer_timings: WorkerTimings,
        consumer_timings: WorkerTimings,
    }

    #[derive(Default)]
    struct WorkerTimings {
        enabled: bool,
        batches: u64,
        prepare: Duration,
        post: Duration,
        wait: Duration,
        auxiliary: Duration,
    }

    struct ProducedBatch {
        count: usize,
        requests: Vec<PendingRequest>,
    }

    struct ConsumerFeedback {
        completed_slots: usize,
        receive_wrs: usize,
    }

    struct PreparedInputs {
        slots: Vec<Slot>,
    }

    impl PreparedInputs {
        fn new(payload_len: usize) -> Self {
            let slots = (0..INPUT_PATTERN_PERIOD)
                .map(|sequence| input_slot(sequence as u64, payload_len))
                .collect();
            Self { slots }
        }

        fn get(&self, sequence: u64) -> Slot {
            self.slots[sequence as usize % INPUT_PATTERN_PERIOD]
        }
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
        ensure!(
            !args.throughput_only || args.skip_validation,
            "--throughput-only requires --skip-validation"
        );

        let prepared_inputs = args.pre_generate_inputs.then(|| {
            println!("pre-generating the {INPUT_PATTERN_PERIOD} distinct input payloads");
            PreparedInputs::new(args.size as usize)
        });

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
        let (mut sender, receiver) = endpoint.split();
        if args.throughput_only {
            let slot = prepared_inputs
                .as_ref()
                .map(|inputs| inputs.get(0))
                .unwrap_or_else(|| input_slot(0, payload_len));
            sender.preload_send_ring(slot);
            println!("preloaded registered send ring for throughput-only mode");
        }
        let (sender, receiver, next_sequence) = if args.warmup_iterations > 0 {
            println!("warming up with {} round trips", args.warmup_iterations);
            let (sender, receiver, _) = run_phase(
                sender,
                receiver,
                &server,
                0,
                args.warmup_iterations,
                payload_len,
                args.in_flight,
                args.batch_size,
                !args.skip_validation,
                false,
                false,
                prepared_inputs.as_ref(),
            )?;
            (sender, receiver, args.warmup_iterations)
        } else {
            (sender, receiver, 0)
        };

        println!(
            "measuring {} round trips with at most {} in flight",
            args.iterations, args.in_flight
        );
        let (_, _, result) = run_phase(
            sender,
            receiver,
            &server,
            next_sequence,
            args.iterations,
            payload_len,
            args.in_flight,
            args.batch_size,
            !args.skip_validation,
            !args.throughput_only,
            args.profile_stages,
            prepared_inputs.as_ref(),
        )?;
        print_result(&args, result);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_phase(
        sender: RdmaSender,
        receiver: RdmaReceiver,
        server: &EndpointBootstrap,
        first_sequence: u64,
        iterations: u64,
        payload_len: usize,
        request_window: usize,
        batch_size: usize,
        validate: bool,
        measure: bool,
        profile_stages: bool,
        prepared_inputs: Option<&PreparedInputs>,
    ) -> Result<(RdmaSender, RdmaReceiver, PhaseResult)> {
        let last_sequence = first_sequence
            .checked_add(iterations)
            .context("sequence number overflow")?;
        let channel_capacity = request_window.div_ceil(batch_size) + 1;
        let (produced_tx, produced_rx) = sync_channel(channel_capacity);
        let (feedback_tx, feedback_rx) = sync_channel(channel_capacity);
        let started = Instant::now();
        let (sender, receiver, latencies_ns, producer_timings, consumer_timings) =
            thread::scope(|scope| -> Result<_> {
                let producer = scope.spawn(|| {
                    run_producer(
                        sender,
                        &server.writable,
                        first_sequence,
                        last_sequence,
                        payload_len,
                        request_window,
                        batch_size,
                        measure,
                        validate || measure,
                        !measure && !validate,
                        profile_stages,
                        prepared_inputs,
                        produced_tx,
                        feedback_rx,
                    )
                });
                let consumer = scope.spawn(|| {
                    run_consumer(
                        receiver,
                        first_sequence,
                        last_sequence,
                        payload_len,
                        validate,
                        measure,
                        profile_stages,
                        produced_rx,
                        feedback_tx,
                    )
                });
                let (sender, producer_timings) = producer.join().map_err(|_| anyhow!("producer thread panicked"))??;
                let (receiver, latencies, consumer_timings) =
                    consumer.join().map_err(|_| anyhow!("consumer thread panicked"))??;
                Ok((sender, receiver, latencies, producer_timings, consumer_timings))
            })?;
        Ok((
            sender,
            receiver,
            PhaseResult {
                elapsed: started.elapsed(),
                latencies_ns,
                producer_timings,
                consumer_timings,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_producer(
        mut sender: RdmaSender,
        remote: &gpu_rdma_runtime::control_protocol::MemoryRegionInfo,
        first_sequence: u64,
        last_sequence: u64,
        payload_len: usize,
        request_window: usize,
        batch_size: usize,
        measure: bool,
        track_requests: bool,
        use_preloaded_ring: bool,
        profile_stages: bool,
        prepared_inputs: Option<&PreparedInputs>,
        produced: SyncSender<ProducedBatch>,
        feedback: Receiver<ConsumerFeedback>,
    ) -> Result<(RdmaSender, WorkerTimings)> {
        let mut next_to_send = first_sequence;
        let mut outstanding = 0usize;
        let mut send_completions = VecDeque::with_capacity(MAX_IN_FLIGHT_SENDS);
        let mut timings = WorkerTimings {
            enabled: profile_stages,
            ..WorkerTimings::default()
        };

        while next_to_send < last_sequence {
            let reclaiming = profile_stages.then(Instant::now);
            drain_feedback(&mut sender, &feedback, &mut outstanding)?;
            add_elapsed(&mut timings.auxiliary, reclaiming);
            while outstanding >= request_window {
                let waiting = profile_stages.then(Instant::now);
                apply_feedback(
                    &mut sender,
                    feedback.recv().context("consumer stopped before producer completed")?,
                    &mut outstanding,
                )?;
                add_elapsed(&mut timings.wait, waiting);
            }
            if send_completions.len() >= MAX_IN_FLIGHT_SENDS {
                let waiting = profile_stages.then(Instant::now);
                wait_oldest_send(&mut sender, &mut send_completions)?;
                add_elapsed(&mut timings.auxiliary, waiting);
            }

            let preparing = profile_stages.then(Instant::now);
            let count = batch_size
                .min(request_window - outstanding)
                .min((last_sequence - next_to_send) as usize);
            let mut requests = Vec::with_capacity(if track_requests { count } else { 0 });
            if use_preloaded_ring {
                sender.publish_preloaded_slots(count)?;
                next_to_send += count as u64;
            } else {
                for _ in 0..count {
                    let request_started = measure.then(Instant::now);
                    let slot = prepared_inputs
                        .map(|inputs| inputs.get(next_to_send))
                        .unwrap_or_else(|| input_slot(next_to_send, payload_len));
                    sender.write_slot_local(slot)?;
                    if track_requests {
                        requests.push(PendingRequest {
                            sequence: next_to_send,
                            started: request_started,
                        });
                    }
                    next_to_send += 1;
                }
            }
            add_elapsed(&mut timings.prepare, preparing);

            let posting = profile_stages.then(Instant::now);
            let wr_id = sender.write_slots_remote(remote, count)?;
            add_elapsed(&mut timings.post, posting);
            timings.batches += 1;
            send_completions.push_back(wr_id);
            outstanding += count;
            let publishing = profile_stages.then(Instant::now);
            produced
                .send(ProducedBatch { count, requests })
                .context("consumer stopped before producer completed")?;
            add_elapsed(&mut timings.auxiliary, publishing);
        }

        while outstanding > 0 {
            let waiting = profile_stages.then(Instant::now);
            apply_feedback(
                &mut sender,
                feedback
                    .recv()
                    .context("consumer stopped before all responses arrived")?,
                &mut outstanding,
            )?;
            add_elapsed(&mut timings.wait, waiting);
        }
        while !send_completions.is_empty() {
            let waiting = profile_stages.then(Instant::now);
            wait_oldest_send(&mut sender, &mut send_completions)?;
            add_elapsed(&mut timings.auxiliary, waiting);
        }
        Ok((sender, timings))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_consumer(
        mut receiver: RdmaReceiver,
        first_sequence: u64,
        last_sequence: u64,
        payload_len: usize,
        validate: bool,
        measure: bool,
        profile_stages: bool,
        produced: Receiver<ProducedBatch>,
        feedback: SyncSender<ConsumerFeedback>,
    ) -> Result<(RdmaReceiver, Vec<u64>, WorkerTimings)> {
        let mut next_sequence = first_sequence;
        let mut pending = VecDeque::new();
        let mut pending_count = 0usize;
        let mut latencies_ns = Vec::with_capacity(if measure {
            (last_sequence - first_sequence) as usize
        } else {
            0
        });
        let mut timings = WorkerTimings {
            enabled: profile_stages,
            ..WorkerTimings::default()
        };

        while next_sequence < last_sequence {
            let waiting = profile_stages.then(Instant::now);
            let batches = loop {
                let batches = receiver.poll_output_batches()?;
                if !batches.is_empty() {
                    break batches;
                }
                std::hint::spin_loop();
            };
            add_elapsed(&mut timings.wait, waiting);
            timings.batches += batches.len() as u64;
            let consuming = profile_stages.then(Instant::now);
            let completed_slots = batches.iter().map(|count| *count as usize).sum();
            while pending_count < completed_slots {
                let batch = produced
                    .recv()
                    .context("producer stopped before all responses arrived")?;
                pending_count += batch.count;
                pending.extend(batch.requests);
            }
            if validate || measure {
                for _ in 0..completed_slots {
                    let slot = receiver
                        .read_slot_local()
                        .context("output notification published a missing slot")?;
                    let request = pending
                        .pop_front()
                        .context("received a response without a pending request")?;
                    ensure!(
                        request.sequence == next_sequence,
                        "response order mismatch: expected {next_sequence}, pending {}",
                        request.sequence
                    );
                    if validate {
                        validate_response(&slot, request.sequence, payload_len)?;
                    }
                    if let Some(started) = request.started {
                        latencies_ns.push(duration_ns(started.elapsed()));
                    }
                    next_sequence += 1;
                }
            } else {
                receiver.discard_slots(completed_slots)?;
                next_sequence += completed_slots as u64;
            }
            pending_count -= completed_slots;
            add_elapsed(&mut timings.prepare, consuming);
            let sending_feedback = profile_stages.then(Instant::now);
            feedback
                .send(ConsumerFeedback {
                    completed_slots,
                    receive_wrs: batches.len(),
                })
                .context("producer stopped before consumer completed")?;
            add_elapsed(&mut timings.post, sending_feedback);
        }
        ensure!(pending_count == 0, "responses completed with pending requests");
        ensure!(pending.is_empty(), "responses completed with pending metadata");
        Ok((receiver, latencies_ns, timings))
    }

    fn drain_feedback(
        sender: &mut RdmaSender,
        feedback: &Receiver<ConsumerFeedback>,
        outstanding: &mut usize,
    ) -> Result<()> {
        loop {
            match feedback.try_recv() {
                Ok(message) => apply_feedback(sender, message, outstanding)?,
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow!("consumer stopped before producer completed"));
                }
            }
        }
    }

    fn apply_feedback(sender: &mut RdmaSender, feedback: ConsumerFeedback, outstanding: &mut usize) -> Result<()> {
        ensure!(
            feedback.completed_slots <= *outstanding,
            "consumer returned more credits than the producer has outstanding"
        );
        sender.complete_round_trips(feedback.completed_slots)?;
        sender.replenish_output_notifications(feedback.receive_wrs)?;
        *outstanding -= feedback.completed_slots;
        Ok(())
    }

    fn wait_oldest_send(sender: &mut RdmaSender, completions: &mut VecDeque<u64>) -> Result<()> {
        let wr_id = completions.pop_front().context("no send completion available")?;
        sender.wait_for_completion(wr_id)
    }

    fn print_result(args: &Args, mut result: PhaseResult) {
        result.latencies_ns.sort_unstable();
        let throughput = args.iterations as f64 / result.elapsed.as_secs_f64();
        let one_way_gbps = throughput * args.size as f64 * 8.0 / 1_000_000_000.0;
        let round_trip_gbps = one_way_gbps * 2.0;

        println!("benchmark result:");
        println!("  tuples:             {}", args.iterations);
        println!("  elapsed:            {:.3?}", result.elapsed);
        println!("  throughput:         {:.2} tuples/s", throughput);
        println!("  input goodput:      {:.3} Gbit/s", one_way_gbps);
        println!("  bidirectional data: {:.3} Gbit/s", round_trip_gbps);
        if result.latencies_ns.is_empty() {
            println!("  latency:            disabled");
        } else {
            let mean_ns = result.latencies_ns.iter().map(|value| *value as u128).sum::<u128>() as f64
                / result.latencies_ns.len() as f64;
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
        }
        println!("  validation:         {}", !args.skip_validation);
        println!("  pre-generated input: {}", args.pre_generate_inputs);
        println!("  throughput only:    {}", args.throughput_only);
        println!("  batch size:         {}", args.batch_size);
        print_worker_timings(
            "producer",
            &result.producer_timings,
            ["prepare", "RDMA post", "credit wait", "control/send CQ"],
        );
        print_worker_timings(
            "consumer",
            &result.consumer_timings,
            ["consume", "feedback", "output CQ wait", "auxiliary"],
        );
    }

    fn print_worker_timings(name: &str, timings: &WorkerTimings, labels: [&str; 4]) {
        if !timings.enabled || timings.batches == 0 {
            return;
        }
        println!("{name} timings (average per batch):");
        for (label, duration) in
            labels
                .into_iter()
                .zip([timings.prepare, timings.post, timings.wait, timings.auxiliary])
        {
            println!("  {label:<16} {:.3} us", average_us(duration, timings.batches));
        }
    }

    fn add_elapsed(total: &mut Duration, started: Option<Instant>) {
        if let Some(started) = started {
            *total += started.elapsed();
        }
    }

    fn average_us(duration: Duration, count: u64) -> f64 {
        duration.as_secs_f64() * 1_000_000.0 / count as f64
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
