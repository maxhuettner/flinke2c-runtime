# GPU RDMA Runtime

The server and test client exchange slots through an RC queue pair. TCP is used
only to exchange QP, memory-region, GID, and path-MTU metadata.

## Data path

1. Rust allocates two GPU-resident ring buffers through the CUDA Driver API.
2. Rust exports and registers them with `ibv_reg_dmabuf_mr`.
3. The client publishes a contiguous input batch with one RDMA
   Write-with-Immediate. Ring wrap requires one preceding plain write.
4. The server drains available CQ notifications, establishes GPUDirect memory
   ordering once for the group, and launches batches onto independent CUDA streams.
5. One CUDA block maps each slot, with its threads cooperating across the payload.
6. CUDA events retire batches in input order while later batches execute concurrently.
7. The server publishes each ordered output batch directly from the GPU ring with
   RDMA Write-with-Immediate, again using one preceding write only at ring wrap.
8. The client receive CQE publishes the batch to its local consumer in order.

The GPU-node CPU only orchestrates complete batches; tuple data never stages in
host memory. Standard `libibverbs` still requires the CPU to launch CUDA work
and post output work requests.

The immediate value carries the batch count. Each peer maintains its monotonic
ring position locally, so producer-head metadata does not cross the network.
The benchmark client runs input production/QP posting and output CQ polling in
separate threads. The producer exclusively owns the QP; the consumer returns
credits and receive-WR replenishment requests once per completed batch group.

## Structure

- `cuda/process_function.cu`: stable cooperative per-slot processing kernel.
- `cuda/process_map.cuh`: replaceable per-element `process_one` implementation.
- `cuda/slot.h`: CUDA data layout matching Rust.
- `src/gpu_runtime/cuda.rs`: CUDA Driver API, PTX, and DMA-BUF ownership.
- `src/gpu_runtime/endpoint.rs`: QP setup and ordered GPU-to-peer writes.
- `src/gpu_runtime/server.rs`: bootstrap and processing loop.
- `src/bin/rdma_gpu_server.rs`: Rust server CLI.

## Requirements

- CUDA toolkit and NVIDIA open kernel driver
- `libibverbs` development files
- GPUDirect RDMA-capable GPU/RNIC topology
- DMA-BUF support in the NVIDIA and RNIC drivers

## Build

From `gpu_rdma_runtime`:

```bash
cmake --fresh -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/build -j
cargo build --release --bins
```

CMake writes `cuda/process_function.ptx`. Rebuilding this file is sufficient
after changing `process_one`; no C++ host executable is involved.

## Run

On the GPU server (GPU1 is closest to `mlx5_0` in the example topology):

```bash
target/release/rdma_gpu_server \
  --kernel cuda/process_function.ptx \
  --ib-device mlx5_0 \
  --ib-port 1 \
  --gid-index 3 \
  --cuda-device 1 \
  --port 50001 \
  --warmup-iterations 10000 \
  --batch-size 16 \
  --pipeline-depth 4 \
  --profile-stages \
  --iterations 1000000
```

On the peer:

```bash
target/release/rdma_test_client \
  --server 192.168.1.17:50001 \
  --ib-device mlx5_0 \
  --ib-port 1 \
  --gid-index 3 \
  --warmup-iterations 10000 \
  --iterations 1000000 \
  --in-flight 256 \
  --batch-size 16 \
  --size 2048
```

The bootstrap negotiates the lower active RDMA MTU. Both peers must still use
compatible RoCE GIDs, normally the same RoCE version and IP-family entry.

The client reports actual per-request p50/p95/p99/p99.9 round-trip latency and
application goodput. Use the same warm-up and measured iteration counts on both
processes. Use the same `--batch-size` on both sides. Sweep batch sizes such as
`1, 4, 16, 64, 256` while keeping `--in-flight` at least as large as the batch.
Validation is enabled by default; use
`--skip-validation` only to isolate transport overhead.
Use `--pre-generate-inputs` to remove payload construction from the timed loop.
The benchmark payload repeats every 251 sequence numbers, so this mode prepares
the 251 distinct slots once instead of allocating about 2 GiB for one million
byte-identical pattern repetitions. Copying into registered send memory remains
inside the measurement.
Use `--throughput-only --skip-validation` for a transport/GPU throughput run
comparable to a pre-encoded buffered source. This disables per-tuple timestamps,
latency storage, response copies, and response payload inspection; ordered batch
CQEs still drive ring credits. The registered send ring is populated once before
warm-up and recycled without payload copies. Run the normal mode separately for
latency, serialization, and correctness results.
Add `--profile-stages` to the client only for diagnosis. It reports batch-level
producer preparation/post/credit times and consumer CQ/processing/feedback times.

`--batch-size` controls how many tuples share one RDMA notification and CUDA
launch. `--pipeline-depth` controls how many server-side CUDA batches can overlap.
The client `--in-flight` window must contain at least two batches to expose useful
pipeline parallelism; a good starting point is
`in-flight >= batch-size * pipeline-depth`. The scheduler never waits for all
pipeline lanes to fill, so smaller client windows remain valid.

For the built-in byte map, edit only `process_one` in `cuda/process_map.cuh` and
rebuild the PTX. The stable kernel distributes tuple bytes across the block.
Use `--profile-stages` only for diagnosis; its per-stage clocks add overhead and
the reported stages overlap, so their averages are not additive.
