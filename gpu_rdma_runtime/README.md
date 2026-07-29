# GPU RDMA Runtime

The server exchanges slots through two RC queue pairs: a pre client publishes
input to the GPU and a post client receives GPU output. TCP exchanges the
client role plus QP, memory-region, GID, and path-MTU metadata and waits for
both roles to complete bootstrap before connecting either QP.

## Data path

1. Rust allocates two GPU-resident ring buffers through the CUDA Driver API.
2. Rust exports and registers them with `ibv_reg_dmabuf_mr`.
3. The pre client publishes a contiguous input batch with one RDMA
   Write-with-Immediate. Ring wrap requires one preceding plain write.
4. The server drains available CQ notifications, establishes GPUDirect memory
   ordering once for the group, and launches batches onto independent CUDA streams.
5. One CUDA block maps each slot, with its threads cooperating across the payload.
6. CUDA events retire batches in input order while later batches execute concurrently.
7. The server publishes each ordered output batch directly from the GPU ring with
   RDMA Write-with-Immediate, again using one preceding write only at ring wrap.
8. The post client receive CQE publishes the batch to its local consumer in order.

The GPU-node CPU only orchestrates complete batches; tuple data never stages in
host memory. Standard `libibverbs` still requires the CPU to launch CUDA work
and post output work requests.

The immediate value carries the batch count. Each peer maintains its monotonic
ring position locally, so producer-head metadata does not cross the network.
The pre and post clients each own one QP. The pre client only produces and the
post client only consumes. Both directions use credit-based flow control: the
GPU returns input credits after a batch has been retired, and the post client
returns output credits after it consumes a batch. The default ring capacity is
65536 slots; clients block only when their corresponding ring is full.

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
# Use `rm -rf cuda/build` first only when replacing an incompatible old
# configuration; `cmake --fresh` is unavailable on older CMake versions.
cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/build -j
cargo build --release --bins
```

The CUDA project requires CMake 3.22 or newer. If configure reports a lower
version, upgrade CMake before building the PTX.

CMake writes `cuda/process_function.ptx`. Rebuilding this file is sufficient
after changing `process_one`; no C++ host executable is involved.

## Flink JNI client

The Flink integration uses the optional `jni` feature. It keeps one
role-specific QP and CQ,
registered rings, and the TCP bootstrap in Rust while exposing only batch
operations to Java:

```bash
cargo build --release --features jni
```

The resulting `libgpu_rdma_runtime` native library must be installed where the
Flink TaskManager can load it. Java loads it with
`System.loadLibrary("gpu_rdma_runtime")`. The native client sends either `pre`
or `post` in the length-prefixed JSON bootstrap. Use
`RustRdmaRingBuffer.Factory.PRE` for the input operator and
`RustRdmaRingBuffer.Factory.POST` for the output operator. No RDMA-CM
connection is used on this path.

## Run

On the GPU server (GPU1 is closest to `mlx5_0` in the example topology):

```bash
target/release/rdma_gpu_server \
  --port 50001 \
  --ib-device mlx5_0 \
  --ib-port 1 \
  --gid-index 3 \
  --cuda-device 1 \
  --warmup-iterations 10000 \
  --batch-size 64 \
  --profile-stages \
  --iterations 1000000
```

On the peer:

```bash
target/release/rdma_test_client \
  --role pre \
  --server 192.168.1.17:50001 \
  --ib-device mlx5_0 \
  --ib-port 1 \
  --gid-index 3 \
  --warmup-iterations 10000 \
  --iterations 1000000 \
  --batch-size 64 \
  --size 2048 \
  --pre-generate-inputs \
  --skip-validation \
  --throughput-only

target/release/rdma_test_client \
  --role post \
  --server 192.168.1.17:50001 \
  --ib-device mlx5_0 \
  --ib-port 1 \
  --gid-index 3 \
  --warmup-iterations 10000 \
  --iterations 1000000 \
  --batch-size 64 \
  --size 2048 \
  --skip-validation \
  --throughput-only
```

The bootstrap negotiates the lower active RDMA MTU. Both peers must still use
compatible RoCE GIDs, normally the same RoCE version and IP-family entry.

The server can also run without `--iterations`. In that mode it processes
input until PRE closes its session and sends the input-done message; POST then
drains the final output batch and can close normally:

```bash
target/release/rdma_gpu_server --port 50001 \
  --ib-device mlx5_1 --ib-port 1 --gid-index 3 --cuda-device 0 \
  --warmup-iterations 0 --batch-size 64 --profile-stages
```

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
launch. The server uses a fixed four-lane CUDA pipeline. The client `--in-flight`
window must contain at least two batches to expose useful pipeline parallelism;
`in-flight >= batch-size * 4` is a good starting point. The scheduler never waits
for all pipeline lanes to fill, so smaller client windows remain valid.

For the built-in byte map, edit only `process_one` in `cuda/process_map.cuh` and
rebuild the PTX. The stable kernel distributes tuple bytes across the block.
Use `--profile-stages` only for diagnosis; its per-stage clocks add overhead and
the reported stages overlap, so their averages are not additive.

## GPU price imputation

The processing kernel also supports the stateful KNN price imputer used by
`org.example.flinke2c.ImputationFunction`. Select it in both RDMA Flink
operators with:

```text
rdmaProcessingSpec={"function":"IMPUTE","field_index":0,"fields":["DECIMAL_BYTES","INT64","INT64","BYTES","BYTES","TIMESTAMP_MILLIS","BYTES"]}
```

The seven fields must be ordered as `price DECIMAL(23,3)`, `auction BIGINT`,
`bidder BIGINT`, `channel STRING`, `url STRING`, `dateTime TIMESTAMP(3)`, and
`extra STRING`. Null non-price fields are replaced with the same defaults as
the Java UDF. A null price uses inverse-distance-weighted KNN with `K=10`, the
newest 512 observations, and a 5,000-observation GPU ring. Only real prices
enter history.

Rows in a batch are logically processed in input order. The GPU evaluates
missing rows in parallel, but each row includes earlier real prices from its
batch. CUDA event dependencies order history commits between pipeline lanes,
so batching and the multi-stream pipeline do not change neighbor selection.
Like the Java UDF, this state is session-local and is not checkpointed.
Distance and weighting use GPU `double`; an imputed value exactly on a decimal
halfway boundary can round differently from `BigDecimal.valueOf(...).setScale`
by one unit in the last (`0.001`) place.
