# GPU RDMA Runtime

The server and test client exchange slots through an RC queue pair. TCP is used
only to exchange QP, memory-region, GID, and path-MTU metadata.

## Data path

1. Rust allocates two GPU-resident ring buffers through the CUDA Driver API.
2. Rust exports and registers them with `ibv_reg_dmabuf_mr`.
3. The client writes input slots and then their producer head directly to GPU memory.
4. The server launches `process_slots`; one CUDA thread maps one slot.
5. The server posts ordered one-sided RDMA writes directly from the GPU output ring.
6. The client consumes the returned slots in their original order.

The CPU posts verbs work requests. Standard `libibverbs` does not let the CUDA
kernel itself initiate these network operations.

## Structure

- `cuda/process_function.cu`: parallel slot processing and ring commits.
- `cuda/process_map.cuh`: replaceable `process_one` implementation.
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
