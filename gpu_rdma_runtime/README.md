# GPU Runtime Scaffold

Minimal scaffold for wiring a framed binary request into a GPU kernel and returning the result.

## Files

- `kernel_scaffold.cu`
  CUDA persistent kernel and host-callable launcher `flinke2c_start_persistent_kernel`.
- `src/rdma_kernel_scaffold.rs`
  Thin host launcher + NCCL interface (no binary header parsing/validation on host).
- `src/gpu_executor.rs`
  Reusable persistent-kernel host executor shared by the NCCL and socket server binaries.
- `src/wire_codec.rs`
  Rust wire encode/decode helpers aligned with the project framing in `src/codec.rs`.
- `src/nccl_transport.rs`
  Linux-only CUDA + NCCL transport helpers (communicator bootstrap, device buffers, send/recv).
- `src/bin/gpu_kernel_cli.rs` (inside `gpu_runtime` crate)
  Rank-0 NCCL endpoint for GPU-to-GPU transport tests.
- `src/bin/gpu_test_client.rs` (inside `gpu_runtime` crate)
  Rank-1 NCCL endpoint that sends framed test rows and prints responses.
- `src/bin/gpu_socket_server.rs` (inside `gpu_runtime` crate)
  TCP server that accepts normal CPU clients and forwards framed requests to the persistent GPU kernel.
- `src/bin/cpu_test_client.rs` (inside `gpu_runtime` crate)
  CPU-only test client for the socket server.
- `src/bin/rdma_gpu_server.rs` (inside `gpu_runtime` crate)
  RDMA bootstrap server that exposes GPU input/output/control buffers to a remote client.
- `src/bin/rdma_test_client.rs` (inside `gpu_runtime` crate)
  CPU-only RDMA test client that writes requests directly into the remote GPU buffers.

## Binary frame contract

This scaffold now uses the same base wire conventions as `src/codec.rs`:

- frame envelope: `[i32_be payload_len][payload_bytes]`
- row payload layout for this demo:
  - `op: i32_be`
  - `row_id: i64_be`
  - `null_bitmap: 1 byte` (bit 0 is field-0 null)
  - `field_0: f32_be` if not null

## CPU client + GPU server flow

1. CPU client encodes one framed row payload and sends it over TCP.
2. Server starts `flinke2c_start_persistent_kernel` once at boot.
3. Server copies the request payload into GPU memory and signals the persistent kernel.
4. Kernel parses/validates the row payload, computes output, and serializes output payload bytes on GPU.
5. Server copies the GPU-produced output bytes back to host memory and returns them to the client.
6. Client decodes the framed response using the same wire codec.

This socket path removes the GPU requirement from the client. It is a transport split, not a full GPUDirect RDMA data path.

## CPU client + GPU server flow (RDMA)

1. RDMA server starts the persistent GPU kernel and registers the GPU input/output/control buffers with libibverbs.
2. Server accepts a TCP bootstrap connection and exchanges queue-pair metadata with the client.
3. CPU client RDMA-writes the request payload into the server GPU input buffer.
4. CPU client RDMA-writes the persistent-kernel control block to signal `request_ready=1`.
5. GPU kernel processes the request and sets `response_ready=1` in the GPU control block.
6. CPU client RDMA-reads the GPU control block until the response is ready, then RDMA-reads the GPU output buffer.

This is the test path that matches the intended architecture most closely.

## GPU peer test flow

The NCCL binaries remain useful for GPU-to-GPU transport experiments:

1. NCCL client encodes one framed row payload and copies it to device memory.
2. Client sends frame length + payload over NCCL `send/recv` (byte transport).
3. Server receives the frame, executes the GPU kernel through the shared `GpuExecutor`, and sends the response frame back over NCCL.

## Build hint

Compile CUDA code into a shared object and resolve symbol `flinke2c_start_persistent_kernel` from Rust runtime code.

Example:

```bash
nvcc -shared -Xcompiler -fPIC \
  -o gpu_runtime/libflinke2c_gpu.so \
  gpu_runtime/kernel_scaffold.cu
```

## CLI test loop (CPU client)

Start socket server:

```bash
cargo run -p flinke2c-gpu-runtime --bin gpu_socket_server -- \
  --cuda-lib gpu_runtime/libflinke2c_gpu.so \
  --bind 0.0.0.0:50051 \
  --requests 3
```

Run CPU client:

```bash
cargo run -p flinke2c-gpu-runtime --bin cpu_test_client -- \
  --server 127.0.0.1:50051 \
  --value 42.5 \
  --count 3
```

## CLI test loop (RDMA CPU client)

Start RDMA GPU server:

```bash
cargo run -p flinke2c-gpu-runtime --bin rdma_gpu_server -- \
  --cuda-lib gpu_runtime/libflinke2c_gpu.so \
  --bind 0.0.0.0:50061
```

Run RDMA client:

```bash
cargo run -p flinke2c-gpu-runtime --bin rdma_test_client -- \
  --server 127.0.0.1:50061 \
  --count 3
```

## CLI test loop (NCCL GPU peer transport)

Start server (rank 0):

```bash
cargo run -p flinke2c-gpu-runtime --bin gpu_kernel_cli -- \
  --cuda-lib gpu_runtime/libflinke2c_gpu.so \
  --bootstrap-file /tmp/flinke2c_nccl.id \
  --nranks 2 \
  --rank 0 \
  --peer-rank 1 \
  --requests 3
```

Run client (rank 1):

```bash
cargo run -p flinke2c-gpu-runtime --bin gpu_test_client -- \
  --bootstrap-file /tmp/flinke2c_nccl.id \
  --nranks 2 \
  --rank 1 \
  --peer-rank 0 \
  --max-frame-size 256 \
  --value 42.5 \
  --count 3
```

## RDMA note

The RDMA path in this scaffold uses raw libibverbs with a TCP bootstrap. For direct NIC access to GPU memory, the target system still needs the required GPUDirect RDMA support in the driver stack, such as DMA-BUF or `nvidia-peermem`, plus an RNIC/firmware configuration that allows GPU memory registration.

The NCCL binaries are still available for GPU-to-GPU transport tests. On InfiniBand/RoCE systems, NCCL can use RDMA-capable network backends. Set your NCCL environment to prefer IB/RDMA (for example `NCCL_IB_DISABLE=0`) according to your cluster setup.

## Persistent-kernel note

`gpu_kernel_cli` starts a persistent kernel at startup and signals work through a device control block, so request handling does not relaunch the kernel each time.
