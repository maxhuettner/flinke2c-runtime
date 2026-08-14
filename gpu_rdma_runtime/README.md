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
5. Stateless increment/currency kernels modify input slots in place with one
   packed CUDA thread per row. Imputation uses one warp per row and eight rows
   per block to copy payloads coalescently into the separate output ring.
6. CUDA events retire batches in input order while later batches execute concurrently.
7. The server publishes each ordered output batch directly from its GPU ring
   with RDMA Write-with-Immediate. Stateless output is sent from the input ring,
   and its input credit is returned only after the RNIC finishes reading those
   slots. Ring wrap may split a batch into multiple writes.
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

- `cuda/process_function.cu`: in-place stateless and warp-per-row imputation kernels.
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

CMake also builds `cuda/build/libflinke2c_imputation_gpu.so`, the JNI library
used by the direct-call imputation UDF described below.
CMake also builds `cuda/build/libflinke2c_currency_conversion_gpu.so`, the JNI
library used by the batched direct-call currency conversion UDF.

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

Use `--profile-stages` only for diagnosis; its per-stage clocks add overhead and
the reported stages overlap, so their averages are not additive.

## GPU RDMA currency conversion

The RDMA processing kernel supports the stateless currency conversion used by
`org.example.flinke2c.CurrencyConversionFunction`. Select it in both RDMA Flink
operators with:

```text
rdmaProcessingSpec={"function":"CURRENCY_CONVERSION","field_index":2,"fields":["INT64","INT64","DECIMAL_BYTES","TIMESTAMP_MILLIS","BYTES","INT64"]}
```

`field_index` may select any `DECIMAL_BYTES` field in the declared wire schema.
The kernel multiplies the unscaled decimal integer by `908 / 1000` and rounds
HALF_UP at the existing scale. The calculation uses integer byte arithmetic,
so `DECIMAL(23,3)` values do not lose precision through a `double` conversion.
Null target fields remain null.

Currency conversion and increment use one packed CUDA thread per row and modify
their input slots in place. The output RDMA write reads the result directly from
the input GPU ring, removing the full input-to-output GPU row copy. The server
waits for that RDMA write's completion before returning the corresponding input
credit, so PRE cannot overwrite a slot while the RNIC is still reading it.
The RDMA transport still sends one fixed-size slot per row, so throughput can
plateau once batching has amortized launch and notification overhead. Increasing
the batch beyond that point does not reduce the bytes transferred; select the
smallest batch at the measured throughput plateau, commonly 1024 for this
workload.

PRE publishes every complete or flushed partial RDMA batch before emitting its
corresponding placeholders downstream. This ordering allows large batches even
when PRE and POST are chained or the Flink network has less buffering than one
batch. Watermarks and checkpoint barriers flush partial batches first.

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

## Direct CUDA price imputation

`org.example.flinke2c.ImputationFunctionGpu` implements the same scalar UDF
interface but directly invokes the standalone kernel in
`cuda/direct_imputation_jni.cu`. It does not use the RDMA runtime. It is a
Flink `AsyncScalarFunction`: calls are held in FIFO order and submitted through
one JNI call and one host-to-device copy per batch. One CUDA kernel computes
the batch in parallel, and a second commits observed prices in source order
after the batch has read the old history, matching the RDMA implementation's
state boundary.

Build both CUDA artifacts as above, build the Flink UDF JAR from `flinke2c`,
and make the shared library visible to every TaskManager:

```bash
cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/build -j
cd ../flinke2c
mvn package

# TaskManager JVM option (the path must be absolute):
-Dflinke2c.imputation.gpu.library=/path/to/gpu_rdma_runtime/cuda/build/libflinke2c_imputation_gpu.so
```

The CUDA JNI target requires JDK headers. If CMake cannot find `jni.h`, set
`JAVA_HOME` to the JDK root before configuring:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
test -f "$JAVA_HOME/include/jni.h"
```

Alternatively, add `cuda/build` to the TaskManager's
`java.library.path`; the UDF then loads `flinke2c_imputation_gpu` by name.
The visible CUDA device defaults to zero. Select another device with
`-Dflinke2c.imputation.gpu.device=N` or by constructing the UDF with a device
index in Java. Direct CUDA batches default to 64 rows with a 1 ms maximum wait
for a partial batch. Configure them on the TaskManager with:

```text
-Dflinke2c.imputation.gpu.batch-size=64
-Dflinke2c.imputation.gpu.batch-delay-micros=1000
```

Flink must allow at least that many async calls to remain outstanding or the
batch cannot fill. For a batch size of 64, use at least:

```sql
SET 'table.exec.async-scalar.max-concurrent-operations' = '128';
SET 'table.exec.async-scalar.retry-strategy' = 'NO_RETRY';
```

The function is stateful, so retrying an individual async invocation could
insert the same observed price twice. Keep retries disabled and let Flink's
normal job-level recovery restart the non-checkpointed history.

Flink 2.2's async scalar code generator normally completes the result with
`null` without invoking the UDF when any argument is null. Price imputation
requires receiving a null `price`, so apply the included planner patch to the
Flink source tree and rebuild the Flink distribution:

```bash
cd /path/to/flink
git apply /path/to/gpu_rdma_runtime/flink-patches/async-scalar-null-arguments.patch
```

The patch forwards nullable boxed arguments to `AsyncScalarFunction.eval`,
matching normal scalar-function invocation. The imputation function already
normalizes null values before placing a row in the CUDA batch. This changes
null-input handling for every async scalar UDF in that Flink distribution, so
keep the patch local to the experimental build.

Register `ImputationFunctionGpu` in the same way as
`ImputationFunction`. Its input and output types are identical. The direct
UDF keeps a 5,000-observation GPU history per UDF instance; like the CPU and
RDMA versions, it is not checkpointed. Only real prices enter history. For a
result and performance comparison, run each implementation with fresh state,
the same input order and parallelism, and preferably parallelism one because
the imputer uses global rather than keyed history.

## Direct CUDA currency conversion

`org.example.flinke2c.CurrencyConversionFunctionGpu` is a stateless,
batched `AsyncScalarFunction` equivalent to
`CurrencyConversionFunction`. It sends one batch of prices through JNI,
multiplies non-null values by `0.908` in the CUDA kernel, and completes each
future in input order. Null prices remain null. The GPU result is rounded to
`DECIMAL(23,3)` in Java, matching the declared SQL result type.

The CUDA build above produces the library. Make it visible to every
TaskManager and configure the direct UDF with:

```text
-Dflinke2c.currency.gpu.library=/path/to/gpu_rdma_runtime/cuda/build/libflinke2c_currency_conversion_gpu.so
-Dflinke2c.currency.gpu.device=0
-Dflinke2c.currency.gpu.batch-size=64
-Dflinke2c.currency.gpu.batch-delay-micros=1000
```

Register `CurrencyConversionFunctionGpu` in place of
`CurrencyConversionFunction`. The async operator should have enough
outstanding calls to fill a batch:

```sql
SET 'table.exec.async-scalar.max-concurrent-operations' = '128';
SET 'table.exec.async-scalar.retry-strategy' = 'NO_RETRY';
```

The currency function is stateless, so retries do not duplicate history, but
disabling retries keeps comparisons with the other GPU paths deterministic.
The imputation-specific planner patch is not required for null currency
inputs when normal SQL null propagation is desired; Flink can complete those
rows as null without invoking the function.
