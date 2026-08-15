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
Flink `AsyncScalarFunction`: calls are held in FIFO order and submitted
through one JNI call and one host-to-device copy per batch. One CUDA kernel
computes the batch in parallel, and a second commits observed prices in
source order after the batch has read the old history, matching the RDMA
implementation's state boundary.

Like the currency conversion UDF, batches are round-robined across
`pipelineDepth` CUDA streams so consecutive batches' copy/kernel/copy phases
can overlap instead of fully serializing behind one synchronize per batch.
The KNN history is one piece of state shared by every lane, so it cannot be
pipelined as freely as the stateless currency conversion path: the native
side makes each batch's compute and history-commit kernels wait on a GPU
event for the previous batch's commit (`cudaStreamWaitEvent`) before running,
regardless of which lane submits it, so history updates still apply in strict
submission order. That wait is enqueued on the GPU, not blocked on the CPU,
so the calling thread can still fill and submit the next lane without
stalling — only the state-touching kernels are serialized, not the batches'
H2D/D2H copies or the CPU-side dispatch.

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
Direct CUDA batches default to 64 rows with a 1 ms maximum wait for a partial
batch. The library path and CUDA device are TaskManager-wide JVM options,
since the native library itself has to be loaded before any job runs:

```text
-Dflinke2c.imputation.gpu.library=/path/to/gpu_rdma_runtime/cuda/build/libflinke2c_imputation_gpu.so
-Dflinke2c.imputation.gpu.device=0
```

Everything else — batch size, batch delay, pipeline depth, threads per block —
is per-job configuration, not a JVM flag. It resolves in this order: an
explicit constructor argument on `ImputationFunctionGpu`, then the
`flinke2c.imputation.gpu.conf` job parameter (set per job/session from SQL,
below), then the matching `-D` system property as a cluster-wide fallback,
then the built-in default. To set it from SQL the same way `RdmaOperator`
takes its `conf` string via `table.exec.external-runtime.conf.<class>`: Flink
only exposes `pipeline.global-job-parameters` to UDF code
(`FunctionContext.getJobParameter`) — arbitrary `SET 'x'='y'` keys are not
visible to a UDF's `open()`, only that one. So the whole conf string goes in
as the value of a single job parameter:

```sql
SET 'pipeline.global-job-parameters' =
    'flinke2c.imputation.gpu.conf:batchsize=1024;pipelinedepth=8;threadsperblock=128';
```

The inner string is parsed exactly like `RdmaOperator.RdmaConfig`'s `conf`
string: semicolon-separated `key=value` pairs, keys lower-cased and trimmed,
malformed entries silently ignored. Recognized keys: `batchsize`,
`batchdelaymicros`, `pipelinedepth`, `threadsperblock`, `device`.

Flink must allow enough async calls to remain outstanding to fill
`pipelineDepth` batches concurrently, not just one, or batches will stay
partial and hit the delay timeout instead of filling by count. As a rule of
thumb, set `max-concurrent-operations` to at least
`batch-size * pipeline-depth * 2`:

```sql
SET 'table.exec.async-scalar.max-concurrent-operations' = '512';
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
`CurrencyConversionFunction`. It sends batches of prices through JNI,
multiplies non-null values by `0.908` in the CUDA kernel, and completes each
call's future once its batch's device-to-host copy lands. Null prices remain
null. The GPU result is rounded to `DECIMAL(23,3)` in Java, matching the
declared SQL result type. The function is deterministic, so futures may
complete out of input order across batches without affecting correctness;
Flink's async operator restores stream order downstream.

The native context keeps `pipelineDepth` independent CUDA streams ("lanes"),
each with its own pinned host and device buffers. Batches are submitted to
lanes round robin; a submission launches the host-to-device copy, kernel, and
device-to-host copy on that lane's stream and returns immediately instead of
blocking, so the next batch can be filled and launched on another lane while
this one is still running on the GPU. A lane is only waited on when it is
about to be reused (or when the call queue drains to empty), so several
batches' copy/kernel/copy phases overlap instead of fully serializing behind
one synchronize per batch. Without this, throughput is bounded by
`batchSize / (H2D + kernel + D2H + JNI overhead)` per round trip; with it,
`pipelineDepth` round trips can be in flight at once.

The CUDA build above produces the library. Make it visible to every
TaskManager. Only the library path and CUDA device are TaskManager-wide JVM
options, since the native library has to be loaded before any job runs:

```text
-Dflinke2c.currency.gpu.library=/path/to/gpu_rdma_runtime/cuda/build/libflinke2c_currency_conversion_gpu.so
-Dflinke2c.currency.gpu.device=0
```

Batch size, batch delay, pipeline depth, and threads per block are per-job
configuration instead, resolved in this order: an explicit constructor
argument on `CurrencyConversionFunctionGpu`, then the
`flinke2c.currency.gpu.conf` job parameter (set per job/session from SQL,
below), then the matching `-D` system property as a cluster-wide fallback,
then the built-in default. To set it from SQL the same way `RdmaOperator`
takes its `conf` string via `table.exec.external-runtime.conf.<class>`: Flink
only exposes `pipeline.global-job-parameters` to UDF code
(`FunctionContext.getJobParameter`) — arbitrary `SET 'x'='y'` keys are not
visible to a UDF's `open()`, only that one. So the whole conf string goes in
as the value of a single job parameter:

```sql
SET 'pipeline.global-job-parameters' =
    'flinke2c.currency.gpu.conf:batchsize=1024;pipelinedepth=8;threadsperblock=256';
```

The inner string is parsed exactly like `RdmaOperator.RdmaConfig`'s `conf`
string: semicolon-separated `key=value` pairs, keys lower-cased and trimmed,
malformed entries silently ignored. Recognized keys: `batchsize`,
`batchdelaymicros`, `pipelinedepth`, `threadsperblock`, `device`.

Register `CurrencyConversionFunctionGpu` in place of
`CurrencyConversionFunction`. The async operator needs enough outstanding
calls to fill `pipelineDepth` batches concurrently, not just one:

```sql
SET 'table.exec.async-scalar.max-concurrent-operations' = '256';
SET 'table.exec.async-scalar.retry-strategy' = 'NO_RETRY';
```

As a rule of thumb, set `max-concurrent-operations` to at least
`batch-size * pipeline-depth * 2` so the queue can keep every lane fed while
the previous round trip is still draining. Too few outstanding calls and
batches stay partial, hitting `batch-delay-micros` on every flush instead of
filling.

The currency function is stateless, so retries do not duplicate history, but
disabling retries keeps comparisons with the other GPU paths deterministic.
The imputation-specific planner patch is not required for null currency
inputs when normal SQL null propagation is desired; Flink can complete those
rows as null without invoking the function.

### Batch-size / pipeline-depth sweeps

Currency conversion is one cheap multiply per row and imputation's per-row
compute is still small relative to a round trip, so a single round trip (JNI
call, host-device copy, kernel launch, RDMA hop, or whatever the path adds)
costs far more than the arithmetic itself; at `batchSize=1` any GPU path will
lose badly to staying on the CPU in the same Flink thread. The question worth
measuring is whether *aggregate* throughput under load can still win by
amortizing that fixed cost over more parallel work. These knobs control how
much parallel work is in flight; all of them are runtime parameters read at
`open()` — none require rebuilding the PTX/shared library to sweep, and on
the direct paths none require a TaskManager restart either: set them per job
via the `flinke2c.*.gpu.conf` job parameter (see above) and just resubmit the
job between sweep points.

- **Batch size** — how many rows one kernel launch (and, on the direct path,
  one JNI call; on the RDMA path, one RDMA notification) processes. Larger
  batches mean fewer round trips, more threads launched per kernel, and this
  is also what total thread count tracks (see below).
- **Pipeline depth** — how many batches can be on the GPU at once
  (`flinke2c.currency.gpu.pipeline-depth` / `flinke2c.imputation.gpu.pipeline-depth`
  on the direct paths, `--cuda-pipeline-depth` on the RDMA server). Depth
  beyond 1 is what lets a new batch's copy/launch overlap a previous batch's
  still-running copy or kernel, instead of the GPU sitting idle between round
  trips. On the imputation path, raising this does not change result
  ordering: the native side still applies history commits in strict
  submission order via a GPU-side event wait between lanes, regardless of how
  many lanes are configured.
- **Threads per block** — on the direct paths,
  `flinke2c.currency.gpu.threads-per-block` /
  `flinke2c.imputation.gpu.threads-per-block` (defaults 256 / 128, max 1024);
  not exposed on the RDMA path, which uses a fixed 256. Every kernel here maps
  one CUDA thread to one row, so *total* threads launched per batch is always
  batch size regardless of this setting — this only changes how those threads
  are grouped into blocks (occupancy/scheduling), it does not add or remove
  parallelism on its own. Sweep it after batch size and pipeline depth are
  already at a good point, not before; it's a second-order effect by
  comparison.

Sweep batch size and pipeline depth together and compare against the plain
CPU baseline (`CurrencyConversionFunction` / `ImputationFunction`) at matching
Flink parallelism (for imputation, parallelism one, since its history is
global rather than keyed):

- Direct paths: vary `batchsize` (e.g. 16, 64, 256, 1024) times
  `pipelinedepth` (e.g. 1, 2, 4, 8) via the `flinke2c.*.gpu.conf` job
  parameter, keeping `table.exec.async-scalar.max-concurrent-operations` well
  above `batchsize * pipelinedepth` at every point (also set per job, via
  `SET`) so batches actually fill by count instead of timing out on the batch
  delay. Once that combination plateaus, optionally sweep `threadsperblock`
  (e.g. 64, 128, 256, 512) at the winning batch size/depth to check for a
  further, smaller gain.
- RDMA path: vary `--batch-size` on both `rdma_gpu_server` and
  `rdma_test_client` together (the README's transport section already
  recommends `1, 4, 16, 64, 256`) times `--cuda-pipeline-depth` on the server.
  Remember the RDMA transport sends one fixed `MAX_ITEM_SIZE` slot per row
  regardless of batch size, so its throughput plateaus once the link is
  saturated; past that point more pipeline depth or batch size will not help,
  and the smallest batch at the plateau is the right operating point.

Expect diminishing returns once either knob is large enough to keep the GPU
continuously fed — at that point the bottleneck has moved to one of: raw
kernel throughput (unlikely for currency conversion; more plausible for
imputation's `O(SEARCH_LIMIT)` neighbor scan per row), the single-threaded
batch-fill/dispatch loop on the direct paths (see below), the single-thread
`commit_batch_history` kernel on the imputation path (its cost scales with
batch size and cannot itself be pipelined across batches), or, on the RDMA
path, link bandwidth for the fixed-size slot format.

Both direct-path UDFs dispatch batches from one dedicated executor thread
(filling pinned buffers and issuing the JNI call). Once GPU-side round trips
overlap via pipelining, that single CPU thread doing the fill-loop and native
dispatch sequentially across lanes can become the new bottleneck at high
depth — if throughput plateaus well before the GPU should plausibly be
saturated, check this before assuming it is a GPU limit. In practice the
biggest single contributor found on that thread was `BigDecimal.valueOf(double)`
on the output side: it's `new BigDecimal(Double.toString(val))` internally, a
full decimal string format-and-reparse per row, run once per output row on
that one thread. Both UDFs now compute the scale-3 unscaled value directly
and use the non-parsing `BigDecimal.valueOf(long, int)` overload instead
(falling back to the exact string-based path only for values whose unscaled
magnitude could overflow a `long`, which no realistic price approaches).
This alone can be the difference between "GPU-bound" and "one Java thread
doing decimal string parsing 2048 times per batch bound."

A second, related fix a CPU flame graph (async-profiler / JFR, attached to
the TaskManager during a real run) surfaced: both UDFs originally ran their
per-batch dispatch (`eval()`'s immediate `execute()` calls) *and* the
delayed partial-batch timer on the same `ScheduledThreadPoolExecutor`. That
class backs *everything* — even zero-delay `execute()` calls — with the same
priority-heap queue (`DelayedWorkQueue`) it uses for the timer, so every
single dispatch paid an O(log n) heap insert/remove
(`ScheduledFutureTask.compareTo`, `DelayedWorkQueue.siftDown`) instead of the
O(1) a plain FIFO queue would cost. Measured on a real profile, that was
~5.8% of total CPU self time before any of the framework overhead below.
Both UDFs now split this into two executors: a plain
`Executors.newSingleThreadExecutor` for dispatch, and a separate
`ScheduledThreadPoolExecutor` used only for the timer — which, holding at
most one pending task, pays negligible heap cost regardless.

Reading a flame graph of either UDF, expect roughly these buckets (order and
exact split will vary by run and hardware):
- **The actual GPU work** — `submitBatch`/`waitBatch` and everything under
  `CurrencyConversionGpuNative`/`ImputationGpuNative` — should be a small
  slice (single-digit percent) if pipelining is doing its job. If this is
  large instead, that points back at the round trip itself, not the JVM side.
- **Flink's async-operator framework** — `AsyncWaitOperator`,
  `OrderedStreamElementQueue`, `DelegatingAsyncResultFuture`, plus JDK
  `CompletableFuture` completion machinery — this is inherent per-row
  `AsyncScalarFunction` cost, not something tunable from inside this UDF; see
  below.
- **Flink row serialization** (`RowDataSerializer`, `GenericRowData`,
  `DecimalData`) converting the returned value back into Flink's internal
  row format — also framework cost, largely unavoidable.
- **JIT compiler activity** (`CompileBroker`, `C2Compiler`, `PhaseChaitin`,
  and similar HotSpot-internal frames) — if this is a large fraction, the
  profiled window likely overlapped JVM warm-up rather than steady state;
  re-profile after several minutes of sustained load before trusting the
  numbers.
- **Our own UDF code** (batching, `ByteBuffer` marshalling, `toScale3`) —
  should also be a small slice; if this grows large, that's the concrete,
  fixable kind of cost the two fixes above were.

If the dispatch thread is still the ceiling after both fixes (check: is the
`flinke2c-*-gpu-*` thread pinned near 100% CPU while GPU utilization is low?
that confirms it), the next-larger lever is architectural, not a tuning knob:
`AsyncScalarFunction.eval()` costs more per row than the plain CPU UDF
inherently pays, independent of anything in this codebase — a `BigDecimal`
argument boxed by Flink's codegen, a `CompletableFuture` allocated by Flink
per call, and the async operator's own in-flight/ordering bookkeeping, all on
top of whatever our own queueing adds. A real profile measured this Flink
async-framework tax (`AsyncWaitOperator`/`OrderedStreamElementQueue`/
`DelegatingAsyncResultFuture` plus `CompletableFuture`) at roughly 14% of
total CPU self time on its own — a real cost, though on its own not close to
explaining a multi-times throughput gap against a synchronous CPU baseline;
expect it to be one contributor among several rather than the single
explanation. The RDMA path doesn't pay any of this:
`RdmaPreOperator`/`RdmaPostOperator` are custom operators that encode whole
batches to bytes once (`ExternalRuntimeBinaryCodec`) and move them as
`byte[]`, with no per-row future and no per-row scalar-function call at all.
Closing the *entire* gap to a CPU baseline that also doesn't pay per-row
async overhead may not be possible from inside the `AsyncScalarFunction`
model — the more faithful fix, if the gap remains large after the above,
would be reshaping the direct-CUDA path into an operator pair structured like
`RdmaPreOperator`/`RdmaPostOperator` (batches of raw bytes in and out, no
per-row `CompletableFuture`) instead of a scalar UDF. That's a substantially
larger change than anything above and worth doing only if the cheaper fixes
don't close enough of the gap.

## Direct CUDA currency conversion operator (batch-native, no AsyncScalarFunction)

`CudaCurrencyConversionOperator` (`java/flink/CudaCurrencyConversionOperator.java`)
is that larger change: a real operator replacement for
`CurrencyConversionFunctionGpu`, built the way the "more faithful fix"
paragraph above describes, once profiling on a real workload confirmed the
`AsyncScalarFunction` per-row completion machinery (one `CompletableFuture`
per row, `AsyncWaitOperator`'s ordered result queue, one mailbox repost per
row) was the remaining ceiling after the batching/pipelining/executor fixes
above — none of which touch that machinery, since it's per-row regardless of
how batched the GPU dispatch is.

**Design.** Unlike `RdmaPreOperator`/`RdmaPostOperator`, this needs only one
operator: the native call is an in-process JNI call to a local GPU, not a
cross-machine RDMA round trip, so there's no reason to split "publish input"
and "consume output" across two operators/TaskManagers. It reuses the exact
lane/pipelining design `CurrencyConversionFunctionGpu` uses (buffer a batch,
submit non-blockingly to one of `pipelineDepth` CUDA streams, collect a
lane's previous batch right before reusing it), but emits results with a
plain `output.collect()` loop instead of completing futures — no
`CompletableFuture`, no ordered async queue, no per-row mailbox repost.
Because `processElement` is guaranteed non-concurrent with itself by Flink's
runtime, none of the batching state needs the locking the async UDF requires.
Row order is preserved by construction (lanes are always collected in
submission order, always before being reused), so unlike `RdmaPostOperator`
there's no explicit sequence-number check needed — that one exists because
RDMA crosses a network boundary, and this doesn't.

**Two important things to know before using this:**

1. **It is written by close analogy to `RdmaOperator`/`RdmaPreOperator`/
   `RdmaPostOperator`, not verified against `ExternalRuntimeOperator`'s
   actual source** (not present in this repository). The constructor shape,
   the inherited `conf`/`output` fields, and the
   `openInternal`/`closeInternal`/`processElementInternal`/`processRow`
   template methods are inferred from how the RDMA operators use them. The
   batching/pipelining/row-conversion logic doesn't depend on getting those
   exactly right, but the class won't compile until they match the real base
   class — check this first.
2. **Wiring it into a query the way `RdmaOperator` is wired in — as a
   transparent swap-in for a plain `SELECT CurrencyConversionFunction(price)`
   call via `table.exec.external-runtime.conf.<class>` — depends on planner
   code that also isn't in this repository.** `usesDirectCudaTransport(conf)`
   is provided (mirroring `RdmaOperator.usesRdmaTransport`) in case the
   dispatch convention expects a predicate like that per candidate operator
   class, but whether the planner's dispatch is a generic lookup (in which
   case this "just works" once wired) or a hardcoded reference to
   `RdmaPreOperator`/`RdmaPostOperator` specifically (in which case adding
   this operator as a candidate needs an edit on the Flink planner side too)
   isn't something this repository can answer.

**Native bridge is intentionally separate from `CurrencyConversionGpuNative`.**
This operator's package (`org.apache.flink.table.runtime.functions.table
.externalruntime`, matching `RdmaPreOperator`/`RdmaPostOperator`) implies it
gets compiled into the Flink distribution itself, the same way
`RustRdmaNative` is for the RDMA path — while `CurrencyConversionGpuNative`
(used by the `AsyncScalarFunction` path) ships in the separate `flinke2c`
user JAR. Flink's own classloader generally can't see classes from a
separately deployed user JAR, so reusing that bridge class directly across
the package boundary isn't reliable. `DirectCudaCurrencyNative`
(`java/flink/DirectCudaCurrencyNative.java`) is a second, self-contained JNI
bridge with its own exported symbol names, calling into the *same*
`direct_currency_conversion_jni.cu` logic — see that file's "Bridge 1"/
"Bridge 2" comments. No CUDA logic is duplicated, only the thin JNI wrapper
functions. One consequence worth testing before relying on both paths in the
same cluster: loading the same `.so` from two different classloaders in one
JVM process can fail with `UnsatisfiedLinkError: Native Library ... already
loaded in another classloader` if a single TaskManager process ever runs both
`CurrencyConversionFunctionGpu` and `CudaCurrencyConversionOperator` over its
lifetime.

**Conf keys** (same semicolon-delimited `key=value` style as
`RdmaOperator.RdmaConfig`, e.g. via whatever conf string your planner rule
passes through): `batchsize` (default 64), `pipelinedepth` (default 4, max
64), `threadsperblock` (default 256, max 1024), `device` (default 0),
`fieldindex` (default 0 — the row position of the `DECIMAL` price column to
convert; must name a `DECIMAL` column or construction fails).
