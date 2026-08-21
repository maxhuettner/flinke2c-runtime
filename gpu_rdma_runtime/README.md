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
rdmaProcessingSpec={"function":"IMPUTE","field_index":0,"fields":["DECIMAL_UNSCALED_I64","INT64","INT64","BYTES","BYTES","TIMESTAMP_MILLIS","BYTES"]}
```

The seven fields must be ordered as `price DECIMAL(23,3)`, `auction BIGINT`,
`bidder BIGINT`, `channel STRING`, `url STRING`, `dateTime TIMESTAMP(3)`, and
`extra STRING`. **`price`'s wire field type is `DECIMAL_UNSCALED_I64`, not
`DECIMAL_BYTES`** — a fixed 8-byte scale-3 unscaled long, unlike every other
decimal-bearing function on this page (currency conversion, Black-Scholes),
which still use the variable-length `DECIMAL_BYTES` format. This is
IMPUTE-specific: `price`'s *declared* SQL type stays `DECIMAL(23,3)`, but
its *wire* representation is narrower (`DECIMAL(18,3)`, comfortably beyond
any realistic bid price) specifically to unlock `DecimalData`'s cheap
compact-decimal path on both ends — see the packed-imputation section below
for the full reasoning and cross-cutting change list. An old client still
sending `DECIMAL_BYTES` at field 0 is rejected at bootstrap (schema
validation), not silently misinterpreted. Null non-price fields are
replaced with the same defaults as the Java UDF. A null price uses
inverse-distance-weighted KNN with `K=10`, the newest 512 observations, and
a 5,000-observation GPU ring. Only real prices enter history.

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

## Direct CUDA currency conversion function (packed, dynamically loaded)

`CurrencyConversionGpuFunction` is that packed implementation: a dynamic
replacement for `CurrencyConversionFunctionGpu`, built the way the "more faithful fix"
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
`CurrencyConversionFunctionGpu` and `CurrencyConversionGpuFunction` over its
lifetime.

**Conf keys** (same semicolon-delimited `key=value` style as
`RdmaOperator.RdmaConfig`, e.g. via whatever conf string your planner rule
passes through): `batchsize` (default 64), `pipelinedepth` (default 4, max
64), `threadsperblock` (default 256, max 1024), `device` (default 0),
`fieldindex` (default 0 — the row position of the `DECIMAL` price column to
convert; must name a `DECIMAL` column or construction fails).

## Packed direct CUDA imputation

`ImputationGpuFunction` and the existing `flinke2c_imputation_gpu` library provide the
corresponding local path for imputation. The dynamically loaded function uses
the same framed seven-field ABI as `RdmaPreOperator`/`RdmaPostOperator`, writes
directly into pinned slot memory, and runs parsing, string hashing, KNN search,
history commit, and decimal encoding on the GPU. Use the normal dynamic
`impl=org.example.flinke2c.ImputationGpuFunction` configuration (with the optional `batchsize`,
`pipelinedepth`, `threadsperblock`, and `device` keys). The default schema is
`DECIMAL/BIGINT price, BIGINT auction, BIGINT bidder, STRING channel, STRING
url, TIMESTAMP, STRING extra`; `pricefield=...` and the corresponding other
`*field` keys can select positions. The planner must register
the dynamic `GpuRuntimeOperator` with the packed function interface described
below.

The same optimization is available through the normal dynamic GPU-function
configuration. `GpuRuntimeOperator` now accepts any class implementing
the packed `GpuRuntimeFunction` contract; it still loads the class from `impl=...`, but
forwards live rows directly and lets the function own native batching. The
updated user functions are `org.example.flinke2c.ImputationGpuFunction` and
`org.example.flinke2c.CurrencyConversionGpuFunction`.

**Submit/complete must run on separate threads.** `submitBatch`
([`direct_imputation_jni.cu`](cuda/direct_imputation_jni.cu)) only enqueues a
lane's H2D copy, kernels, and D2H copy on that lane's stream and returns —
`waitBatch` is the only call that blocks (`cudaEventSynchronize`). The
original implementation nonetheless ran `submitBatch` immediately followed by
`waitBatch` and `decode()` on one background thread. That serialized GPU
dispatch behind GPU completion: the thread could not enqueue lane N+1's work
until it had finished blocking on lane N, so the GPU's command queue only ever
held one lane's work regardless of `pipelinedepth`, and every batch paid a
full host round-trip stall between the previous batch finishing and the next
one being launched. Measured effect: capped around 800k tps against the same
kernels' >1M tps over the RDMA path, whose server dispatches batches without
blocking a single thread on each one's completion. `ImputationGpuFunction` now
splits this into a submitter thread (calls `submitBatch` only, loops
immediately) and a completer thread (drains a `submittedQueue` in the same
FIFO order and does `waitBatch`/`decode`), so the GPU stays fed the same way
the RDMA server's loop does. Submission order is still strictly
single-threaded and FIFO, which the native side depends on: lanes are chained
with `cudaStreamWaitEvent` off `context->last_lane` to keep KNN history
commits ordered, and that ordering is only correct if `submitBatch` calls
happen in submission order.
`CurrencyConversionGpuFunction` never had this bug — it waits on a lane lazily,
inline on the Flink thread, only right before reusing it, which works because
its lanes are independent (no shared history to serialize).

**The submit/complete split alone did not close the gap to RDMA.** Both paths
run the *same* CUDA kernels (`direct_imputation_jni.cu` includes
`process_function.cu`, the file the RDMA server also uses), so once dispatch
no longer stalls the GPU, a remaining gap points at the JVM/JNI side, not the
kernels. Comparing `decode()` against `RdmaPostOperator` found it doing
avoidable per-row work on the completer thread, which is now the thread that
actually gates throughput:
- `ExternalRuntimeBinaryCodec.copyBytes(ByteBuffer, ...)`, used for every
  `STRING`/`BYTES`/`DECIMAL_UNSCALED_BYTES` field (three strings plus the
  price per bid), copied one byte at a time in a Java loop instead of a bulk
  `ByteBuffer.get(byte[], off, len)`, which the JIT intrinsifies for a direct
  buffer and the loop does not. This cost scales with row count, not batch
  count, which fits `ImputationFunctionGpu`'s RDMA config using
  `rdmabatchsize=64` against the direct path's `4096` and still coming out
  ahead — a bigger batch doesn't amortize a per-row cost.
- `decode()` allocated a fresh 7-field `GenericRowData` for the wire row on
  every row, on top of the fresh result-row allocation it also needs. The
  wire row is only ever read immediately and discarded, so it's now a single
  reused scratch container instead (`wireScratch`, one per function instance,
  passed as `decodeCodec`'s `reuseRow`) — safe because `decode()` processes
  one row at a time on the one completer thread and copies every value it
  needs out of it before touching the next slot.

**Full buffer-level `pipeline.object-reuse` (matching `RdmaPostOperator`'s
`reuseRow`) was initially not enabled here, since it looked like a
correctness bug given this class's architecture, not just an optimization.**
`RdmaPostOperator` decodes and immediately collects one row at a time, so
reusing a row's backing buffers before the next decode is always safe. This
class decodes an entire batch ahead of time onto a queue that the Flink
thread drains later (`completedQueue`), so a decoded row must stay
independently valid — with its own backing bytes — until it is actually
collected, which can be several batches later. `StringData.fromBytes` wraps
its byte array rather than copying it, so naively sharing one scratch array
across rows the way `ExternalRuntimeBinaryCodec.reuseBytes` does for
`RdmaPostOperator` would let a later row's decode silently overwrite an
earlier row's still-queued string field.

**`perfcsv` profiling confirmed `decode()` as the single largest stage in
the whole pipeline** (bigger than GPU `wait` time, bigger than `frame`,
bigger than `collect`) — the per-row allocation this section already
suspected, now measured rather than guessed. That justified doing the
"real change" this section originally deferred: a **bounded pool of
`GenericRowData` result rows**, sized `(pipelinedepth + 2) * batchsize`
(the `+2` is headroom above the provable minimum — see below — not a
required margin).

The safety argument: `completedQueue` holds at most `pipelinedepth` batches
before the completer thread blocks offering another, so at most
`pipelinedepth * batchsize` rows can ever be "decoded but not yet
collected" *once already queued* — plus up to one more batch's worth that
finished decoding but hasn't been offered yet, if the queue happened to be
momentarily full at that instant. A pool slot is assigned from a monotonic
per-row counter, wrapped modulo the pool size, so slot `i % poolSize` is
never reassigned to a new row until `poolSize` further rows have been
decoded since — which, given that bound, always means every row that
previously used that slot has already been collected. This is enforced by
construction (the arithmetic bound above), not by a runtime check, so
getting `poolSize` wrong would be a silent correctness bug rather than a
crash — that's what the `+2` headroom and the field comments in
`ImputationGpuFunction` (`resultPool`/`poolSize`) are for: cheap insurance,
since a few thousand extra pre-allocated rows costs nothing next to the
millions of row-allocations this exists to avoid.

**Pooling the variable-length field `byte[]`s the same way, with an *exact*
length match, was tried first and reverted after measuring it with
`perfcsv` — it made `decode()` slower, not faster** (617 → 937 micros/batch
on the same workload, total wall time slightly *up*). The reuse check only
fired on an exact length match (`existing.length != len` → allocate fresh
anyway), and `channel`/`url`/`extra` vary in length row to row, with a pool
slot only recurring every `poolSize` (~18k) rows — unrelated to whatever
length happened to occupy it last. The result: fresh allocation happened
almost as often as with no pooling at all, plus the added cost of an extra
array indirection and, likely the bigger factor, touching a large,
rarely-revisited pool array that's cold in cache, in place of the JVM's TLAB
bump-allocator — about as fast as allocation gets, and it keeps
freshly-allocated short-lived objects cache-hot, which is exactly the case
this is (an object read once and discarded almost immediately).

**Fixed by switching STRING fields to *grow-only* pooling instead of exact-
match** — `ExternalRuntimeBinaryCodec#copyBytesGrowable` reuses a slot's
array whenever it's already `>= len` bytes, only reallocating when it's too
small. That converges to each field's max-seen length and then stops
allocating almost entirely, instead of needing an exact hit every time. This
is safe specifically because `StringData.fromBytes(bytes, 0, len)` takes an
explicit length and never reads past it — an oversized backing array is
harmless. It is **not** safe for `BYTES` or `DECIMAL_UNSCALED_BYTES`
(`price`): those hand the array off as the field value itself, with no
length carried alongside it, so an oversized array wouldn't just waste
space — for `price` specifically it's fed straight into `BigInteger(byte[])`,
which reads every byte as part of the encoded value, so a stale oversized
array would silently produce the *wrong number*. Both wire types keep using
`copyBytes`'s exact-match-or-fresh behavior via a new `copyBytesGrowable`
sibling; `price`'s call site passes `null` unconditionally, ignoring
whatever `fieldScratch` the caller passed for the row as a whole, so its
behavior can't drift from the known-safe baseline no matter what pooling
is enabled for the row's other fields.

Lesson for whoever reaches for this pattern next: a reuse pool only pays
for itself when the reuse check actually *hits* most of the time — an
exact-match check on a variable-length payload is a near-guaranteed miss;
a grow-only check is a near-guaranteed hit once the pool has warmed up,
but is only sound when the consumer of the returned array always carries
an explicit length alongside it rather than trusting `array.length`.

`ExternalRuntimeBinaryCodec`'s `readFramedRow(ByteBuffer, int, RowKind,
GenericRowData, byte[][] fieldScratch)` overload (and `copyBytesGrowable`)
are purely additive — the existing 4-argument overload still delegates to
the codec's own internal `reuseObjects`/`reuseBytes` mechanism unchanged,
and `RdmaPostOperator`'s byte[]-based decode path (`decodeFrame(byte[],
...)`, `readValueIntoRow(byte[], ...)`, `copyBytes(int, byte[], ...)`)
isn't touched at all.

If the bottleneck moves after this, profile the completer thread again with
`perfcsv` before guessing further — `frame` (submitter thread) does
comparable per-row work encoding the wire format and hasn't had any pooling
treatment, so it's a plausible next target if `decode()` stops dominating.

**On rewriting this as GPU-side ("zero-serde") serialization instead:**
considered and deliberately not attempted. The strongest version of that
idea — the GPU writing bytes Flink's row type can consume with no parsing
at all — means replicating `BinaryRowData`'s exact internal byte layout
(null-bitmap width, fixed-region alignment, variable-length offset
encoding) in CUDA code, without verified access to that layout for this
Flink version. Getting it subtly wrong wouldn't fail loudly; it would
silently hand downstream operators corrupted rows, which is a worse failure
mode than a measured performance regression and not one to risk on a guess.
The GPU kernels already do the actual interpretation work (string hashing,
decimal parsing, the KNN search) — what's left on the JVM side is memory-
layout packing/unpacking into Flink's row abstraction, which some row
representation still has to do somewhere for `emitter.collect()` to have
anything to hand downstream.

**What was done instead: fixing the one field whose *format*, not its
allocation, was the actual cost — `price`'s wire type.** Every field except
`price` decodes as a fixed-width read plus a cheap cast (`readLongBE` +
box, or a zero-parse `StringData.fromBytes` wrap). `price` alone went
through `new BigInteger(bytes) → new BigDecimal(BigInteger, scale) →
DecimalData.fromBigDecimal(...)` on every row, both directions — required
only because `DECIMAL(23,3)` exceeds Flink's 18-digit compact-decimal
threshold, forcing the full `BigDecimal`-backed `DecimalData`
representation instead of the cheap `DecimalData.fromUnscaledLong(long,
precision, scale)` path every other numeric field already gets. 18 digits
before the decimal point is far beyond any realistic bid price, so the wire
representation for `price` is now `DECIMAL(18,3)` /
`DECIMAL_UNSCALED_I64` (a fixed 8-byte scale-3 unscaled long, no length
prefix) instead of `DECIMAL(23,3)` / `DECIMAL_UNSCALED_BYTES` (a
length-prefixed variable-length two's-complement byte array). This is
**not** the same category of change as the pooling attempts — it removes
work (two fewer object constructions and a digit-counting precision check
per row) rather than trying to avoid paying for the same work through
reuse, which is exactly why the pooling attempts couldn't have fixed this
regardless of strategy.

This is the one lever so far that isn't JVM-only: the wire *bytes* for
`price` are produced by the GPU kernel, and that kernel
(`process_function.cu`) is shared verbatim between the RDMA and
direct-CUDA paths, so the change touches both.

- **`process_function.cu`**: `BidView::price_unscaled` (an `int64_t`)
  replaces the old `price`/`price_length` byte-pointer pair. `parse_bid`
  reads a fixed 8 bytes (`be64`) instead of `read_variable`. A new
  `decimal_to_double_i64`/`encode_decimal_i64` pair (fixed-width
  counterparts to the existing `decimal_to_double`/`encode_decimal`, which
  stay exactly as they were) handles the `int64_t ↔ double` conversion for
  the KNN math and the imputed-value re-encode. `write_imputed_bid` writes
  price as a plain `put_be64` — no length prefix — for both the
  pass-through (observed) and imputed cases; the output-length formula and
  field offsets were updated to match the now-fixed width. None of this
  touches `ImputationObservation` (still plain `double`) or the actual KNN
  logic — only the wire parsing/writing boundary.
- **RDMA protocol** (`control_protocol.rs`): new `WireFieldType::
  DecimalUnscaledI64` variant, distinct from the existing `DecimalBytes`
  (which currency conversion and Black-Scholes keep using unchanged).
  `gpu_runtime/cuda.rs`'s `IMPUTATION_SCHEMA` and its validation error
  message now expect `DECIMAL_UNSCALED_I64` at field 0; an
  `rdmaProcessingSpec` for IMPUTE must be updated to match (`"fields":
  ["DECIMAL_UNSCALED_I64","INT64","INT64","BYTES","BYTES",
  "TIMESTAMP_MILLIS","BYTES"]` instead of the old `"DECIMAL_BYTES"` at
  index 0) — an **old RDMA client config using the previous spec will now
  be rejected at bootstrap**, not silently misinterpreted, since
  `from_protocol` validates the whole fields array against the exact
  expected schema. The new variant's numeric `field_types` tag (6) isn't
  actually read by `parse_bid`/`write_imputed_bid` (IMPUTE's field walk is
  hardcoded, not driven by that array — only `process_currency_conversion`/
  `process_increment` consult it generically), but is assigned and kept
  accurate anyway for anyone reading the schema. `direct_imputation_jni.cu`'s
  own hardcoded `imputation_spec()` (the local JNI path's equivalent, unrelated
  to the Rust protocol) was updated to the same tag for the same reason.
- **`ImputationGpuFunction.java`**: `WIRES[0]` changed from
  `DECIMAL_UNSCALED_BYTES` to `DECIMAL_UNSCALED_I64`; `writeTargets[0]`/
  `readSources[0]` (the wire's own declared type, now factored into a
  `WIRE_PRICE_TYPE` constant) changed from `DecimalType(23, 3)` to
  `DecimalType(18, 3)`. Deliberately **not** changed: `readTargets[0]`,
  still derived from `resultType` — i.e., whatever `ImputationFunction`'s
  actual declared SQL result type is (`DECIMAL(23,3)`, per
  `ImputationFunction.ImputedBid`'s `@DataTypeHint`) stays exactly as
  declared; nothing about the SQL-visible result type or the source
  `bids.price` column's type changes. This works because
  `ExternalRuntimeBinaryCodec#castIfNeeded`'s `DECIMAL` branch materializes
  using the *source* type first (cheap, since the wire source is now
  compact) and only then widens via `DecimalDataUtils.castFrom` to the
  target's actual precision/scale — narrow-compact-in,
  wide-if-declared-out, with the expensive representation only where the
  SQL contract actually requires it.

**Measured: the two bulk-copy fixes above did not close the gap.** That rules
out the copy loops as the dominant cost and points at something structural
instead: RDMA gets PRE (encode) and POST (decode+collect) on two genuinely
separate Flink operators/threads (possibly two different TaskManagers - see
"Data path" above), while `ImputationGpuFunction` is deliberately one
operator (see its class Javadoc, and `GpuRuntimeOperator`'s), so its
Flink-thread-side work (`processElement`, plus the trivial `collect()` loop
in `poll()`) and its completer-thread-side work (`waitBatch` + decode) are
the only two places CPU cost can land - there's no way to add a third
Flink-visible thread for output the way RDMA has one, because Flink requires
`output.collect()` to run only on the operator's own thread
(`GpuRuntimeOperator`'s comment: "all output collection remains on this
Flink operator thread").

**A true PRE/POST operator pair for this path was already tried and
abandoned.** `git log` shows `CudaCurrencyConversionOperator` +
`java/flink/DirectCudaCurrencyNative.java` existed on this branch: a single
operator extending Flink's `ExternalRuntimeOperator` directly, same as
`RdmaPreOperator`/`RdmaPostOperator` do. Commit `a8f5ac8` ("fixed direct
path") deleted both and introduced `GpuRuntimeFunction`/`GpuRuntimeOperator`
instead, because the `table.exec.external-runtime.*` dispatch to a
hand-written operator class never actually wired into the planner - there
was no rule picking it up the way `RdmaPreOperator`/`RdmaPostOperator` are
picked up for `type=rdma`. `GpuRuntimeOperator`'s `impl=` reflection loader
is the thing that's actually confirmed to reach the planner, and a new
PRE/POST pair for the direct-CUDA path would need exactly the same
`ExternalRuntimeOperator` route that was already found to be a dead end. Do
not repeat that route without first confirming (with real Flink planner
source, not this repository) that the dispatch rule now exists.

**What was done instead, staying entirely inside `GpuRuntimeFunction`:** the
one Flink-thread-mandatory step - reading the live `RowData`, which
`pipeline.object-reuse` means cannot be deferred past `processElement`
returning - is now split from the wire-format framing step, which has no
such constraint once values are out as independent Java objects.
`ExternalRuntimeBinaryCodec.extractRowValues`/`writeExtractedRow` are the two
halves; `processElement` now only calls the former (a handful of typed
getters plus a list append), and the submitter thread calls the latter
(null-bitmap layout, wire-format packing, the buffer write) right before
`submitBatch`, in parallel with the Flink thread already filling the next
batch. This moves real, previously Flink-thread-only CPU work onto a second
thread without needing a second Flink operator at all - the closest
approximation of RDMA's PRE/POST core split available within one operator.
Traded off: each row's `Pending` now carries a small `Object[]` of extracted
values (boxing the `Long`s that used to go straight into `outBuf` as
primitives), a modest allocation cost this doesn't try to avoid.

**Per-stage timing CSV.** Set `perfcsv=/path/to/file.csv` in the conf string
to stop guessing where time goes and measure it directly:

```sql
SET 'table.exec.gpu-runtime.conf.org.example.flinke2c.ImputationFunction' =
  'impl=org.example.flinke2c.ImputationGpuFunction;batchsize=1024;pipelinedepth=16;threadsperblock=128;device=0;perfcsv=/tmp/imputation-perf.csv';
```

`PerfStats` (`flinke2c/.../PerfStats.java`) accumulates wall-clock time and a
call count per named stage with `LongAdder`s, so recording from the Flink
thread, the submitter thread, and the completer thread never contends on a
lock. `ImputationGpuFunction` records eleven stages, matching the pipeline
above: `extract` (Flink thread, per row - reading the live `RowData`),
`frame` (submitter thread, per batch - wire-format packing),
`submit` (submitter thread, per batch - the async JNI launch call itself;
should stay near zero, since it only enqueues GPU work),
`wait` (completer thread, per batch - `cudaEventSynchronize`, the CPU wall
time blocked waiting on the GPU), five `gpu*` stages breaking that GPU time
down further (see below), `decode` (completer thread, per batch - parsing
the output slots back into `GenericRowData`), and `collect` (Flink thread,
per completed batch - the `emitter.collect()` loop). One row is appended to
the CSV per `open()`..`close()` lifecycle (the local-GPU equivalent of "a new
connection"), with `timestamp,host,label,wallMillis`, then `rows`, `batches`,
`batchSize`, `pipelineDepth`, `threadsPerBlock`, `device`, then
`<stage>TotalMillis,<stage>AvgMicros,<stage>Count` for each stage - total
tells you where the time actually went, average tells you the per-call cost
once you already know which stage dominates. A header line is written only
if the file doesn't exist yet, so the same path can be reused across runs to
build up a comparison table. Disabled (zero overhead beyond a null check per
call site) unless `perfcsv` is set. Unset by default; nothing changes for
existing jobs that don't pass it.

**Rows are written periodically, not only at `close()`.** A streaming query
over a source that's never stopped (still running, no cancel, no bounded
end reached yet) never calls `close()` at all, by definition - there's
nothing to catch there short of stopping the job. For that case: a separate
daemon thread (`gpu-imputation-perf-flush`) writes a growing cumulative
snapshot every `perfflushseconds` (default 30; the label column reads
`ImputationGpuFunction-periodic` for these and `ImputationGpuFunction-final`
for the one `close()` always writes last, once the pipeline has fully
drained). Set `perfflushseconds=0` to disable periodic snapshots and only
ever write the final row.

`close()` itself was also missing a robustness fix: it calls
`Thread.join(5000L)` on the flusher/submitter/completer threads while
tearing down, and an uncaught `InterruptedException` there (if the closing
thread itself gets interrupted mid-wait) would abort the rest of the
`finally` block, skipping both the final `perfcsv` row and the native
`DirectCudaImputationNative.destroy(handle)` call. Each join now goes
through `interruptAndJoinQuietly`, which restores the closing thread's
interrupted status for its own caller to see but always lets the rest of
cleanup - the destroy call and the final CSV row - run regardless.

Two other things worth checking if a row still doesn't show up where
expected. First, on a cluster the file is written on whichever TaskManager
the operator instance actually runs on - the CUDA device it drives is local
to that machine - which is not necessarily the machine running the SQL
client; both the `host` column and the log line below name that host
directly so this doesn't have to be worked out from the Flink UI's task
list. Second, `PerfStats` logs at WARN on that TaskManager if a write fails
(permissions, a bad path, ...) instead of swallowing it silently, and at
INFO with the host and the resolved absolute path the first time a write
succeeds - check there if a file that should exist doesn't.

(An earlier version of this note claimed `GpuRuntimeOperator` should also
hook `StreamOperator.dispose()`, reasoning that `close()` is only guaranteed
on a graceful finish and `dispose()` is Flink's hook for every termination
path including a plain cancel. That doesn't hold for the Flink version this
targets - there is no `dispose()` to override - so that change was reverted.
If cancelling a job (rather than letting it finish or stopping it with a
savepoint) still turns out to skip `close()` here, that's a real gap worth
revisiting against the actual `StreamOperator` lifecycle for this Flink
version specifically, not by assuming an API shape from a different one.)

**GPU-side kernel timing, combined into the same row.** `wait` alone only
says how long the CPU blocked; it doesn't say whether that time went into
the H2D copy, one of the three kernels, or the D2H copy. `perfcsv` also
switches on GPU-side timing (`direct_imputation_jni.cu`'s `Context::profiling`,
set from `DirectCudaImputationNative.create`'s new `profiling` argument -
`perfCsvPath != null`, the same flag that creates the Java-side `PerfStats`),
which adds five extra timed `cudaEvent_t`s per lane (the lane-reuse/history-
ordering event stays `cudaEventDisableTiming`, since that one is recorded on
every `submitBatch` call regardless of profiling and timed events are
marginally heavier). `submitBatch` records one around each stage; a new
native call, `waitBatchTimed`, reads them back with `cudaEventElapsedTime`
once `waitBatch`'s own `cudaEventSynchronize` confirms the batch is done
(safe to do unsynchronized at that point - events recorded earlier on the
same stream are already known complete, since a CUDA stream executes and
records events in issue order) and returns `[h2dMs, prepareMs, processMs,
commitMs, d2hMs]`. The completer thread feeds these straight into the same
`PerfStats` instance as `gpuH2D`/`gpuPrepare`/`gpuProcess`/`gpuCommit`/
`gpuD2H`, so they land as columns in the exact same CSV row as the
CPU-side stages - "combine" here means literally the same row, not a
separate file to cross-reference. Comparing `wait`'s total against the sum
of the five `gpu*` totals separates two different problems that look
identical from tps alone: if they're close, the GPU kernels themselves are
the ceiling (matches expectations from `commit_imputation_history_state`
being single-threaded and unable to pipeline across batches - see the
batch-size/pipeline-depth sweep guidance above); if `wait` is
meaningfully larger, the gap is host-side launch/scheduling overhead the
GPU-side numbers can't see. Both `waitBatch` (untimed) and `waitBatchTimed`
still exist as separate native calls - profiling off costs nothing beyond
the five null pointer checks in `submitBatch`, since `waitBatch` is used
unchanged in that case.

## GPU Black-Scholes

`org.example.flinke2c.BlackScholesFunction` prices a European call option via
Black-Scholes, treating the bid price as spot (30-day maturity, strike 10%
out of the money, 3% risk-free rate, 25% volatility). Unlike currency
conversion's single multiply, this is compute-bound — a `log`, two `exp`, a
`sqrt`, and an `erf` approximation per row — making it a heavier stand-in map
operator for CPU/GPU comparison. It's a single-DECIMAL-field transform, so
both GPU paths reuse currency conversion's exact plumbing (lane design, wire
ABI, field-walk loop) with only the kernel body swapped.

**RDMA.** Select it in both RDMA Flink operators the same way as currency
conversion:

```text
rdmaProcessingSpec={"function":"BLACK_SCHOLES","field_index":2,"fields":["INT64","INT64","DECIMAL_BYTES","TIMESTAMP_MILLIS","BYTES","INT64"]}
```

`field_index` may select any `DECIMAL_BYTES` field in the declared wire
schema. The kernel (`process_black_scholes`/`convert_black_scholes_decimal_field`
in `cuda/process_function.cu`) walks the framed row exactly like
`process_currency_conversion` and replaces the target field in place, growing
or shrinking the slot the same way. Unlike currency conversion's exact
integer arithmetic, this necessarily goes through `decimal_to_double`/
`encode_decimal` — `log`/`exp`/`erf` have no exact fixed-point form — so a
result exactly on a rounding boundary can differ from the CPU path by one
unit in the last (`0.001`) place, the same caveat already documented for
imputation. A null price stays null; the field-walk loop returns without
touching it, matching `process_currency_conversion`.

**Direct CUDA (packed, dynamically loaded).** `BlackScholesGpuFunction` is
`CurrencyConversionGpuFunction` with the kernel swapped — same lane/batching
design, same `Input`/`Output` ABI (`direct_black_scholes_jni.cu`), same
`GpuRuntimeFunction` packed-operator model:

```sql
SET 'table.exec.gpu-runtime.function-class' = 'org.example.flinke2c.BlackScholesFunction';
SET 'table.exec.gpu-runtime.conf.org.example.flinke2c.BlackScholesFunction' =
  'impl=org.example.flinke2c.BlackScholesGpuFunction;batchsize=1024;pipelinedepth=8;threadsperblock=256;device=0;fieldindex=0';
```

**Conf keys** (same style as `CurrencyConversionGpuFunction`): `batchsize`
(default is the operator's configured batch size), `pipelinedepth` (default
4, max 64), `threadsperblock` (default 256, max 1024), `device` (default 0),
`fieldindex`/`pricefield` (default 2 — the row position of the `DECIMAL`
column to price; must name a `DECIMAL` or `BIGINT` column).

Build both new artifacts the same way as the other direct-CUDA libraries —
`cuda/CMakeLists.txt` now has a `flinke2c_black_scholes_gpu` target alongside
the currency/imputation ones, producing
`cuda/build/libflinke2c_black_scholes_gpu.so`. Make it visible to every
TaskManager the same way as `libflinke2c_currency_conversion_gpu.so`.

**Not ported:** an `AsyncScalarFunction` version
(`BlackScholesFunctionGpu`, analogous to `CurrencyConversionFunctionGpu`) —
only the CPU `ScalarFunction` and the two GPU paths above exist for this
function. Add one by close analogy if a three-way comparison against the
async-UDF path specifically is needed; nothing about the kernel or wire
format below would change.
