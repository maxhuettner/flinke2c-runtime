#include <cuda_runtime.h>
#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr uint32_t HISTORY_SIZE = 5000;
constexpr uint32_t SEARCH_LIMIT = 512;
constexpr uint32_t K = 10;
constexpr double EPS = 1e-6;
constexpr double W_BIDDER = 0.25;
constexpr double W_TIME = 1.0;
constexpr double W_STRING = 0.25;

struct Observation {
    double price;
    int64_t bidder;
    double timestamp_seconds;
    uint32_t channel_hash;
    uint32_t url_hash;
    uint32_t extra_hash;
};

struct Input {
    Observation observation;
    uint32_t has_price;
};

static_assert(sizeof(Observation) == 40, "Java/native Observation layout changed");
static_assert(sizeof(Input) == 48, "Java/native Input layout changed");
static_assert(
    offsetof(Input, observation) + offsetof(Observation, price) == 0,
    "price offset changed");
static_assert(
    offsetof(Input, observation) + offsetof(Observation, bidder) == 8,
    "bidder offset changed");
static_assert(
    offsetof(Input, observation) +
        offsetof(Observation, timestamp_seconds) == 16,
    "timestamp offset changed");
static_assert(
    offsetof(Input, observation) + offsetof(Observation, channel_hash) == 24,
    "channel hash offset changed");
static_assert(
    offsetof(Input, observation) + offsetof(Observation, url_hash) == 28,
    "URL hash offset changed");
static_assert(
    offsetof(Input, observation) + offsetof(Observation, extra_hash) == 32,
    "extra hash offset changed");
static_assert(offsetof(Input, has_price) == 40, "has-price offset changed");

struct DeviceState {
    Observation history[HISTORY_SIZE];
    uint32_t start;
    uint32_t size;
};

// Each lane owns an independent stream, event, and pinned/device input+output
// buffer pair, exactly like the currency conversion lanes. Unlike currency
// conversion, imputation reads and mutates one DeviceState shared by every
// lane (the KNN history), so lanes cannot simply run free: a later batch's
// impute kernel must observe an earlier batch's committed history, and two
// commits must not interleave. Ordering that on the CPU (block until the
// previous batch fully finishes before submitting the next) would collapse
// back to the fully-serial design this replaces. Instead, ordering is
// enforced on the GPU: submitBatch makes the lane's own stream wait on the
// previous imputation lane's completion event before launching its kernels,
// via cudaStreamWaitEvent. That wait is asynchronous from the CPU's
// perspective (it just enqueues a dependency), so the CPU can still fill and
// submit the next lane without blocking, while the GPU itself serializes only
// the state-touching kernels. Independent lanes' H2D/D2H copies are
// unaffected and can overlap freely. This mirrors the cross-stream
// last_imputation_lane dependency gpu_runtime/cuda.rs uses for the RDMA path.
struct Lane {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    bool event_pending = false;
    Input* device_input = nullptr;
    double* device_output = nullptr;
    Input* host_input = nullptr;
    double* host_output = nullptr;
};

struct Context {
    int device = 0;
    uint32_t capacity = 0;
    // CUDA block size for the batch kernel. A compile-time constant would
    // require rebuilding the PTX/shared library to sweep; this is instead
    // set once at create() time from Java, alongside batch capacity and
    // pipeline depth. Imputation uses substantially more registers per tuple
    // than currency conversion, so its Java-side default is smaller (128
    // rather than 256) while still filling every warp.
    uint32_t threads_per_block = 0;
    DeviceState* state = nullptr;
    std::vector<Lane> lanes;
    // Index into lanes of the most recently submitted imputation batch, or
    // -1 if none has been submitted yet on this context.
    int last_lane = -1;
};

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass error_class = env->FindClass(class_name);
    if (error_class != nullptr) {
        env->ThrowNew(error_class, message.c_str());
    }
}

bool cuda_ok(
    JNIEnv* env, cudaError_t status, const char* operation,
    const char* exception_class = "java/lang/IllegalStateException") {
    if (status == cudaSuccess) {
        return true;
    }
    std::string message(operation);
    message.append(": ");
    message.append(cudaGetErrorString(status));
    throw_java(env, exception_class, message);
    return false;
}

void release_lane(Context* context, Lane& lane) {
    cudaSetDevice(context->device);
    if (lane.stream != nullptr) {
        cudaStreamSynchronize(lane.stream);
    }
    if (lane.device_output != nullptr) {
        cudaFree(lane.device_output);
    }
    if (lane.device_input != nullptr) {
        cudaFree(lane.device_input);
    }
    if (lane.host_output != nullptr) {
        cudaFreeHost(lane.host_output);
    }
    if (lane.host_input != nullptr) {
        cudaFreeHost(lane.host_input);
    }
    if (lane.event != nullptr) {
        cudaEventDestroy(lane.event);
    }
    if (lane.stream != nullptr) {
        cudaStreamDestroy(lane.stream);
    }
}

void release_context(Context* context) {
    if (context == nullptr) {
        return;
    }
    for (Lane& lane : context->lanes) {
        release_lane(context, lane);
    }
    cudaSetDevice(context->device);
    if (context->state != nullptr) {
        cudaFree(context->state);
    }
    delete context;
}

__device__ double distance(
    const Observation& target, const Observation& candidate) {
    double result =
        W_BIDDER * (target.bidder == candidate.bidder ? 0.0 : 1.0);
    const double delta =
        target.timestamp_seconds - candidate.timestamp_seconds;
    result += W_TIME * delta * delta * 1e-8;
    result += W_STRING *
        (target.channel_hash == candidate.channel_hash ? 0.0 : 1.0);
    result += W_STRING *
        (target.url_hash == candidate.url_hash ? 0.0 : 1.0);
    result += W_STRING *
        (target.extra_hash == candidate.extra_hash ? 0.0 : 1.0);
    return result;
}

__device__ void consider_neighbor(
    const Observation& target, const Observation& candidate,
    double* best_distance, double* best_price, uint32_t& found) {
    const double candidate_distance = distance(target, candidate);
    if (found < K) {
        best_distance[found] = candidate_distance;
        best_price[found] = candidate.price;
        ++found;
        return;
    }

    uint32_t worst_index = 0;
    double worst = best_distance[0];
    for (uint32_t j = 1; j < K; ++j) {
        if (best_distance[j] > worst) {
            worst = best_distance[j];
            worst_index = j;
        }
    }
    if (candidate_distance < worst) {
        best_distance[worst_index] = candidate_distance;
        best_price[worst_index] = candidate.price;
    }
}

__device__ double impute(
    const DeviceState& state, const Input* batch, uint32_t item) {
    const Observation& target = batch[item].observation;
    double best_distance[K];
    double best_price[K];
    uint32_t found = 0;
    uint32_t remaining = SEARCH_LIMIT;

    // Earlier observed rows from this batch are logically newer than the
    // committed history and must be considered newest-first.
    for (uint32_t prior = item; prior > 0 && remaining > 0; --prior) {
        const Input& candidate = batch[prior - 1];
        if (candidate.has_price != 0) {
            consider_neighbor(
                target, candidate.observation,
                best_distance, best_price, found);
            --remaining;
        }
    }

    const uint32_t history_count =
        state.size < remaining ? state.size : remaining;
    for (uint32_t i = 0; i < history_count; ++i) {
        const uint32_t index =
            (state.start + state.size - 1 - i) % HISTORY_SIZE;
        consider_neighbor(
            target, state.history[index],
            best_distance, best_price, found);
    }

    if (found == 0) {
        return nan("");
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t i = 0; i < found; ++i) {
        const double weight = 1.0 / (best_distance[i] + EPS);
        numerator += best_price[i] * weight;
        denominator += weight;
    }
    return denominator == 0.0 ? nan("") : numerator / denominator;
}

__device__ void add_observation(
    DeviceState& state, const Observation& observation) {
    uint32_t index;
    if (state.size < HISTORY_SIZE) {
        index = (state.start + state.size) % HISTORY_SIZE;
        ++state.size;
    } else {
        index = state.start;
        state.start = (state.start + 1) % HISTORY_SIZE;
    }
    state.history[index] = observation;
}

// One thread processes one tuple. Each thread retains the deterministic
// newest-first neighbor scan and floating-point accumulation order.
__global__ void direct_imputation_batch_kernel(
    const DeviceState* state, const Input* input, double* output,
    uint32_t count) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }
    output[item] = input[item].has_price != 0
        ? input[item].observation.price
        : impute(*state, input, item);
}

// Commit only after every output in the batch has read the old history.
// Source-order commits form the state boundary used by the RDMA path. The
// caller (submitBatch) is responsible for ordering this against other lanes'
// commits via cudaStreamWaitEvent; this kernel itself assumes it is the only
// one touching *state at a time.
__global__ void commit_batch_history(
    DeviceState* state, const Input* input, uint32_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (uint32_t item = 0; item < count; ++item) {
        if (input[item].has_price != 0) {
            add_observation(*state, input[item].observation);
        }
    }
}

Context* require_context(JNIEnv* env, jlong handle) {
    Context* context = reinterpret_cast<Context*>(handle);
    if (context == nullptr) {
        throw_java(
            env, "java/lang/IllegalStateException",
            "Direct CUDA imputation context is closed");
    }
    return context;
}

bool require_lane(JNIEnv* env, Context* context, jint lane, uint32_t& out) {
    if (lane < 0 || static_cast<uint32_t>(lane) >= context->lanes.size()) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "Direct CUDA imputation lane index is out of range");
        return false;
    }
    out = static_cast<uint32_t>(lane);
    return true;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_ImputationGpuNative_create(
    JNIEnv* env, jclass, jint cuda_device, jint batch_capacity, jint pipeline_depth,
    jint threads_per_block) {
    if (cuda_device < 0 || batch_capacity <= 0 || pipeline_depth <= 0) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "CUDA device must be non-negative, and batch capacity and pipeline "
            "depth must be positive");
        return 0;
    }
    // 1024 is the max threads per block on every CUDA compute capability this
    // library targets (compute_80+); Java validates this too, but check here
    // as well since this is a JNI entry point Java isn't the only caller of.
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "threadsPerBlock must be in 1..1024");
        return 0;
    }

    Context* context = new (std::nothrow) Context();
    if (context == nullptr) {
        throw_java(
            env, "java/lang/OutOfMemoryError",
            "Failed to allocate the direct CUDA imputation context");
        return 0;
    }
    context->device = cuda_device;
    context->capacity = static_cast<uint32_t>(batch_capacity);
    context->threads_per_block = static_cast<uint32_t>(threads_per_block);
    context->lanes.resize(static_cast<uint32_t>(pipeline_depth));

    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice") ||
        !cuda_ok(
            env,
            cudaMalloc(reinterpret_cast<void**>(&context->state), sizeof(DeviceState)),
            "cudaMalloc(DeviceState)") ||
        !cuda_ok(
            env, cudaMemset(context->state, 0, sizeof(DeviceState)),
            "cudaMemset(DeviceState)")) {
        release_context(context);
        return 0;
    }

    const size_t input_bytes = sizeof(Input) * context->capacity;
    const size_t output_bytes = sizeof(double) * context->capacity;
    for (Lane& lane : context->lanes) {
        if (!cuda_ok(
                env,
                cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags") ||
            !cuda_ok(
                env,
                cudaEventCreateWithFlags(&lane.event, cudaEventDisableTiming),
                "cudaEventCreateWithFlags") ||
            !cuda_ok(
                env,
                cudaMalloc(reinterpret_cast<void**>(&lane.device_input), input_bytes),
                "cudaMalloc(batch input)") ||
            !cuda_ok(
                env,
                cudaMalloc(reinterpret_cast<void**>(&lane.device_output), output_bytes),
                "cudaMalloc(batch output)") ||
            !cuda_ok(
                env,
                cudaHostAlloc(
                    reinterpret_cast<void**>(&lane.host_input), input_bytes,
                    cudaHostAllocPortable),
                "cudaHostAlloc(batch input)") ||
            !cuda_ok(
                env,
                cudaHostAlloc(
                    reinterpret_cast<void**>(&lane.host_output), output_bytes,
                    cudaHostAllocPortable),
                "cudaHostAlloc(batch output)")) {
            release_context(context);
            return 0;
        }
    }

    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_ImputationGpuNative_inputBuffer(
    JNIEnv* env, jclass, jlong handle, jint lane) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
        context->lanes[lane_index].host_input,
        static_cast<jlong>(sizeof(Input) * context->capacity));
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_ImputationGpuNative_outputBuffer(
    JNIEnv* env, jclass, jlong handle, jint lane) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
        context->lanes[lane_index].host_output,
        static_cast<jlong>(sizeof(double) * context->capacity));
}

// Launches one lane's H2D copy, impute kernel, history commit, and D2H copy
// asynchronously on that lane's own stream and returns immediately; call
// waitBatch to block for the result. The impute/commit kernels wait on the
// previous imputation batch's completion event (on any lane) before running,
// so history updates apply in submission order even though lanes overlap.
extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_submitBatch(
    JNIEnv* env, jclass, jlong handle, jint lane, jint count) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return;
    }
    if (count <= 0 || static_cast<uint32_t>(count) > context->capacity) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "CUDA imputation batch count is outside the configured capacity");
        return;
    }
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) {
        return;
    }

    Lane& active = context->lanes[lane_index];
    const uint32_t batch_count = static_cast<uint32_t>(count);
    const size_t input_bytes = sizeof(Input) * batch_count;
    const size_t output_bytes = sizeof(double) * batch_count;

    // Independent of history ordering: this lane's own buffers are not
    // touched by any other lane, so the copy can start immediately.
    if (!cuda_ok(
            env,
            cudaMemcpyAsync(
                active.device_input, active.host_input, input_bytes,
                cudaMemcpyHostToDevice, active.stream),
            "cudaMemcpyAsync(batch input)")) {
        return;
    }

    if (context->last_lane >= 0) {
        // Delay only the state-touching kernels below until the previous
        // imputation batch (possibly on a different lane) has committed.
        if (!cuda_ok(
                env,
                cudaStreamWaitEvent(
                    active.stream, context->lanes[context->last_lane].event, 0),
                "cudaStreamWaitEvent")) {
            return;
        }
    }
    context->last_lane = static_cast<int>(lane_index);

    const uint32_t threads_per_block = context->threads_per_block;
    const uint32_t block_count =
        (batch_count + threads_per_block - 1) / threads_per_block;
    direct_imputation_batch_kernel<<<block_count, threads_per_block, 0, active.stream>>>(
        context->state, active.device_input, active.device_output, batch_count);
    if (!cuda_ok(env, cudaGetLastError(), "direct_imputation_batch_kernel launch")) {
        return;
    }
    commit_batch_history<<<1, 1, 0, active.stream>>>(
        context->state, active.device_input, batch_count);
    if (!cuda_ok(env, cudaGetLastError(), "commit_batch_history launch")) {
        return;
    }
    if (!cuda_ok(
            env,
            cudaMemcpyAsync(
                active.host_output, active.device_output, output_bytes,
                cudaMemcpyDeviceToHost, active.stream),
            "cudaMemcpyAsync(batch output)")) {
        return;
    }
    if (!cuda_ok(env, cudaEventRecord(active.event, active.stream), "cudaEventRecord")) {
        return;
    }
    active.event_pending = true;
}

// Blocks until the lane's most recent submitBatch has finished; a no-op if
// nothing is outstanding on the lane.
extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_waitBatch(
    JNIEnv* env, jclass, jlong handle, jint lane) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return;
    }
    Lane& active = context->lanes[lane_index];
    if (!active.event_pending) {
        return;
    }
    if (cuda_ok(env, cudaEventSynchronize(active.event), "cudaEventSynchronize")) {
        active.event_pending = false;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_destroy(
    JNIEnv*, jclass, jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}
