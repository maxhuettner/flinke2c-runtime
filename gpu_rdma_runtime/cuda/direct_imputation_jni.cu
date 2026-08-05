#include <cuda_runtime.h>
#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

namespace {

constexpr uint32_t HISTORY_SIZE = 5000;
constexpr uint32_t SEARCH_LIMIT = 512;
constexpr uint32_t K = 10;
constexpr double EPS = 1e-6;
constexpr double W_BIDDER = 0.25;
constexpr double W_TIME = 1.0;
constexpr double W_STRING = 0.25;
// Imputation uses substantially more registers per tuple than currency
// conversion, so use a smaller block while still filling every warp.
constexpr uint32_t THREADS_PER_BLOCK = 128;

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

struct Context {
    int device = 0;
    uint32_t capacity = 0;
    cudaStream_t stream = nullptr;
    DeviceState* state = nullptr;
    Input* device_input = nullptr;
    double* device_output = nullptr;
    Input* host_input = nullptr;
    double* host_output = nullptr;
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

void release_context(Context* context) {
    if (context == nullptr) {
        return;
    }
    cudaSetDevice(context->device);
    if (context->stream != nullptr) {
        cudaStreamSynchronize(context->stream);
    }
    if (context->device_output != nullptr) {
        cudaFree(context->device_output);
    }
    if (context->device_input != nullptr) {
        cudaFree(context->device_input);
    }
    if (context->state != nullptr) {
        cudaFree(context->state);
    }
    if (context->host_output != nullptr) {
        cudaFreeHost(context->host_output);
    }
    if (context->host_input != nullptr) {
        cudaFreeHost(context->host_input);
    }
    if (context->stream != nullptr) {
        cudaStreamDestroy(context->stream);
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
// Source-order commits form the state boundary used by the RDMA path.
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

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_ImputationGpuNative_create(
    JNIEnv* env, jclass, jint cuda_device, jint batch_capacity) {
    if (cuda_device < 0 || batch_capacity <= 0) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "CUDA device must be non-negative and batch capacity must be positive");
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
    const size_t input_bytes = sizeof(Input) * context->capacity;
    const size_t output_bytes = sizeof(double) * context->capacity;

    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice") ||
        !cuda_ok(
            env,
            cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags") ||
        !cuda_ok(
            env,
            cudaMalloc(
                reinterpret_cast<void**>(&context->state),
                sizeof(DeviceState)),
            "cudaMalloc(DeviceState)") ||
        !cuda_ok(
            env,
            cudaMalloc(
                reinterpret_cast<void**>(&context->device_input),
                input_bytes),
            "cudaMalloc(batch input)") ||
        !cuda_ok(
            env,
            cudaMalloc(
                reinterpret_cast<void**>(&context->device_output),
                output_bytes),
            "cudaMalloc(batch output)") ||
        !cuda_ok(
            env,
            cudaHostAlloc(
                reinterpret_cast<void**>(&context->host_input),
                input_bytes, cudaHostAllocPortable),
            "cudaHostAlloc(batch input)") ||
        !cuda_ok(
            env,
            cudaHostAlloc(
                reinterpret_cast<void**>(&context->host_output),
                output_bytes, cudaHostAllocPortable),
            "cudaHostAlloc(batch output)") ||
        !cuda_ok(
            env,
            cudaMemsetAsync(
                context->state, 0, sizeof(DeviceState), context->stream),
            "cudaMemsetAsync(DeviceState)") ||
        !cuda_ok(
            env, cudaStreamSynchronize(context->stream),
            "cudaStreamSynchronize")) {
        release_context(context);
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_ImputationGpuNative_inputBuffer(
    JNIEnv* env, jclass, jlong handle) {
    Context* context = require_context(env, handle);
    if (context == nullptr) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
        context->host_input,
        static_cast<jlong>(sizeof(Input) * context->capacity));
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_ImputationGpuNative_outputBuffer(
    JNIEnv* env, jclass, jlong handle) {
    Context* context = require_context(env, handle);
    if (context == nullptr) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
        context->host_output,
        static_cast<jlong>(sizeof(double) * context->capacity));
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_processBatch(
    JNIEnv* env, jclass, jlong handle, jint count) {
    Context* context = require_context(env, handle);
    if (context == nullptr) {
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

    const uint32_t batch_count = static_cast<uint32_t>(count);
    const size_t input_bytes = sizeof(Input) * batch_count;
    const size_t output_bytes = sizeof(double) * batch_count;
    if (!cuda_ok(
            env,
            cudaMemcpyAsync(
                context->device_input, context->host_input, input_bytes,
                cudaMemcpyHostToDevice, context->stream),
            "cudaMemcpyAsync(batch input)")) {
        return;
    }

    const uint32_t block_count =
        (batch_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    direct_imputation_batch_kernel
        <<<block_count, THREADS_PER_BLOCK, 0, context->stream>>>(
        context->state, context->device_input,
        context->device_output, batch_count);
    if (!cuda_ok(
            env, cudaGetLastError(),
            "direct_imputation_batch_kernel launch")) {
        return;
    }
    commit_batch_history<<<1, 1, 0, context->stream>>>(
        context->state, context->device_input, batch_count);
    if (!cuda_ok(env, cudaGetLastError(), "commit_batch_history launch") ||
        !cuda_ok(
            env,
            cudaMemcpyAsync(
                context->host_output, context->device_output, output_bytes,
                cudaMemcpyDeviceToHost, context->stream),
            "cudaMemcpyAsync(batch output)") ||
        !cuda_ok(
            env, cudaStreamSynchronize(context->stream),
            "cudaStreamSynchronize")) {
        return;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_destroy(
    JNIEnv*, jclass, jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}
