#include <cuda_runtime.h>
#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

namespace {

// Same fixed contract as BlackScholesFunction.java: a 30-day, 10%-out-of-
// the-money European call, treating the bid price as spot. Keep in sync
// with that file.
constexpr double STRIKE_RATIO = 1.10;
constexpr double TIME_TO_MATURITY_YEARS = 30.0 / 365.0;
constexpr double RISK_FREE_RATE = 0.03;
constexpr double VOLATILITY = 0.25;

// Keep this layout in sync with BlackScholesGpuFunction.java and
// DirectCudaBlackScholesNative.java. Deliberately identical shape to
// direct_currency_conversion_jni.cu's Input/Output - same "one double price
// plus a validity flag" contract, just a different kernel.
struct Input {
    double price;
    uint32_t valid;
    uint32_t reserved;
};

struct Output {
    double price;
    uint32_t valid;
    uint32_t reserved;
};

static_assert(sizeof(Input) == 16, "Java/native black-scholes input layout changed");
static_assert(sizeof(Output) == 16, "Java/native black-scholes output layout changed");
static_assert(offsetof(Input, price) == 0, "black-scholes price offset changed");
static_assert(offsetof(Input, valid) == 8, "black-scholes input validity offset changed");
static_assert(offsetof(Output, price) == 0, "black-scholes output price offset changed");
static_assert(offsetof(Output, valid) == 8, "black-scholes output validity offset changed");

// Same lane/pipeline design as direct_currency_conversion_jni.cu: each lane
// owns an independent stream, event, and pinned/device buffer pair so
// consecutive batches overlap on the GPU's copy and compute engines instead
// of fully serializing behind one cudaStreamSynchronize per batch.
struct Lane {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    bool event_pending = false;
    Input* device_input = nullptr;
    Output* device_output = nullptr;
    Input* host_input = nullptr;
    Output* host_output = nullptr;
};

struct Context {
    int device = 0;
    uint32_t capacity = 0;
    uint32_t threads_per_block = 0;
    std::vector<Lane> lanes;
};

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass error_class = env->FindClass(class_name);
    if (error_class != nullptr) {
        env->ThrowNew(error_class, message.c_str());
    }
}

bool cuda_ok(
        JNIEnv* env,
        cudaError_t status,
        const char* operation,
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
    delete context;
}

// Abramowitz & Stegun approximation 7.1.26 (max absolute error 1.5e-7),
// matching BlackScholesFunction.erf exactly so the GPU and CPU paths agree
// to floating-point precision.
__device__ double erf_approx(double x) {
    const double sign = x < 0.0 ? -1.0 : 1.0;
    x = fabs(x);
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double p = 0.3275911;
    const double t = 1.0 / (1.0 + p * x);
    const double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}

__device__ double normal_cdf(double x) {
    return 0.5 * (1.0 + erf_approx(x / sqrt(2.0)));
}

__device__ double call_price(double spot) {
    const double strike = spot * STRIKE_RATIO;
    const double t = TIME_TO_MATURITY_YEARS;
    const double sqrt_t = sqrt(t);
    const double d1 =
            (log(spot / strike) + (RISK_FREE_RATE + 0.5 * VOLATILITY * VOLATILITY) * t)
            / (VOLATILITY * sqrt_t);
    const double d2 = d1 - VOLATILITY * sqrt_t;
    return spot * normal_cdf(d1) - strike * exp(-RISK_FREE_RATE * t) * normal_cdf(d2);
}

// Mirrors BlackScholesFunction.eval: null in -> null out (valid=0), a
// non-positive spot -> BigDecimal.ZERO (valid=1, price=0.0), otherwise the
// computed call price.
__global__ void direct_black_scholes_batch_kernel(
        const Input* input, Output* output, uint32_t count) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }
    if (input[item].valid == 0) {
        output[item].price = 0.0;
        output[item].valid = 0;
        return;
    }
    const double spot = input[item].price;
    output[item].price = spot <= 0.0 ? 0.0 : call_price(spot);
    output[item].valid = 1;
}

Context* require_context(JNIEnv* env, jlong handle) {
    Context* context = reinterpret_cast<Context*>(handle);
    if (context == nullptr) {
        throw_java(
                env,
                "java/lang/IllegalStateException",
                "Direct CUDA Black-Scholes context is closed");
    }
    return context;
}

bool require_lane(JNIEnv* env, Context* context, jint lane, uint32_t& out) {
    if (lane < 0 || static_cast<uint32_t>(lane) >= context->lanes.size()) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "Direct CUDA Black-Scholes lane index is out of range");
        return false;
    }
    out = static_cast<uint32_t>(lane);
    return true;
}

jlong create_impl(
        JNIEnv* env, jint cuda_device, jint batch_capacity, jint pipeline_depth,
        jint threads_per_block) {
    if (cuda_device < 0 || batch_capacity <= 0 || pipeline_depth <= 0) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "CUDA device must be non-negative, and batch capacity and pipeline "
                "depth must be positive");
        return 0;
    }
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "threadsPerBlock must be in 1..1024");
        return 0;
    }

    Context* context = new (std::nothrow) Context();
    if (context == nullptr) {
        throw_java(
                env,
                "java/lang/OutOfMemoryError",
                "Failed to allocate the direct CUDA Black-Scholes context");
        return 0;
    }
    context->device = cuda_device;
    context->capacity = static_cast<uint32_t>(batch_capacity);
    context->threads_per_block = static_cast<uint32_t>(threads_per_block);
    context->lanes.resize(static_cast<uint32_t>(pipeline_depth));

    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) {
        release_context(context);
        return 0;
    }

    const size_t input_bytes = sizeof(Input) * context->capacity;
    const size_t output_bytes = sizeof(Output) * context->capacity;
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
                    cudaMalloc(
                            reinterpret_cast<void**>(&lane.device_input), input_bytes),
                    "cudaMalloc(black-scholes input)") ||
            !cuda_ok(
                    env,
                    cudaMalloc(
                            reinterpret_cast<void**>(&lane.device_output), output_bytes),
                    "cudaMalloc(black-scholes output)") ||
            !cuda_ok(
                    env,
                    cudaHostAlloc(
                            reinterpret_cast<void**>(&lane.host_input),
                            input_bytes,
                            cudaHostAllocPortable),
                    "cudaHostAlloc(black-scholes input)") ||
            !cuda_ok(
                    env,
                    cudaHostAlloc(
                            reinterpret_cast<void**>(&lane.host_output),
                            output_bytes,
                            cudaHostAllocPortable),
                    "cudaHostAlloc(black-scholes output)")) {
            release_context(context);
            return 0;
        }
    }

    return reinterpret_cast<jlong>(context);
}

jobject input_buffer_impl(JNIEnv* env, jlong handle, jint lane) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
            context->lanes[lane_index].host_input,
            static_cast<jlong>(sizeof(Input) * context->capacity));
}

jobject output_buffer_impl(JNIEnv* env, jlong handle, jint lane) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
            context->lanes[lane_index].host_output,
            static_cast<jlong>(sizeof(Output) * context->capacity));
}

// Launches one lane's H2D copy, kernel, and D2H copy asynchronously on that
// lane's own stream and returns immediately; call waitBatch to block for the
// result. The caller must not touch this lane's input/output buffers again
// until waitBatch returns.
void submit_batch_impl(JNIEnv* env, jlong handle, jint lane, jint count) {
    Context* context = require_context(env, handle);
    uint32_t lane_index;
    if (context == nullptr || !require_lane(env, context, lane, lane_index)) {
        return;
    }
    if (count <= 0 || static_cast<uint32_t>(count) > context->capacity) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "CUDA Black-Scholes batch count is outside the configured capacity");
        return;
    }
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) {
        return;
    }

    Lane& active = context->lanes[lane_index];
    const uint32_t batch_count = static_cast<uint32_t>(count);
    const size_t input_bytes = sizeof(Input) * batch_count;
    const size_t output_bytes = sizeof(Output) * batch_count;
    if (!cuda_ok(
                env,
                cudaMemcpyAsync(
                        active.device_input,
                        active.host_input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        active.stream),
                "cudaMemcpyAsync(black-scholes input)")) {
        return;
    }

    const uint32_t threads_per_block = context->threads_per_block;
    const uint32_t block_count =
            (batch_count + threads_per_block - 1) / threads_per_block;
    direct_black_scholes_batch_kernel
            <<<block_count, threads_per_block, 0, active.stream>>>(
            active.device_input,
            active.device_output,
            batch_count);
    if (!cuda_ok(
                env,
                cudaGetLastError(),
                "direct_black_scholes_batch_kernel launch")) {
        return;
    }
    if (!cuda_ok(
                env,
                cudaMemcpyAsync(
                        active.host_output,
                        active.device_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        active.stream),
                "cudaMemcpyAsync(black-scholes output)")) {
        return;
    }
    if (!cuda_ok(env, cudaEventRecord(active.event, active.stream), "cudaEventRecord")) {
        return;
    }
    active.event_pending = true;
}

// Blocks until the lane's most recent submitBatch has finished; a no-op if
// nothing is outstanding on the lane.
void wait_batch_impl(JNIEnv* env, jlong handle, jint lane) {
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

void destroy_impl(jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}

}  // namespace

// ---------------------------------------------------------------------
// org.example.flinke2c.DirectCudaBlackScholesNative
// Used by the GpuRuntimeFunction path (BlackScholesGpuFunction).
// ---------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_create(
        JNIEnv* env, jclass, jint cuda_device, jint batch_capacity, jint pipeline_depth,
        jint threads_per_block) {
    return create_impl(env, cuda_device, batch_capacity, pipeline_depth, threads_per_block);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_inputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return input_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_outputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return output_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_submitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane, jint count) {
    submit_batch_impl(env, handle, lane, count);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_waitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    wait_batch_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaBlackScholesNative_destroy(
        JNIEnv*, jclass, jlong handle) {
    destroy_impl(handle);
}
