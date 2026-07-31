#include <cuda_runtime.h>
#include <jni.h>

#include <cmath>
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

struct DeviceState {
    Observation history[HISTORY_SIZE];
    uint32_t start;
    uint32_t size;
};

struct Context {
    int device = 0;
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

__device__ double impute(
    const DeviceState& state, const Observation& target) {
    double best_distance[K];
    double best_price[K];
    uint32_t found = 0;
    const uint32_t search_count =
        state.size < SEARCH_LIMIT ? state.size : SEARCH_LIMIT;

    // Traverse newest-first, like BoundedRing.snapshotLast in the Java UDF.
    for (uint32_t i = 0; i < search_count; ++i) {
        const uint32_t index =
            (state.start + state.size - 1 - i) % HISTORY_SIZE;
        const Observation& candidate = state.history[index];
        const double candidate_distance = distance(target, candidate);
        if (found < K) {
            best_distance[found] = candidate_distance;
            best_price[found] = candidate.price;
            ++found;
            continue;
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

__global__ void direct_imputation_kernel(
    DeviceState* state, const Input* input, double* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (input->has_price != 0) {
        add_observation(*state, input->observation);
        *output = input->observation.price;
    } else {
        *output = impute(*state, input->observation);
    }
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_ImputationGpuNative_create(
    JNIEnv* env, jclass, jint cuda_device) {
    if (cuda_device < 0) {
        throw_java(
            env, "java/lang/IllegalArgumentException",
            "CUDA device index must be non-negative");
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
                sizeof(Input)),
            "cudaMalloc(Input)") ||
        !cuda_ok(
            env,
            cudaMalloc(
                reinterpret_cast<void**>(&context->device_output),
                sizeof(double)),
            "cudaMalloc(output)") ||
        !cuda_ok(
            env,
            cudaHostAlloc(
                reinterpret_cast<void**>(&context->host_input),
                sizeof(Input), cudaHostAllocPortable),
            "cudaHostAlloc(Input)") ||
        !cuda_ok(
            env,
            cudaHostAlloc(
                reinterpret_cast<void**>(&context->host_output),
                sizeof(double), cudaHostAllocPortable),
            "cudaHostAlloc(output)") ||
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

extern "C" JNIEXPORT jdouble JNICALL
Java_org_example_flinke2c_ImputationGpuNative_process(
    JNIEnv* env, jclass, jlong handle, jboolean has_price, jdouble price,
    jlong bidder, jdouble timestamp_seconds, jint channel_hash, jint url_hash,
    jint extra_hash) {
    Context* context = reinterpret_cast<Context*>(handle);
    if (context == nullptr) {
        throw_java(
            env, "java/lang/IllegalStateException",
            "Direct CUDA imputation context is closed");
        return NAN;
    }

    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) {
        return NAN;
    }

    context->host_input->observation.price = price;
    context->host_input->observation.bidder = bidder;
    context->host_input->observation.timestamp_seconds = timestamp_seconds;
    context->host_input->observation.channel_hash =
        static_cast<uint32_t>(channel_hash);
    context->host_input->observation.url_hash =
        static_cast<uint32_t>(url_hash);
    context->host_input->observation.extra_hash =
        static_cast<uint32_t>(extra_hash);
    context->host_input->has_price = has_price == JNI_TRUE ? 1U : 0U;

    if (!cuda_ok(
            env,
            cudaMemcpyAsync(
                context->device_input, context->host_input, sizeof(Input),
                cudaMemcpyHostToDevice, context->stream),
            "cudaMemcpyAsync(input)")) {
        return NAN;
    }
    direct_imputation_kernel<<<1, 1, 0, context->stream>>>(
        context->state, context->device_input, context->device_output);
    if (!cuda_ok(env, cudaGetLastError(), "direct_imputation_kernel launch") ||
        !cuda_ok(
            env,
            cudaMemcpyAsync(
                context->host_output, context->device_output, sizeof(double),
                cudaMemcpyDeviceToHost, context->stream),
            "cudaMemcpyAsync(output)") ||
        !cuda_ok(
            env, cudaStreamSynchronize(context->stream),
            "cudaStreamSynchronize")) {
        return NAN;
    }
    return *context->host_output;
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_ImputationGpuNative_destroy(
    JNIEnv*, jclass, jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}
