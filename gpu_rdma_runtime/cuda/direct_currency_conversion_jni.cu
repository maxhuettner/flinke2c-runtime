#include <cuda_runtime.h>
#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

namespace {

constexpr double CONVERSION_FACTOR = 0.908;

// Keep these layouts in sync with CurrencyConversionFunctionGpu.java.
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

static_assert(sizeof(Input) == 16, "Java/native currency input layout changed");
static_assert(sizeof(Output) == 16, "Java/native currency output layout changed");
static_assert(offsetof(Input, price) == 0, "currency price offset changed");
static_assert(offsetof(Input, valid) == 8, "currency input validity offset changed");
static_assert(offsetof(Output, price) == 0, "currency output price offset changed");
static_assert(offsetof(Output, valid) == 8, "currency output validity offset changed");

struct Context {
    int device = 0;
    uint32_t capacity = 0;
    cudaStream_t stream = nullptr;
    Input* device_input = nullptr;
    Output* device_output = nullptr;
    Input* host_input = nullptr;
    Output* host_output = nullptr;
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

__global__ void direct_currency_conversion_batch_kernel(
        const Input* input, Output* output, uint32_t count) {
    const uint32_t item = blockIdx.x;
    if (item >= count || threadIdx.x != 0) {
        return;
    }
    if (input[item].valid == 0) {
        output[item].price = 0.0;
        output[item].valid = 0;
        return;
    }
    output[item].price = input[item].price * CONVERSION_FACTOR;
    output[item].valid = 1;
}

Context* require_context(JNIEnv* env, jlong handle) {
    Context* context = reinterpret_cast<Context*>(handle);
    if (context == nullptr) {
        throw_java(
                env,
                "java/lang/IllegalStateException",
                "Direct CUDA currency conversion context is closed");
    }
    return context;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_create(
        JNIEnv* env, jclass, jint cuda_device, jint batch_capacity) {
    if (cuda_device < 0 || batch_capacity <= 0) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "CUDA device must be non-negative and batch capacity must be positive");
        return 0;
    }

    Context* context = new (std::nothrow) Context();
    if (context == nullptr) {
        throw_java(
                env,
                "java/lang/OutOfMemoryError",
                "Failed to allocate the direct CUDA currency conversion context");
        return 0;
    }
    context->device = cuda_device;
    context->capacity = static_cast<uint32_t>(batch_capacity);
    const size_t input_bytes = sizeof(Input) * context->capacity;
    const size_t output_bytes = sizeof(Output) * context->capacity;

    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice") ||
        !cuda_ok(
                env,
                cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags") ||
        !cuda_ok(
                env,
                cudaMalloc(
                        reinterpret_cast<void**>(&context->device_input), input_bytes),
                "cudaMalloc(currency input)") ||
        !cuda_ok(
                env,
                cudaMalloc(
                        reinterpret_cast<void**>(&context->device_output), output_bytes),
                "cudaMalloc(currency output)") ||
        !cuda_ok(
                env,
                cudaHostAlloc(
                        reinterpret_cast<void**>(&context->host_input),
                        input_bytes,
                        cudaHostAllocPortable),
                "cudaHostAlloc(currency input)") ||
        !cuda_ok(
                env,
                cudaHostAlloc(
                        reinterpret_cast<void**>(&context->host_output),
                        output_bytes,
                        cudaHostAllocPortable),
                "cudaHostAlloc(currency output)")) {
        release_context(context);
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_inputBuffer(
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
Java_org_example_flinke2c_CurrencyConversionGpuNative_outputBuffer(
        JNIEnv* env, jclass, jlong handle) {
    Context* context = require_context(env, handle);
    if (context == nullptr) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(
            context->host_output,
            static_cast<jlong>(sizeof(Output) * context->capacity));
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_processBatch(
        JNIEnv* env, jclass, jlong handle, jint count) {
    Context* context = require_context(env, handle);
    if (context == nullptr) {
        return;
    }
    if (count <= 0 || static_cast<uint32_t>(count) > context->capacity) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "CUDA currency conversion batch count is outside the configured capacity");
        return;
    }
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) {
        return;
    }

    const uint32_t batch_count = static_cast<uint32_t>(count);
    const size_t input_bytes = sizeof(Input) * batch_count;
    const size_t output_bytes = sizeof(Output) * batch_count;
    if (!cuda_ok(
                env,
                cudaMemcpyAsync(
                        context->device_input,
                        context->host_input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        context->stream),
                "cudaMemcpyAsync(currency input)")) {
        return;
    }

    direct_currency_conversion_batch_kernel<<<batch_count, 1, 0, context->stream>>>(
            context->device_input,
            context->device_output,
            batch_count);
    if (!cuda_ok(
                env,
                cudaGetLastError(),
                "direct_currency_conversion_batch_kernel launch")) {
        return;
    }
    if (!cuda_ok(
                env,
                cudaMemcpyAsync(
                        context->host_output,
                        context->device_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        context->stream),
                "cudaMemcpyAsync(currency output)")) {
        return;
    }
    cuda_ok(env, cudaStreamSynchronize(context->stream), "cudaStreamSynchronize");
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_destroy(
        JNIEnv*, jclass, jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}
