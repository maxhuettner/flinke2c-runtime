#include <cuda_runtime.h>
#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr double CONVERSION_FACTOR = 0.908;

// Keep these layouts in sync with CurrencyConversionFunctionGpu.java and
// DirectCudaCurrencyNative.java (both native bridges below share this ABI).
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

// Each lane owns an independent stream, event, and pinned/device buffer pair.
// A single lane is exactly as serial as the original implementation: fill
// host_input, launch, wait. With several lanes the caller can submit lane 1
// while lane 0's H2D copy/kernel/D2H copy are still running, so consecutive
// batches overlap on the GPU's copy and compute engines instead of fully
// serializing behind one cudaStreamSynchronize per batch. This mirrors the
// multi-stream CudaLane pipeline in gpu_runtime/cuda.rs.
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
    // CUDA block size for the batch kernel. A compile-time constant would
    // require rebuilding the PTX/shared library to sweep; this is instead
    // set once at create() time from Java, alongside batch capacity and
    // pipeline depth.
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

__global__ void direct_currency_conversion_batch_kernel(
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

bool require_lane(JNIEnv* env, Context* context, jint lane, uint32_t& out) {
    if (lane < 0 || static_cast<uint32_t>(lane) >= context->lanes.size()) {
        throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "Direct CUDA currency conversion lane index is out of range");
        return false;
    }
    out = static_cast<uint32_t>(lane);
    return true;
}

// ---------------------------------------------------------------------
// Shared implementation. Exposed to Java through two independent sets of
// JNI wrappers below: CurrencyConversionGpuNative (used by the
// AsyncScalarFunction path, org.example.flinke2c) and
// DirectCudaCurrencyNative (used by CudaCurrencyConversionOperator,
// org.apache.flink.table.runtime.functions.table.externalruntime). Both
// bridge classes call into the exact same context/lane logic; only the
// JNI-mangled entry-point names differ, one per calling class. This exists
// because the operator is expected to be compiled into the Flink
// distribution itself (matching how RdmaPreOperator/RdmaPostOperator and
// their RustRdmaNative bridge are structured) while the async UDF ships in
// a separate user JAR — two different classloaders in general, so a single
// shared Java bridge class isn't reliably visible to both callers.
// ---------------------------------------------------------------------

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
    // 1024 is the max threads per block on every CUDA compute capability this
    // library targets (compute_80+); Java validates this too, but check here
    // as well since this is a JNI entry point Java isn't the only caller of.
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
                "Failed to allocate the direct CUDA currency conversion context");
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
                    "cudaMalloc(currency input)") ||
            !cuda_ok(
                    env,
                    cudaMalloc(
                            reinterpret_cast<void**>(&lane.device_output), output_bytes),
                    "cudaMalloc(currency output)") ||
            !cuda_ok(
                    env,
                    cudaHostAlloc(
                            reinterpret_cast<void**>(&lane.host_input),
                            input_bytes,
                            cudaHostAllocPortable),
                    "cudaHostAlloc(currency input)") ||
            !cuda_ok(
                    env,
                    cudaHostAlloc(
                            reinterpret_cast<void**>(&lane.host_output),
                            output_bytes,
                            cudaHostAllocPortable),
                    "cudaHostAlloc(currency output)")) {
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
                "CUDA currency conversion batch count is outside the configured capacity");
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
                "cudaMemcpyAsync(currency input)")) {
        return;
    }

    const uint32_t threads_per_block = context->threads_per_block;
    const uint32_t block_count =
            (batch_count + threads_per_block - 1) / threads_per_block;
    direct_currency_conversion_batch_kernel
            <<<block_count, threads_per_block, 0, active.stream>>>(
            active.device_input,
            active.device_output,
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
                        active.host_output,
                        active.device_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        active.stream),
                "cudaMemcpyAsync(currency output)")) {
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
// Bridge 1: org.example.flinke2c.CurrencyConversionGpuNative
// Used by the AsyncScalarFunction path (CurrencyConversionFunctionGpu),
// shipped in the flinke2c user JAR.
// ---------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_create(
        JNIEnv* env, jclass, jint cuda_device, jint batch_capacity, jint pipeline_depth,
        jint threads_per_block) {
    return create_impl(env, cuda_device, batch_capacity, pipeline_depth, threads_per_block);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_inputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return input_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_outputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return output_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_submitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane, jint count) {
    submit_batch_impl(env, handle, lane, count);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_waitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    wait_batch_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_CurrencyConversionGpuNative_destroy(
        JNIEnv*, jclass, jlong handle) {
    destroy_impl(handle);
}

// ---------------------------------------------------------------------
// Bridge 1b: org.example.flinke2c.DirectCudaCurrencyNative
// Used by the GpuRuntimeFunction path.  This has the same ABI and behavior
// as CurrencyConversionGpuNative above; the JNI symbol prefix must still
// match the Java class name exactly.
// ---------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_create(
        JNIEnv* env, jclass, jint cuda_device, jint batch_capacity, jint pipeline_depth,
        jint threads_per_block) {
    return create_impl(env, cuda_device, batch_capacity, pipeline_depth, threads_per_block);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_inputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return input_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_outputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return output_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_submitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane, jint count) {
    submit_batch_impl(env, handle, lane, count);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_waitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    wait_batch_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaCurrencyNative_destroy(
        JNIEnv*, jclass, jlong handle) {
    destroy_impl(handle);
}

// ---------------------------------------------------------------------
// Bridge 2: org.apache.flink.table.runtime.functions.table.externalruntime
//           .DirectCudaCurrencyNative
// Used by CudaCurrencyConversionOperator, expected to be compiled into the
// Flink distribution itself (see that class's Javadoc). Identical ABI and
// behavior to Bridge 1 above; only the exported symbol names differ, since
// JNI resolves native methods by the calling Java class's fully-qualified
// name.
// ---------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_create(
        JNIEnv* env, jclass, jint cuda_device, jint batch_capacity, jint pipeline_depth,
        jint threads_per_block) {
    return create_impl(env, cuda_device, batch_capacity, pipeline_depth, threads_per_block);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_inputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return input_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_outputBuffer(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    return output_buffer_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_submitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane, jint count) {
    submit_batch_impl(env, handle, lane, count);
}

extern "C" JNIEXPORT void JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_waitBatch(
        JNIEnv* env, jclass, jlong handle, jint lane) {
    wait_batch_impl(env, handle, lane);
}

extern "C" JNIEXPORT void JNICALL
Java_org_apache_flink_table_runtime_functions_table_externalruntime_DirectCudaCurrencyNative_destroy(
        JNIEnv*, jclass, jlong handle) {
    destroy_impl(handle);
}
