#include <cuda_runtime.h>
#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "process_function.cu"

namespace {

constexpr uint32_t MAX_PIPELINE_DEPTH = 64;
constexpr uint32_t MIN_THREADS_PER_BLOCK = 32;
constexpr uint32_t MAX_THREADS_PER_BLOCK = 1024;

static_assert(offsetof(Slot, value) == 16, "Java slot value offset changed");
static_assert(sizeof(Slot) == 2064, "Java slot stride changed");

struct Lane {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    bool pending = false;
    Slot* host_input = nullptr;
    Slot* host_output = nullptr;
    RingBuffer* device_input = nullptr;
    RingBuffer* device_output = nullptr;
    ImputationObservation* device_observations = nullptr;
    uint8_t* device_has_price = nullptr;
    // Only created/recorded when Context::profiling is set (see
    // waitBatchTimed). event above stays timing-disabled regardless -
    // that's the one every submitBatch call touches for lane-reuse and
    // cross-lane history ordering, and cudaEventDisableTiming is lighter
    // weight for that hot path. These five give the same breakdown
    // PerfStats does on the Java side, but for what actually runs on the
    // GPU between the host-observed "submit" and "wait" stages.
    cudaEvent_t t_h2d_start = nullptr;
    cudaEvent_t t_h2d_end = nullptr;
    cudaEvent_t t_prepare_end = nullptr;
    cudaEvent_t t_process_end = nullptr;
    cudaEvent_t t_commit_end = nullptr;
    cudaEvent_t t_d2h_end = nullptr;
};

struct Context {
    int device = 0;
    uint32_t capacity = 0;
    uint32_t threads_per_block = 0;
    ImputationState* state = nullptr;
    std::vector<Lane> lanes;
    int last_lane = -1;
    bool profiling = false;
};

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass cls = env->FindClass(class_name);
    if (cls != nullptr) env->ThrowNew(cls, message.c_str());
}

bool cuda_ok(JNIEnv* env, cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::string message(operation);
    message.append(": ");
    message.append(cudaGetErrorString(status));
    throw_java(env, "java/lang/IllegalStateException", message);
    return false;
}

void release_lane(Context* context, Lane& lane) {
    cudaSetDevice(context->device);
    if (lane.stream != nullptr) cudaStreamSynchronize(lane.stream);
    if (lane.device_output != nullptr) cudaFree(lane.device_output);
    if (lane.device_input != nullptr) cudaFree(lane.device_input);
    if (lane.device_observations != nullptr) cudaFree(lane.device_observations);
    if (lane.device_has_price != nullptr) cudaFree(lane.device_has_price);
    if (lane.host_output != nullptr) cudaFreeHost(lane.host_output);
    if (lane.host_input != nullptr) cudaFreeHost(lane.host_input);
    if (lane.event != nullptr) cudaEventDestroy(lane.event);
    if (lane.t_h2d_start != nullptr) cudaEventDestroy(lane.t_h2d_start);
    if (lane.t_h2d_end != nullptr) cudaEventDestroy(lane.t_h2d_end);
    if (lane.t_prepare_end != nullptr) cudaEventDestroy(lane.t_prepare_end);
    if (lane.t_process_end != nullptr) cudaEventDestroy(lane.t_process_end);
    if (lane.t_commit_end != nullptr) cudaEventDestroy(lane.t_commit_end);
    if (lane.t_d2h_end != nullptr) cudaEventDestroy(lane.t_d2h_end);
    if (lane.stream != nullptr) cudaStreamDestroy(lane.stream);
    lane = Lane{};
}

void release_context(Context* context) {
    if (context == nullptr) return;
    for (Lane& lane : context->lanes) release_lane(context, lane);
    cudaSetDevice(context->device);
    if (context->state != nullptr) cudaFree(context->state);
    delete context;
}

bool valid_handle(JNIEnv* env, jlong handle, Context*& context) {
    context = reinterpret_cast<Context*>(handle);
    if (context != nullptr) return true;
    throw_java(env, "java/lang/IllegalStateException", "null CUDA imputation context");
    return false;
}

bool valid_lane(JNIEnv* env, Context* context, jint lane, Lane*& result) {
    if (lane < 0 || static_cast<uint32_t>(lane) >= context->lanes.size()) {
        throw_java(env, "java/lang/IllegalArgumentException", "invalid CUDA lane");
        return false;
    }
    result = &context->lanes[static_cast<uint32_t>(lane)];
    return true;
}

ProcessSpec imputation_spec() {
    ProcessSpec spec{};
    spec.function = FUNCTION_IMPUTE;
    spec.field_index = 0;
    spec.field_count = 7;
    // Not actually consulted by parse_bid/write_imputed_bid (IMPUTE's field
    // walk is hardcoded, not driven by field_types), but kept accurate: 6 =
    // DECIMAL_UNSCALED_I64, matching gpu_runtime/cuda.rs's field_types
    // mapping on the RDMA side. price moved off the DECIMAL_BYTES (3)
    // variable-length format - see BidView::price_unscaled's comment in
    // process_function.cu.
    spec.field_types[0] = 6;
    spec.field_types[1] = 2;
    spec.field_types[2] = 2;
    spec.field_types[3] = 4;
    spec.field_types[4] = 4;
    spec.field_types[5] = 5;
    spec.field_types[6] = 4;
    return spec;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_create(
    JNIEnv* env, jclass, jint device, jint capacity, jint pipeline_depth,
    jint threads_per_block, jboolean profiling) {
    if (device < 0 || capacity <= 0 || capacity > static_cast<jint>(RING_BUFFER_ELEMENTS) ||
        pipeline_depth <= 0 || pipeline_depth > static_cast<jint>(MAX_PIPELINE_DEPTH) ||
        threads_per_block < static_cast<jint>(MIN_THREADS_PER_BLOCK) ||
        threads_per_block > static_cast<jint>(MAX_THREADS_PER_BLOCK) ||
        (threads_per_block % 32) != 0) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "invalid CUDA imputation capacity, pipeline depth, or block size");
        return 0;
    }
    if (!cuda_ok(env, cudaSetDevice(device), "cudaSetDevice")) return 0;

    Context* context = new (std::nothrow) Context();
    if (context == nullptr) {
        throw_java(env, "java/lang/OutOfMemoryError", "cannot allocate CUDA context");
        return 0;
    }
    context->device = device;
    context->capacity = static_cast<uint32_t>(capacity);
    context->threads_per_block = static_cast<uint32_t>(threads_per_block);
    context->profiling = profiling != JNI_FALSE;
    context->lanes.resize(static_cast<size_t>(pipeline_depth));

    if (!cuda_ok(env, cudaMalloc(reinterpret_cast<void**>(&context->state), sizeof(ImputationState)), "cudaMalloc state") ||
        !cuda_ok(env, cudaMemset(context->state, 0, sizeof(ImputationState)), "cudaMemset state")) {
        release_context(context);
        return 0;
    }

    const size_t bytes = offsetof(RingBuffer, slots) + sizeof(Slot) * static_cast<size_t>(capacity);
    for (Lane& lane : context->lanes) {
        if (!cuda_ok(env, cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking), "cudaStreamCreate") ||
            !cuda_ok(env, cudaEventCreateWithFlags(&lane.event, cudaEventDisableTiming), "cudaEventCreate") ||
            !cuda_ok(env, cudaHostAlloc(reinterpret_cast<void**>(&lane.host_input), sizeof(Slot) * static_cast<size_t>(capacity), cudaHostAllocPortable), "cudaHostAlloc input") ||
            !cuda_ok(env, cudaHostAlloc(reinterpret_cast<void**>(&lane.host_output), sizeof(Slot) * static_cast<size_t>(capacity), cudaHostAllocPortable), "cudaHostAlloc output") ||
            !cuda_ok(env, cudaMalloc(reinterpret_cast<void**>(&lane.device_input), bytes), "cudaMalloc input") ||
            !cuda_ok(env, cudaMalloc(reinterpret_cast<void**>(&lane.device_output), bytes), "cudaMalloc output") ||
            !cuda_ok(env, cudaMalloc(reinterpret_cast<void**>(&lane.device_observations), sizeof(ImputationObservation) * static_cast<size_t>(capacity)), "cudaMalloc observations") ||
            !cuda_ok(env, cudaMalloc(reinterpret_cast<void**>(&lane.device_has_price), sizeof(uint8_t) * static_cast<size_t>(capacity)), "cudaMalloc price flags")) {
            release_context(context);
            return 0;
        }
        if (context->profiling &&
            (!cuda_ok(env, cudaEventCreate(&lane.t_h2d_start), "cudaEventCreate t_h2d_start") ||
             !cuda_ok(env, cudaEventCreate(&lane.t_h2d_end), "cudaEventCreate t_h2d_end") ||
             !cuda_ok(env, cudaEventCreate(&lane.t_prepare_end), "cudaEventCreate t_prepare_end") ||
             !cuda_ok(env, cudaEventCreate(&lane.t_process_end), "cudaEventCreate t_process_end") ||
             !cuda_ok(env, cudaEventCreate(&lane.t_commit_end), "cudaEventCreate t_commit_end") ||
             !cuda_ok(env, cudaEventCreate(&lane.t_d2h_end), "cudaEventCreate t_d2h_end"))) {
            release_context(context);
            return 0;
        }
    }
    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_inputBuffer(
    JNIEnv* env, jclass, jlong handle, jint lane_index) {
    Context* context; Lane* lane;
    if (!valid_handle(env, handle, context) || !valid_lane(env, context, lane_index, lane)) return nullptr;
    return env->NewDirectByteBuffer(lane->host_input, static_cast<jlong>(sizeof(Slot)) * context->capacity);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_outputBuffer(
    JNIEnv* env, jclass, jlong handle, jint lane_index) {
    Context* context; Lane* lane;
    if (!valid_handle(env, handle, context) || !valid_lane(env, context, lane_index, lane)) return nullptr;
    return env->NewDirectByteBuffer(lane->host_output, static_cast<jlong>(sizeof(Slot)) * context->capacity);
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_submitBatch(
    JNIEnv* env, jclass, jlong handle, jint lane_index, jint count) {
    Context* context; Lane* lane;
    if (!valid_handle(env, handle, context) || !valid_lane(env, context, lane_index, lane)) return;
    if (count <= 0 || static_cast<uint32_t>(count) > context->capacity) {
        throw_java(env, "java/lang/IllegalArgumentException", "invalid CUDA imputation batch count");
        return;
    }
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice")) return;
    const bool profiling = context->profiling;
    const size_t offset = offsetof(RingBuffer, slots);
    const size_t bytes = sizeof(Slot) * static_cast<size_t>(count);
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_h2d_start, lane->stream), "cudaEventRecord t_h2d_start")) return;
    if (!cuda_ok(env, cudaMemcpyAsync(reinterpret_cast<uint8_t*>(lane->device_input) + offset, lane->host_input, bytes, cudaMemcpyHostToDevice, lane->stream), "cudaMemcpyAsync input")) return;
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_h2d_end, lane->stream), "cudaEventRecord t_h2d_end")) return;
    if (context->last_lane >= 0) {
        Lane& previous = context->lanes[static_cast<size_t>(context->last_lane)];
        if (!cuda_ok(env, cudaStreamWaitEvent(lane->stream, previous.event, 0), "cudaStreamWaitEvent")) return;
    }
    const ProcessSpec spec = imputation_spec();
    const uint32_t warps_per_block = context->threads_per_block / 32;
    const uint32_t blocks = (static_cast<uint32_t>(count) + warps_per_block - 1) / warps_per_block;
    const uint32_t observation_blocks =
        (static_cast<uint32_t>(count) + context->threads_per_block - 1) /
        context->threads_per_block;
    prepare_imputation_observations<<<observation_blocks, context->threads_per_block, 0, lane->stream>>>(
        lane->device_input, lane->device_observations, lane->device_has_price,
        static_cast<uint32_t>(count));
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_prepare_end, lane->stream), "cudaEventRecord t_prepare_end")) return;
    process_slots_state<<<blocks, context->threads_per_block, 0, lane->stream>>>(
        lane->device_input, lane->device_output, context->state,
        lane->device_observations, lane->device_has_price,
        0, 0, static_cast<uint32_t>(count), spec);
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_process_end, lane->stream), "cudaEventRecord t_process_end")) return;
    commit_imputation_history_state<<<1, 1, 0, lane->stream>>>(
        lane->device_observations, lane->device_has_price, context->state,
        0, static_cast<uint32_t>(count), spec);
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_commit_end, lane->stream), "cudaEventRecord t_commit_end")) return;
    if (!cuda_ok(env, cudaGetLastError(), "launch imputation kernels")) return;
    if (!cuda_ok(env, cudaMemcpyAsync(lane->host_output, reinterpret_cast<uint8_t*>(lane->device_output) + offset, bytes, cudaMemcpyDeviceToHost, lane->stream), "cudaMemcpyAsync output")) return;
    if (profiling && !cuda_ok(env, cudaEventRecord(lane->t_d2h_end, lane->stream), "cudaEventRecord t_d2h_end")) return;
    if (!cuda_ok(env, cudaEventRecord(lane->event, lane->stream), "cudaEventRecord")) return;
    lane->pending = true;
    context->last_lane = lane_index;
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_waitBatch(
    JNIEnv* env, jclass, jlong handle, jint lane_index) {
    Context* context; Lane* lane;
    if (!valid_handle(env, handle, context) || !valid_lane(env, context, lane_index, lane)) return;
    if (!lane->pending) return;
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice") ||
        !cuda_ok(env, cudaEventSynchronize(lane->event), "cudaEventSynchronize")) return;
    lane->pending = false;
}

// Same wait as waitBatch, but for a context created with profiling=true:
// also returns the elapsed milliseconds of each GPU-side stage between the
// host-observed "submit" and "wait" (see PerfStats on the Java side for the
// stages surrounding this one). Safe to read all five events with
// cudaEventElapsedTime unsynchronized here - lane->event above is recorded
// after all of them on the same stream, and cudaEventSynchronize on it
// above already guarantees a CUDA stream's earlier work (and its event
// records) has completed, since a stream executes in issue order. Returns
// an empty array if the context wasn't created with profiling, or if this
// lane has nothing pending.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_waitBatchTimed(
    JNIEnv* env, jclass, jlong handle, jint lane_index) {
    Context* context; Lane* lane;
    if (!valid_handle(env, handle, context) || !valid_lane(env, context, lane_index, lane)) return nullptr;
    jfloatArray empty = env->NewFloatArray(0);
    if (!lane->pending) return empty;
    if (!cuda_ok(env, cudaSetDevice(context->device), "cudaSetDevice") ||
        !cuda_ok(env, cudaEventSynchronize(lane->event), "cudaEventSynchronize")) return nullptr;
    lane->pending = false;
    if (!context->profiling) return empty;

    float h2d_ms = 0.0f, prepare_ms = 0.0f, process_ms = 0.0f, commit_ms = 0.0f, d2h_ms = 0.0f;
    if (!cuda_ok(env, cudaEventElapsedTime(&h2d_ms, lane->t_h2d_start, lane->t_h2d_end), "cudaEventElapsedTime h2d") ||
        !cuda_ok(env, cudaEventElapsedTime(&prepare_ms, lane->t_h2d_end, lane->t_prepare_end), "cudaEventElapsedTime prepare") ||
        !cuda_ok(env, cudaEventElapsedTime(&process_ms, lane->t_prepare_end, lane->t_process_end), "cudaEventElapsedTime process") ||
        !cuda_ok(env, cudaEventElapsedTime(&commit_ms, lane->t_process_end, lane->t_commit_end), "cudaEventElapsedTime commit") ||
        !cuda_ok(env, cudaEventElapsedTime(&d2h_ms, lane->t_commit_end, lane->t_d2h_end), "cudaEventElapsedTime d2h")) {
        return nullptr;
    }
    jfloatArray result = env->NewFloatArray(5);
    if (result == nullptr) return nullptr;
    const float values[5] = {h2d_ms, prepare_ms, process_ms, commit_ms, d2h_ms};
    env->SetFloatArrayRegion(result, 0, 5, values);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_org_example_flinke2c_DirectCudaImputationNative_destroy(
    JNIEnv*, jclass, jlong handle) {
    release_context(reinterpret_cast<Context*>(handle));
}
