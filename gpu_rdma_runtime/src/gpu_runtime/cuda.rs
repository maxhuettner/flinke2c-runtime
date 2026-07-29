use std::ffi::{c_char, c_void, CStr, CString};
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use libloading::Library;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::control_protocol::{ProcessingFunction, ProcessingSpec, WireFieldType};
use crate::ring_buffer::RingBuffer;

use super::THREADS_PER_BLOCK;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const DMA_BUF_HANDLE_TYPE: u32 = 1;
const GPU_DMA_PAGE_SIZE: usize = 2 * 1024 * 1024;
pub const MAX_PROCESS_FIELDS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaProcessSpec {
    pub function: u32,
    pub field_index: u32,
    pub field_count: u32,
    pub field_types: [u32; MAX_PROCESS_FIELDS],
}

impl CudaProcessSpec {
    pub fn from_protocol(spec: &ProcessingSpec) -> Result<Self> {
        ensure!(
            spec.field_index < spec.fields.len() as u32,
            "processing field index is out of range"
        );
        ensure!(
            spec.fields.len() <= MAX_PROCESS_FIELDS,
            "processing schema has too many fields"
        );
        let function = match spec.function {
            ProcessingFunction::Increment => 1,
            ProcessingFunction::Impute => {
                const IMPUTATION_SCHEMA: [WireFieldType; 7] = [
                    WireFieldType::DecimalBytes,
                    WireFieldType::Int64,
                    WireFieldType::Int64,
                    WireFieldType::Bytes,
                    WireFieldType::Bytes,
                    WireFieldType::TimestampMillis,
                    WireFieldType::Bytes,
                ];
                ensure!(
                    spec.field_index == 0 && spec.fields.as_slice() == IMPUTATION_SCHEMA.as_slice(),
                    "IMPUTE requires field_index 0 and fields \
                     [DECIMAL_BYTES, INT64, INT64, BYTES, BYTES, TIMESTAMP_MILLIS, BYTES]"
                );
                2
            }
        };
        let mut field_types = [0u32; MAX_PROCESS_FIELDS];
        for (index, field) in spec.fields.iter().enumerate() {
            field_types[index] = match field {
                WireFieldType::Int32 => 1,
                WireFieldType::Int64 => 2,
                WireFieldType::DecimalBytes => 3,
                WireFieldType::Bytes => 4,
                WireFieldType::TimestampMillis => 5,
            };
        }
        Ok(Self {
            function,
            field_index: spec.field_index,
            field_count: spec.fields.len() as u32,
            field_types,
        })
    }
}

type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult;
type CuMemGetHandleForAddressRange = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, u32, u64) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, u32) -> CuResult;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamWaitEvent = unsafe extern "C" fn(CuStream, CuEvent, u32) -> CuResult;
type CuEventCreate = unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult;
type CuEventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type CuEventQuery = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuEventSynchronize = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuFlushGpudirectRdmaWrites = unsafe extern "C" fn(u32, u32) -> CuResult;
type CuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

struct CudaApi {
    _library: Library,
    init: CuInit,
    device_get: CuDeviceGet,
    primary_context_retain: CuDevicePrimaryCtxRetain,
    primary_context_release: CuDevicePrimaryCtxRelease,
    context_set_current: CuCtxSetCurrent,
    memory_allocate: CuMemAlloc,
    memory_free: CuMemFree,
    memory_set: CuMemsetD8,
    memory_get_range_handle: CuMemGetHandleForAddressRange,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    stream_wait_event: CuStreamWaitEvent,
    event_create: CuEventCreate,
    event_destroy: CuEventDestroy,
    event_record: CuEventRecord,
    event_query: CuEventQuery,
    event_synchronize: CuEventSynchronize,
    launch_kernel: CuLaunchKernel,
    flush_gpudirect_writes: Option<CuFlushGpudirectRdmaWrites>,
    error_string: CuGetErrorString,
}

impl CudaApi {
    fn load() -> Result<Arc<Self>> {
        let library = unsafe { Library::new("libcuda.so.1") }.context("load libcuda.so.1")?;
        unsafe {
            Ok(Arc::new(Self {
                init: load_symbol(&library, b"cuInit\0")?,
                device_get: load_symbol(&library, b"cuDeviceGet\0")?,
                primary_context_retain: load_symbol(&library, b"cuDevicePrimaryCtxRetain\0")?,
                primary_context_release: load_symbol(&library, b"cuDevicePrimaryCtxRelease_v2\0")?,
                context_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
                memory_allocate: load_symbol(&library, b"cuMemAlloc_v2\0")?,
                memory_free: load_symbol(&library, b"cuMemFree_v2\0")?,
                memory_set: load_symbol(&library, b"cuMemsetD8_v2\0")?,
                memory_get_range_handle: load_symbol(&library, b"cuMemGetHandleForAddressRange\0")?,
                module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
                module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
                module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
                stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
                stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
                stream_wait_event: load_symbol(&library, b"cuStreamWaitEvent\0")?,
                event_create: load_symbol(&library, b"cuEventCreate\0")?,
                event_destroy: load_symbol(&library, b"cuEventDestroy_v2\0")?,
                event_record: load_symbol(&library, b"cuEventRecord\0")?,
                event_query: load_symbol(&library, b"cuEventQuery\0")?,
                event_synchronize: load_symbol(&library, b"cuEventSynchronize\0")?,
                launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
                flush_gpudirect_writes: load_optional_symbol(&library, b"cuFlushGPUDirectRDMAWrites\0"),
                error_string: load_symbol(&library, b"cuGetErrorString\0")?,
                _library: library,
            }))
        }
    }

    fn check(&self, status: CuResult, operation: &str) -> Result<()> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }

        let mut description = ptr::null();
        unsafe { (self.error_string)(status, &mut description) };
        let description = if description.is_null() {
            "unknown CUDA error".into()
        } else {
            unsafe { CStr::from_ptr(description) }.to_string_lossy()
        };
        anyhow::bail!("{operation}: {description} (CUDA status {status})")
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(name) }?)
}

unsafe fn load_optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(name) }.ok().map(|symbol| *symbol)
}

struct CudaContext {
    api: Arc<CudaApi>,
    device: CuDevice,
    context: CuContext,
}

impl CudaContext {
    fn create(ordinal: u32) -> Result<Self> {
        let api = CudaApi::load()?;
        api.check(unsafe { (api.init)(0) }, "initialize CUDA")?;

        let mut device = 0;
        api.check(
            unsafe { (api.device_get)(&mut device, ordinal as i32) },
            "select CUDA device",
        )?;
        let mut context = ptr::null_mut();
        api.check(
            unsafe { (api.primary_context_retain)(&mut context, device) },
            "retain CUDA primary context",
        )?;
        api.check(unsafe { (api.context_set_current)(context) }, "activate CUDA context")?;
        Ok(Self { api, device, context })
    }

    fn make_current(&self) -> Result<()> {
        self.api.check(
            unsafe { (self.api.context_set_current)(self.context) },
            "activate CUDA context",
        )
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        let _ = unsafe { (self.api.primary_context_release)(self.device) };
    }
}

pub struct CudaBuffer {
    api: Arc<CudaApi>,
    pointer: CuDevicePtr,
    allocation_size: usize,
    dma_buf_fd: OwnedFd,
}

impl CudaBuffer {
    fn allocate(context: &CudaContext) -> Result<Self> {
        context.make_current()?;
        let allocation_size = round_up(
            std::mem::size_of::<RingBuffer<RING_BUFFER_ELEMENTS>>(),
            GPU_DMA_PAGE_SIZE,
        );

        let mut pointer = 0;
        context.api.check(
            unsafe { (context.api.memory_allocate)(&mut pointer, allocation_size) },
            "allocate CUDA ring",
        )?;

        let result = (|| {
            ensure!(
                (pointer as usize).is_multiple_of(GPU_DMA_PAGE_SIZE),
                "CUDA allocation is not aligned to the GPU DMA page size"
            );
            context.api.check(
                unsafe { (context.api.memory_set)(pointer, 0, allocation_size) },
                "clear CUDA ring",
            )?;

            let mut dma_buf_fd = -1;
            context.api.check(
                unsafe {
                    (context.api.memory_get_range_handle)(
                        (&mut dma_buf_fd as *mut i32).cast(),
                        pointer,
                        allocation_size,
                        DMA_BUF_HANDLE_TYPE,
                        0,
                    )
                },
                "export CUDA DMA-BUF",
            )?;
            ensure!(dma_buf_fd >= 0, "CUDA returned an invalid DMA-BUF descriptor");
            Ok(unsafe { OwnedFd::from_raw_fd(dma_buf_fd) })
        })();

        match result {
            Ok(dma_buf_fd) => Ok(Self {
                api: Arc::clone(&context.api),
                pointer,
                allocation_size,
                dma_buf_fd,
            }),
            Err(error) => {
                unsafe { (context.api.memory_free)(pointer) };
                Err(error)
            }
        }
    }

    pub fn pointer(&self) -> u64 {
        self.pointer
    }

    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    pub fn dma_buf_fd(&self) -> i32 {
        self.dma_buf_fd.as_raw_fd()
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            (self.api.memory_free)(self.pointer);
        }
    }
}

struct CudaKernel {
    api: Arc<CudaApi>,
    module: CuModule,
    process_slots: CuFunction,
    commit_imputation_history: CuFunction,
    publish_output_head: CuFunction,
}

impl CudaKernel {
    fn load(context: &CudaContext, path: &Path) -> Result<Self> {
        context.make_current()?;
        let ptx = fs::read(path).with_context(|| format!("read CUDA kernel {}", path.display()))?;
        let ptx = CString::new(ptx).context("CUDA PTX contains an interior NUL byte")?;
        let mut module = ptr::null_mut();
        context.api.check(
            unsafe { (context.api.module_load_data)(&mut module, ptx.as_ptr().cast()) },
            "load CUDA PTX module",
        )?;

        let result = (|| {
            let mut process_slots = ptr::null_mut();
            context.api.check(
                unsafe { (context.api.module_get_function)(&mut process_slots, module, c"process_slots".as_ptr()) },
                "resolve process_slots kernel",
            )?;
            let mut publish_output_head = ptr::null_mut();
            context.api.check(
                unsafe {
                    (context.api.module_get_function)(&mut publish_output_head, module, c"publish_output_head".as_ptr())
                },
                "resolve publish_output_head kernel",
            )?;
            let mut commit_imputation_history = ptr::null_mut();
            context.api.check(
                unsafe {
                    (context.api.module_get_function)(
                        &mut commit_imputation_history,
                        module,
                        c"commit_imputation_history".as_ptr(),
                    )
                },
                "resolve commit_imputation_history kernel",
            )?;
            Ok((process_slots, commit_imputation_history, publish_output_head))
        })();

        match result {
            Ok((process_slots, commit_imputation_history, publish_output_head)) => Ok(Self {
                api: Arc::clone(&context.api),
                module,
                process_slots,
                commit_imputation_history,
                publish_output_head,
            }),
            Err(error) => {
                unsafe { (context.api.module_unload)(module) };
                Err(error)
            }
        }
    }
}

struct CudaLane {
    api: Arc<CudaApi>,
    stream: CuStream,
    event: CuEvent,
}

impl CudaLane {
    fn create(context: &CudaContext) -> Result<Self> {
        const STREAM_NON_BLOCKING: u32 = 1;
        const EVENT_DISABLE_TIMING: u32 = 2;

        let mut stream = ptr::null_mut();
        context.api.check(
            unsafe { (context.api.stream_create)(&mut stream, STREAM_NON_BLOCKING) },
            "create CUDA stream",
        )?;
        let mut event = ptr::null_mut();
        if let Err(error) = context.api.check(
            unsafe { (context.api.event_create)(&mut event, EVENT_DISABLE_TIMING) },
            "create CUDA event",
        ) {
            unsafe { (context.api.stream_destroy)(stream) };
            return Err(error);
        }
        Ok(Self {
            api: Arc::clone(&context.api),
            stream,
            event,
        })
    }
}

impl Drop for CudaLane {
    fn drop(&mut self) {
        unsafe {
            (self.api.event_destroy)(self.event);
            (self.api.stream_destroy)(self.stream);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaBatch {
    lane: usize,
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        unsafe { (self.api.module_unload)(self.module) };
    }
}

pub struct CudaRuntime {
    lanes: Vec<CudaLane>,
    kernel: CudaKernel,
    input: CudaBuffer,
    output: CudaBuffer,
    next_lane: usize,
    last_imputation_lane: Option<usize>,
    context: CudaContext,
}

impl CudaRuntime {
    pub fn create(device: u32, pipeline_depth: usize, kernel_path: &Path) -> Result<Self> {
        ensure!(
            (1..=64).contains(&pipeline_depth),
            "CUDA pipeline depth must be in 1..=64"
        );
        let context = CudaContext::create(device)?;
        let input = CudaBuffer::allocate(&context)?;
        let output = CudaBuffer::allocate(&context)?;
        let kernel = CudaKernel::load(&context, kernel_path)?;
        // A single ordered stream guarantees that the GPU-published producer
        // head can never skip an unfinished batch.
        let lanes = (0..pipeline_depth)
            .map(|_| CudaLane::create(&context))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            kernel,
            input,
            output,
            next_lane: 0,
            last_imputation_lane: None,
            context,
        })
    }

    pub fn input(&self) -> &CudaBuffer {
        &self.input
    }

    pub fn output(&self) -> &CudaBuffer {
        &self.output
    }

    pub fn flush_gpudirect_writes(&self) -> Result<()> {
        const TARGET_CURRENT_CONTEXT: u32 = 0;
        const SCOPE_TO_OWNER: u32 = 100;
        const CUDA_ERROR_NOT_SUPPORTED: CuResult = 801;

        self.context.make_current()?;
        let Some(flush) = self.context.api.flush_gpudirect_writes else {
            return Ok(());
        };
        let status = unsafe { flush(TARGET_CURRENT_CONTEXT, SCOPE_TO_OWNER) };
        if status == CUDA_ERROR_NOT_SUPPORTED {
            // The following CPU-submitted kernel launch still establishes ordering.
            return Ok(());
        }
        self.context.api.check(status, "flush GPUDirect RDMA writes")
    }

    pub fn pipeline_depth(&self) -> usize {
        self.lanes.len()
    }

    pub fn submit(
        &mut self,
        input_tail: u64,
        output_head: u64,
        count: u32,
        spec: CudaProcessSpec,
    ) -> Result<CudaBatch> {
        ensure!(count > 0, "CUDA batch must not be empty");
        self.context.make_current()?;

        let lane_index = self.next_lane;
        self.next_lane = (self.next_lane + 1) % self.lanes.len();
        let lane = &self.lanes[lane_index];

        if spec.function == 2 {
            if let Some(previous_lane) = self.last_imputation_lane {
                self.context.api.check(
                    unsafe { (self.context.api.stream_wait_event)(lane.stream, self.lanes[previous_lane].event, 0) },
                    "order stateful imputation batches",
                )?;
            }
            self.last_imputation_lane = Some(lane_index);
        }

        let mut input = self.input.pointer;
        let mut output = self.output.pointer;
        let mut input_tail_arg = input_tail;
        let mut output_head_arg = output_head;
        let mut count_arg = count;
        let mut spec_arg = spec;
        let mut arguments = [
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut input_tail_arg as *mut u64).cast(),
            (&mut output_head_arg as *mut u64).cast(),
            (&mut count_arg as *mut u32).cast(),
            (&mut spec_arg as *mut CudaProcessSpec).cast(),
        ];
        self.context.api.check(
            unsafe {
                (self.context.api.launch_kernel)(
                    self.kernel.process_slots,
                    count,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    lane.stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch process_slots",
        )?;
        if spec.function == 2 {
            let mut history_arguments = [
                (&mut input as *mut u64).cast(),
                (&mut input_tail_arg as *mut u64).cast(),
                (&mut count_arg as *mut u32).cast(),
                (&mut spec_arg as *mut CudaProcessSpec).cast(),
            ];
            self.context.api.check(
                unsafe {
                    (self.context.api.launch_kernel)(
                        self.kernel.commit_imputation_history,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        lane.stream,
                        history_arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "launch commit_imputation_history",
            )?;
        }
        self.context.api.check(
            unsafe { (self.context.api.event_record)(lane.event, lane.stream) },
            "record CUDA batch completion",
        )?;
        Ok(CudaBatch { lane: lane_index })
    }

    pub fn is_complete(&self, batch: CudaBatch) -> Result<bool> {
        self.context.make_current()?;
        let status = unsafe { (self.context.api.event_query)(self.lanes[batch.lane].event) };
        if status == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        self.context.api.check(status, "query CUDA batch completion")?;
        Ok(true)
    }

    pub fn wait(&self, batch: CudaBatch) -> Result<()> {
        self.context.make_current()?;
        self.context.api.check(
            unsafe { (self.context.api.event_synchronize)(self.lanes[batch.lane].event) },
            "wait for CUDA batch completion",
        )
    }
}

fn round_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imputation_spec() -> ProcessingSpec {
        ProcessingSpec {
            function: ProcessingFunction::Impute,
            field_index: 0,
            fields: vec![
                WireFieldType::DecimalBytes,
                WireFieldType::Int64,
                WireFieldType::Int64,
                WireFieldType::Bytes,
                WireFieldType::Bytes,
                WireFieldType::TimestampMillis,
                WireFieldType::Bytes,
            ],
        }
    }

    #[test]
    fn accepts_imputation_schema() {
        let spec = CudaProcessSpec::from_protocol(&imputation_spec()).unwrap();
        assert_eq!(spec.function, 2);
        assert_eq!(spec.field_index, 0);
        assert_eq!(spec.field_count, 7);
        assert_eq!(&spec.field_types[..7], &[3, 2, 2, 4, 4, 5, 4]);
    }

    #[test]
    fn rejects_imputation_schema_with_wrong_price_type() {
        let mut spec = imputation_spec();
        spec.fields[0] = WireFieldType::Int64;
        let error = match CudaProcessSpec::from_protocol(&spec) {
            Ok(_) => panic!("invalid imputation schema was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("IMPUTE requires"));
    }
}
