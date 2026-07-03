use std::ffi::{c_char, c_void, CStr, CString};
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use libloading::Library;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::ring_buffer::RingBuffer;

use super::THREADS_PER_BLOCK;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const DMA_BUF_HANDLE_TYPE: u32 = 1;
const GPU_DMA_PAGE_SIZE: usize = 2 * 1024 * 1024;

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
type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
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
    launch_kernel: CuLaunchKernel,
    context_synchronize: CuCtxSynchronize,
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
                launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
                context_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
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
    commit_positions: CuFunction,
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
            let mut commit_positions = ptr::null_mut();
            context.api.check(
                unsafe {
                    (context.api.module_get_function)(&mut commit_positions, module, c"commit_ring_positions".as_ptr())
                },
                "resolve commit_ring_positions kernel",
            )?;
            Ok((process_slots, commit_positions))
        })();

        match result {
            Ok((process_slots, commit_positions)) => Ok(Self {
                api: Arc::clone(&context.api),
                module,
                process_slots,
                commit_positions,
            }),
            Err(error) => {
                unsafe { (context.api.module_unload)(module) };
                Err(error)
            }
        }
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        unsafe { (self.api.module_unload)(self.module) };
    }
}

pub struct CudaRuntime {
    kernel: CudaKernel,
    input: CudaBuffer,
    output: CudaBuffer,
    context: CudaContext,
}

impl CudaRuntime {
    pub fn create(device: u32, kernel_path: &Path) -> Result<Self> {
        let context = CudaContext::create(device)?;
        let input = CudaBuffer::allocate(&context)?;
        let output = CudaBuffer::allocate(&context)?;
        let kernel = CudaKernel::load(&context, kernel_path)?;
        Ok(Self {
            kernel,
            input,
            output,
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

    pub fn process(&self, input_tail: u64, output_head: u64, count: u32) -> Result<()> {
        ensure!(count > 0, "CUDA batch must not be empty");
        self.context.make_current()?;

        let mut input = self.input.pointer;
        let mut output = self.output.pointer;
        let mut input_tail_arg = input_tail;
        let mut output_head_arg = output_head;
        let mut count_arg = count;
        let mut arguments = [
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut input_tail_arg as *mut u64).cast(),
            (&mut output_head_arg as *mut u64).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        let blocks = count.div_ceil(THREADS_PER_BLOCK);
        self.context.api.check(
            unsafe {
                (self.context.api.launch_kernel)(
                    self.kernel.process_slots,
                    blocks,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch process_slots",
        )?;

        input_tail_arg = (input_tail + count as u64) & (RING_BUFFER_ELEMENTS as u64 - 1);
        output_head_arg = (output_head + count as u64) & (RING_BUFFER_ELEMENTS as u64 - 1);
        let mut commit_arguments = [
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut input_tail_arg as *mut u64).cast(),
            (&mut output_head_arg as *mut u64).cast(),
        ];
        self.context.api.check(
            unsafe {
                (self.context.api.launch_kernel)(
                    self.kernel.commit_positions,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    commit_arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch commit_ring_positions",
        )?;
        self.context.api.check(
            unsafe { (self.context.api.context_synchronize)() },
            "synchronize CUDA processing",
        )
    }
}

fn round_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
