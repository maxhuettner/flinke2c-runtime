use std::io;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, Result};
use rdma_mummy_sys::{ibv_dereg_mr, ibv_mr, ibv_reg_dmabuf_mr};
use sideway::ibverbs::protection_domain::ProtectionDomain;

use super::cuda::CudaBuffer;

pub struct DmaBufMemoryRegion {
    mr: NonNull<ibv_mr>,
    _pd: Arc<ProtectionDomain>,
}

impl DmaBufMemoryRegion {
    /// The CUDA allocation and its DMA-BUF descriptor must outlive this MR.
    pub unsafe fn register(pd: Arc<ProtectionDomain>, buffer: &CudaBuffer, access: i32) -> Result<Self> {
        let mr = unsafe {
            ibv_reg_dmabuf_mr(
                pd.pd().as_ptr(),
                0,
                buffer.allocation_size(),
                buffer.pointer(),
                buffer.dma_buf_fd(),
                access,
            )
        };
        let mr = NonNull::new(mr)
            .ok_or_else(io::Error::last_os_error)
            .context("register CUDA DMA-BUF memory region (verify open NVIDIA modules and GPUDirect topology)")?;
        Ok(Self { mr, _pd: pd })
    }

    pub fn lkey(&self) -> u32 {
        unsafe { self.mr.as_ref().lkey }
    }

    pub fn rkey(&self) -> u32 {
        unsafe { self.mr.as_ref().rkey }
    }
}

impl Drop for DmaBufMemoryRegion {
    fn drop(&mut self) {
        unsafe {
            ibv_dereg_mr(self.mr.as_ptr());
        }
    }
}
