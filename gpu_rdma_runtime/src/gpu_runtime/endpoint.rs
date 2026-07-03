use std::mem::size_of;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use rdma_mummy_sys::ibv_access_flags;
use sideway::ibverbs::{
    address::AddressHandleAttribute,
    completion::{ExtendedCompletionQueue, GenericCompletionQueue, PollCompletionQueueError, WorkCompletionStatus},
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, Mtu},
    protection_domain::ProtectionDomain,
    queue_pair::{
        ExtendedQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState, SetScatterGatherEntry,
        WorkRequestFlags,
    },
    AccessFlags,
};

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::control_protocol::{MemoryRegionInfo, RdmaDestination};
use crate::ring_buffer::{slot::Slot, RingBuffer};

use super::cuda::CudaRuntime;
use super::memory_region::DmaBufMemoryRegion;

type RuntimeRing = RingBuffer<RING_BUFFER_ELEMENTS>;

pub struct GpuRdmaEndpoint {
    pub ctx: Arc<DeviceContext>,
    pub qp: ExtendedQueuePair,
    cq: Arc<ExtendedCompletionQueue>,
    input_mr: DmaBufMemoryRegion,
    output_mr: DmaBufMemoryRegion,
    _pd: Arc<ProtectionDomain>,
    cuda: CudaRuntime,
    next_write_id: u64,
}

impl GpuRdmaEndpoint {
    pub fn build(
        ib_device: Option<&str>,
        ib_port: u8,
        cuda_device: u32,
        kernel_path: &std::path::Path,
    ) -> Result<Self> {
        let cuda = CudaRuntime::create(cuda_device, kernel_path)?;
        let device_list = DeviceList::new().context("get RDMA device list")?;
        let device = match ib_device {
            Some(name) => device_list
                .iter()
                .find(|device| device.name() == name)
                .with_context(|| format!("RDMA device {name} not found"))?,
            None => device_list.iter().next().context("no RDMA device found")?,
        };
        let ctx = device
            .open()
            .with_context(|| format!("open RDMA device {}", device.name()))?;
        let pd = ctx.alloc_pd().context("allocate RDMA protection domain")?;

        let remote_access = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ)
            .0 as i32;
        let local_access = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 as i32;
        let input_mr = unsafe { DmaBufMemoryRegion::register(Arc::clone(&pd), cuda.input(), remote_access) }
            .context("register GPUDirect input ring")?;
        let output_mr = unsafe { DmaBufMemoryRegion::register(Arc::clone(&pd), cuda.output(), local_access) }
            .context("register GPUDirect output ring")?;

        let cq = ctx.create_cq_builder().build_ex().context("create completion queue")?;
        let generic_cq = GenericCompletionQueue::from(Arc::clone(&cq));
        let mut qp = pd
            .create_qp_builder()
            .setup_max_inline_data(128)
            .setup_max_recv_wr(1)
            .setup_max_send_wr(1024)
            .setup_send_cq(generic_cq.clone())
            .setup_recv_cq(generic_cq)
            .build_ex()
            .context("create RDMA queue pair")?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::Init)
            .setup_pkey_index(0)
            .setup_port(ib_port)
            .setup_access_flags(AccessFlags::RemoteWrite | AccessFlags::RemoteRead);
        qp.modify(&attr).context("move QP to INIT")?;

        Ok(Self {
            ctx,
            qp,
            cq,
            input_mr,
            output_mr,
            _pd: pd,
            cuda,
            next_write_id: 0,
        })
    }

    pub fn connect(
        &mut self,
        remote: &RdmaDestination,
        ib_port: u8,
        packet_seq_num: u32,
        mtu: Mtu,
        gid_index: u8,
    ) -> Result<()> {
        let mut address = AddressHandleAttribute::new();
        address
            .setup_dest_lid(0)
            .setup_port(ib_port)
            .setup_service_level(0)
            .setup_grh_src_gid_index(gid_index)
            .setup_grh_dest_gid(&remote.gid)
            .setup_grh_hop_limit(1);

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::ReadyToReceive)
            .setup_path_mtu(mtu)
            .setup_dest_qp_num(remote.qp_number)
            .setup_rq_psn(remote.packet_seq_num)
            .setup_max_dest_read_atomic(1)
            .setup_min_rnr_timer(0)
            .setup_address_vector(&address);
        self.qp.modify(&attr).context("move QP to RTR")?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::ReadyToSend)
            .setup_sq_psn(packet_seq_num)
            .setup_timeout(12)
            .setup_retry_cnt(7)
            .setup_rnr_retry(7)
            .setup_max_read_atomic(1);
        self.qp.modify(&attr).context("move QP to RTS")
    }

    pub fn input_region_info(&self) -> MemoryRegionInfo {
        MemoryRegionInfo {
            addr: self.cuda.input().pointer(),
            rkey: self.input_mr.rkey(),
            size: self.cuda.input().allocation_size() as u32,
        }
    }

    pub fn read_input_head(&self) -> Result<u64> {
        self.cuda.read_input_head()
    }

    pub fn process(&self, input_tail: u64, output_head: u64, count: u32) -> Result<()> {
        self.cuda.process(input_tail, output_head, count)
    }

    pub fn write_output_batch(&mut self, remote: &MemoryRegionInfo, output_head: u64, count: u32) -> Result<()> {
        ensure!(count > 0, "output batch must not be empty");
        ensure!(
            count < RING_BUFFER_ELEMENTS as u32,
            "output batch exceeds ring capacity"
        );
        ensure!(
            remote.size as usize >= size_of::<RuntimeRing>(),
            "remote output region is smaller than the ring buffer"
        );

        let wr_id = self.next_write_id;
        self.next_write_id = self.next_write_id.wrapping_add(1);
        let local_base = self.cuda.output().pointer();
        let mask = RING_BUFFER_ELEMENTS as u64 - 1;
        let lkey = self.output_mr.lkey();
        let mut guard = self.qp.start_post_send();

        let first_index = (output_head & mask) as usize;
        let first_count = (count as usize).min(RING_BUFFER_ELEMENTS - first_index);
        let second_count = count as usize - first_count;
        post_output_segment(&mut guard, lkey, local_base, remote, first_index, first_count);
        if second_count > 0 {
            post_output_segment(&mut guard, lkey, local_base, remote, 0, second_count);
        }

        let head_offset = RuntimeRing::producer_head_offset() as u64;
        let head_write = guard
            .construct_wr(wr_id, WorkRequestFlags::Signaled)
            .setup_write(remote.rkey, remote.addr + head_offset);
        unsafe {
            head_write.setup_sge(lkey, local_base + head_offset, size_of::<u64>() as u32);
        }
        guard.post().context("post GPU output RDMA writes")?;
        self.wait_for_completion(wr_id)
    }

    fn wait_for_completion(&self, expected_wr_id: u64) -> Result<()> {
        loop {
            match self.cq.start_poll() {
                Ok(mut poller) => {
                    if let Some(completion) = poller.next() {
                        if completion.status() != WorkCompletionStatus::Success as u32 {
                            bail!(
                                "RDMA write failed: status={}, vendor_err={}",
                                completion.status(),
                                completion.vendor_err()
                            );
                        }
                        ensure!(
                            completion.wr_id() == expected_wr_id,
                            "unexpected RDMA completion {} (expected {expected_wr_id})",
                            completion.wr_id()
                        );
                        return Ok(());
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => std::hint::spin_loop(),
                Err(error) => return Err(error).context("poll RDMA completion queue"),
            }
        }
    }
}

fn post_output_segment<G: PostSendGuard>(
    guard: &mut G,
    lkey: u32,
    local_base: u64,
    remote: &MemoryRegionInfo,
    index: usize,
    count: usize,
) {
    let offset = RuntimeRing::slot_offset(index) as u64;
    let byte_count = count * size_of::<Slot>();
    let write = guard
        .construct_wr(0, WorkRequestFlags::none())
        .setup_write(remote.rkey, remote.addr + offset);
    unsafe {
        write.setup_sge(lkey, local_base + offset, byte_count as u32);
    }
}
