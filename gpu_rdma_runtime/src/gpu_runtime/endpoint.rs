use std::collections::VecDeque;
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

use crate::constants::{RECEIVE_WR_DEPTH, RING_BUFFER_ELEMENTS};
use crate::control_protocol::{MemoryRegionInfo, RdmaDestination};
use crate::ring_buffer::{slot::Slot, RingBuffer};

use super::cuda::{CudaBatch, CudaProcessSpec, CudaRuntime};
use super::memory_region::DmaBufMemoryRegion;

type RuntimeRing = RingBuffer<RING_BUFFER_ELEMENTS>;
const OUTPUT_SIGNAL_INTERVAL: usize = 32;

pub struct GpuRdmaEndpoint {
    pub ctx: Arc<DeviceContext>,
    pub input_qp: ExtendedQueuePair,
    pub output_qp: ExtendedQueuePair,
    send_cq: Arc<ExtendedCompletionQueue>,
    receive_cq: Arc<ExtendedCompletionQueue>,
    input_mr: DmaBufMemoryRegion,
    output_mr: DmaBufMemoryRegion,
    _pd: Arc<ProtectionDomain>,
    cuda: CudaRuntime,
    next_write_id: u64,
    output_batches_since_signal: usize,
    pending_output_completions: VecDeque<u64>,
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
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ
            | ibv_access_flags::IBV_ACCESS_RELAXED_ORDERING)
            .0 as i32;
        let local_access = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 as i32;
        let input_mr = unsafe { DmaBufMemoryRegion::register(Arc::clone(&pd), cuda.input(), remote_access) }
            .context("register GPUDirect input ring")?;
        let output_mr = unsafe { DmaBufMemoryRegion::register(Arc::clone(&pd), cuda.output(), local_access) }
            .context("register GPUDirect output ring")?;

        let send_cq = ctx
            .create_cq_builder()
            .build_ex()
            .context("create send completion queue")?;
        let receive_cq = ctx
            .create_cq_builder()
            .build_ex()
            .context("create receive completion queue")?;
        let mut input_qp = pd
            .create_qp_builder()
            .setup_max_inline_data(128)
            .setup_max_recv_wr(RECEIVE_WR_DEPTH as u32)
            .setup_max_send_wr(1024)
            .setup_send_cq(GenericCompletionQueue::from(Arc::clone(&send_cq)))
            .setup_recv_cq(GenericCompletionQueue::from(Arc::clone(&receive_cq)))
            .build_ex()
            .context("create RDMA queue pair")?;

        let mut output_qp = pd
            .create_qp_builder()
            .setup_max_inline_data(128)
            .setup_max_recv_wr(1)
            .setup_max_send_wr(1024)
            .setup_send_cq(GenericCompletionQueue::from(Arc::clone(&send_cq)))
            .setup_recv_cq(GenericCompletionQueue::from(Arc::clone(&receive_cq)))
            .build_ex()
            .context("create RDMA output queue pair")?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::Init)
            .setup_pkey_index(0)
            .setup_port(ib_port)
            .setup_access_flags(AccessFlags::RemoteWrite | AccessFlags::RemoteRead);
        input_qp.modify(&attr).context("move input QP to INIT")?;
        post_receive_notifications(&mut input_qp, RECEIVE_WR_DEPTH)?;
        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::Init)
            .setup_pkey_index(0)
            .setup_port(ib_port)
            .setup_access_flags(AccessFlags::RemoteWrite | AccessFlags::RemoteRead);
        output_qp.modify(&attr).context("move output QP to INIT")?;

        Ok(Self {
            ctx,
            input_qp,
            output_qp,
            send_cq,
            receive_cq,
            input_mr,
            output_mr,
            _pd: pd,
            cuda,
            next_write_id: 0,
            output_batches_since_signal: 0,
            pending_output_completions: VecDeque::new(),
        })
    }

    pub fn connect_input(
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
        self.input_qp.modify(&attr).context("move input QP to RTR")?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::ReadyToSend)
            .setup_sq_psn(packet_seq_num)
            .setup_timeout(12)
            .setup_retry_cnt(7)
            .setup_rnr_retry(7)
            .setup_max_read_atomic(1);
        self.input_qp.modify(&attr).context("move input QP to RTS")
    }

    pub fn connect_output(
        &mut self,
        remote: &RdmaDestination,
        ib_port: u8,
        packet_seq_num: u32,
        mtu: Mtu,
        gid_index: u8,
    ) -> Result<()> {
        connect_qp(&mut self.output_qp, remote, ib_port, packet_seq_num, mtu, gid_index)
    }

    pub fn input_region_info(&self) -> MemoryRegionInfo {
        MemoryRegionInfo {
            addr: self.cuda.input().pointer(),
            rkey: self.input_mr.rkey(),
            size: self.cuda.input().allocation_size() as u32,
        }
    }

    pub fn output_region_info(&self) -> MemoryRegionInfo {
        MemoryRegionInfo {
            addr: self.cuda.output().pointer(),
            rkey: self.output_mr.rkey(),
            size: self.cuda.output().allocation_size() as u32,
        }
    }

    pub fn pipeline_depth(&self) -> usize {
        self.cuda.pipeline_depth()
    }

    pub fn submit_process(&mut self, input_tail: u64, output_head: u64, count: u32, spec: CudaProcessSpec) -> Result<CudaBatch> {
        self.cuda.submit(input_tail, output_head, count, spec)
    }

    pub fn process_complete(&self, batch: CudaBatch) -> Result<bool> {
        self.cuda.is_complete(batch)
    }

    pub fn wait_for_process(&self, batch: CudaBatch) -> Result<()> {
        self.cuda.wait(batch)
    }

    pub fn flush_input_writes(&self) -> Result<()> {
        self.cuda.flush_gpudirect_writes()
    }

    pub fn wait_for_input_batch(&mut self) -> Result<u32> {
        loop {
            if let Some(count) = self.try_input_batch()? {
                return Ok(count);
            }
            std::hint::spin_loop();
        }
    }

    pub fn try_input_batch(&mut self) -> Result<Option<u32>> {
        let completion = match self.receive_cq.start_poll() {
            Ok(mut poller) => poller
                .next()
                .map(|completion| (completion.status(), completion.vendor_err(), completion.imm_data())),
            Err(PollCompletionQueueError::CompletionQueueEmpty) => return Ok(None),
            Err(error) => return Err(error).context("poll RDMA input notifications"),
        };
        let Some((status, vendor_err, immediate)) = completion else {
            return Ok(None);
        };
        if status != WorkCompletionStatus::Success as u32 {
            bail!("RDMA input notification failed: status={status}, vendor_err={vendor_err}");
        }
        let count = u32::from_be(immediate);
        ensure!(count > 0, "RDMA input notification contains an empty batch");
        post_receive_notifications(&mut self.input_qp, 1)?;
        Ok(Some(count))
    }

    pub fn write_output_batch(
        &mut self,
        remote: &MemoryRegionInfo,
        output_head: u64,
        count: u32,
        phase_end: bool,
    ) -> Result<()> {
        ensure!(count > 0, "output batch must not be empty");
        ensure!(
            count < RING_BUFFER_ELEMENTS as u32,
            "output batch exceeds ring capacity"
        );
        ensure!(
            remote.size as usize >= size_of::<RuntimeRing>(),
            "remote output region is smaller than the ring buffer"
        );

        let signaled = phase_end || self.output_batches_since_signal + 1 >= OUTPUT_SIGNAL_INTERVAL;
        let wr_id = if signaled {
            let id = self.next_write_id;
            self.next_write_id = self.next_write_id.wrapping_add(1);
            Some(id)
        } else {
            None
        };
        let local_base = self.cuda.output().pointer();
        let mask = RING_BUFFER_ELEMENTS as u64 - 1;
        let lkey = self.output_mr.lkey();
        let mut guard = self.output_qp.start_post_send();

        let first_index = (output_head & mask) as usize;
        let first_count = (count as usize).min(RING_BUFFER_ELEMENTS - first_index);
        let second_count = count as usize - first_count;
        if second_count > 0 {
            post_output_segment(&mut guard, lkey, local_base, remote, first_index, first_count, None);
            post_output_segment(
                &mut guard,
                lkey,
                local_base,
                remote,
                0,
                second_count,
                Some((wr_id.unwrap_or(0), signaled, count)),
            );
        } else {
            post_output_segment(
                &mut guard,
                lkey,
                local_base,
                remote,
                first_index,
                first_count,
                Some((wr_id.unwrap_or(0), signaled, count)),
            );
        }
        guard.post().context("post GPU output RDMA writes")?;

        if let Some(wr_id) = wr_id {
            self.pending_output_completions.push_back(wr_id);
            self.output_batches_since_signal = 0;
        } else {
            self.output_batches_since_signal += 1;
        }
        self.reap_output_completions().map(|_| ())
    }

    pub fn finish_output(&mut self) -> Result<()> {
        ensure!(
            self.output_batches_since_signal == 0,
            "the final output batch must request a completion"
        );
        while !self.pending_output_completions.is_empty() {
            if self.reap_output_completions()? == 0 {
                std::hint::spin_loop();
            }
        }
        Ok(())
    }

    fn reap_output_completions(&mut self) -> Result<usize> {
        let completions = match self.send_cq.start_poll() {
            Ok(mut poller) => poller
                .by_ref()
                .map(|completion| (completion.status(), completion.vendor_err(), completion.wr_id()))
                .collect::<Vec<_>>(),
            Err(PollCompletionQueueError::CompletionQueueEmpty) => return Ok(0),
            Err(error) => return Err(error).context("poll RDMA completion queue"),
        };
        for (status, vendor_err, wr_id) in &completions {
            if *status != WorkCompletionStatus::Success as u32 {
                bail!("RDMA write failed: status={status}, vendor_err={vendor_err}");
            }
            let expected = self
                .pending_output_completions
                .pop_front()
                .context("received an unexpected RDMA output completion")?;
            ensure!(
                *wr_id == expected,
                "unexpected RDMA completion {wr_id} (expected {expected})"
            );
        }
        Ok(completions.len())
    }
}

fn connect_qp(
    qp: &mut ExtendedQueuePair,
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
    qp.modify(&attr).context("move QP to RTR")?;
    let mut attr = QueuePairAttribute::new();
    attr.setup_state(QueuePairState::ReadyToSend)
        .setup_sq_psn(packet_seq_num)
        .setup_timeout(12)
        .setup_retry_cnt(7)
        .setup_rnr_retry(7)
        .setup_max_read_atomic(1);
    qp.modify(&attr).context("move QP to RTS")
}

fn post_receive_notifications(qp: &mut ExtendedQueuePair, count: usize) -> Result<()> {
    let mut guard = qp.start_post_recv();
    for _ in 0..count {
        guard.construct_wr(0);
    }
    guard.post().context("post RDMA input notification receives")
}

fn post_output_segment<G: PostSendGuard>(
    guard: &mut G,
    lkey: u32,
    local_base: u64,
    remote: &MemoryRegionInfo,
    index: usize,
    count: usize,
    notification: Option<(u64, bool, u32)>,
) {
    let offset = RuntimeRing::slot_offset(index) as u64;
    let byte_count = count * size_of::<Slot>();
    if let Some((wr_id, signaled, immediate)) = notification {
        let flags = if signaled {
            WorkRequestFlags::Signaled
        } else {
            WorkRequestFlags::none()
        };
        let write =
            guard
                .construct_wr(wr_id, flags)
                .setup_write_imm(remote.rkey, remote.addr + offset, immediate.to_be());
        unsafe {
            write.setup_sge(lkey, local_base + offset, byte_count as u32);
        }
    } else {
        let write = guard
            .construct_wr(0, WorkRequestFlags::none())
            .setup_write(remote.rkey, remote.addr + offset);
        unsafe {
            write.setup_sge(lkey, local_base + offset, byte_count as u32);
        }
    }
}
