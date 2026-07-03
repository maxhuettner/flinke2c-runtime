use std::collections::HashSet;
use std::sync::Arc;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::{
    control_protocol::{MemoryRegionInfo, RdmaDestination},
    ring_buffer::{slot::Slot, RingBuffer},
};
use anyhow::{bail, ensure, Context, Result};
use sideway::ibverbs::{
    address::AddressHandleAttribute,
    completion::{ExtendedCompletionQueue, GenericCompletionQueue, PollCompletionQueueError, WorkCompletionStatus},
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, Mtu},
    memory_region::MemoryRegion,
    protection_domain::ProtectionDomain,
    queue_pair::{
        ExtendedQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState, SetInlineData,
        SetScatterGatherEntry, WorkRequestFlags,
    },
    AccessFlags,
};

pub struct RdmaEndpoint {
    pub ctx: Arc<DeviceContext>,
    _pd: Arc<ProtectionDomain>,
    send_rb: Box<RingBuffer<RING_BUFFER_ELEMENTS>>,
    send_mr: Arc<MemoryRegion>,
    recv_rb: Box<RingBuffer<RING_BUFFER_ELEMENTS>>,
    recv_mr: Arc<MemoryRegion>,
    cq: Arc<ExtendedCompletionQueue>,
    pub qp: ExtendedQueuePair,
    current_write_id: u64,
    completed_write_ids: HashSet<u64>,
    posted_slots: u64,
}

impl RdmaEndpoint {
    pub fn build(ib_device: Option<&str>, ib_port: u8) -> anyhow::Result<Self> {
        let device_list = DeviceList::new().expect("Failed to get IB devices list");
        let device = match ib_device {
            Some(ib_dev) => device_list
                .iter()
                .find(|dev| dev.name().eq(&ib_dev))
                .with_context(|| format!("IB device {ib_dev} not found"))?,
            None => device_list.iter().next().context("No IB device found")?,
        };

        let context = device
            .open()
            .with_context(|| format!("Couldn't get context for {}", device.name()))?;

        let pd = context.alloc_pd()?;

        let mr_access = AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead;

        let send_rb = RingBuffer::new_boxed();
        let send_mr = unsafe { pd.reg_mr(send_rb.as_ptr() as usize, send_rb.len(), mr_access) }
            .context("Failed to register send memory region")?;

        let recv_rb = RingBuffer::new_boxed();
        let recv_mr = unsafe { pd.reg_mr(recv_rb.as_ptr() as usize, recv_rb.len(), mr_access) }
            .context("Failed to register receive memory region")?;

        let cq_builder = context.create_cq_builder();
        let cq = cq_builder.build_ex()?;
        let cq_for_qp = GenericCompletionQueue::from(Arc::clone(&cq));

        let mut builder = pd.create_qp_builder();

        let mut qp = builder
            .setup_max_inline_data(128)
            .setup_max_recv_wr(1024)
            .setup_max_send_wr(1024)
            .setup_send_cq(cq_for_qp.clone())
            .setup_recv_cq(cq_for_qp)
            .build_ex()?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::Init)
            .setup_pkey_index(0)
            .setup_port(ib_port)
            .setup_access_flags(AccessFlags::RemoteWrite | AccessFlags::RemoteRead);
        qp.modify(&attr)?;

        Ok(RdmaEndpoint {
            ctx: context,
            _pd: pd,
            send_rb,
            send_mr,
            recv_rb,
            recv_mr,
            cq,
            qp,
            current_write_id: 0,
            completed_write_ids: HashSet::new(),
            posted_slots: 0,
        })
    }

    pub fn connect(
        &mut self,
        remote_context: &RdmaDestination,
        ib_port: u8,
        packet_seq_num: u32,
        mtu: Mtu,
        sl: u8,
        gid_idx: u8,
    ) -> Result<()> {
        let mut ah_attr = AddressHandleAttribute::new();
        ah_attr
            .setup_dest_lid(0)
            .setup_port(ib_port)
            .setup_service_level(sl)
            .setup_grh_src_gid_index(gid_idx)
            .setup_grh_dest_gid(&remote_context.gid)
            .setup_grh_hop_limit(1);

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::ReadyToReceive)
            .setup_path_mtu(mtu)
            .setup_dest_qp_num(remote_context.qp_number)
            .setup_rq_psn(remote_context.packet_seq_num)
            .setup_max_dest_read_atomic(1)
            .setup_min_rnr_timer(0)
            .setup_address_vector(&ah_attr);

        self.qp
            .modify(&attr)
            .context("Failed to modify QP to Ready-To-Receive (RTR) state")?;

        let mut attr = QueuePairAttribute::new();
        attr.setup_state(QueuePairState::ReadyToSend)
            .setup_sq_psn(packet_seq_num)
            .setup_timeout(12)
            .setup_retry_cnt(7)
            .setup_rnr_retry(7)
            .setup_max_read_atomic(1);

        self.qp
            .modify(&attr)
            .context("Failed to modify QP to Ready-To-Send (RTS) state")?;

        Ok(())
    }

    pub fn write_slot_remote(&mut self, remote: &MemoryRegionInfo) -> Result<u64> {
        self.write_slots_remote(remote, 1)
    }

    pub fn write_slots_remote(&mut self, remote: &MemoryRegionInfo, count: usize) -> Result<u64> {
        ensure!(count > 0, "RDMA batch must not be empty");
        ensure!(count < RING_BUFFER_ELEMENTS, "RDMA batch exceeds ring capacity");
        ensure!(
            self.posted_slots + count as u64 <= self.send_rb.available_read_slots(),
            "not enough unposted slots for RDMA batch"
        );
        ensure!(
            remote.size as usize >= self.send_rb.len(),
            "remote input region is smaller than the ring buffer"
        );

        let lkey = self.send_mr.lkey();
        let head_wr_id = self.create_write_id();
        let slot_index = ((self.send_rb.tail_idx() + self.posted_slots) & (RING_BUFFER_ELEMENTS as u64 - 1)) as usize;
        let first_count = count.min(RING_BUFFER_ELEMENTS - slot_index);
        let second_count = count - first_count;
        let head = self.send_rb.head_idx().to_ne_bytes();
        let mut guard = self.qp.start_post_send();

        post_slot_segment(
            &mut guard,
            lkey,
            self.send_rb.slots_ptr(),
            remote,
            slot_index,
            first_count,
        );
        if second_count > 0 {
            post_slot_segment(&mut guard, lkey, self.send_rb.slots_ptr(), remote, 0, second_count);
        }

        let head_update_wr = guard
            .construct_wr(head_wr_id, WorkRequestFlags::Signaled)
            .setup_write(remote.rkey, remote.addr + self.send_rb.abs_head_offset() as u64);
        head_update_wr.setup_inline_data(&head);

        guard.post().context("failed to post RDMA write")?;
        self.posted_slots += count as u64;

        Ok(head_wr_id)
    }

    pub fn complete_round_trip_local(&mut self) {
        assert!(self.posted_slots > 0, "no outstanding slot to complete");
        self.posted_slots -= 1;
        self.send_rb.advance_tail();
    }

    fn create_write_id(&mut self) -> u64 {
        let id = self.current_write_id;
        self.current_write_id = self.current_write_id.wrapping_add(1);
        id
    }

    pub fn memory_region_info(&self) -> MemoryRegionInfo {
        MemoryRegionInfo {
            addr: self.recv_mr.get_ptr() as u64,
            rkey: self.recv_mr.rkey(),
            size: self.recv_mr.region_len() as u32,
        }
    }

    pub fn write_slot_local(&mut self, value: Slot) -> Result<()> {
        self.send_rb.write_slot(value)
    }

    pub fn read_slot_local(&mut self) -> Option<Slot> {
        self.recv_rb.read_slot()
    }

    pub fn wait_for_completion(&mut self, expected_wr_id: u64) -> Result<()> {
        loop {
            if self.completed_write_ids.remove(&expected_wr_id) {
                return Ok(());
            }

            match self.cq.start_poll() {
                Ok(mut poller) => {
                    for wc in &mut poller {
                        if wc.status() != WorkCompletionStatus::Success as u32 {
                            bail!(
                                "send completion failed: status={}, vendor_err={}",
                                wc.status(),
                                wc.vendor_err()
                            );
                        }
                        self.completed_write_ids.insert(wc.wr_id());
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {
                    std::hint::spin_loop();
                }
                Err(err) => {
                    return Err(err).context("failed to poll completion queue");
                }
            }
        }
    }
}

fn post_slot_segment<G: PostSendGuard>(
    guard: &mut G,
    lkey: u32,
    slots: *const Slot,
    remote: &MemoryRegionInfo,
    index: usize,
    count: usize,
) {
    let byte_count = count * std::mem::size_of::<Slot>();
    let offset = RingBuffer::<RING_BUFFER_ELEMENTS>::slot_offset(index) as u64;
    let write = guard
        .construct_wr(0, WorkRequestFlags::none())
        .setup_write(remote.rkey, remote.addr + offset);
    unsafe {
        write.setup_sge(lkey, slots.add(index) as u64, byte_count as u32);
    }
}
