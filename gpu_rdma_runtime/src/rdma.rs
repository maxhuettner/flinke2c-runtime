use std::sync::Arc;

use crate::constants::RING_BUFFER_ELEMENTS;
use crate::{
    control_protocol::{MemoryRegionInfo, RdmaDestination},
    ring_buffer::{slot::Slot, RingBuffer},
};
use anyhow::{bail, Context, Result};
use sideway::ibverbs::{
    address::AddressHandleAttribute,
    completion::{ExtendedCompletionQueue, GenericCompletionQueue, PollCompletionQueueError, WorkCompletionStatus},
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, Mtu},
    memory_region::MemoryRegion,
    protection_domain::ProtectionDomain,
    queue_pair::{
        ExtendedQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState, SetScatterGatherEntry,
        WorkRequestFlags,
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
            .setup_max_dest_read_atomic(0)
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
            .setup_max_read_atomic(0);

        self.qp
            .modify(&attr)
            .context("Failed to modify QP to Ready-To-Send (RTS) state")?;

        Ok(())
    }

    pub fn write_slot_remote(&mut self, remote: &MemoryRegionInfo) -> Result<u64> {
        // let lkey = self.send_mr.lkey();

        // debug_assert!(local_offset + len <= self.send_rb.len());
        // debug_assert!(remote_offset + len <= remote.size as usize);

        //

        // let mut guard = self.qp.start_post_send();

        // let wr = guard
        //     .construct_wr(wr_id, WorkRequestFlags::Signaled)
        //     .setup_write(remote.rkey, remote.addr + remote_offset as u64);

        // unsafe { wr.setup_sge(lkey, base_ptr + local_offset as u64, len as u32) };

        // guard.post().context("failed to post RDMA write")?;

        // Ok(wr_id)
        let lkey = self.send_mr.lkey();
        let wr_id = self.create_write_id();
        let head_wr_id = self.create_write_id();

        let segments = self.send_rb.wrapping_read_slot_base_ptrs(1)?;
        let base_ptr = segments.first_part.ptr as u64;
        let offset = segments.first_part.offset as u64;

        let mut guard = self.qp.start_post_send();

        let wr = guard
            .construct_wr(wr_id, WorkRequestFlags::none())
            .setup_write(remote.rkey, remote.addr + offset);
        unsafe { wr.setup_sge(lkey, base_ptr, std::mem::size_of::<Slot>() as u32 * 1) };

        let head_update_wr = guard
            .construct_wr(head_wr_id, WorkRequestFlags::Signaled)
            .setup_write(remote.rkey, remote.addr + self.send_rb.abs_head_offset() as u64);
        unsafe { head_update_wr.setup_sge(lkey, self.send_rb.abs_head_ptr() as u64, 8) };

        guard.post().context("failed to post RDMA write")?;

        Ok(head_wr_id)
    }

    pub fn complete_sent_slot_local(&mut self) {
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

    pub fn wait_for_completion(&self, expected_wr_id: u64) -> Result<()> {
        loop {
            match self.cq.start_poll() {
                Ok(mut poller) => {
                    for wc in &mut poller {
                        if wc.wr_id() != expected_wr_id {
                            continue;
                        }

                        if wc.status() != WorkCompletionStatus::Success as u32 {
                            bail!(
                                "send completion failed: status={}, vendor_err={}",
                                wc.status(),
                                wc.vendor_err()
                            );
                        }

                        return Ok(());
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
