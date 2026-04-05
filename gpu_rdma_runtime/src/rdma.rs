use std::{sync::Arc, thread, time::Duration};

use crate::control_protocol::{MemoryRegionInfo, RdmaDestination, Slot};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sideway::ibverbs::{
    address::{AddressHandleAttribute, Gid},
    completion::{
        ExtendedCompletionQueue, ExtendedWorkCompletion, GenericCompletionQueue, PollCompletionQueueError,
        WorkCompletionStatus,
    },
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, Mtu},
    memory_region::MemoryRegion,
    protection_domain::ProtectionDomain,
    queue_pair::{
        ExtendedQueuePair, PostSendError, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState,
        SetScatterGatherEntry, WorkRequestFlags,
    },
    AccessFlags,
};

const RING_BUFFER_ELEMENTS: usize = 16;
const MEM_REGION_SIZE: usize = size_of::<Slot>() * RING_BUFFER_ELEMENTS;

pub struct RdmaEndpoint {
    pub ctx: Arc<DeviceContext>,
    _pd: Arc<ProtectionDomain>,
    _send_buf: Arc<Vec<u8>>,
    send_mr: Arc<MemoryRegion>,
    _recv_buf: Arc<Vec<u8>>,
    recv_mr: Arc<MemoryRegion>,
    cq: Arc<ExtendedCompletionQueue>,
    pub qp: ExtendedQueuePair,
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

        let send_buf = Arc::new(vec![0; MEM_REGION_SIZE]);
        let send_mr = unsafe { pd.reg_mr(send_buf.as_ptr() as usize, send_buf.len(), mr_access) }
            .context("Failed to register send memory region")?;

        let recv_buf = Arc::new(vec![0; MEM_REGION_SIZE]);
        let recv_mr = unsafe { pd.reg_mr(recv_buf.as_ptr() as usize, recv_buf.len(), mr_access) }
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
            _send_buf: send_buf,
            send_mr,
            _recv_buf: recv_buf,
            recv_mr,
            cq,
            qp,
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

    pub fn write_slot_remote(&mut self, remote: &MemoryRegionInfo, write_id: u64, slot_id: u32) -> Result<()> {
        let mut guard = self.qp.start_post_send();
        let lkey = self.send_mr.lkey();
        let base_ptr = self.send_mr.get_ptr() as u64;

        let len = size_of::<Slot>();
        let local_offset = (slot_id as usize % RING_BUFFER_ELEMENTS) * len;
        let remote_offset = (slot_id as usize % RING_BUFFER_ELEMENTS) * len;

        debug_assert!(local_offset + len <= MEM_REGION_SIZE);
        debug_assert!(remote_offset + len <= remote.size as usize);

        let wr = guard
            .construct_wr(write_id, WorkRequestFlags::Signaled)
            .setup_write(remote.rkey, remote.addr + remote_offset as u64);

        unsafe { wr.setup_sge(lkey, base_ptr + local_offset as u64, len as u32) };

        guard.post().context("failed to post RDMA write")
    }

    // pub fn write(&mut self, remote: &MemoryRegionInfo, write_id: u64) -> Result<()> {
    //     let mut guard = self.qp.start_post_send();
    //     let lkey = self.send_mr.lkey();
    //     let ptr = self.send_mr.get_ptr() as u64;
    //     let len = remote.size;

    //     let wr = guard
    //         .construct_wr(write_id, WorkRequestFlags::Signaled)
    //         .setup_write(remote.rkey, remote.addr);

    //     unsafe { wr.setup_sge(lkey, ptr, len) };

    //     guard.post().context("failed to post RDMA write")
    // }

    // pub fn read(&mut self, remote: &MemoryRegionInfo, read_id: u64) -> Result<()> {
    //     let mut guard = self.qp.start_post_send();
    //     let lkey = self.recv_mr.lkey();
    //     let ptr = self.recv_mr.get_ptr() as u64;
    //     let len = self.size;

    //     let wr = guard
    //         .construct_wr(read_id, WorkRequestFlags::Signaled)
    //         .setup_read(remote.rkey, remote.addr);

    //     unsafe { wr.setup_sge(lkey, ptr, len) };

    //     guard.post().context("failed to post RDMA read")
    // }

    pub fn memory_region_info(&self) -> MemoryRegionInfo {
        MemoryRegionInfo {
            addr: self.recv_mr.get_ptr() as u64,
            rkey: self.recv_mr.rkey(),
            size: MEM_REGION_SIZE as u32,
        }
    }

    pub fn write_slot_local(&mut self, slot_id: u32, value: Slot) {
        let slot_index = slot_id as usize % RING_BUFFER_ELEMENTS;
        unsafe {
            (self.send_mr.get_ptr() as *mut Slot)
                .add(slot_index)
                .write_volatile(value);
        }
    }

    pub fn local_send_slot_mut(&mut self, slot_id: u32) -> &mut Slot {
        let slot_index = slot_id as usize % RING_BUFFER_ELEMENTS;
        unsafe { &mut *((self.send_mr.get_ptr() as *mut Slot).add(slot_index)) }
    }

    pub fn read_slot_local(&self, slot_id: u32) -> Slot {
        let slot_index = slot_id as usize % RING_BUFFER_ELEMENTS;
        unsafe { (self.recv_mr.get_ptr() as *const Slot).add(slot_index).read_volatile() }
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
