use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};
use sideway::ibverbs::address::Gid;
use sideway::ibverbs::device_context::Mtu;

pub const MAX_ITEM_SIZE: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointBootstrap {
    pub dest: RdmaDestination,
    pub writable: MemoryRegionInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegionInfo {
    pub addr: u64,
    pub rkey: u32,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Slot {
    pub seq: u64,
    pub len: u32,
    pub value: [u8; MAX_ITEM_SIZE],
}

impl Display for Slot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let len = (self.len as usize).min(MAX_ITEM_SIZE);
        let bytes = &self.value[..len];

        match std::str::from_utf8(bytes) {
            Ok(s) => write!(f, "Slot {{ seq: {}, len: {}, value: {:?} }}", self.seq, self.len, s),
            Err(_) => write!(
                f,
                "Slot {{ seq: {}, len: {}, value_bytes: {:?} }}",
                self.seq, self.len, bytes
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaDestination {
    pub gid: Gid,
    pub qp_number: u32,
    pub packet_seq_num: u32,
}
