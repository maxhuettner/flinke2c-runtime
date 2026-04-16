use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};
use sideway::ibverbs::address::Gid;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaDestination {
    pub gid: Gid,
    pub qp_number: u32,
    pub packet_seq_num: u32,
}
