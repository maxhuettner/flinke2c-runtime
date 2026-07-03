use serde::{Deserialize, Serialize};
use sideway::ibverbs::address::Gid;

pub const MAX_ITEM_SIZE: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointBootstrap {
    pub dest: RdmaDestination,
    pub writable: MemoryRegionInfo,
    pub path_mtu: u32,
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
