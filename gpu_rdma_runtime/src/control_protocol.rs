use serde::{Deserialize, Serialize};
use sideway::ibverbs::address::Gid;

pub const MAX_ITEM_SIZE: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClientRole {
    Pre,
    Post,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BootstrapHello {
    pub role: ClientRole,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InputDone {
    pub done: bool,
    pub slots: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointBootstrap {
    pub dest: RdmaDestination,
    pub writable: MemoryRegionInfo,
    pub path_mtu: u32,
    pub processing: ProcessingSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSpec {
    pub function: ProcessingFunction,
    pub field_index: u32,
    pub fields: Vec<WireFieldType>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProcessingFunction {
    Increment,
    Impute,
    CurrencyConversion,
    BlackScholes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WireFieldType {
    Int32,
    Int64,
    DecimalBytes,
    // Fixed 8-byte scale-3 unscaled decimal, no length prefix - unlike
    // DecimalBytes above. Only IMPUTE's price field (index 0) uses this;
    // see process_function.cu's BidView::price_unscaled and the README's
    // imputation section for why price specifically moved off the
    // variable-length format the other decimal-bearing paths still use.
    DecimalUnscaledI64,
    Bytes,
    TimestampMillis,
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
