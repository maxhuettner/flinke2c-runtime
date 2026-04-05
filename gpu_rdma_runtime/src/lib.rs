use anyhow::Result;

pub mod control_helpers;
pub mod control_protocol;
pub mod rdma;
pub mod wire_codec;

pub struct TestStruct {
    pub a: u32,
    pub b: String,
}

impl TestStruct {
    pub fn new(a: u32, b: String) -> Self {
        Self { a, b }
    }

    pub fn to_memory(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.a.to_le_bytes());
        bytes.extend_from_slice(&(self.b.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.b.as_bytes());
        bytes
    }

    pub fn from_memory(bytes: &[u8]) -> Result<Self> {
        let a = u32::from_le_bytes(bytes[0..4].try_into()?);
        let b_len = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
        let b = String::from_utf8(bytes[8..8 + b_len].to_vec())?;
        Ok(Self { a, b })
    }
}
