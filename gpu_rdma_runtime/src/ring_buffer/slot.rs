use std::fmt::{self, Display, Formatter};

use crate::control_protocol::MAX_ITEM_SIZE;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub len: u32,
    pub value: [u8; MAX_ITEM_SIZE],
}

impl Default for Slot {
    fn default() -> Self {
        Slot {
            len: 0,
            value: [0; MAX_ITEM_SIZE],
        }
    }
}

impl Display for Slot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let len = (self.len as usize).min(MAX_ITEM_SIZE);
        let bytes = &self.value[..len];

        match std::str::from_utf8(bytes) {
            Ok(s) => write!(f, "Slot {{ len: {}, value: {:?} }}", self.len, s),
            Err(_) => write!(f, "Slot {{ len: {}, value_bytes: {:?} }}", self.len, bytes),
        }
    }
}