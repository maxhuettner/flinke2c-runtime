use std::{
    mem::{offset_of, size_of},
    ptr::read_volatile,
};

use crate::ring_buffer::slot::Slot;

pub mod slot;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct RingBuffer<const N: usize> {
    producer_head: u64,
    consumer_tail: u64,
    slots: [Slot; N],
}

impl<const N: usize> RingBuffer<N> {
    pub const fn producer_head_offset() -> usize {
        offset_of!(RingBuffer<N>, producer_head)
    }

    pub const fn consumer_tail_offset() -> usize {
        offset_of!(RingBuffer<N>, consumer_tail)
    }

    pub const fn slots_offset() -> usize {
        offset_of!(RingBuffer<N>, slots)
    }

    pub const fn slot_offset(index: usize) -> usize {
        Self::slots_offset() + index * size_of::<Slot>()
    }

    /// SAFETY: If the RingBuffer only has zero-valid fields, this is safe to call and the resulting RingBuffer can be used as normal.
    /// When the RingBuffer implementation changes and non-zero-valid fields are added, this function must be updated to properly initialize those fields before returning the RingBuffer.
    pub fn new_boxed() -> Box<Self> {
        assert!(N.is_power_of_two(), "ring buffer size must be a power of 2");
        unsafe { Box::<Self>::new_zeroed().assume_init() }
    }

    pub fn head_idx(&self) -> u64 {
        unsafe { read_volatile(&self.producer_head) }
    }

    pub fn tail_idx(&self) -> u64 {
        unsafe { read_volatile(&self.consumer_tail) }
    }

    pub fn advance_head_by(&mut self, count: u64) -> anyhow::Result<()> {
        if count > self.available_write_slots() {
            return Err(anyhow::anyhow!(
                "cannot publish {count} elements into a ring with {} writable slots",
                self.available_write_slots()
            ));
        }
        self.producer_head = (self.head_idx() + count) & (N as u64 - 1);
        Ok(())
    }

    pub fn advance_tail_by(&mut self, count: u64) -> anyhow::Result<()> {
        if count > self.available_read_slots() {
            return Err(anyhow::anyhow!(
                "cannot consume {count} elements from a ring with {} readable slots",
                self.available_read_slots()
            ));
        }
        self.consumer_tail = (self.tail_idx() + count) & (N as u64 - 1);
        Ok(())
    }

    pub fn write_slot(&mut self, value: Slot) -> anyhow::Result<()> {
        if self.available_write_slots() == 0 {
            Err(anyhow::anyhow!("ring buffer is full"))?;
        }
        let slot_index = self.head_idx() as usize & (N - 1);
        self.slots[slot_index] = value;
        self.producer_head = (self.head_idx() + 1) & (N as u64 - 1);
        Ok(())
    }

    pub fn fill_slots(&mut self, value: Slot) {
        self.slots.fill(value);
    }

    pub fn read_slot(&mut self) -> Option<Slot> {
        if self.available_read_slots() == 0 {
            return None;
        }
        let slot_index = self.tail_idx() as usize & (N - 1);
        let value = self.slots[slot_index];
        self.consumer_tail = (self.tail_idx() + 1) & (N as u64 - 1);
        Some(value)
    }

    pub fn len(&self) -> usize {
        size_of::<Self>()
    }

    pub fn is_empty(&self) -> bool {
        self.available_read_slots() == 0
    }
}

impl<const N: usize> RingBuffer<N> {
    pub fn as_ptr(&self) -> *const Self {
        self as *const Self
    }

    pub fn slots_ptr(&self) -> *const Slot {
        self.slots.as_ptr()
    }

    pub fn available_write_slots(&self) -> u64 {
        let head = self.head_idx();
        let tail = self.tail_idx();
        (tail + N as u64 - head - 1) & (N as u64 - 1)
    }

    pub fn available_read_slots(&self) -> u64 {
        let head = self.head_idx();
        let tail = self.tail_idx();
        (head + N as u64 - tail) & (N as u64 - 1)
    }
}

#[cfg(test)]
mod tests {
    use crate::control_protocol::MAX_ITEM_SIZE;

    use super::*;

    #[test]
    fn basic_functionalities() {
        const N_ELEMENTS: usize = 2;

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        assert_eq!(rb.available_read_slots(), 0);
        assert_eq!(rb.available_write_slots(), N_ELEMENTS as u64 - 1);
        let value = "test".as_bytes();

        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..value.len()].copy_from_slice(value);

        for _ in 0..N_ELEMENTS as u64 - 1 {
            rb.write_slot(Slot {
                len: value.len() as u32,
                timestamp_ns: 0,
                value: payload,
            })
            .unwrap();
            assert!(rb.available_read_slots() > 0);
        }

        assert_eq!(rb.available_write_slots(), 0);
        assert!(rb.write_slot(Slot { len: 0, timestamp_ns: 0, value: payload }).is_err());

        for _ in 0..N_ELEMENTS as u64 - 1 {
            let slot = rb.read_slot().unwrap();
            assert_eq!(slot.len, value.len() as u32);
            assert_eq!(&slot.value[..slot.len as usize], value);
        }

        assert_eq!(rb.available_read_slots(), 0);
        assert!(rb.read_slot().is_none());
    }

    #[test]
    fn write_and_read() {
        const N_ELEMENTS: usize = 4;

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        let value = "test".as_bytes();

        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..value.len()].copy_from_slice(value);

        for _ in 0..N_ELEMENTS as u64 - 1 {
            rb.write_slot(Slot {
                len: value.len() as u32,
                timestamp_ns: 0,
                value: payload,
            })
            .unwrap();
        }

        for _ in 0..N_ELEMENTS as u64 - 1 {
            let slot = rb.read_slot().unwrap();
            assert_eq!(slot.len, value.len() as u32);
            assert_eq!(&slot.value[..slot.len as usize], value);
        }
    }

    #[test]
    fn wrapping_behavior() {
        const N_ELEMENTS: usize = 4;

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        let test_values = ["test1", "test2", "test3"].map(|s| s.as_bytes());
        let wrap_value = "test4".as_bytes();

        for value in test_values {
            let mut payload = [0u8; MAX_ITEM_SIZE];
            payload[..value.len()].copy_from_slice(value);
            rb.write_slot(Slot {
                len: value.len() as u32,
                timestamp_ns: 0,
                value: payload,
            })
            .unwrap();
        }

        assert_eq!(rb.available_write_slots(), 0);
        assert_eq!(rb.head_idx(), 3);
        assert_eq!(rb.tail_idx(), 0);

        let first = rb.read_slot().unwrap();
        assert_eq!(&first.value[..first.len as usize], test_values[0]);
        assert_eq!(rb.available_write_slots(), 1);
        assert_eq!(rb.head_idx(), 3);
        assert_eq!(rb.tail_idx(), 1);

        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..wrap_value.len()].copy_from_slice(wrap_value);
        rb.write_slot(Slot {
            len: wrap_value.len() as u32,
            timestamp_ns: 0,
            value: payload,
        })
        .unwrap();

        assert_eq!(rb.available_write_slots(), 0);
        assert_eq!(rb.head_idx(), 0);
        assert_eq!(rb.tail_idx(), 1);

        for expected in [test_values[1], test_values[2], wrap_value] {
            let slot = rb.read_slot().unwrap();
            assert_eq!(slot.len, expected.len() as u32);
            assert_eq!(&slot.value[..slot.len as usize], expected);
        }

        assert_eq!(rb.available_read_slots(), 0);
        assert_eq!(rb.head_idx(), 0);
        assert_eq!(rb.tail_idx(), 0);
        assert!(rb.read_slot().is_none());
    }

    #[test]
    fn publishes_completed_batches() {
        let mut rb = RingBuffer::<8>::new_boxed();

        rb.advance_head_by(6).unwrap();
        assert_eq!(rb.head_idx(), 6);
        assert_eq!(rb.available_read_slots(), 6);
        assert!(rb.advance_head_by(2).is_err());

        rb.advance_tail_by(4).unwrap();
        rb.advance_head_by(4).unwrap();
        assert_eq!(rb.head_idx(), 2);
        assert_eq!(rb.available_read_slots(), 6);

        rb.advance_tail_by(5).unwrap();
        assert_eq!(rb.tail_idx(), 1);
        assert_eq!(rb.available_read_slots(), 1);
        assert!(rb.advance_tail_by(2).is_err());
    }

    #[test]
    fn pointer_calculations() {
        const N_ELEMENTS: usize = 4;

        let rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        let base_ptr = rb.as_ptr() as usize;
        let slots_offset = RingBuffer::<N_ELEMENTS>::slots_offset();

        assert_eq!(base_ptr + slots_offset, rb.slots_ptr() as usize);
    }
}
