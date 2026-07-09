use anyhow::Result;
use std::{mem::offset_of, ptr::read_volatile};

use crate::ring_buffer::slot::Slot;

pub mod slot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSegment {
    pub offset: usize,
    pub ptr: *const Slot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSegments {
    pub first_part: SlotSegment,
    pub second_part: Option<SlotSegment>,
}

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

    pub fn is_full(&self) -> bool {
        let head = self.head_idx();
        let tail = self.tail_idx();

        ((head + 1) & (N as u64 - 1)) == tail
    }

    pub fn is_empty(&self) -> bool {
        self.head_idx() == self.tail_idx()
    }

    pub fn head_idx(&self) -> u64 {
        unsafe { read_volatile(&self.producer_head) }
    }

    pub fn tail_idx(&self) -> u64 {
        unsafe { read_volatile(&self.consumer_tail) }
    }

    pub fn advance_head(&mut self) {
        let head = self.head_idx();
        self.producer_head = (head + 1) & (N as u64 - 1)
    }

    pub fn advance_head_by(&mut self, count: u64) -> Result<()> {
        if count > self.available_write_slots() {
            return Err(anyhow::anyhow!(
                "cannot publish {count} elements into a ring with {} writable slots",
                self.available_write_slots()
            ));
        }
        self.producer_head = (self.head_idx() + count) & (N as u64 - 1);
        Ok(())
    }

    pub fn advance_tail(&mut self) {
        let tail = self.tail_idx();
        self.consumer_tail = (tail + 1) & (N as u64 - 1)
    }

    pub fn advance_tail_by(&mut self, count: u64) -> Result<()> {
        if count > self.available_read_slots() {
            return Err(anyhow::anyhow!(
                "cannot consume {count} elements from a ring with {} readable slots",
                self.available_read_slots()
            ));
        }
        self.consumer_tail = (self.tail_idx() + count) & (N as u64 - 1);
        Ok(())
    }

    pub fn write_slot(&mut self, value: Slot) -> Result<()> {
        if self.is_full() {
            Err(anyhow::anyhow!("ring buffer is full"))?;
        }
        let slot_index = self.head_idx() as usize & (N - 1);
        self.slots[slot_index] = value;
        self.advance_head();
        Ok(())
    }

    pub fn fill_slots(&mut self, value: Slot) {
        self.slots.fill(value);
    }

    pub fn read_slot(&mut self) -> Option<Slot> {
        if self.is_empty() {
            return None;
        }
        let slot_index = self.tail_idx() as usize & (N - 1);
        let value = self.slots[slot_index];
        self.advance_tail();
        Some(value)
    }

    pub fn slots_len(&self) -> usize {
        size_of_val(&self.slots)
    }

    pub fn len(&self) -> usize {
        size_of::<Self>()
    }
}

impl<const N: usize> RingBuffer<N> {
    pub fn abs_head_ptr(&self) -> *const u64 {
        &self.producer_head as *const u64
    }

    pub fn abs_tail_ptr(&self) -> *const u64 {
        &self.consumer_tail as *const u64
    }

    pub fn rel_slot_head_ptr(&self) -> *const Slot {
        unsafe { self.slots_ptr().add(self.head_idx() as usize) }
    }

    pub fn rel_slot_tail_ptr(&self) -> *const Slot {
        unsafe { self.slots_ptr().add(self.tail_idx() as usize) }
    }

    pub fn abs_slot_offset(&self) -> usize {
        Self::slots_offset()
    }

    pub fn abs_head_offset(&self) -> usize {
        Self::producer_head_offset()
    }

    pub fn abs_tail_offset(&self) -> usize {
        Self::consumer_tail_offset()
    }

    pub fn rel_slot_head_offset(&self) -> usize {
        self.abs_slot_offset() + self.head_idx() as usize * size_of::<Slot>()
    }

    pub fn rel_slot_tail_offset(&self) -> usize {
        self.abs_slot_offset() + self.tail_idx() as usize * size_of::<Slot>()
    }

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

    pub fn wrapping_write_slot_base_ptrs(&self, num_elements: u64) -> Result<SlotSegments> {
        if num_elements > self.available_write_slots() {
            return Err(anyhow::anyhow!(
                "requested more writable elements than available in the ring buffer"
            ));
        }

        let head = self.head_idx();
        let abs_head_offset = self.rel_slot_head_offset();
        let slots_ptr = self.slots_ptr();
        let abs_slot_offset = self.abs_slot_offset();

        let first_part_ptr = self.rel_slot_head_ptr();

        if head + num_elements <= N as u64 {
            Ok(SlotSegments {
                first_part: SlotSegment {
                    offset: abs_head_offset,
                    ptr: first_part_ptr,
                },
                second_part: None,
            })
        } else {
            Ok(SlotSegments {
                first_part: SlotSegment {
                    offset: abs_head_offset,
                    ptr: first_part_ptr,
                },
                second_part: Some(SlotSegment {
                    offset: abs_slot_offset,
                    ptr: slots_ptr,
                }),
            })
        }
    }

    pub fn wrapping_read_slot_base_ptrs(&self, num_elements: u64) -> Result<SlotSegments> {
        if num_elements > self.available_read_slots() {
            return Err(anyhow::anyhow!(
                "requested more readable elements than available in the ring buffer"
            ));
        }

        let tail = self.tail_idx();
        let abs_tail_offset = self.rel_slot_tail_offset();
        let slots_ptr = self.slots_ptr();
        let abs_slot_offset = self.abs_slot_offset();

        let first_part_ptr = self.rel_slot_tail_ptr();

        if tail + num_elements <= N as u64 {
            Ok(SlotSegments {
                first_part: SlotSegment {
                    offset: abs_tail_offset,
                    ptr: first_part_ptr,
                },
                second_part: None,
            })
        } else {
            Ok(SlotSegments {
                first_part: SlotSegment {
                    offset: abs_tail_offset,
                    ptr: first_part_ptr,
                },
                second_part: Some(SlotSegment {
                    offset: abs_slot_offset,
                    ptr: slots_ptr,
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{constants::RING_BUFFER_ELEMENTS, control_protocol::MAX_ITEM_SIZE};

    use super::*;

    #[test]
    fn basic_functionalities() {
        const N_ELEMENTS: usize = 2;

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        assert!(rb.is_empty());
        assert!(!rb.is_full());
        let value = "test".as_bytes();

        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..value.len()].copy_from_slice(value);

        for _ in 0..N_ELEMENTS as u64 - 1 {
            rb.write_slot(Slot {
                len: value.len() as u32,
                value: payload,
            })
            .unwrap();
            assert!(!rb.is_empty());
        }

        assert!(rb.is_full());
        assert!(rb.write_slot(Slot { len: 0, value: payload }).is_err());

        for _ in 0..N_ELEMENTS as u64 - 1 {
            let slot = rb.read_slot().unwrap();
            assert_eq!(slot.len, value.len() as u32);
            assert_eq!(&slot.value[..slot.len as usize], value);
        }

        assert!(rb.is_empty());
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
                value: payload,
            })
            .unwrap();
        }

        assert!(rb.is_full());
        assert_eq!(rb.head_idx(), 3);
        assert_eq!(rb.tail_idx(), 0);

        let first = rb.read_slot().unwrap();
        assert_eq!(&first.value[..first.len as usize], test_values[0]);
        assert!(!rb.is_full());
        assert_eq!(rb.head_idx(), 3);
        assert_eq!(rb.tail_idx(), 1);

        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..wrap_value.len()].copy_from_slice(wrap_value);
        rb.write_slot(Slot {
            len: wrap_value.len() as u32,
            value: payload,
        })
        .unwrap();

        assert!(rb.is_full());
        assert_eq!(rb.head_idx(), 0);
        assert_eq!(rb.tail_idx(), 1);

        for expected in [test_values[1], test_values[2], wrap_value] {
            let slot = rb.read_slot().unwrap();
            assert_eq!(slot.len, expected.len() as u32);
            assert_eq!(&slot.value[..slot.len as usize], expected);
        }

        assert!(rb.is_empty());
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

        for _ in 0..4 {
            rb.advance_tail();
        }
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
        let slots_offset = rb.abs_slot_offset();
        let head_offset = rb.rel_slot_head_offset();
        let tail_offset = rb.rel_slot_tail_offset();

        assert_eq!(base_ptr + slots_offset, rb.slots_ptr() as usize);
        assert_eq!(base_ptr + head_offset, rb.rel_slot_head_ptr() as usize);
        assert_eq!(base_ptr + tail_offset, rb.rel_slot_tail_ptr() as usize);
    }

    #[test]
    fn ptrs() {
        const N_ELEMENTS: usize = RING_BUFFER_ELEMENTS;

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        let base_ptr = rb.as_ptr() as usize;

        let non_wrapping_write = rb
            .wrapping_write_slot_base_ptrs(2)
            .expect("should have space for 2 elements");
        assert_eq!(non_wrapping_write.first_part.offset, rb.abs_slot_offset());
        assert_eq!(
            base_ptr + non_wrapping_write.first_part.offset,
            non_wrapping_write.first_part.ptr as usize
        );
        assert_eq!(non_wrapping_write.first_part.ptr, rb.slots_ptr());
        assert_eq!(non_wrapping_write.second_part, None);

        for _ in 0..N_ELEMENTS as u64 - 1 {
            rb.advance_head();
        }

        assert_eq!(rb.head_idx(), N_ELEMENTS as u64 - 1);
        assert!(rb.wrapping_write_slot_base_ptrs(2).is_err());

        rb.advance_tail();
        rb.advance_tail();
        rb.advance_tail();

        assert_eq!(rb.tail_idx(), 3);

        let wrapping_write = rb
            .wrapping_write_slot_base_ptrs(2)
            .expect("should have space for 2 elements after tail advances");
        assert_eq!(wrapping_write.first_part.offset, rb.rel_slot_head_offset());
        assert_eq!(
            base_ptr + wrapping_write.first_part.offset,
            wrapping_write.first_part.ptr as usize
        );
        assert_eq!(wrapping_write.first_part.ptr, rb.rel_slot_head_ptr());
        let second_part = wrapping_write
            .second_part
            .expect("should wrap to the start of the slots array");
        assert_eq!(second_part.offset, rb.abs_slot_offset());
        assert_eq!(base_ptr + second_part.offset, second_part.ptr as usize);
        assert_eq!(second_part.ptr, rb.slots_ptr());

        let mut rb = RingBuffer::<N_ELEMENTS>::new_boxed();
        let base_ptr = rb.as_ptr() as usize;

        assert!(rb.wrapping_read_slot_base_ptrs(1).is_err());

        rb.advance_head();
        rb.advance_head();

        let non_wrapping_read = rb
            .wrapping_read_slot_base_ptrs(2)
            .expect("should have 2 readable elements");
        assert_eq!(non_wrapping_read.first_part.offset, rb.abs_slot_offset());
        assert_eq!(
            base_ptr + non_wrapping_read.first_part.offset,
            non_wrapping_read.first_part.ptr as usize
        );
        assert_eq!(non_wrapping_read.first_part.ptr, rb.slots_ptr());
        assert_eq!(non_wrapping_read.second_part, None);

        for _ in 0..N_ELEMENTS as u64 - 1 {
            rb.advance_tail();
        }

        assert_eq!(rb.tail_idx(), N_ELEMENTS as u64 - 1);

        rb.advance_head();
        rb.advance_head();
        rb.advance_head();

        let wrapping_read = rb
            .wrapping_read_slot_base_ptrs(2)
            .expect("should have wrapped readable elements");
        assert_eq!(wrapping_read.first_part.offset, rb.rel_slot_tail_offset());
        assert_eq!(
            base_ptr + wrapping_read.first_part.offset,
            wrapping_read.first_part.ptr as usize
        );
        assert_eq!(wrapping_read.first_part.ptr, rb.rel_slot_tail_ptr());
        let second_part = wrapping_read
            .second_part
            .expect("should wrap to the start of the slots array");
        assert_eq!(second_part.offset, rb.abs_slot_offset());
        assert_eq!(base_ptr + second_part.offset, second_part.ptr as usize);
        assert_eq!(second_part.ptr, rb.slots_ptr());
    }
}
