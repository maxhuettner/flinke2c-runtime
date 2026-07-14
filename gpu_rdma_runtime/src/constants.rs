pub const BATCH_SIZE: usize = 16;
/// Large enough to absorb bursts while the GPU pipeline and both clients run
/// independently. This value must match cuda/slot.h.
pub const RING_BUFFER_ELEMENTS: usize = 65_536;

/// Receive WQEs carry batch notifications, not ring slots. Keep this below
/// common RNIC max_qp_wr limits; completed WQEs are replenished continuously.
pub const RECEIVE_WR_DEPTH: usize = 1024;
