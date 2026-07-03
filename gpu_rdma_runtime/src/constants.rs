pub const BATCH_SIZE: usize = 16;
pub const RING_BUFFER_ELEMENTS: usize = const_max(BATCH_SIZE * 8, 1024);

const fn const_max(lhs: usize, rhs: usize) -> usize {
    if lhs < rhs {
        rhs
    } else {
        lhs
    }
}
