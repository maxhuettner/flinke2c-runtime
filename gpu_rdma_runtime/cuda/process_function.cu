#include "slot.h"
#include "process_map.cuh"

extern "C" __global__
void process_slots(
    const RingBuffer* input,
    RingBuffer* output,
    uint64_t input_tail,
    uint64_t output_head,
    uint32_t count) {
    const uint32_t item = blockIdx.x;
    if (item >= count) return;

    const uint32_t input_index = (input_tail + item) & (RING_BUFFER_ELEMENTS - 1);
    const uint32_t output_index = (output_head + item) & (RING_BUFFER_ELEMENTS - 1);
    const Slot& source = input->slots[input_index];
    Slot& destination = output->slots[output_index];

    const uint32_t len = source.len < MAX_ITEM_SIZE ? source.len : MAX_ITEM_SIZE;
    for (uint32_t offset = threadIdx.x; offset < len; offset += blockDim.x) {
        destination.value[offset] = process_one(source.value[offset]);
    }
    if (threadIdx.x == 0) {
        destination.len = len;
    }
}
