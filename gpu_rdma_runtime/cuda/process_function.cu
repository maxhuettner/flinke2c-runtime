#include "slot.h"

struct ProcessSpec {
    uint32_t function;
    uint32_t field_index;
    uint32_t field_count;
    uint32_t field_types[32];
};

__device__ uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

__device__ void put_be32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >> 8); p[3] = uint8_t(v);
}

__device__ void increment_int32(uint8_t* p) {
    put_be32(p, be32(p) + 1u);
}

__device__ void increment_int64(uint8_t* p) {
    for (int i = 7; i >= 0; --i) {
        p[i]++;
        if (p[i] != 0) break;
    }
}

// Increments a minimal big-endian two's-complement integer in place. The
// surrounding field length is updated when a positive value needs a sign byte.
__device__ void increment_decimal(uint8_t* row, uint32_t& row_len,
                                  uint32_t length_pos, uint32_t bytes_pos) {
    uint32_t n = be32(row + length_pos);
    if (n == 0 || bytes_pos + n > row_len || n > MAX_ITEM_SIZE) return;
    bool positive = (row[bytes_pos] & 0x80u) == 0;
    for (uint32_t i = bytes_pos + n; i > bytes_pos; --i) {
        row[i - 1]++;
        if (row[i - 1] != 0) break;
    }
    if (positive && (row[bytes_pos] & 0x80u)) {
        if (row_len == MAX_ITEM_SIZE) return;
        for (uint32_t i = row_len; i > bytes_pos; --i) row[i] = row[i - 1];
        row[bytes_pos] = 0;
        ++n; ++row_len;
        put_be32(row + length_pos, n);
    }
}

extern "C" __global__ void process_slots(
    const RingBuffer* input, RingBuffer* output,
    uint64_t input_tail, uint64_t output_head, uint32_t count,
    ProcessSpec spec) {
    const uint32_t item = blockIdx.x;
    if (item >= count) return;
    const uint32_t in_index = (input_tail + item) & (RING_BUFFER_ELEMENTS - 1);
    const uint32_t out_index = (output_head + item) & (RING_BUFFER_ELEMENTS - 1);
    const Slot& source = input->slots[in_index];
    Slot& destination = output->slots[out_index];
    const uint32_t copy_len = source.len < MAX_ITEM_SIZE ? source.len : MAX_ITEM_SIZE;
    for (uint32_t i = threadIdx.x; i < copy_len; i += blockDim.x)
        destination.value[i] = source.value[i];
    if (threadIdx.x != 0) return;
    destination.len = copy_len;
    destination.timestamp_ns = source.timestamp_ns;

    const bool framed = copy_len >= 4 && be32(destination.value) == copy_len - 4;
    const uint32_t base = framed ? 4 : 0;
    const uint32_t null_bytes = (spec.field_count + 7) / 8;
    uint32_t pos = base + 12 + null_bytes;
    if (spec.field_count == 0 || pos > copy_len) return;
    for (uint32_t field = 0; field < spec.field_count; ++field) {
        const bool is_null = (destination.value[base + 12 + field / 8] >> (field % 8)) & 1;
        const uint32_t type = spec.field_types[field];
        if (is_null) continue;
        if (field == spec.field_index && spec.function == 1) {
            if (type == 1 && pos + 4 <= destination.len) increment_int32(destination.value + pos);
            else if (type == 2 && pos + 8 <= destination.len) increment_int64(destination.value + pos);
            else if (type == 3 && pos + 4 <= destination.len) increment_decimal(destination.value, destination.len, pos, pos + 4);
        }
        if (type == 1) pos += 4;
        else if (type == 2 || type == 5) pos += 8;
        else if (type == 3 || type == 4) {
            if (pos + 4 > destination.len) return;
            pos += 4 + be32(destination.value + pos);
        }
        if (pos > destination.len) return;
    }
    if (framed) put_be32(destination.value, destination.len - 4);
}
