#ifndef SLOT_H
#define SLOT_H

#include <cstdint>
#include <cstddef>

constexpr uint32_t MAX_ITEM_SIZE = 2048;
constexpr uint32_t RING_BUFFER_ELEMENTS = 65536;
constexpr uint32_t BATCH_SIZE = 16;

struct Slot {
    uint32_t len;
    uint64_t timestamp_ns;
    uint8_t value[MAX_ITEM_SIZE];
};

static_assert(sizeof(Slot) == 16 + MAX_ITEM_SIZE, "Slot layout must match Rust");

struct RingBuffer {
    uint64_t producer_head;
    uint64_t consumer_tail;
    Slot slots[RING_BUFFER_ELEMENTS];
};

static_assert(offsetof(RingBuffer, producer_head) == 0, "producer head offset must match Rust");
static_assert(offsetof(RingBuffer, consumer_tail) == 8, "consumer tail offset must match Rust");
static_assert(offsetof(RingBuffer, slots) == 16, "slot offset must match Rust");

#endif
