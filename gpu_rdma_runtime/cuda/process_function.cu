#include "slot.h"

#include <cmath>
#include <cstdint>

struct ProcessSpec {
    uint32_t function;
    uint32_t field_index;
    uint32_t field_count;
    uint32_t field_types[32];
};

constexpr uint32_t FUNCTION_INCREMENT = 1;
constexpr uint32_t FUNCTION_IMPUTE = 2;
constexpr uint32_t FUNCTION_CURRENCY_CONVERSION = 3;

constexpr uint32_t CURRENCY_NUMERATOR = 908;
constexpr uint32_t CURRENCY_DENOMINATOR = 1000;
// DECIMAL(23,3) needs at most ten bytes. Leave ample room while keeping the
// per-thread local arrays bounded.
constexpr uint32_t MAX_CURRENCY_DECIMAL_BYTES = 40;

constexpr uint32_t IMPUTATION_HISTORY_SIZE = 5000;
constexpr uint32_t IMPUTATION_SEARCH_LIMIT = 512;
constexpr uint32_t IMPUTATION_K = 10;
constexpr double IMPUTATION_EPS = 1e-6;
constexpr double W_BIDDER = 0.25;
constexpr double W_TIME = 1.0;
constexpr double W_STRING = 0.25;

__device__ __constant__ uint8_t DEFAULT_CHANNEL_BYTES[7] = {
    uint8_t('u'), uint8_t('n'), uint8_t('k'), uint8_t('n'),
    uint8_t('o'), uint8_t('w'), uint8_t('n')
};

struct ImputationObservation {
    double price;
    int64_t bidder;
    double timestamp_seconds;
    uint32_t channel_hash;
    uint32_t url_hash;
    uint32_t extra_hash;
};

// Module globals are zero-initialized when a server session loads the PTX.
// Stateful batches are ordered by CUDA events on the host before these values
// are read or updated.
__device__ ImputationObservation imputation_history[IMPUTATION_HISTORY_SIZE];
__device__ uint32_t imputation_history_start;
__device__ uint32_t imputation_history_size;

struct BidView {
    bool valid;
    bool price_missing;
    const uint8_t* price;
    uint32_t price_length;
    int64_t auction;
    int64_t bidder;
    const uint8_t* channel;
    uint32_t channel_length;
    const uint8_t* url;
    uint32_t url_length;
    int64_t timestamp_millis;
    const uint8_t* extra;
    uint32_t extra_length;
};

__device__ __forceinline__ uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

__device__ __forceinline__ uint64_t be64(const uint8_t* p) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; ++i) value = (value << 8) | uint64_t(p[i]);
    return value;
}

__device__ __forceinline__ void put_be32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v >> 24);
    p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >> 8);
    p[3] = uint8_t(v);
}

__device__ __forceinline__ void put_be64(uint8_t* p, uint64_t value) {
    for (int i = 7; i >= 0; --i) {
        p[i] = uint8_t(value);
        value >>= 8;
    }
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
        ++n;
        ++row_len;
        put_be32(row + length_pos, n);
    }
}

// Multiplies a minimal big-endian two's-complement unscaled decimal integer
// by 0.908 and rounds to the same scale with BigDecimal HALF_UP semantics.
// Arithmetic is performed on the integer bytes so DECIMAL(23,3) does not lose
// precision through a double conversion.
__device__ uint32_t convert_currency_decimal(
    const uint8_t* value, uint32_t length, uint8_t* output) {
    if (length == 0 || length > MAX_CURRENCY_DECIMAL_BYTES) return 0;

    const bool negative = (value[0] & 0x80u) != 0;
    uint8_t magnitude[MAX_CURRENCY_DECIMAL_BYTES + 2];
    uint32_t digits = length;
    for (uint32_t i = 0; i < length; ++i) {
        magnitude[i] = value[length - 1 - i];
    }

    if (negative) {
        uint32_t carry = 1;
        for (uint32_t i = 0; i < digits; ++i) {
            const uint32_t converted =
                uint32_t(uint8_t(~magnitude[i])) + carry;
            magnitude[i] = uint8_t(converted);
            carry = converted >> 8;
        }
    }
    while (digits > 0 && magnitude[digits - 1] == 0) --digits;
    if (digits == 0) {
        output[0] = 0;
        return 1;
    }

    uint32_t carry = 0;
    for (uint32_t i = 0; i < digits; ++i) {
        const uint32_t product =
            uint32_t(magnitude[i]) * CURRENCY_NUMERATOR + carry;
        magnitude[i] = uint8_t(product);
        carry = product >> 8;
    }
    while (carry != 0) {
        magnitude[digits++] = uint8_t(carry);
        carry >>= 8;
    }

    uint32_t remainder = 0;
    for (uint32_t i = digits; i > 0; --i) {
        const uint32_t dividend =
            remainder * 256u + uint32_t(magnitude[i - 1]);
        magnitude[i - 1] = uint8_t(dividend / CURRENCY_DENOMINATOR);
        remainder = dividend % CURRENCY_DENOMINATOR;
    }
    while (digits > 0 && magnitude[digits - 1] == 0) --digits;

    // HALF_UP increments the magnitude on a tie, which rounds negative
    // values away from zero just like BigDecimal.
    if (remainder * 2u >= CURRENCY_DENOMINATOR) {
        uint32_t index = 0;
        uint32_t round_carry = 1;
        while (round_carry != 0 && index < digits) {
            const uint32_t rounded =
                uint32_t(magnitude[index]) + round_carry;
            magnitude[index] = uint8_t(rounded);
            round_carry = rounded >> 8;
            ++index;
        }
        if (round_carry != 0 || digits == 0) {
            magnitude[digits++] = uint8_t(round_carry == 0 ? 1 : round_carry);
        }
    }
    if (digits == 0) {
        output[0] = 0;
        return 1;
    }

    uint32_t width = digits;
    if (!negative) {
        if ((magnitude[digits - 1] & 0x80u) != 0) {
            output[0] = 0;
            ++width;
        }
        for (uint32_t i = 0; i < digits; ++i) {
            output[width - 1 - i] = magnitude[i];
        }
        return width;
    }

    bool lower_nonzero = false;
    for (uint32_t i = 0; i + 1 < digits; ++i) {
        lower_nonzero |= magnitude[i] != 0;
    }
    const uint8_t top = magnitude[digits - 1];
    if (top > 0x80u || (top == 0x80u && lower_nonzero)) ++width;
    for (uint32_t i = 0; i < width; ++i) {
        const uint8_t magnitude_byte =
            i < digits ? magnitude[i] : uint8_t(0);
        output[width - 1 - i] = uint8_t(~magnitude_byte);
    }
    for (uint32_t i = width; i > 0; --i) {
        output[i - 1]++;
        if (output[i - 1] != 0) break;
    }
    return width;
}

__device__ void convert_currency_decimal_field(
    uint8_t* row, uint32_t& row_len,
    uint32_t length_pos, uint32_t bytes_pos) {
    if (length_pos > row_len || row_len - length_pos < 4 ||
        bytes_pos > row_len) {
        return;
    }
    const uint32_t old_length = be32(row + length_pos);
    if (old_length == 0 || old_length > row_len - bytes_pos) return;

    uint8_t converted[MAX_CURRENCY_DECIMAL_BYTES + 1];
    const uint32_t new_length =
        convert_currency_decimal(row + bytes_pos, old_length, converted);
    if (new_length == 0) return;

    const uint32_t old_end = bytes_pos + old_length;
    if (new_length > old_length) {
        const uint32_t growth = new_length - old_length;
        if (growth > MAX_ITEM_SIZE - row_len) return;
        for (uint32_t i = row_len; i > old_end; --i) {
            row[i + growth - 1] = row[i - 1];
        }
        row_len += growth;
    } else if (new_length < old_length) {
        const uint32_t shrink = old_length - new_length;
        for (uint32_t i = old_end; i < row_len; ++i) {
            row[i - shrink] = row[i];
        }
        row_len -= shrink;
    }
    for (uint32_t i = 0; i < new_length; ++i) {
        row[bytes_pos + i] = converted[i];
    }
    put_be32(row + length_pos, new_length);
}

__device__ __forceinline__ bool null_field(const uint8_t* bitmap, uint32_t field) {
    return ((bitmap[field / 8] >> (field % 8)) & 1u) != 0;
}

__device__ bool blank_utf8(const uint8_t* value, uint32_t length) {
    if (length == 0) return true;
    for (uint32_t i = 0; i < length; ++i) {
        // This matches String.trim().isEmpty() for ASCII/UTF-8 whitespace.
        if (value[i] > 0x20u) return false;
    }
    return true;
}

__device__ bool read_variable(
    const uint8_t* row, uint32_t row_length, uint32_t& position,
    const uint8_t*& value, uint32_t& length) {
    if (position + 4 > row_length) return false;
    length = be32(row + position);
    position += 4;
    if (length > row_length - position) return false;
    value = row + position;
    position += length;
    return true;
}

__device__ bool parse_bid(const Slot& slot, BidView& bid) {
    bid.valid = false;
    bid.price_missing = true;
    bid.price = nullptr;
    bid.price_length = 0;
    bid.auction = 0;
    bid.bidder = 0;
    bid.channel = nullptr;
    bid.channel_length = 0;
    bid.url = nullptr;
    bid.url_length = 0;
    bid.timestamp_millis = 0;
    bid.extra = nullptr;
    bid.extra_length = 0;

    const uint32_t row_length =
        slot.len < MAX_ITEM_SIZE ? slot.len : MAX_ITEM_SIZE;
    if (row_length < 17 || be32(slot.value) != row_length - 4) return false;

    // Framed payload: int32 op, int64 row id, one-byte null bitmap, fields.
    const uint8_t* nulls = slot.value + 16;
    uint32_t position = 17;

    bid.price_missing = null_field(nulls, 0);
    if (!bid.price_missing &&
        (!read_variable(
             slot.value, row_length, position, bid.price, bid.price_length) ||
         bid.price_length == 0)) {
        return false;
    }

    if (!null_field(nulls, 1)) {
        if (position + 8 > row_length) return false;
        bid.auction = int64_t(be64(slot.value + position));
        position += 8;
    }
    if (!null_field(nulls, 2)) {
        if (position + 8 > row_length) return false;
        bid.bidder = int64_t(be64(slot.value + position));
        position += 8;
    }
    if (!null_field(nulls, 3)) {
        if (!read_variable(
                slot.value, row_length, position,
                bid.channel, bid.channel_length)) {
            return false;
        }
        if (blank_utf8(bid.channel, bid.channel_length)) {
            bid.channel = nullptr;
            bid.channel_length = 0;
        }
    }
    if (!null_field(nulls, 4)) {
        if (!read_variable(
                slot.value, row_length, position, bid.url, bid.url_length)) {
            return false;
        }
        if (blank_utf8(bid.url, bid.url_length)) {
            bid.url = nullptr;
            bid.url_length = 0;
        }
    }
    if (!null_field(nulls, 5)) {
        if (position + 8 > row_length) return false;
        bid.timestamp_millis = int64_t(be64(slot.value + position));
        position += 8;
    }
    if (!null_field(nulls, 6)) {
        if (!read_variable(
                slot.value, row_length, position, bid.extra, bid.extra_length)) {
            return false;
        }
        if (blank_utf8(bid.extra, bid.extra_length)) {
            bid.extra = nullptr;
            bid.extra_length = 0;
        }
    }

    bid.valid = position == row_length;
    return bid.valid;
}

__device__ uint32_t hash_bytes(const uint8_t* value, uint32_t length) {
    uint32_t hash = 0x9747b28cU;
    for (uint32_t i = 0; i < length; ++i) {
        // Java byte is signed before promotion in "h ^= b".
        const uint32_t signed_byte = uint32_t(int32_t(int8_t(value[i])));
        hash ^= signed_byte;
        hash *= 0x5bd1e995U;
        hash ^= hash >> 15;
    }
    return hash;
}

__device__ uint32_t bid_channel_hash(const BidView& bid) {
    return bid.channel == nullptr
        ? hash_bytes(DEFAULT_CHANNEL_BYTES, sizeof(DEFAULT_CHANNEL_BYTES))
        : hash_bytes(bid.channel, bid.channel_length);
}

__device__ __forceinline__ uint32_t optional_hash(
    const uint8_t* value, uint32_t length) {
    return value == nullptr || length == 0 ? 0 : hash_bytes(value, length);
}

__device__ double decimal_to_double(const uint8_t* value, uint32_t length) {
    if (value == nullptr || length == 0) return 0.0;
    double result = double(int8_t(value[0]));
    for (uint32_t i = 1; i < length; ++i) {
        result = result * 256.0 + double(value[i]);
    }
    // The specialized imputer is for DECIMAL(23,3).
    return result / 1000.0;
}

__device__ ImputationObservation observation_from(const BidView& bid) {
    ImputationObservation observation;
    observation.price = decimal_to_double(bid.price, bid.price_length);
    observation.bidder = bid.bidder;
    observation.timestamp_seconds = double(bid.timestamp_millis) / 1000.0;
    observation.channel_hash = bid_channel_hash(bid);
    observation.url_hash = optional_hash(bid.url, bid.url_length);
    observation.extra_hash = optional_hash(bid.extra, bid.extra_length);
    return observation;
}

__device__ double observation_distance(
    const ImputationObservation& target,
    const ImputationObservation& candidate) {
    double distance =
        W_BIDDER * (target.bidder == candidate.bidder ? 0.0 : 1.0);
    const double delta =
        target.timestamp_seconds - candidate.timestamp_seconds;
    distance += W_TIME * delta * delta * 1e-8;
    distance += W_STRING *
        (target.channel_hash == candidate.channel_hash ? 0.0 : 1.0);
    distance += W_STRING *
        (target.url_hash == candidate.url_hash ? 0.0 : 1.0);
    distance += W_STRING *
        (target.extra_hash == candidate.extra_hash ? 0.0 : 1.0);
    return distance;
}

__device__ void consider_neighbor(
    const ImputationObservation& target,
    const ImputationObservation& candidate,
    double* best_distance, double* best_price, uint32_t& found) {
    const double distance = observation_distance(target, candidate);
    if (found < IMPUTATION_K) {
        best_distance[found] = distance;
        best_price[found] = candidate.price;
        ++found;
        return;
    }

    uint32_t worst_index = 0;
    double worst = best_distance[0];
    for (uint32_t i = 1; i < IMPUTATION_K; ++i) {
        if (best_distance[i] > worst) {
            worst = best_distance[i];
            worst_index = i;
        }
    }
    if (distance < worst) {
        best_distance[worst_index] = distance;
        best_price[worst_index] = candidate.price;
    }
}

__device__ double impute_price(
    const RingBuffer* input, uint64_t input_tail, uint32_t item,
    const BidView& target_bid) {
    const ImputationObservation target = observation_from(target_bid);
    double best_distance[IMPUTATION_K];
    double best_price[IMPUTATION_K];
    uint32_t found = 0;

    // snapshotLast(512) first sees earlier batch rows newest-first, then the
    // already-committed ring. Limit the combined scan to the Java UDF's 512.
    uint32_t remaining = IMPUTATION_SEARCH_LIMIT;
    for (uint32_t prior = item; prior > 0 && remaining > 0; --prior) {
        const uint32_t index =
            (input_tail + prior - 1) & (RING_BUFFER_ELEMENTS - 1);
        BidView candidate_bid;
        if (parse_bid(input->slots[index], candidate_bid) &&
            !candidate_bid.price_missing) {
            consider_neighbor(
                target, observation_from(candidate_bid),
                best_distance, best_price, found);
            --remaining;
        }
    }

    const uint32_t search_count =
        imputation_history_size < remaining
            ? imputation_history_size
            : remaining;
    for (uint32_t i = 0; i < search_count; ++i) {
        const uint32_t index =
            (imputation_history_start + imputation_history_size - 1 - i) %
            IMPUTATION_HISTORY_SIZE;
        consider_neighbor(
            target, imputation_history[index],
            best_distance, best_price, found);
    }

    if (found == 0) {
        return 0.0;
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t i = 0; i < found; ++i) {
        const double weight = 1.0 / (best_distance[i] + IMPUTATION_EPS);
        numerator += best_price[i] * weight;
        denominator += weight;
    }
    return denominator != 0.0 ? numerator / denominator : 0.0;
}

// Encodes round-HALF_UP(price * 1000) as a minimal big-endian two's-complement
// integer. The base-256 path also covers imputed values outside int64 range.
__device__ uint32_t encode_decimal(double price, uint8_t* output) {
    if (!isfinite(price)) {
        output[0] = 0;
        return 1;
    }
    const double scaled = price * 1000.0;
    const bool negative = scaled < 0.0;
    double magnitude = negative ? -scaled : scaled;
    magnitude = floor(magnitude + 0.5);
    if (magnitude == 0.0) {
        output[0] = 0;
        return 1;
    }

    uint8_t little_endian[40];
    uint32_t digits = 0;
    while (magnitude >= 1.0 && digits < sizeof(little_endian)) {
        const double quotient = floor(magnitude / 256.0);
        little_endian[digits++] =
            uint8_t(magnitude - quotient * 256.0);
        magnitude = quotient;
    }
    if (digits == 0 || magnitude >= 1.0) {
        output[0] = 0;
        return 1;
    }

    uint32_t width = digits;
    if (!negative) {
        const bool sign_byte = (little_endian[digits - 1] & 0x80u) != 0;
        if (sign_byte) {
            ++width;
            output[0] = 0;
        }
        for (uint32_t i = 0; i < digits; ++i) {
            output[width - 1 - i] = little_endian[i];
        }
        return width;
    }

    bool lower_nonzero = false;
    for (uint32_t i = 0; i + 1 < digits; ++i) {
        lower_nonzero |= little_endian[i] != 0;
    }
    const uint8_t top = little_endian[digits - 1];
    if (top > 0x80u || (top == 0x80u && lower_nonzero)) ++width;
    for (uint32_t i = 0; i < width; ++i) {
        const uint8_t magnitude_byte =
            i < digits ? little_endian[i] : uint8_t(0);
        output[width - 1 - i] = uint8_t(~magnitude_byte);
    }
    for (uint32_t i = width; i > 0; --i) {
        output[i - 1]++;
        if (output[i - 1] != 0) break;
    }
    return width;
}

__device__ void copy_bytes(
    uint8_t* destination, uint32_t& position,
    const uint8_t* source, uint32_t length) {
    for (uint32_t i = 0; i < length; ++i) {
        destination[position + i] = source[i];
    }
    position += length;
}

__device__ bool write_imputed_bid(
    const Slot& source, Slot& destination, const BidView& bid,
    double imputed_price) {
    uint8_t price_bytes[40];
    const uint8_t* price = bid.price;
    uint32_t price_length = bid.price_length;
    if (bid.price_missing) {
        price_length = encode_decimal(imputed_price, price_bytes);
        price = price_bytes;
    }
    const uint8_t* channel =
        bid.channel == nullptr ? DEFAULT_CHANNEL_BYTES : bid.channel;
    const uint32_t channel_length =
        bid.channel == nullptr
            ? uint32_t(sizeof(DEFAULT_CHANNEL_BYTES))
            : bid.channel_length;
    const uint32_t output_length =
        17 + 4 + price_length + 8 + 8 +
        4 + channel_length + 4 + bid.url_length + 8 +
        4 + bid.extra_length;
    if (output_length > MAX_ITEM_SIZE) return false;

    // Preserve frame header's op and row id, then emit a fully non-null row.
    for (uint32_t i = 0; i < 16; ++i) {
        destination.value[i] = source.value[i];
    }
    destination.value[16] = 0;
    uint32_t position = 17;

    put_be32(destination.value + position, price_length);
    position += 4;
    copy_bytes(destination.value, position, price, price_length);
    put_be64(destination.value + position, uint64_t(bid.auction));
    position += 8;
    put_be64(destination.value + position, uint64_t(bid.bidder));
    position += 8;
    put_be32(destination.value + position, channel_length);
    position += 4;
    copy_bytes(destination.value, position, channel, channel_length);
    put_be32(destination.value + position, bid.url_length);
    position += 4;
    if (bid.url_length != 0) {
        copy_bytes(destination.value, position, bid.url, bid.url_length);
    }
    put_be64(destination.value + position, uint64_t(bid.timestamp_millis));
    position += 8;
    put_be32(destination.value + position, bid.extra_length);
    position += 4;
    if (bid.extra_length != 0) {
        copy_bytes(destination.value, position, bid.extra, bid.extra_length);
    }

    destination.len = position;
    put_be32(destination.value, position - 4);
    return true;
}

__device__ void process_increment(Slot& destination, const ProcessSpec& spec) {
    const bool framed =
        destination.len >= 4 &&
        be32(destination.value) == destination.len - 4;
    const uint32_t base = framed ? 4 : 0;
    const uint32_t null_bytes = (spec.field_count + 7) / 8;
    uint32_t position = base + 12 + null_bytes;
    if (spec.field_count == 0 || position > destination.len) return;
    for (uint32_t field = 0; field < spec.field_count; ++field) {
        const bool is_null =
            (destination.value[base + 12 + field / 8] >> (field % 8)) & 1;
        const uint32_t type = spec.field_types[field];
        if (is_null) {
            if (field == spec.field_index) return;
            continue;
        }
        if (field == spec.field_index) {
            if (type == 1 && position + 4 <= destination.len) {
                increment_int32(destination.value + position);
            } else if (type == 2 && position + 8 <= destination.len) {
                increment_int64(destination.value + position);
            } else if (type == 3 && position + 4 <= destination.len) {
                increment_decimal(
                    destination.value, destination.len,
                    position, position + 4);
            }
            if (framed) put_be32(destination.value, destination.len - 4);
            return;
        }
        if (type == 1) {
            position += 4;
        } else if (type == 2 || type == 5) {
            position += 8;
        } else if (type == 3 || type == 4) {
            if (position + 4 > destination.len) return;
            position += 4 + be32(destination.value + position);
        }
        if (position > destination.len) return;
    }
}

__device__ void process_currency_conversion(
    Slot& destination, const ProcessSpec& spec) {
    const bool framed =
        destination.len >= 4 &&
        be32(destination.value) == destination.len - 4;
    const uint32_t base = framed ? 4 : 0;
    const uint32_t null_bytes = (spec.field_count + 7) / 8;
    uint32_t position = base + 12 + null_bytes;
    if (spec.field_count == 0 || position > destination.len) return;
    for (uint32_t field = 0; field < spec.field_count; ++field) {
        const bool is_null =
            (destination.value[base + 12 + field / 8] >> (field % 8)) & 1;
        const uint32_t type = spec.field_types[field];
        if (is_null) {
            if (field == spec.field_index) return;
            continue;
        }
        if (field == spec.field_index) {
            if (type == 3) {
                convert_currency_decimal_field(
                    destination.value, destination.len,
                    position, position + 4);
            }
            if (framed) put_be32(destination.value, destination.len - 4);
            return;
        }
        if (type == 1) {
            position += 4;
        } else if (type == 2 || type == 5) {
            position += 8;
        } else if (type == 3 || type == 4) {
            if (position + 4 > destination.len) return;
            position += 4 + be32(destination.value + position);
        }
        if (position > destination.len) return;
    }
}

// Stateless functions modify their input slots directly. With no row copy to
// cooperate on, tightly pack one row per CUDA thread.
extern "C" __global__ void process_slots_in_place(
    RingBuffer* input, uint64_t input_tail, uint32_t count,
    ProcessSpec spec) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    const uint32_t input_index =
        (input_tail + item) & (RING_BUFFER_ELEMENTS - 1);
    Slot& row = input->slots[input_index];
    row.len = row.len < MAX_ITEM_SIZE ? row.len : MAX_ITEM_SIZE;
    if (spec.function == FUNCTION_INCREMENT) {
        process_increment(row, spec);
    } else if (spec.function == FUNCTION_CURRENCY_CONVERSION) {
        process_currency_conversion(row, spec);
    }
}

// Functions that materialize a separate output use one warp per row. Each warp
// copies adjacent bytes with coalesced accesses, and a block handles eight rows.
extern "C" __global__ void process_slots(
    const RingBuffer* input, RingBuffer* output,
    uint64_t input_tail, uint64_t output_head, uint32_t count,
    ProcessSpec spec) {
    constexpr uint32_t WARP_SIZE = 32;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warp = threadIdx.x / WARP_SIZE;
    const uint32_t warps_per_block = blockDim.x / WARP_SIZE;
    const uint32_t item = blockIdx.x * warps_per_block + warp;
    if (item >= count) return;
    const uint32_t input_index =
        (input_tail + item) & (RING_BUFFER_ELEMENTS - 1);
    const uint32_t output_index =
        (output_head + item) & (RING_BUFFER_ELEMENTS - 1);
    const Slot& source = input->slots[input_index];
    Slot& destination = output->slots[output_index];
    const uint32_t copy_length =
        source.len < MAX_ITEM_SIZE ? source.len : MAX_ITEM_SIZE;
    for (uint32_t i = lane; i < copy_length; i += WARP_SIZE) {
        destination.value[i] = source.value[i];
    }
    __syncwarp();
    if (lane != 0) return;

    destination.len = copy_length;
    destination.timestamp_ns = source.timestamp_ns;
    if (spec.function != FUNCTION_IMPUTE) return;

    BidView bid;
    if (!parse_bid(source, bid)) return;
    double price = bid.price_missing
        ? impute_price(input, input_tail, item, bid)
        : decimal_to_double(bid.price, bid.price_length);
    // A missing/invalid neighborhood maps to the Java UDF's 0.000 default.
    if (!isfinite(price)) price = 0.0;
    write_imputed_bid(source, destination, bid, price);
}

// A second kernel creates the batch boundary needed for deterministic state:
// all output rows read the old history plus earlier input rows, then observed
// prices are committed in source order for the next batch.
extern "C" __global__ void commit_imputation_history(
    const RingBuffer* input, uint64_t input_tail, uint32_t count,
    ProcessSpec spec) {
    if (blockIdx.x != 0 || threadIdx.x != 0 ||
        spec.function != FUNCTION_IMPUTE) {
        return;
    }
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t input_index =
            (input_tail + item) & (RING_BUFFER_ELEMENTS - 1);
        BidView bid;
        if (!parse_bid(input->slots[input_index], bid) || bid.price_missing) {
            continue;
        }
        uint32_t history_index;
        if (imputation_history_size < IMPUTATION_HISTORY_SIZE) {
            history_index =
                (imputation_history_start + imputation_history_size) %
                IMPUTATION_HISTORY_SIZE;
            ++imputation_history_size;
        } else {
            history_index = imputation_history_start;
            imputation_history_start =
                (imputation_history_start + 1) % IMPUTATION_HISTORY_SIZE;
        }
        imputation_history[history_index] = observation_from(bid);
    }
}
