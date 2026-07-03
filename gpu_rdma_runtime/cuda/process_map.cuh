static __device__ __forceinline__
uint32_t process_one(const uint8_t* in, uint32_t len, uint8_t* out) {
    for (uint32_t i = 0; i < len; ++i) {
        out[i] = in[i] + 1;
    }
    return len;
}
