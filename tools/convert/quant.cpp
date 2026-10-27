#include "basalt/ggml/ggml.h"
#include <cstddef>
#include <cstdint>

extern "C" size_t basalt_quantize_q4_k(const float *src, void *dst, int64_t rows, int64_t cols) {
    if (!src || !dst || rows <= 0 || cols <= 0 || cols % 256 != 0)
        return 0;
    return ggml_quantize_chunk(GGML_TYPE_Q4_K, src, dst, 0, rows, cols);
}
