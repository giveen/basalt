// basalt/program/engine_local.hpp - helpers shared by generate.cpp and the serve engine's translation unit.
// Private to src/program: nothing here is part of the engine's public surface.
#pragma once

#include <cstdint>

#include "basalt/program/multi_gpu_plan.hpp"   // PsPrice

namespace basalt::program::engine_local {

/// One candidate split price: the plan's cost, the prompt chunk and the head chunk it was measured with.
struct SplitPricing {
    basalt::program::PsPrice price;
    int64_t chunk = 0, head = 0;
};

/// The largest multiple of 256 in [0, ceiling] that `ok` accepts.  `ok` is monotone: false once too large.
template <class Ok>
int64_t biggest_chunk(int64_t ceiling, Ok&& ok) {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
}

}  // namespace basalt::program::engine_local
