#pragma once

#include <cstdint>
#include <string>

namespace basalt::models::deepseek4 {

/// The most positions a step may rewind: the rows a decode window processed past what the engine kept (the drafts it
/// rejected, or the rows it ran past a stop token). Its window and compressor rings cannot go back further.
constexpr int64_t kMaxRewind = 8;

/// Where a decode window, prompt chunk or image may start, given the `held` positions the session processed: at 0 (a
/// new sequence), at `held` (it continues), or up to kMaxRewind before it, overwriting that tail from `pos` on. Empty
/// when it may start there, else why not.
inline std::string start_refusal(int64_t pos, int64_t held) {
    if (pos < 0) return "a negative position";
    if (pos > held) return "position " + std::to_string(pos) + " is beyond the " + std::to_string(held) + " processed";
    if (pos > 0 && held - pos > kMaxRewind)
        return "position " + std::to_string(pos) + " rewinds " + std::to_string(held - pos) + " of the " +
               std::to_string(held) + " processed; at most " + std::to_string(kMaxRewind) + " go back";
    return {};
}

}  // namespace basalt::models::deepseek4
