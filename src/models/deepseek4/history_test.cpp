#include "models/deepseek4/history.hpp"

#include <cstdio>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::printf("deepseek4 history: %s\n", what);
    ++failures;
}

}  // namespace

int main() {
    using basalt::models::deepseek4::kMaxRewind;
    using basalt::models::deepseek4::start_refusal;
    expect(start_refusal(0, 0).empty(), "a first read starts at 0");
    expect(start_refusal(0, 3965).empty(), "position 0 starts a new sequence");
    expect(start_refusal(3965, 3965).empty(), "a read continues the held positions");
    // the server's next read after a reply that stopped inside a decode window: the window's rows past the stop go back
    expect(start_refusal(3963, 3965).empty(), "a read rewinds the tail of the last window");
    expect(start_refusal(72, 73).empty(), "a read rewinds the token after a stop");
    expect(start_refusal(3965 - kMaxRewind, 3965).empty(), "a whole window's rows go back");
    expect(!start_refusal(3965 - kMaxRewind - 1, 3965).empty(), "no deeper rewind than a window");
    expect(!start_refusal(100, 3965).empty(), "no rewind into the rings' past");
    expect(!start_refusal(3966, 3965).empty(), "no gap past the processed positions");
    expect(!start_refusal(-1, 3965).empty(), "no negative position");
    if (failures) return 1;
    std::puts("deepseek4 history: ok");
    return 0;
}
