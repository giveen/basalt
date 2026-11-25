// LoopCD against a double CPU reference, including window placement, graphs and timing.
#include "basalt/models/inkling2/kernels.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace basalt::models::inkling2;

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
void test(int rows) {
    const size_t n = (size_t) rows * kWidth, bytes = n * sizeof(float);
    std::mt19937 rng(41);
    std::normal_distribution<float> normal;
    std::vector<float> mid(n), out(n), rm(n), ro(n), got_m(n), got_o(n);
    for (auto* v : {&mid, &out, &rm, &ro})
        for (float& x : *v) x = normal(rng);
    float *dm = nullptr, *dh = nullptr, *drm = nullptr, *dro = nullptr;
    for (float** p : {&dm, &dh, &drm, &dro}) check(cudaMalloc((void**) p, bytes));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    auto reset = [&] {
        check(cudaMemcpyAsync(dm, mid.data(), bytes, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dh, out.data(), bytes, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(drm, rm.data(), bytes, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dro, ro.data(), bytes, cudaMemcpyHostToDevice, stream));
    };
    for (float w : {0.f, 0.1f, 0.3f, 0.5f, 1.f}) {
        reset();
        loopcd_hidden(dm, dh, w == 0.f ? nullptr : drm, w == 0.f ? nullptr : dro, rows, w, stream);
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(got_m.data(), dm, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(got_o.data(), dh, bytes, cudaMemcpyDeviceToHost));
        double abs_err = 0, rel_err = 0;
        for (size_t i = 0; i < n; ++i)
            for (int b = 0; b < 2; ++b) {
                const double strong = b ? out[i] : mid[i], weak = b ? ro[i] : rm[i];
                const double ref = strong + (double) w * (strong - weak);
                const double err = std::abs((b ? got_o[i] : got_m[i]) - ref);
                abs_err = std::max(abs_err, err);
                rel_err = std::max(rel_err, err / std::max(1e-8, std::abs(ref)));
                if (err > 1e-6 * std::max(1.0, std::abs(ref))) throw std::runtime_error("CPU double parity");
            }
        if (w == 0.f && (!same(mid, got_m) || !same(out, got_o))) throw std::runtime_error("zero strength identity");
        // Every row alone must produce the same bits as that row in a wider window.
        reset();
        for (int r = 0; r < rows; ++r)
            loopcd_hidden(dm + (size_t) r * kWidth, dh + (size_t) r * kWidth,
                          drm + (size_t) r * kWidth, dro + (size_t) r * kWidth, 1, w, stream);
        check(cudaStreamSynchronize(stream));
        std::vector<float> alone(n);
        check(cudaMemcpy(alone.data(), dm, bytes, cudaMemcpyDeviceToHost));
        if (!same(alone, got_m)) throw std::runtime_error("mid window identity");
        check(cudaMemcpy(alone.data(), dh, bytes, cudaMemcpyDeviceToHost));
        if (!same(alone, got_o)) throw std::runtime_error("output window identity");
        check(cudaMemcpy(alone.data(), drm, bytes, cudaMemcpyDeviceToHost));
        if (!same(alone, rm)) throw std::runtime_error("mid reference modified");
        check(cudaMemcpy(alone.data(), dro, bytes, cudaMemcpyDeviceToHost));
        if (!same(alone, ro)) throw std::runtime_error("output reference modified");
        std::printf("rows=%d strength=%.1f max_abs=%.3g max_rel=%.3g; row identity exact\n", rows, w, abs_err, rel_err);
    }
    // Replaying a graph must consume fresh inputs rather than a previous window's reference.
    reset();
    check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    loopcd_hidden(dm, dh, drm, dro, rows, 0.3f, stream);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int k = 0; k < 2; ++k) {
        reset();
        check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(got_m.data(), dm, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            const float delta = mid[i] - rm[i], step = 0.3f * delta;
            const float ref = mid[i] + step;
            if (got_m[i] != ref) throw std::runtime_error("graph replay parity");
        }
    }
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    // Equal strong/reference states are a fixed point, so repeated timing stays finite.
    check(cudaMemcpyAsync(dm, drm, bytes, cudaMemcpyDeviceToDevice, stream));
    check(cudaMemcpyAsync(dh, dro, bytes, cudaMemcpyDeviceToDevice, stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 100; ++i) loopcd_hidden(dm, dh, drm, dro, rows, 0.3f, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(exec, stream));
    check(cudaEventRecord(start, stream));
    for (int i = 0; i < 10; ++i) check(cudaGraphLaunch(exec, stream));
    check(cudaEventRecord(stop, stream));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    std::printf("rows=%d: %.3f us/call, %.2f GB/s (four reads + two writes)\n", rows, ms, 6.0 * bytes / (ms * 1e3));
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
    for (float* p : {dm, dh, drm, dro}) check(cudaFree(p));
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        for (float w : {-1.f, 2.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            bool rejected = false;
            try { loopcd_hidden(nullptr, nullptr, nullptr, nullptr, 0, w, nullptr); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid strength accepted");
        }
        for (int rows : {1, 3, 8, 4096}) test(rows);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "LoopCD parity: %s\n", e.what());
        return 1;
    }
    return 0;
}
