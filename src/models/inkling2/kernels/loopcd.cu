// Fixed hidden-state LoopCD over the coda's pass-dependent block states.
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

__device__ __forceinline__ void hidden_stage(float* mid, float* out, const float* ref_mid, const float* ref_out,
                                             int n, float strength, int block, int thread) {
    const int i = block * 256 + thread;
    if (i < n) {
        const float m = mid[i], h = out[i];
        mid[i] = __fadd_rn(m, __fmul_rn(strength, __fsub_rn(m, ref_mid[i])));
        out[i] = __fadd_rn(h, __fmul_rn(strength, __fsub_rn(h, ref_out[i])));
    }
}

__global__ void hidden_kernel(float* mid, float* out, const float* ref_mid, const float* ref_out, int n, float strength) {
    basalt::kernels::pdl_wait();
    hidden_stage(mid, out, ref_mid, ref_out, n, strength, blockIdx.x, threadIdx.x);
    basalt::kernels::pdl_launch_dependents();
}

}  // namespace

void loopcd_hidden(float* mid, float* out, const float* ref_mid, const float* ref_out, int rows, float strength, void* stream) {
    if (!std::isfinite(strength) || strength < 0.f || strength > 1.f)
        throw std::invalid_argument("inkling2 LoopCD: strength must be finite and in [0, 1]");
    if (rows <= 0 || strength == 0.f) return;
    const int n = rows * kWidth;
    basalt::kernels::launch_pdl(hidden_kernel, dim3((n + 255) / 256), dim3(256), 0, (cudaStream_t) stream,
                               mid, out, ref_mid, ref_out, n, strength);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 LoopCD: ") + cudaGetErrorString(e));
}

}  // namespace basalt::models::inkling2
