// Copyright (c) 2026 Niko1221 and the Strata contributors. MIT licensed.

#include "basalt/kernels/dflash2.hpp"

#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace basalt::kernels {
namespace {

constexpr int HD = kDf2Hd;

void check(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("dflash2 kernel ") + what + ": " + cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ void conv_kernel_stage(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride,
                                  const float *__restrict__ base, const float *__restrict__ resid, float *__restrict__ out, int rows, int H,
                                  int group, dim3 block_idx) {
    const int i = block_idx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H)
        return;
    const int b = i / H, c = i % H;
    const int groups = H / group;
    float acc = 0.0f;
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        if (b - o < 0)
            continue;
        const float k = base[o * H + c] + dyn[(size_t)b * dyn_stride + o * groups + c / group];
        acc = fmaf(k, x[(size_t)(b - o) * H + c], acc);
    }
    out[i] = resid ? resid[i] + acc : acc;
}
__global__ void conv_kernel(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride, const float *__restrict__ base,
                            const float *__restrict__ resid, float *__restrict__ out, int rows, int H, int group) {
    conv_kernel_stage(x, dyn, dyn_stride, base, resid, out, rows, H, group, blockIdx);
}

__device__ void rope_table_kernel_stage(const int32_t *__restrict__ pos, int rows, float theta, float *__restrict__ table, dim3 block_idx) {
    const int r = block_idx.x, i = threadIdx.x;
    if (r >= rows)
        return;
    const double inv = pow((double)theta, -(double)(2 * i) / (double)HD);
    double ang = (double)pos[r] * inv;
    const double two_pi = 6.283185307179586476925286766559;
    ang -= two_pi * rint(ang / two_pi);
    float s, c;
    sincosf((float)ang, &s, &c);
    table[(r * 2 + 0) * 64 + i] = c;
    table[(r * 2 + 1) * 64 + i] = s;
}
__global__ void rope_table_kernel(const int32_t *__restrict__ pos, int rows, float theta, float *__restrict__ table) {
    rope_table_kernel_stage(pos, rows, theta, table, blockIdx);
}

__device__ void norm_rope_kernel_stage(const float *__restrict__ in, int in_stride, const float *__restrict__ gamma,
                                       const int32_t *__restrict__ pos, const float *__restrict__ table, float *__restrict__ out,
                                       int slot_mod, int heads, float eps, dim3 block_idx) {
    const int r = block_idx.x, h = block_idx.y, d = threadIdx.x;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = in[(size_t)r * in_stride + h * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0)
        red[d >> 5] = ss;
    __syncthreads();
    const float tot = red[0] + red[1] + red[2] + red[3];
    const float v = x * rsqrtf(tot / (float)HD + eps) * gamma[d];
    float res = v;
    if (table != nullptr) {
        y[d] = v;
        __syncthreads();
        const int i = d & 63;
        const float c = table[(r * 2 + 0) * 64 + i], s = table[(r * 2 + 1) * 64 + i];
        res = d < 64 ? v * c - y[d + 64] * s : v * c + y[d - 64] * s;
    }
    const int slot = slot_mod > 0 ? (int)(pos[r] % slot_mod) : r;

    out[slot_mod > 0 ? ((size_t)h * slot_mod + slot) * HD + d : ((size_t)slot * heads + h) * HD + d] = res;
}
__global__ void norm_rope_kernel(const float *__restrict__ in, int in_stride, const float *__restrict__ gamma,
                                 const int32_t *__restrict__ pos, const float *__restrict__ table, float *__restrict__ out, int slot_mod,
                                 int heads, float eps) {
    norm_rope_kernel_stage(in, in_stride, gamma, pos, table, out, slot_mod, heads, eps, blockIdx);
}

__device__ void copy_rows_kernel_stage(const float *__restrict__ in, float *__restrict__ out, const int32_t *__restrict__ pos, int slot_mod,
                                       int rows, int cols, dim3 block_idx) {
    const int i = block_idx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols)
        return;
    const int r = i / cols, c = i % cols;
    const int slot = slot_mod > 0 ? (int)(pos[r] % slot_mod) : r;
    out[slot_mod > 0 ? ((size_t)(c / HD) * slot_mod + slot) * HD + c % HD : (size_t)slot * cols + c] = in[i];
}
__global__ void copy_rows_kernel(const float *__restrict__ in, float *__restrict__ out, const int32_t *__restrict__ pos, int slot_mod,
                                 int rows, int cols) {
    copy_rows_kernel_stage(in, out, pos, slot_mod, rows, cols, blockIdx);
}

constexpr int AS = 132;
constexpr int PS = 36;
__device__ void attn_part_kernel_stage(const float *__restrict__ q, const float *__restrict__ kring, const float *__restrict__ vring,
                                       const float *__restrict__ kblk, const float *__restrict__ vblk,
                                       const int32_t *__restrict__ anchor_pos, int ring_slots, int window, float *__restrict__ part,
                                       int rows, int variant, dim3 block_idx) {
    __shared__ __align__(16) float Qs[32 * AS];
    __shared__ __align__(16) float Ks[32 * AS];
    __shared__ __align__(16) float Ps[32 * PS];
    __shared__ float Sc[32 * 33];
    __shared__ float m_s[32], l_s[32], corr_s[32];
    const int kvh = block_idx.x, chunk = block_idx.y, tid = threadIdx.x;
    const int row_base = block_idx.z * kDf2Rows;
    const int s = *anchor_pos;
    const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
    const int nctx = s - lo;
    const int nkeys = nctx + rows;
    const int k_begin = chunk * kDf2Chunk;
    float *const pbase = part + ((((size_t)block_idx.z * kDf2MaxChunks + chunk) * kDf2Kv + kvh) * 32) * 130;
    if (k_begin >= nkeys) {
        for (int i = tid; i < 32 * 130; i += 128)
            pbase[i] = (i % 130 == 0) ? -INFINITY : 0.0f;
        return;
    }
    const float scale = rsqrtf((float)HD);
    for (int i = tid; i < 32 * HD / 4; i += 128) {
        const int a = i / (HD / 4), d4 = i % (HD / 4);
        const int row = row_base + (a >> 2), g = a & 3;
        float4 v = row < rows ? *reinterpret_cast<const float4 *>(q + ((size_t)row * 32 + kvh * kDf2Gqa + g) * HD + d4 * 4)
                              : make_float4(0.f, 0.f, 0.f, 0.f);
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        v.w *= scale;
        *reinterpret_cast<float4 *>(&Qs[a * AS + d4 * 4]) = v;
    }
    if (tid < 32) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    float O[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            O[i][j] = 0.0f;
    const int qg_pv = tid >> 4, dg_pv = tid & 15;
    const int k_end = min(k_begin + kDf2Chunk, nkeys);

    auto fetch = [&](int k0, float4(&reg)[8], const float *ring, const float *blk) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = tid + u * 128, t = i / (HD / 4), d4 = i % (HD / 4), k = k0 + t;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < nctx)
                v = *reinterpret_cast<const float4 *>(ring + ((size_t)kvh * ring_slots + (lo + k) % ring_slots) * HD + d4 * 4);
            else if (k < nkeys)
                v = *reinterpret_cast<const float4 *>(blk + ((size_t)(k - nctx) * kDf2Kv + kvh) * HD + d4 * 4);
            reg[u] = v;
        }
    };
    auto put = [&](const float4(&reg)[8]) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = tid + u * 128, t = i / (HD / 4), d4 = i % (HD / 4);
            *reinterpret_cast<float4 *>(&Ks[t * AS + d4 * 4]) = reg[u];
        }
    };
    float4 kreg[8], vreg[8];
    fetch(k_begin, kreg, kring, kblk);
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        __syncthreads();
        put(kreg);
        fetch(k0, vreg, vring, vblk);
        __syncthreads();
        if (tid < 64) {
            const int qg = tid >> 3, kg = tid & 7;
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = 0.0f;
#pragma unroll 4
            for (int d = 0; d < ((variant & 1) ? 0 : HD); d += 4) {
                float4 qv[4], kv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    qv[i] = *reinterpret_cast<const float4 *>(&Qs[(qg * 4 + i) * AS + d]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    kv[j] = *reinterpret_cast<const float4 *>(&Ks[(kg * 4 + j) * AS + d]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[i][j] += qv[i].x * kv[j].x + qv[i].y * kv[j].y + qv[i].z * kv[j].z + qv[i].w * kv[j].w;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int qi = qg * 4 + i, t = kg * 4 + j, k = k0 + t, row = row_base + (qi >> 2);
                    bool ok = k < nkeys;
                    if (ok && k < nctx)
                        ok = (lo + k) > (s + row - window);
                    Sc[qi * 33 + t] = ok ? acc[i][j] : -INFINITY;
                }
        }
        __syncthreads();
        {
            const int qi = tid >> 2, sub = tid & 3;
            float v[8], tmax = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                v[u] = Sc[qi * 33 + sub * 8 + u];
                tmax = fmaxf(tmax, v[u]);
            }
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float m_old = m_s[qi];
            const float mn = fmaxf(m_old, tmax);
            const float corr = (m_old == -INFINITY) ? 0.0f : __expf(m_old - mn);
            float psum = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float p = v[u] == -INFINITY ? 0.0f : __expf(v[u] - mn);
                Ps[(sub * 8 + u) * PS + qi] = p;
                psum += p;
            }
            psum += __shfl_xor_sync(0xffffffffu, psum, 1);
            psum += __shfl_xor_sync(0xffffffffu, psum, 2);
            if (sub == 0) {
                l_s[qi] = l_s[qi] * corr + psum;
                m_s[qi] = mn;
                corr_s[qi] = corr;
            }
        }
        __syncthreads();
        put(vreg);
        if (k0 + 32 < k_end)
            fetch(k0 + 32, kreg, kring, kblk);
        __syncthreads();
        {
            const float4 c4 = *reinterpret_cast<const float4 *>(&corr_s[qg_pv * 4]);
            const float cr[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    O[i][j] *= cr[i];
#pragma unroll 4
            for (int t = 0; t < ((variant & 2) ? 0 : 32); ++t) {
                const float4 p4 = *reinterpret_cast<const float4 *>(&Ps[t * PS + qg_pv * 4]);
                const float4 va = *reinterpret_cast<const float4 *>(&Ks[t * AS + dg_pv * 8]);
                const float4 vb = *reinterpret_cast<const float4 *>(&Ks[t * AS + dg_pv * 8 + 4]);
                const float pv[4] = {p4.x, p4.y, p4.z, p4.w};
                const float vv[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        O[i][j] = fmaf(pv[i], vv[j], O[i][j]);
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        pbase[tid * 130] = m_s[tid];
        pbase[tid * 130 + 1] = l_s[tid];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float *o = pbase + (size_t)(qg_pv * 4 + i) * 130 + 2 + dg_pv * 8;
        *reinterpret_cast<float2 *>(o) = make_float2(O[i][0], O[i][1]);
        *reinterpret_cast<float2 *>(o + 2) = make_float2(O[i][2], O[i][3]);
        *reinterpret_cast<float2 *>(o + 4) = make_float2(O[i][4], O[i][5]);
        *reinterpret_cast<float2 *>(o + 6) = make_float2(O[i][6], O[i][7]);
    }
}
__global__ void __launch_bounds__(128)
    attn_part_kernel(const float *__restrict__ q, const float *__restrict__ kring, const float *__restrict__ vring,
                     const float *__restrict__ kblk, const float *__restrict__ vblk, const int32_t *__restrict__ anchor_pos, int ring_slots,
                     int window, int rows, float *__restrict__ part, int variant) {
    attn_part_kernel_stage(q, kring, vring, kblk, vblk, anchor_pos, ring_slots, window, part, rows, variant, blockIdx);
}

__device__ __forceinline__ void store_input(float v, int blk, int col, int K, float *__restrict__ out);

__device__ void attn_merge_kernel_stage(const float *__restrict__ part, float *__restrict__ out, int rows, dim3 block_idx) {
    const int a = block_idx.x;
    const int kvh = a >> 5, qi = a & 31, d = threadIdx.x;
    const int row = block_idx.z * kDf2Rows + (qi >> 2), g = qi & 3;
    if (row >= rows)
        return;
    part += (size_t)block_idx.z * kDf2MaxChunks * kDf2Kv * 32 * 130;
    float M = -INFINITY;
    for (int c = 0; c < kDf2MaxChunks; ++c)
        M = fmaxf(M, part[(((size_t)c * kDf2Kv + kvh) * 32 + qi) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < kDf2MaxChunks; ++c) {
        const float *p = part + (((size_t)c * kDf2Kv + kvh) * 32 + qi) * 130;
        const float mc = p[0];
        const float w = (mc == -INFINITY) ? 0.0f : __expf(mc - M);
        L += p[1] * w;
        o = fmaf(p[2 + d], w, o);
    }

    store_input(L > 0.0f ? o / L : 0.0f, (kvh * kDf2Gqa + g) * (HD / 32) + (d >> 5), row, 32 * HD, out);
}
__global__ void attn_merge_kernel(const float *__restrict__ part, float *__restrict__ out, int rows) { attn_merge_kernel_stage(part, out, rows, blockIdx); }

__device__ __forceinline__ void store_input(float v, int blk, int col, int K, float *__restrict__ out) {
    out[(size_t)col * K + blk * 32 + (threadIdx.x & 31)] = v;
}

__device__ void copy_input_kernel_stage(const float *__restrict__ x, int K, float *__restrict__ out, dim3 block_idx) {
    const int blk = block_idx.x * 8 + (threadIdx.x >> 5), col = block_idx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= K)
        return;
    store_input(x[(size_t)col * K + blk * 32 + lane], blk, col, K, out);
}
__global__ void copy_input_kernel(const float *__restrict__ x, int K, float *__restrict__ out) {
    copy_input_kernel_stage(x, K, out, blockIdx);
}

__device__ void rms_input_kernel_stage(const float *__restrict__ h, const float *__restrict__ gamma, float *__restrict__ xn, int N,
                                       float eps, float *__restrict__ out, dim3 block_idx) {
    const int col = block_idx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float red[8];
    float ss = 0.0f;
    for (int i = tid; i < N; i += 256) {
        const float v = h[(size_t)col * N + i];
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0)
        red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < 8; ++w)
        tot += red[w];
    const float scale = rsqrtf(tot / (float)N + eps);
    for (int blk = warp; blk * 32 < N; blk += 8) {
        const int i = blk * 32 + lane;
        const float v = h[(size_t)col * N + i] * scale * gamma[i];
        if (xn)
            xn[(size_t)col * N + i] = v;
        store_input(v, blk, col, N, out);
    }
}
__global__ void rms_input_kernel(const float *__restrict__ h, const float *__restrict__ gamma, float *__restrict__ xn, int N, float eps,
                                 float *__restrict__ out) {
    rms_input_kernel_stage(h, gamma, xn, N, eps, out, blockIdx);
}

__device__ void conv_input_kernel_stage(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride,
                                        const float *__restrict__ base, int H, int group, float *__restrict__ out, dim3 block_idx) {
    const int blk = block_idx.x * 8 + (threadIdx.x >> 5), b = block_idx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= H)
        return;
    const int c = blk * 32 + lane, groups = H / group;
    float acc = 0.0f;
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        if (b - o < 0)
            continue;
        const float k = base[o * H + c] + dyn[(size_t)b * dyn_stride + o * groups + c / group];
        acc = fmaf(k, x[(size_t)(b - o) * H + c], acc);
    }
    store_input(acc, blk, b, H, out);
}
__global__ void conv_input_kernel(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride,
                                  const float *__restrict__ base, int H, int group, float *__restrict__ out) {
    conv_input_kernel_stage(x, dyn, dyn_stride, base, H, group, out, blockIdx);
}

__device__ void conv_rms_input_kernel_stage(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride,
                                            const float *__restrict__ base, float *__restrict__ h, const float *__restrict__ gamma,
                                            float *__restrict__ xn, int H, int group, float eps, float *__restrict__ out, dim3 block_idx) {
    const int b = block_idx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int groups = H / group, nb = H / 32;
    __shared__ float red[32];
    float v[3];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int blk = warp + j * 32;
        v[j] = 0.0f;
        if (blk < nb) {
            const int c = blk * 32 + lane;
            float acc = 0.0f;
#pragma unroll
            for (int o = 0; o < 2; ++o) {
                if (b - o < 0)
                    continue;
                const float k = base[o * H + c] + dyn[(size_t)b * dyn_stride + o * groups + c / group];
                acc = fmaf(k, x[(size_t)(b - o) * H + c], acc);
            }
            v[j] = h[(size_t)b * H + c] + acc;
            h[(size_t)b * H + c] = v[j];
            ss += v[j] * v[j];
        }
    }
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0)
        red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < 32; ++w)
        tot += red[w];
    const float scale = rsqrtf(tot / (float)H + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int blk = warp + j * 32;
        if (blk < nb) {
            const int c = blk * 32 + lane;
            const float y = v[j] * scale * gamma[c];
            if (xn)
                xn[(size_t)b * H + c] = y;
            store_input(y, blk, b, H, out);
        }
    }
}
__global__ void __launch_bounds__(1024)
    conv_rms_input_kernel(const float *__restrict__ x, const float *__restrict__ dyn, int dyn_stride, const float *__restrict__ base,
                          float *__restrict__ h, const float *__restrict__ gamma, float *__restrict__ xn, int H, int group, float eps,
                          float *__restrict__ out) {
    conv_rms_input_kernel_stage(x, dyn, dyn_stride, base, h, gamma, xn, H, group, eps, out, blockIdx);
}

__device__ void norm_rope_qk_kernel_stage(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ gq,
                                          const float *__restrict__ gk, const float *__restrict__ table, float *__restrict__ qo,
                                          float *__restrict__ ko, int q_heads, int k_heads, float eps, dim3 block_idx) {
    const int r = block_idx.x, hh = block_idx.y, d = threadIdx.x;
    const bool isq = hh < q_heads;
    const int h = isq ? hh : hh - q_heads, heads = isq ? q_heads : k_heads;
    const float *in = isq ? q : k;
    const float *gamma = isq ? gq : gk;
    float *out = isq ? qo : ko;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = in[((size_t)r * heads + h) * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0)
        red[d >> 5] = ss;
    __syncthreads();
    const float v = x * rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)HD + eps) * gamma[d];
    y[d] = v;
    __syncthreads();
    const int i = d & 63;
    const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
    out[((size_t)r * heads + h) * HD + d] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
}
__global__ void norm_rope_qk_kernel(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ gq,
                                    const float *__restrict__ gk, const float *__restrict__ table, float *__restrict__ qo,
                                    float *__restrict__ ko, int q_heads, int k_heads, float eps) {
    norm_rope_qk_kernel_stage(q, k, gq, gk, table, qo, ko, q_heads, k_heads, eps, blockIdx);
}

__device__ void kv_ring_kernel_stage(const float *__restrict__ kin, const float *__restrict__ vin, const float *__restrict__ gk,
                                     const int32_t *__restrict__ pos, const float *__restrict__ table, float *__restrict__ kring,
                                     float *__restrict__ vring, int slots, int heads, float eps, dim3 block_idx) {
    const int r = block_idx.x, h = block_idx.y, d = threadIdx.x;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = kin[((size_t)r * heads + h) * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0)
        red[d >> 5] = ss;
    __syncthreads();
    const float v = x * rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)HD + eps) * gk[d];
    y[d] = v;
    __syncthreads();
    const int i = d & 63;
    const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
    const int slot = (int)(pos[r] % slots);
    const size_t at = ((size_t)h * slots + slot) * HD + d;
    kring[at] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
    vring[at] = vin[((size_t)r * heads + h) * HD + d];
}
__global__ void kv_ring_kernel(const float *__restrict__ kin, const float *__restrict__ vin, const float *__restrict__ gk,
                               const int32_t *__restrict__ pos, const float *__restrict__ table, float *__restrict__ kring,
                               float *__restrict__ vring, int slots, int heads, float eps) {
    kv_ring_kernel_stage(kin, vin, gk, pos, table, kring, vring, slots, heads, eps, blockIdx);
}

__device__ void swiglu_input_kernel_stage(const float *__restrict__ gate, const float *__restrict__ up, int I, float *__restrict__ out,
                                          dim3 block_idx) {
    const int blk = block_idx.x * 8 + (threadIdx.x >> 5), b = block_idx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= I)
        return;
    const size_t i = (size_t)b * I + blk * 32 + lane;
    const float g = gate[i];
    store_input(g / (1.0f + __expf(-g)) * up[i], blk, b, I, out);
}
__global__ void swiglu_input_kernel(const float *__restrict__ gate, const float *__restrict__ up, int I, float *__restrict__ out) {
    swiglu_input_kernel_stage(gate, up, I, out, blockIdx);
}

constexpr int TK_CHUNK_MAX = 4096;
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

__device__ void topk_part_kernel_stage(const float *__restrict__ logits, int n, float *__restrict__ cval, int32_t *__restrict__ cidx,
                                       dim3 block_idx) {
    __shared__ float vals[TK_CHUNK_MAX];
    __shared__ float wv[8];
    __shared__ int wi[8];
    __shared__ int win;
    const int row = block_idx.x, blk = block_idx.y, tid = threadIdx.x;
    const int cs = (n + kDf2TopBlocks - 1) / kDf2TopBlocks;
    const int c0 = blk * cs, c1 = min(c0 + cs, n);
    for (int i = tid; i < cs; i += 256)
        vals[i] = (c0 + i < c1) ? logits[(size_t)row * n + c0 + i] : -INFINITY;
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < cs; i += 256)
            if (better(vals[i], i, bv, bi)) {
                bv = vals[i];
                bi = i;
            }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if ((tid & 31) == 0) {
            wv[tid >> 5] = bv;
            wi[tid >> 5] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) {
                    v = wv[w];
                    ix = wi[w];
                }
            const size_t o = ((size_t)row * kDf2TopBlocks + blk) * kDf2TopK + k;
            cval[o] = v;
            cidx[o] = v == -INFINITY ? -1 : c0 + ix;
            win = ix;
            if (v != -INFINITY)
                vals[ix] = -INFINITY;
        }
        __syncthreads();
    }
}
__global__ void topk_part_kernel(const float *__restrict__ logits, int n, float *__restrict__ cval, int32_t *__restrict__ cidx) {
    topk_part_kernel_stage(logits, n, cval, cidx, blockIdx);
}

__device__ void topk_merge_kernel_stage(const float *__restrict__ cval, const int32_t *__restrict__ cidx,
                                        const int32_t *__restrict__ idx_map, float *__restrict__ val, int32_t *__restrict__ tok,
                                        dim3 block_idx) {
    constexpr int NC = kDf2TopBlocks * kDf2TopK;
    __shared__ float sv[NC];
    __shared__ int si[NC];
    __shared__ float wv[8];
    __shared__ int wi[8], wp[8];
    const int row = block_idx.x, tid = threadIdx.x;
    for (int i = tid; i < NC; i += 256) {
        sv[i] = cval[(size_t)row * NC + i];
        si[i] = cidx[(size_t)row * NC + i];
    }
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for (int i = tid; i < NC; i += 256)
            if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) {
                bv = sv[i];
                bi = si[i];
                bp = i;
            }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bp = op;
            }
        }
        if ((tid & 31) == 0) {
            wv[tid >> 5] = bv;
            wi[tid >> 5] = bi;
            wp[tid >> 5] = bp;
        }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0], p = wp[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) {
                    v = wv[w];
                    ix = wi[w];
                    p = wp[w];
                }
            if (p >= 0) {
                si[p] = -1;
            }
            val[row * kDf2TopK + k] = v;
            tok[row * kDf2TopK + k] = p >= 0 ? (idx_map ? idx_map[ix] : ix) : 0;
        }
        __syncthreads();
    }
}
__global__ void topk_merge_kernel(const float *__restrict__ cval, const int32_t *__restrict__ cidx, const int32_t *__restrict__ idx_map,
                                  float *__restrict__ val, int32_t *__restrict__ tok) {
    topk_merge_kernel_stage(cval, cidx, idx_map, val, tok, blockIdx);
}

__device__ void select_kernel_stage(const float *__restrict__ hp, const float *__restrict__ unary, const int32_t *__restrict__ tok,
                                    const uint16_t *__restrict__ pred_cb, const uint16_t *__restrict__ succ_cb, int rank, int rows,
                                    const int32_t *__restrict__ anchor_tok, int32_t *__restrict__ out_tok, float *__restrict__ out_prob,
                                    dim3 block_idx) {
    __shared__ float vp[256];
    __shared__ float sc[kDf2TopK];
    __shared__ int prev;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    if (tid == 0)
        prev = *anchor_tok;
    __syncthreads();
    for (int i = 0; i < rows; ++i) {
        vp[tid] = bf16f(pred_cb[(size_t)prev * rank + tid]) * hp[(size_t)i * rank + tid];
        __syncthreads();
        for (int j = w * 2; j < w * 2 + 2; ++j) {
            const int t = tok[i * kDf2TopK + j];
            const uint4 raw = *reinterpret_cast<const uint4 *>(succ_cb + (size_t)t * rank + lane * 8);
            const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
            float acc = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc = fmaf(__uint_as_float(u[e] << 16), vp[lane * 8 + e * 2], acc);
                acc = fmaf(__uint_as_float(u[e] & 0xffff0000u), vp[lane * 8 + e * 2 + 1], acc);
            }
            for (int o = 16; o > 0; o >>= 1)
                acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0)
                sc[j] = unary[i * kDf2TopK + j] + acc;
        }
        __syncthreads();
        if (tid == 0) {
            int bj = 0;
            float bv = sc[0];
            for (int j = 1; j < kDf2TopK; ++j)
                if (sc[j] > bv) {
                    bv = sc[j];
                    bj = j;
                }
            float z = 0.0f;
            for (int j = 0; j < kDf2TopK; ++j)
                z += __expf(sc[j] - bv);
            out_tok[i] = tok[i * kDf2TopK + bj];
            out_prob[i] = 1.0f / z;
            prev = out_tok[i];
        }
        __syncthreads();
    }
}
__global__ void select_kernel(const float *__restrict__ hp, const float *__restrict__ unary, const int32_t *__restrict__ tok,
                              const uint16_t *__restrict__ pred_cb, const uint16_t *__restrict__ succ_cb, int rank, int rows,
                              const int32_t *__restrict__ anchor_tok, int32_t *__restrict__ out_tok, float *__restrict__ out_prob) {
    select_kernel_stage(hp, unary, tok, pred_cb, succ_cb, rank, rows, anchor_tok, out_tok, out_prob, blockIdx);
}

__device__ void tap_rows_kernel_stage(const float *__restrict__ R, float *__restrict__ out, int64_t out_stride, int64_t n, int hc,
                                      dim3 block_idx) {
    const int64_t i = (int64_t)block_idx.x * blockDim.x + threadIdx.x;
    const int r = block_idx.y;
    if (i >= n)
        return;
    float s = 0.0f;
    for (int c = 0; c < hc; ++c)
        s += R[((size_t)r * hc + c) * n + i];
    out[(size_t)r * out_stride + i] = s / (float)hc;
}
__global__ void tap_rows_kernel(const float *__restrict__ R, float *__restrict__ out, int64_t out_stride, int64_t n, int hc) {
    tap_rows_kernel_stage(R, out, out_stride, n, hc, blockIdx);
}

__device__ void tap_gr_kernel_stage(const float *__restrict__ R, int64_t R_stride, const float *__restrict__ bo, int64_t bo_stride,
                                    const float *__restrict__ inject, int64_t inj_stride, float *__restrict__ out, int64_t out_stride,
                                    int64_t n, int hc, dim3 block_idx) {
    const int64_t i = (int64_t)block_idx.x * blockDim.x + threadIdx.x;
    const int r = block_idx.y;
    if (i >= n)
        return;
    float wsum = 0.0f, rs = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const float x = inject[(size_t)r * inj_stride + c] / (float)hc;
        wsum += 2.0f / (1.0f + expf(-x));
        rs += R[(size_t)r * R_stride + (size_t)c * n + i];
    }
    out[(size_t)r * out_stride + i] = (rs + bo[(size_t)r * bo_stride + i] * wsum) / (float)hc;
}
__global__ void tap_gr_kernel(const float *__restrict__ R, int64_t R_stride, const float *__restrict__ bo, int64_t bo_stride,
                              const float *__restrict__ inject, int64_t inj_stride, float *__restrict__ out, int64_t out_stride, int64_t n,
                              int hc) {
    tap_gr_kernel_stage(R, R_stride, bo, bo_stride, inject, inj_stride, out, out_stride, n, hc, blockIdx);
}

} // namespace

void df2_tap_gr(const float *R, int64_t R_stride, const float *bo, int64_t bo_stride, const float *inject, int64_t inj_stride, float *out,
                int64_t out_stride, int rows, int64_t n_embd, int hc, void *stream) {
    tap_gr_kernel<<<dim3((unsigned)((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t)stream>>>(R, R_stride, bo, bo_stride, inject,
                                                                                                  inj_stride, out, out_stride, n_embd, hc);
    check("tap_gr");
}

void df2_conv(const float *x, const float *dyn, int dyn_stride, const float *base, const float *resid, float *out, int rows, int H,
              int group, void *stream) {
    const int n = rows * H;
    conv_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(x, dyn, dyn_stride, base, resid, out, rows, H, group);
    check("conv");
}

void df2_rope_table(const int32_t *pos, int rows, float theta, float *table, void *stream) {
    rope_table_kernel<<<rows, 64, 0, (cudaStream_t)stream>>>(pos, rows, theta, table);
    check("rope_table");
}

void df2_norm_rope(const float *in, int in_stride, const float *gamma, const int32_t *pos, const float *rope_table, float *out,
                   int slot_mod, int rows, int heads, float eps, void *stream) {
    norm_rope_kernel<<<dim3(rows, heads), HD, 0, (cudaStream_t)stream>>>(in, in_stride, gamma, pos, rope_table, out, slot_mod, heads, eps);
    check("norm_rope");
}

void df2_copy_rows(const float *in, float *out, const int32_t *pos, int slot_mod, int rows, int cols, void *stream) {
    const int n = rows * cols;
    copy_rows_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(in, out, pos, slot_mod, rows, cols);
    check("copy_rows");
}

void df2_attention(const float *q, const float *kring, const float *vring, const float *kblk, const float *vblk, const int32_t *anchor_pos,
                   int ring_slots, int window, int rows, float *part, float *out, void *stream) {
    if (rows < 2 || rows > 16)
        throw std::invalid_argument("df2_attention: rows out of range");
    const int tiles = (rows + kDf2Rows - 1) / kDf2Rows;
    attn_part_kernel<<<dim3(kDf2Kv, kDf2MaxChunks, tiles), 128, 0, (cudaStream_t)stream>>>(q, kring, vring, kblk, vblk, anchor_pos, ring_slots,
                                                                                    window, rows, part, 0);
    check("attn_part");
    attn_merge_kernel<<<dim3(kDf2Kv * 32, 1, tiles), HD, 0, (cudaStream_t)stream>>>(part, static_cast<float *>(out), rows);
    check("attn_merge");
}

void df2_conv_rms_input(const float *x, const float *dyn, int dyn_stride, const float *base, float *h, const float *gamma, float *xn,
                        int rows, int H, int group, float eps, float *out, void *stream) {
    if (H > 3 * 1024)
        throw std::invalid_argument("df2_conv_rms_input: H too large");
    conv_rms_input_kernel<<<rows, 1024, 0, (cudaStream_t)stream>>>(x, dyn, dyn_stride, base, h, gamma, xn, H, group, eps,
                                                                   static_cast<float *>(out));
    check("conv_rms_input");
}

void df2_norm_rope_qk(const float *q, const float *k, const float *gq, const float *gk, const float *table, float *qo, float *ko, int rows,
                      int q_heads, int k_heads, float eps, void *stream) {
    norm_rope_qk_kernel<<<dim3(rows, q_heads + k_heads), HD, 0, (cudaStream_t)stream>>>(q, k, gq, gk, table, qo, ko, q_heads, k_heads, eps);
    check("norm_rope_qk");
}

void df2_kv_ring(const float *kin, const float *vin, const float *gk, const int32_t *pos, const float *table, float *kring, float *vring,
                 int slots, int rows, int heads, float eps, void *stream) {
    kv_ring_kernel<<<dim3(rows, heads), HD, 0, (cudaStream_t)stream>>>(kin, vin, gk, pos, table, kring, vring, slots, heads, eps);
    check("kv_ring");
}

void df2_copy_input(const float *x, int K, int ncols, float *out, void *stream) {
    copy_input_kernel<<<dim3((K / 32 + 7) / 8, ncols), 256, 0, (cudaStream_t)stream>>>(x, K, static_cast<float *>(out));
    check("copy_input");
}

void df2_rms_input(const float *h, const float *gamma, float *xn, int rows, int N, float eps, float *out, void *stream) {
    rms_input_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(h, gamma, xn, N, eps, static_cast<float *>(out));
    check("rms_input");
}

void df2_conv_input(const float *x, const float *dyn, int dyn_stride, const float *base, int rows, int H, int group, float *out,
                    void *stream) {
    conv_input_kernel<<<dim3((H / 32 + 7) / 8, rows), 256, 0, (cudaStream_t)stream>>>(x, dyn, dyn_stride, base, H, group,
                                                                                      static_cast<float *>(out));
    check("conv_input");
}

void df2_swiglu_input(const float *gate, const float *up, int rows, int I, float *out, void *stream) {
    swiglu_input_kernel<<<dim3((I / 32 + 7) / 8, rows), 256, 0, (cudaStream_t)stream>>>(gate, up, I, static_cast<float *>(out));
    check("swiglu_input");
}

void df2_topk(const float *logits, int n, const int32_t *idx_map, int rows, uint32_t *scratch, float *val, int32_t *tok, void *stream) {
    if ((n + kDf2TopBlocks - 1) / kDf2TopBlocks > TK_CHUNK_MAX)
        throw std::invalid_argument("df2_topk: too many logits per row");
    float *cval = reinterpret_cast<float *>(scratch);
    int32_t *cidx = reinterpret_cast<int32_t *>(scratch + (size_t)rows * kDf2TopBlocks * kDf2TopK);
    topk_part_kernel<<<dim3(rows, kDf2TopBlocks), 256, 0, (cudaStream_t)stream>>>(logits, n, cval, cidx);
    check("topk_part");
    topk_merge_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(cval, cidx, idx_map, val, tok);
    check("topk_merge");
}

void df2_select(const float *hp, const float *unary, const int32_t *tok, const uint16_t *pred_cb, const uint16_t *succ_cb, int rank,
                int rows, const int32_t *anchor_tok, int32_t *out_tok, float *out_prob, void *stream) {
    if (rank != 256)
        throw std::invalid_argument("df2_select: the selector rank must be 256");
    select_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(hp, unary, tok, pred_cb, succ_cb, rank, rows, anchor_tok, out_tok, out_prob);
    check("select");
}

void df2_tap_rows(const float *R, float *out, int64_t out_stride, int64_t n_embd, int hc, int rows, void *stream) {
    tap_rows_kernel<<<dim3((unsigned)((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t)stream>>>(R, out, out_stride, n_embd, hc);
    check("tap_rows");
}

} // namespace basalt::kernels
