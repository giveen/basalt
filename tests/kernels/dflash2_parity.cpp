#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/dflash2.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace basalt::kernels;
static void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Dev {
  T *p = nullptr;
  size_t n;
  explicit Dev(size_t count) : n(count) {
    ck(cudaMalloc((void **)&p, n * sizeof(T)));
  }
  explicit Dev(const std::vector<T> &x) : Dev(x.size()) {
    ck(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Dev() { cudaFree(p); }
  std::vector<T> get() const {
    std::vector<T> x(n);
    ck(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return x;
  }
};
static std::mt19937 rng(95);
static std::vector<float> randoms(size_t n) {
  std::vector<float> x(n);
  for (auto &v : x) {
    const float unit = std::min((float)rng() * (1.0f / 4294967296.0f), 16777215.0f / 16777216.0f);
    v = unit * 0.5f - 0.25f;
  }
  return x;
}
static void compare(const char *name, const std::vector<float> &got,
                    const std::vector<double> &ref, double tol = 2e-5) {
  double ae = 0, re = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::abs((double)got[i] - ref[i]);
    if (!std::isfinite(got[i]))
      throw std::runtime_error(name);
    ae = std::max(ae, e);
    re = std::max(re, e / std::max(1e-12, std::abs(ref[i])));
  }
  std::printf("%-24s max_abs %.9g max_rel %.9g\n", name, ae, re);
  if (ae > tol)
    throw std::runtime_error(name);
}
static double bf(uint16_t x) {
  uint32_t b = (uint32_t)x << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

template <class Fn>
static void bench(const char *name, size_t bytes, Fn launch) {
  for (int i = 0; i < 3; ++i)
    launch();
  cudaEvent_t a, b;
  ck(cudaEventCreate(&a));
  ck(cudaEventCreate(&b));
  ck(cudaEventRecord(a));
  for (int i = 0; i < 50; ++i)
    launch();
  ck(cudaEventRecord(b));
  ck(cudaEventSynchronize(b));
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, a, b));
  const double us = ms * 20.0;
  std::printf("%-24s %.3f us/call, %.3f GB/s payload\n", name, us,
              bytes / (us * 1000));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

static void projection() {
  constexpr int IN = 12800, OUT = 2560, B = 8;
  auto weights = randoms((size_t)IN * OUT), x = randoms(B * IN);
  std::vector<uint16_t> w(weights.size());
  for (size_t i = 0; i < w.size(); ++i) {
    uint32_t b;
    std::memcpy(&b, &weights[i], 4);
    w[i] = (uint16_t)(b >> 16);
  }
  Dev<uint16_t> dw(w);
  Dev<float> dx(x), out(B * OUT), single(OUT);
  bf16_gemv_fp32_mmvf_multi(dx.p, IN, dw.p, out.p, OUT, IN, OUT, B, nullptr);
  const auto got = out.get();
  std::vector<double> ref(B * OUT);
  for (int row = 0; row < B; ++row)
    for (int o = 0; o < OUT; ++o)
      for (int i = 0; i < IN; ++i)
        ref[row * OUT + o] += bf(w[(size_t)o * IN + i]) * x[row * IN + i];
  compare("BF16 feature projection", got, ref, 1e-4);
  for (int row = 0; row < B; ++row) {
    bf16_gemv_fp32_mmvf(dx.p + row * IN, dw.p, single.p, IN, OUT, nullptr);
    const auto one = single.get();
    if (std::memcmp(one.data(), got.data() + row * OUT, OUT * 4))
      throw std::runtime_error("projection batch invariance");
  }
  std::puts(
      "BF16 projection          exact vs current single-row Basalt kernel");
  bench("BF16 projection", w.size() * 2 + x.size() * 4 + got.size() * 4, [&] {
    bf16_gemv_fp32_mmvf_multi(dx.p, IN, dw.p, out.p, OUT, IN, OUT, B, nullptr);
  });
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = (float)bf(w[i]);
  const auto type = GGML_TYPE_Q4_K;
  const char *label = "Q4_K";
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = (float)bf(w[i]);
  std::vector<uint8_t> packed((size_t)OUT * IN / 256 * 144);
  if (ggml_quantize_chunk(type, weights.data(), packed.data(), 0, OUT, IN) !=
      packed.size())
    throw std::runtime_error("quantized byte count");
  Dev<uint8_t> dpacked(packed), x8(native_q8_1_bytes(IN, B));
  cudaStream_t cs;
  ck(cudaStreamCreate(&cs));
  native_quantize_q8_1(dx.p, x8.p, IN, B, cs);
  dense_tc(type, dpacked.p, x8.p, out.p, IN, OUT, B, cs);
  ck(cudaStreamSynchronize(cs));
  const auto qgot = out.get();
  const auto activation = x8.get();
  ggml_get_type_traits(type)->to_float(packed.data(), weights.data(),
                                       (int64_t)OUT * IN);
  std::vector<double> qref(B * OUT);
  std::vector<double> qinput(B * IN);
  for (size_t i = 0; i < qinput.size(); ++i) {
    const size_t block = i / 32 * 36;
    uint16_t scale;
    std::memcpy(&scale, activation.data() + block, 2);
    qinput[i] =
        ggml_fp16_to_fp32(scale) * (int8_t)activation[block + 4 + i % 32];
  }
  for (int row = 0; row < B; ++row)
    for (int o = 0; o < OUT; ++o)
      for (int i = 0; i < IN; ++i) {
        qref[row * OUT + o] +=
            weights[(size_t)o * IN + i] * qinput[row * IN + i];
      }
  compare((std::string(label) + " vs double CPU").c_str(), qgot, qref, 2e-4);
  native_mmvq(type, dpacked.p, x8.p, out.p, IN, OUT, B, cs);
  ck(cudaStreamSynchronize(cs));
  const auto old = out.get();
  compare((std::string(label) + " vs current mmvq").c_str(), qgot,
          std::vector<double>(old.begin(), old.end()), 2e-2);
  for (int width = 1; width <= B; ++width) {
    dense_tc(type, dpacked.p, x8.p, out.p, IN, OUT, width, cs);
    ck(cudaStreamSynchronize(cs));
    const auto current = out.get();
    if (std::memcmp(current.data(), qgot.data(), (size_t)width * OUT * 4))
      throw std::runtime_error("quantized projection batch invariance");
  }
  bench((std::string(label) + " projection").c_str(),
        packed.size() + activation.size() + qgot.size() * 4,
        [&] { dense_tc(type, dpacked.p, x8.p, out.p, IN, OUT, B, cs); });
  ck(cudaStreamSynchronize(cs));
  ck(cudaStreamDestroy(cs));
}

static void elementwise(int B) {
  constexpr int H = 2560, G = 16, I = 7680;
  auto x = randoms(B * H), dyn = randoms(B * 640), base = randoms(2 * H),
       gamma = randoms(H);
  Dev<float> dx(x), dd(dyn), db(base), dg(gamma), y(B * I), xn(B * H),
      scales(B * I);
  std::vector<double> conv(B * H), norm(B * H);
  for (int r = 0; r < B; ++r)
    for (int c = 0; c < H; ++c) {
      for (int t = 0; t < 2 && t <= r; ++t)
        conv[r * H + c] +=
            ((double)base[t * H + c] + dyn[r * 640 + t * (H / G) + c / G]) *
            x[(r - t) * H + c];
    }
  df2_conv(dx.p, dd.p, 640, db.p, nullptr, y.p, B, H, G, nullptr);
  compare("conv", y.get(), conv);
  df2_conv_input(dx.p, dd.p, 640, db.p, B, H, G, y.p, nullptr);
  compare("conv input", y.get(), conv);
  for (int r = 0; r < B; ++r) {
    double ss = 0;
    for (int c = 0; c < H; ++c)
      ss += (double)x[r * H + c] * x[r * H + c];
    for (int c = 0; c < H; ++c)
      norm[r * H + c] = x[r * H + c] / std::sqrt(ss / H + 1e-6) * gamma[c];
  }
  df2_rms_input(dx.p, dg.p, xn.p, B, H, 1e-6f, y.p, nullptr);
  const auto full = y.get();
  compare("rms", full, norm);
  for (int width : {1, 3, 7}) {
    df2_rms_input(dx.p, dg.p, xn.p, width, H, 1e-6f, y.p, nullptr);
    const auto part = y.get();
    if (std::memcmp(part.data(), full.data(), (size_t)width * H * 4))
      throw std::runtime_error("rms batch invariance");
  }
  auto gate = randoms(B * I), up = randoms(B * I);
  Dev<float> g(gate), u(up);
  std::vector<double> sw(B * I);
  for (size_t i = 0; i < sw.size(); ++i)
    sw[i] = gate[i] / (1 + std::exp(-(double)gate[i])) * up[i];
  df2_swiglu_input(g.p, u.p, B, I, y.p, nullptr);
  compare("swiglu", y.get(), sw);
  df2_copy_input(dx.p, H, B, y.p, nullptr);
  compare("input copy", y.get(), std::vector<double>(x.begin(), x.end()), 0);
  Dev<float> residual(x);
  df2_conv_rms_input(dx.p, dd.p, 640, db.p, residual.p, dg.p, xn.p, B, H, G,
                     1e-6f, y.p, nullptr);
  for (int r = 0; r < B; ++r) {
    double ss = 0;
    for (int c = 0; c < H; ++c) {
      const double v = x[r * H + c] + conv[r * H + c];
      ss += v * v;
    }
    for (int c = 0; c < H; ++c)
      norm[r * H + c] = (x[r * H + c] + conv[r * H + c]) /
                        std::sqrt(ss / H + 1e-6) * gamma[c];
  }
  compare("conv residual rms", y.get(), norm);
  bench("conv", (x.size() * 2 + dyn.size() + base.size()) * 4, [&] {
    df2_conv(dx.p, dd.p, 640, db.p, nullptr, y.p, B, H, G, nullptr);
  });
  bench("conv input", (x.size() * 2 + dyn.size() + base.size()) * 4,
        [&] { df2_conv_input(dx.p, dd.p, 640, db.p, B, H, G, y.p, nullptr); });
  bench("rms input", (x.size() * 3 + gamma.size()) * 4,
        [&] { df2_rms_input(dx.p, dg.p, xn.p, B, H, 1e-6f, y.p, nullptr); });
  bench("input copy", x.size() * 8,
        [&] { df2_copy_input(dx.p, H, B, y.p, nullptr); });
  bench("swiglu input", gate.size() * 12,
        [&] { df2_swiglu_input(g.p, u.p, B, I, y.p, nullptr); });
  bench("conv residual rms",
        (x.size() * 5 + dyn.size() + base.size() + gamma.size()) * 4, [&] {
          df2_conv_rms_input(dx.p, dd.p, 640, db.p, residual.p, dg.p, xn.p, B,
                             H, G, 1e-6f, y.p, nullptr);
        });
}

static void attention(int B) {
  constexpr int HD = 128, KV = 8, Q = 32, RING = 2064, WINDOW = 2048;
  auto q = randoms(B * Q * HD), k = randoms(KV * RING * HD),
       v = randoms(KV * RING * HD);
  auto kb = randoms(B * KV * HD), vb = randoms(B * KV * HD);
  Dev<float> dq(q), dk(k), dv(v), dkb(kb), dvb(vb), out(B * Q * HD),
      scales(B * Q * HD);
  Dev<float> part((size_t)((B + 7) / 8) * kDf2MaxChunks * KV * 32 * 130);
  Dev<int32_t> pos(std::vector<int32_t>{2090});
  df2_attention(dq.p, dk.p, dv.p, dkb.p, dvb.p, pos.p, RING, WINDOW, B, part.p,
                out.p, nullptr);
  std::vector<double> ref(B * Q * HD);
  for (int r = 0; r < B; ++r)
    for (int h = 0; h < Q; ++h) {
      const int lo = std::max(0, 2090 + r - WINDOW + 1), nctx = 2090 - lo;
      std::vector<double> logits(nctx + B);
      auto key = [&](int j) {
        return j < nctx ? &k[((size_t)(h / 4) * RING + (lo + j) % RING) * HD]
                        : &kb[((size_t)(j - nctx) * KV + h / 4) * HD];
      };
      auto val = [&](int j) {
        return j < nctx ? &v[((size_t)(h / 4) * RING + (lo + j) % RING) * HD]
                        : &vb[((size_t)(j - nctx) * KV + h / 4) * HD];
      };
      for (int j = 0; j < nctx + B; ++j)
        for (int d = 0; d < HD; ++d)
          logits[j] += (double)q[((size_t)r * Q + h) * HD + d] * key(j)[d] /
                       std::sqrt((double)HD);
      const double m = *std::max_element(logits.begin(), logits.end());
      double z = 0;
      for (double &p : logits) {
        p = std::exp(p - m);
        z += p;
      }
      for (int j = 0; j < nctx + B; ++j)
        for (int d = 0; d < HD; ++d)
          ref[((size_t)r * Q + h) * HD + d] += logits[j] / z * val(j)[d];
    }
  compare("windowed attention", out.get(), ref);
  uint64_t hash = 14695981039346656037ull;
  const auto attention_output = out.get();
  for (size_t i = 0; i < attention_output.size() * sizeof(float); ++i) {
    hash ^= reinterpret_cast<const uint8_t *>(attention_output.data())[i];
    hash *= 1099511628211ull;
  }
  std::printf("attention frozen hash %016llx\n", (unsigned long long)hash);
  if (B == 8 && hash != 0xc866564303e32fd0ull)
    throw std::runtime_error(
        "attention changed from frozen eight-row Basalt output");
  cudaEvent_t a, b;
  ck(cudaEventCreate(&a));
  ck(cudaEventCreate(&b));
  ck(cudaEventRecord(a));
  for (int i = 0; i < 100; ++i)
    df2_attention(dq.p, dk.p, dv.p, dkb.p, dvb.p, pos.p, RING, WINDOW, B,
                  part.p, out.p, nullptr);
  ck(cudaEventRecord(b));
  ck(cudaEventSynchronize(b));
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, a, b));
  const double us = ms * 10.0;
  std::printf("attention microbench %.3f us/call, %.3f GB/s K/V payload\n", us,
              (double)KV * (WINDOW - 1 + B) * HD * 8 / (us * 1000));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

static void rotary(int B) {
  constexpr int HD = 128, Q = 32, KV = 8, RING = 2064;
  std::vector<int32_t> positions = {0,    1,      31,     4096,
                                    8192, 131071, 262143, 262144};
  for (int32_t next = positions.back() + 1; (int)positions.size() < B; ++next)
    if (std::none_of(positions.begin(), positions.end(),
                     [&](int32_t p) { return p % RING == next % RING; }))
      positions.push_back(next);
  auto q = randoms(B * Q * HD), k = randoms(B * KV * HD),
       v = randoms(B * KV * HD);
  auto gq = randoms(HD), gk = randoms(HD);
  Dev<float> dq(q), dk(k), dv(v), dgq(gq), dgk(gk), table(B * HD),
      qo(B * Q * HD), ko(B * KV * HD);
  Dev<int32_t> pos(positions);
  df2_rope_table(pos.p, B, 10000, table.p, nullptr);
  auto reference = [&](const std::vector<float> &x,
                       const std::vector<float> &gamma, int heads) {
    std::vector<double> y(x.size());
    for (int r = 0; r < B; ++r)
      for (int h = 0; h < heads; ++h) {
        double sum = 0;
        for (int d = 0; d < HD; ++d) {
          const double a = x[(r * heads + h) * HD + d];
          sum += a * a;
        }
        const double scale = 1 / std::sqrt(sum / HD + 1e-6);
        for (int d = 0; d < 64; ++d) {
          const double angle = positions[r] * std::pow(10000.0, -2.0 * d / HD);
          const double a = x[(r * heads + h) * HD + d] * scale * gamma[d];
          const double b =
              x[(r * heads + h) * HD + d + 64] * scale * gamma[d + 64];
          y[(r * heads + h) * HD + d] =
              a * std::cos(angle) - b * std::sin(angle);
          y[(r * heads + h) * HD + d + 64] =
              b * std::cos(angle) + a * std::sin(angle);
        }
      }
    return y;
  };
  const auto qr = reference(q, gq, Q), kr = reference(k, gk, KV);
  df2_norm_rope(dq.p, Q * HD, dgq.p, pos.p, table.p, qo.p, 0, B, Q, 1e-6f,
                nullptr);
  compare("norm rotary", qo.get(), qr);
  df2_norm_rope_qk(dq.p, dk.p, dgq.p, dgk.p, table.p, qo.p, ko.p, B, Q, KV,
                   1e-6f, nullptr);
  compare("query rotary", qo.get(), qr);
  compare("key rotary", ko.get(), kr);
  Dev<float> kring((size_t)KV * RING * HD), vring((size_t)KV * RING * HD);
  ck(cudaMemset(kring.p, 0, kring.n * 4));
  ck(cudaMemset(vring.p, 0, vring.n * 4));
  df2_kv_ring(dk.p, dv.p, dgk.p, pos.p, table.p, kring.p, vring.p, RING, B, KV,
              1e-6f, nullptr);
  std::vector<double> keys(kring.n), vals(vring.n);
  for (int r = 0; r < B; ++r)
    for (int h = 0; h < KV; ++h)
      for (int d = 0; d < HD; ++d) {
        const size_t at = ((size_t)h * RING + positions[r] % RING) * HD + d;
        keys[at] = kr[(r * KV + h) * HD + d];
        vals[at] = v[(r * KV + h) * HD + d];
      }
  compare("ring keys", kring.get(), keys);
  compare("ring values", vring.get(), vals, 0);
  ck(cudaMemset(vring.p, 0, vring.n * 4));
  df2_copy_rows(dv.p, vring.p, pos.p, RING, B, KV * HD, nullptr);
  compare("ring row copy", vring.get(), vals, 0);
  bench("rotary table", B * HD * 4,
        [&] { df2_rope_table(pos.p, B, 10000, table.p, nullptr); });
  bench("norm rotary", q.size() * 8 + gq.size() * 4, [&] {
    df2_norm_rope(dq.p, Q * HD, dgq.p, pos.p, table.p, qo.p, 0, B, Q, 1e-6f,
                  nullptr);
  });
  bench("query/key rotary", (q.size() + k.size()) * 8, [&] {
    df2_norm_rope_qk(dq.p, dk.p, dgq.p, dgk.p, table.p, qo.p, ko.p, B, Q, KV,
                     1e-6f, nullptr);
  });
  bench("ring key/value", (k.size() + v.size()) * 8, [&] {
    df2_kv_ring(dk.p, dv.p, dgk.p, pos.p, table.p, kring.p, vring.p, RING, B,
                KV, 1e-6f, nullptr);
  });
  bench("ring row copy", v.size() * 8, [&] {
    df2_copy_rows(dv.p, vring.p, pos.p, RING, B, KV * HD, nullptr);
  });
}

static void taps() {
  constexpr int B = 8, H = 2560, HC = 4;
  auto r = randoms(B * H * HC), bo = randoms(B * H), inj = randoms(B * HC);
  Dev<float> dr(r), db(bo), di(inj), out(B * H);
  std::vector<double> ref(B * H);
  for (int row = 0; row < B; ++row)
    for (int c = 0; c < H; ++c)
      for (int h = 0; h < HC; ++h)
        ref[row * H + c] += r[(row * HC + h) * H + c] / (double)HC;
  df2_tap_rows(dr.p, out.p, H, H, HC, B, nullptr);
  compare("feature mean", out.get(), ref);
  for (int row = 0; row < B; ++row)
    for (int c = 0; c < H; ++c)
      for (int h = 0; h < HC; ++h)
        ref[row * H + c] += bo[row * H + c] * 2.0 /
                            (1 + std::exp(-inj[row * HC + h] / (double)HC)) /
                            HC;
  df2_tap_gr(dr.p, H * HC, db.p, H, di.p, HC, out.p, H, B, H, HC, nullptr);
  compare("completed feature", out.get(), ref);
  bench("feature mean", (r.size() + ref.size()) * 4,
        [&] { df2_tap_rows(dr.p, out.p, H, H, HC, B, nullptr); });
  bench("completed feature",
        (r.size() + bo.size() + inj.size() + ref.size()) * 4, [&] {
          df2_tap_gr(dr.p, H * HC, db.p, H, di.p, HC, out.p, H, B, H, HC,
                     nullptr);
        });
}

static void topk_select(int B) {
  constexpr int V = 248320, K = 16, R = 256;
  auto logits = randoms((size_t)B * V);
  for (int i = 0; i < 32; ++i)
    logits[i] = 1;
  Dev<float> dl(logits), val(B * K);
  Dev<int32_t> tok(B * K);
  Dev<uint32_t> scratch(B * kDf2TopBlocks * K * 2);
  df2_topk(dl.p, V, nullptr, B, scratch.p, val.p, tok.p, nullptr);
  const auto ids = tok.get();
  const auto values = val.get();
  for (int row = 0; row < B; ++row) {
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + K, order.end(),
                      [&](int a, int b) {
                        const float x = logits[(size_t)row * V + a],
                                    y = logits[(size_t)row * V + b];
                        return x > y || (x == y && a < b);
                      });
    for (int i = 0; i < K; ++i)
      if (ids[row * K + i] != order[i] ||
          values[row * K + i] != logits[(size_t)row * V + order[i]])
        throw std::runtime_error("top-k");
  }
  std::printf(
      "top-k                    exact IDs and logits, including ties\n");
  auto hp = randoms(B * R), unary = randoms(B * K);
  std::vector<int32_t> candidates(B * K);
  for (size_t i = 0; i < candidates.size(); ++i)
    candidates[i] = (int32_t)(i % 32);
  std::vector<uint16_t> pred(32 * R), succ(32 * R);
  for (size_t i = 0; i < pred.size(); ++i) {
    pred[i] = (uint16_t)(0x3c00 + i % 512);
    succ[i] = (uint16_t)(0x3c00 + i % 300);
  }
  Dev<float> dh(hp), du(unary), probs(B);
  Dev<int32_t> dc(candidates), anchor(std::vector<int32_t>{3}), out(B);
  Dev<uint16_t> dp(pred), ds(succ);
  df2_select(dh.p, du.p, dc.p, dp.p, ds.p, R, B, anchor.p, out.p, probs.p,
             nullptr);
  const auto got = out.get();
  std::vector<double> ref(B);
  int previous = 3;
  for (int row = 0; row < B; ++row) {
    double scores[K];
    for (int j = 0; j < K; ++j) {
      scores[j] = unary[row * K + j];
      for (int d = 0; d < R; ++d)
        scores[j] += bf(pred[previous * R + d]) * hp[row * R + d] *
                     bf(succ[candidates[row * K + j] * R + d]);
    }
    const int best = (int)(std::max_element(scores, scores + K) - scores);
    double z = 0;
    for (double score : scores)
      z += std::exp(score - scores[best]);
    previous = candidates[row * K + best];
    ref[row] = 1 / z;
    if (got[row] != previous)
      throw std::runtime_error("selector IDs");
  }
  compare("selector probabilities", probs.get(), ref);
  bench("top-k", logits.size() * 4, [&] {
    df2_topk(dl.p, V, nullptr, B, scratch.p, val.p, tok.p, nullptr);
  });
  bench("selector", (B * K * R * 2 + B * R * 4 + B * K * 8), [&] {
    df2_select(dh.p, du.p, dc.p, dp.p, ds.p, R, B, anchor.p, out.p, probs.p,
               nullptr);
  });
}

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
    return 77;
  try {
    projection();
    elementwise(8);
    rotary(8);
    attention(8);
    for (int rows = 2; rows <= 16; ++rows)
      if (rows != 8) {
        std::printf("block rows %d\n", rows);
        attention(rows);
      }
    elementwise(16);
    rotary(16);
    taps();
    topk_select(7);
    topk_select(15);
    ck(cudaDeviceSynchronize());
  } catch (const std::exception &e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("dflash2 parity: ok");
  return 0;
}
