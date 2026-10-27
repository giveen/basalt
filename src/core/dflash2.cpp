// Copyright (c) 2026 Niko1221 and the Strata contributors. MIT licensed.

#include "basalt/core/dflash2.hpp"
#include "basalt/core/load_progress.hpp"

#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/dflash2.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/q80_head.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <stdexcept>

namespace basalt::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

bool mapped(size_t bytes, void **h, void **d) {
  if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  std::memset(*h, 0, bytes);
  return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

} // namespace

Dflash2Drafter::~Dflash2Drafter() {
  if (device_ < 0)
    return;
  const OnDevice on_device(device_);
  if (cs_)
    cudaStreamSynchronize(cs_);
  for (cudaGraphExec_t &e : exec_)
    if (e)
      cudaGraphExecDestroy(e);
  auto fr = [](void *p) {
    if (p)
      cudaFree(p);
  };
  fr(blob_);
  if (shortlist_)
    cudaFree(shortlist_);
  fr(block_q8_);
  fr(context_q8_);
  fr(d_meta_);
  fr(xq_);
  fr(block_input_);
  fr(context_input_);
  fr(rt_ctx_);
  fr(rt_blk_);
  for (int l = 0; l < 8; ++l) {
    fr(kring_[l]);
    fr(vring_[l]);
  }
  fr(fcy_);
  fr(hctx_);
  fr(kc_);
  fr(vc_);
  fr(h_);
  fr(xn_);
  fr(xc_);
  fr(dyn_);
  fr(q_);
  fr(qn_);
  fr(k_);
  fr(v_);
  fr(kblk_);
  fr(ao_);
  fr(o_);
  fr(gate_);
  fr(up_);
  fr(mlp_);
  fr(part_);
  fr(fn_);
  fr(logits_);
  fr(topval_);
  fr(hsel_);
  fr(toptok_);
  fr(topscratch_);
  fr(dvocab_);
  fr(dhead_);
  if (h_meta_)
    cudaFreeHost(h_meta_);
  if (h_out_)
    cudaFreeHost(h_out_);
  if (cs_)
    cudaStreamDestroy(cs_);
  if (cs2_)
    cudaStreamDestroy(cs2_);
  if (ev_fork_)
    cudaEventDestroy(ev_fork_);
  if (ev_join_)
    cudaEventDestroy(ev_join_);
  for (cudaEvent_t e : ev_ctx_)
    if (e)
      cudaEventDestroy(e);
  if (ev_a_)
    cudaEventDestroy(ev_a_);
  if (ev_b_)
    cudaEventDestroy(ev_b_);
}

const Dflash2Drafter::Tensor &
Dflash2Drafter::find(const std::string &name) const {
  for (const Tensor &t : tensors_)
    if (t.name == name)
      return t;
  throw std::runtime_error("dflash2: tensor " + name +
                           " is missing from the pack");
}

bool Dflash2Drafter::load(const FileRange &model, const FileRange &config,
                          int device, int block_rows, std::string &err) {
  if (block_rows < 2 || block_rows > kMaxRows) {
    err = "dflash2: block rows out of range";
    return false;
  }
  block_rows_ = block_rows;
  device_ = device;
  const OnDevice on_device(device_);
  std::vector<uint8_t> bytes, cb;
  if (!read_range(model, bytes, err) || !read_range(config, cb, err))
    return false;
  Json cfg, index, formats;
  if (!parse_json((const char *)cb.data(), cb.size(), cfg, err) ||
      bytes.size() < 8)
    return false;
  uint64_t ni = 0;
  std::memcpy(&ni, bytes.data(), 8);
  if (ni > (64u << 20) || ni > bytes.size() - 8 ||
      !parse_json((const char *)bytes.data() + 8, (size_t)ni, index, err)) {
    err = "dflash2: invalid safetensors header";
    return false;
  }
  if (const Json *metadata = index.get("__metadata__")) {
    if (const Json *quant = metadata->get("basalt.dflash2.quantization")) {
      if (!quant->is_string() ||
          !parse_json(quant->s.data(), quant->s.size(), formats, err)) {
        err = "dflash2: invalid quantization metadata";
        return false;
      }
    }
  }
  auto number = [&](const Json &j, const char *key) -> int64_t {
    const Json *v = j.get(key);
    if (!v || (v->type != Json::Type::Int && v->type != Json::Type::Float))
      throw std::runtime_error(std::string("dflash2: missing numeric config ") +
                               key);
    return v->i64();
  };
  try {
    const Json *dc = cfg.get("dflash_config");
    if (!dc)
      throw std::runtime_error("dflash2: missing configuration");
    N_ = (int)number(cfg, "hidden_size");
    L_ = (int)number(cfg, "num_hidden_layers");
    NH_ = (int)number(cfg, "num_attention_heads");
    NKV_ = (int)number(cfg, "num_key_value_heads");
    HD_ = (int)number(cfg, "head_dim");
    I_ = (int)number(cfg, "intermediate_size");
    window_ = (int)number(cfg, "sliding_window");
    group_ = (int)number(*dc, "conv_group_size");
    rank_ = (int)number(*dc, "selector_rank");
    mask_token_ = (int)number(*dc, "mask_token_id");
    vocab_ = number(cfg, "vocab_size");
    const Json *eps = cfg.get("rms_norm_eps");
    const Json *rope = cfg.get("rope_parameters");
    if (!eps || !rope)
      throw std::runtime_error(
          "dflash2: missing normalization or rotary configuration");
    eps_ = (float)eps->num();
    theta_ = (float)number(*rope, "rope_theta");
    const Json *taps = dc->get("target_layer_ids");
    if (!taps)
      throw std::runtime_error("dflash2: missing feature layers");
    for (const Json &t : taps->items)
      taps_.push_back((int)t.i64());
    if (N_ != 2560 || L_ != 5 || NH_ != 32 || NKV_ != 8 || HD_ != 128 ||
        I_ != 7680 || window_ != 2048 || group_ != 16 || rank_ != 256 ||
        vocab_ != 248320 || taps_ != std::vector<int>({5, 15, 25, 35, 45}) ||
        number(*dc, "block_size") != 8 ||
        number(*dc, "conv_kernel_size") != 2 ||
        number(*dc, "selector_top_k") != 16 || mask_token_ < 0 ||
        mask_token_ >= vocab_) {
      throw std::runtime_error("dflash2: unsupported model geometry");
    }
    // the config may name the target-head rows the drafter scores (a proposal shortlist).  The ids are a
    // comma-separated string: the pack's config is a JSON document read with a bounded array parser, and a
    // shortlist is far longer than that bound.  They are global ids, so the block's head pass reads only those rows
    // and the top-k still reports global tokens
    if (const Json *ids = cfg.get("proposal_ids")) {
      if (!ids->is_string())
        throw std::runtime_error("dflash2: invalid proposal shortlist");
      std::vector<char> seen((size_t)vocab_, 0);
      const char *at = ids->s.c_str();
      while (*at != '\0') {
        char *end = nullptr;
        const long long t = std::strtoll(at, &end, 10);
        if (end == at || t < 0 || t >= vocab_ || t == mask_token_ ||
            seen[(size_t)t])
          throw std::runtime_error("dflash2: invalid proposal id");
        seen[(size_t)t] = 1;
        proposal_ids_.push_back((int32_t)t);
        at = *end == ',' ? end + 1 : end;
        if (*at != '\0' && *end != ',')
          throw std::runtime_error("dflash2: invalid proposal shortlist");
      }
      if (proposal_ids_.empty())
        throw std::runtime_error("dflash2: empty proposal shortlist");
    }
    uint64_t total = 0;
    std::vector<std::vector<uint8_t>> values;
    for (const auto &item : index.members) {
      if (item.first == "__metadata__")
        continue;
      const Json *dtype = item.second.get("dtype");
      const Json *shape = item.second.get("shape");
      const Json *offsets = item.second.get("data_offsets");
      const Json *format = formats.get(item.first);
      const bool quant = format != nullptr;
      if (!dtype || dtype->s != (quant ? "U8" : "BF16") || !shape ||
          shape->items.empty() || !offsets || offsets->items.size() != 2)
        throw std::runtime_error("dflash2: invalid tensor storage");
      uint64_t count = 1;
      for (const Json &dim : shape->items) {
        const int64_t n = dim.i64();
        if (n <= 0 || (uint64_t)n > bytes.size() / count)
          throw std::runtime_error("dflash2: invalid tensor shape");
        count *= (uint64_t)n;
      }
      const int64_t lo = offsets->items[0].i64(), hi = offsets->items[1].i64();
      if (lo < 0 || hi < lo || (uint64_t)hi > bytes.size() - 8 - ni ||
          (uint64_t)(hi - lo) != count * (quant ? 1 : 2))
        throw std::runtime_error("dflash2: invalid tensor range");
      int ggml_type = -1;
      if (quant) {
        const Json *kind = format->get("format");
        const Json *logical = format->get("shape");
        if (!kind || kind->s != "Q4_K")
          throw std::runtime_error("dflash2: unsupported projection format");
        const int block = 256;
        const int block_bytes = 144;
        ggml_type = 12;
        if (!logical || logical->items.size() != 2 ||
            shape->items.size() != 2 || logical->items[0].i64() <= 0 ||
            logical->items[1].i64() <= 0 || logical->items[1].i64() % block ||
            logical->items[0].i64() != shape->items[0].i64() ||
            logical->items[1].i64() / block * block_bytes !=
                shape->items[1].i64())
          throw std::runtime_error(
              "dflash2: invalid quantized tensor geometry");
        shape = logical;
        quantized_ = true;
      }
      const bool norm =
          !quant && (shape->items.size() == 1 ||
                     item.first.find("base_kernel") != std::string::npos);
      Tensor t;
      t.name = item.first;
      t.dtype = quant ? "U8" : norm ? "f32" : "bf16";
      t.ggml_type = ggml_type;
      t.cols = shape->items.back().i64();
      t.rows = quant ? shape->items[0].i64() : (int64_t)count / t.cols;
      t.off = (total + 255) & ~uint64_t(255);
      t.bytes = quant ? (uint64_t)(hi - lo) : count * (norm ? 4 : 2);
      total = t.off + t.bytes;
      std::vector<uint8_t> value((size_t)t.bytes);
      if (norm) {
        for (uint64_t i = 0; i < count; ++i) {
          uint16_t b;
          std::memcpy(&b, bytes.data() + 8 + ni + (uint64_t)lo + i * 2, 2);
          const uint32_t f = (uint32_t)b << 16;
          std::memcpy(value.data() + i * 4, &f, 4);
        }
      } else
        std::memcpy(value.data(), bytes.data() + 8 + ni + (uint64_t)lo,
                    value.size());
      tensors_.push_back(t);
      values.push_back(std::move(value));
    }
    if (cudaMalloc((void **)&blob_, total) != cudaSuccess)
      throw std::runtime_error("dflash2: weights do not fit in VRAM");
    vram_ += total;
    LoadProgress progress(LoadProgress::cuda(device_), total);
    for (size_t i = 0; i < tensors_.size(); ++i) {
      if (cudaMemcpy(blob_ + tensors_[i].off, values[i].data(),
                     values[i].size(), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("dflash2: weight upload failed");
      progress.add(values[i].size());
    }
    progress.finish();
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
  int low = 0, high = 0;
  cudaDeviceGetStreamPriorityRange(&low, &high);
  if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, high) !=
          cudaSuccess ||
      cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, high) !=
          cudaSuccess) {
    err = "dflash2: stream allocation failed";
    return false;
  }
  if (cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) !=
          cudaSuccess ||
      cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) !=
          cudaSuccess) {
    err = "dflash2: event allocation failed";
    return false;
  }
  if (cudaEventCreate(&ev_a_) != cudaSuccess ||
      cudaEventCreate(&ev_b_) != cudaSuccess) {
    err = "dflash2: event allocation failed";
    return false;
  }
  for (int i = 0; i < 8; ++i)
    if (cudaEventCreateWithFlags(&ev_ctx_[i], cudaEventDisableTiming) !=
        cudaSuccess) {
      err = "dflash2: event allocation failed";
      return false;
    }
  return alloc(err);
}

bool Dflash2Drafter::alloc(std::string &err) {
  auto dev = [&](auto **p, size_t n) {
    using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
    if (cudaMalloc((void **)p, n * sizeof(T)) != cudaSuccess)
      return false;
    vram_ += n * sizeof(T);
    return true;
  };
  const size_t B = (size_t)std::max(kRows, block_rows_);
  const size_t drafts = (size_t)block_rows_ - 1;
  if (quantized_ &&
      (!dev(&block_q8_, kernels::native_q8_1_bytes(5 * N_, kRows)) ||
       !dev(&context_q8_, kernels::native_q8_1_bytes(5 * N_, kRows)))) {
    err = "dflash2: quantized activation buffers do not fit";
    return false;
  }
  bool ok = dev(&d_meta_, kMetaWords) &&
            dev(&xq_, kernels::native_q8_1_bytes(N_, kRows) + 4096) &&
            dev(&block_input_, (size_t)5 * N_ * B + 4096) &&
            dev(&context_input_, (size_t)5 * N_ * 8 + 4096) &&
            dev(&rt_ctx_, B * 128) && dev(&rt_blk_, B * 128) &&
            dev(&fcy_, B * N_) && dev(&hctx_, B * N_) &&
            dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) &&
            dev(&h_, B * N_) && dev(&xn_, B * N_) && dev(&xc_, B * N_) &&
            dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) &&
            dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) &&
            dev(&v_, B * NKV_ * HD_) && dev(&kblk_, B * NKV_ * HD_) &&
            dev(&ao_, B * NH_ * HD_) && dev(&o_, B * N_) &&
            dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
            dev(&part_, (size_t)((block_rows_ + kRows - 1) / kRows) *
                            kernels::kDf2MaxChunks * 8 * 32 * 130) &&
            dev(&fn_, B * N_) && dev(&topval_, drafts * 16) &&
            dev(&toptok_, drafts * 16) && dev(&hsel_, drafts * rank_) &&
            dev(&topscratch_,
                drafts * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2);
  for (int l = 0; ok && l < L_; ++l)
    ok = dev(&kring_[l], (size_t)kRing * NKV_ * HD_) &&
         dev(&vring_[l], (size_t)kRing * NKV_ * HD_);
  if (!ok) {
    err = "dflash2: the drafter's buffers do not fit in VRAM";
    return false;
  }
  void *hd = nullptr;
  void *dd = nullptr;
  if (!mapped(kMetaWords * 4, (void **)&hd, &dd)) {
    err = "dflash2: mapped staging";
    return false;
  }
  h_meta_ = (int32_t *)hd;
  m_meta_ = (int32_t *)dd;
  std::fill_n(h_meta_, kMetaWords, 0);
  if (!mapped(64 * 4, &hd, &dd)) {
    err = "dflash2: mapped output";
    return false;
  }
  h_out_ = (int32_t *)hd;
  out_tok_ = (int32_t *)dd;
  out_prob_ = (float *)dd + 16;
  cudaMemset(rt_ctx_, 0, B * 128 * 4);
  cudaMemset(block_input_, 0, (size_t)5 * N_ * B + 4096);
  cudaMemset(context_input_, 0, (size_t)5 * N_ * 8 + 4096);
  for (int l = 0; l < L_; ++l) {
    cudaMemset(kring_[l], 0, (size_t)kRing * NKV_ * HD_ * 4);
    cudaMemset(vring_[l], 0, (size_t)kRing * NKV_ * HD_ * 4);
  }
  return cudaDeviceSynchronize() == cudaSuccess;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void *head_w, int head_type,
                              int64_t head_rows,
                              const std::vector<int32_t> &vocab,
                              std::string &err) {
  const OnDevice on_device(device_);
  embed_ = std::move(embed);
  head_w_ = head_w;
  head_type_ = head_type;
  n_dvocab_ = head_rows;
  if (!vocab.empty()) {
    n_dvocab_ = (int64_t)vocab.size();
    if (cudaMalloc((void **)&dvocab_, vocab.size() * 4) != cudaSuccess) {
      err = "dflash2: draft vocabulary";
      return false;
    }
    vram_ += vocab.size() * 4;
    cudaMemcpy(dvocab_, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
  }
  if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks >
      4096) {
    err = "dflash2: the head has too many rows for the top-k";
    return false;
  }
  if (!logits_ &&
      cudaMalloc((void **)&logits_, (size_t)(block_rows_ - 1) *
                                        (size_t)n_dvocab_ * 4) != cudaSuccess) {
    err = "dflash2: the draft logits do not fit";
    return false;
  }
  vram_ += (uint64_t)(block_rows_ - 1) * (uint64_t)n_dvocab_ * 4;
  bound_ = true;
  return true;
}

bool Dflash2Drafter::bind(const WeightTable &wt, const NativeHead *head,
                          std::string &err) {
  const OnDevice on_device(device_);
  if (head == nullptr || !head->loaded()) {
    err = "dflash2: the drafter needs the native head";
    return false;
  }
  const WeightRef *we = wt.find("token_embd.weight");
  if (we == nullptr) {
    err = "dflash2: token_embd.weight is missing";
    return false;
  }
  EmbedFn embed = [we](const int32_t *tok, int n, float *out, cudaStream_t cs) {
    if (const NativeEmbed *ne = native_embed()) {
      ne->gather_dev(tok, n, out, cs);
    } else {
      const auto *codes = (const uint8_t *)we->data;
      const auto *scales = (const float *)(codes + we->codes_bytes);
      const auto *offsets =
          we->has_offset
              ? (const float *)(codes + we->codes_bytes + we->scales_bytes)
              : nullptr;
      kernels::embedding_gather_dev(
          codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias,
          we->group_elems, (uint64_t)(we->ne0 / (8 / we->code_bits)),
          (uint64_t)(we->ne0 / we->group_elems), out, cs);
    }
  };
  const void *hw = head->weights();
  const int64_t row_bytes = (int64_t)head->row_bytes();
  const int64_t head_rows =
      row_bytes > 0 ? (int64_t)(head->weight_bytes() / (uint64_t)row_bytes) : 0;
  int64_t rows = head_rows;
  if (!proposal_ids_.empty()) {
    // the config's shortlist: gather those head rows once, so the block's head pass reads only them.  Their bytes are
    // the head's own, so a shortlisted token's logits are the full head's logits
    rows = (int64_t)proposal_ids_.size();
    if (rows > head_rows) {
      err = "dflash2: the proposal shortlist exceeds the head";
      return false;
    }
    const uint64_t bytes = (uint64_t)rows * (uint64_t)row_bytes;
    if (cudaMalloc(&shortlist_, bytes) != cudaSuccess) {
      err = "dflash2: the proposal shortlist does not fit in VRAM";
      return false;
    }
    auto *dst = (uint8_t *)shortlist_;
    const auto *src = (const uint8_t *)hw;
    for (int64_t i = 0; i < rows; ++i) {
      const uint64_t at = (uint64_t)proposal_ids_[(size_t)i] * (uint64_t)row_bytes;
      if (cudaMemcpyAsync(dst + (size_t)i * (size_t)row_bytes, src + at,
                          (size_t)row_bytes, cudaMemcpyDeviceToDevice,
                          nullptr) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess) {
        err = "dflash2: gathering the proposal rows failed";
        return false;
      }
    }
    vram_ += bytes;
    hw = shortlist_;
  }
  if (!bind_raw(embed, hw, head->type(), rows, proposal_ids_, err))
    return false;
  basalt::log::info(
      "basalt dflash2",
      "%s drafter loaded, %lld of %lld head rows%s, %.0f MiB VRAM, drafts=%d",
      quantized_ ? "Q4_K" : "BF16", (long long)n_dvocab_, (long long)head_rows,
      proposal_ids_.empty() ? "" : " (proposal shortlist)",
      (double)vram_ / 1048576.0, block_rows_ - 1);
  return true;
}

void Dflash2Drafter::gemm(std::initializer_list<const char *> names,
                          std::initializer_list<float *> outs, int columns,
                          cudaStream_t cs, const float *values) {
  for (int col = 0; col < columns; col += kRows) {
    const int ncols = std::min(kRows, columns - col);
    auto out = outs.begin();
    const int n_in = (int)find(*names.begin()).cols;
    const float *input = values + (size_t)col * n_in;
    const bool quant =
        std::any_of(names.begin(), names.end(), [&](const char *name) {
          return find(name).ggml_type >= 0;
        });
    void *scratch = cs == cs2_ ? context_q8_ : block_q8_;
    if (quant)
      kernels::native_quantize_q8_1(input, scratch, n_in, ncols, cs);
    if (names.size() == 2 && quant) {
      const Tensor &a = find(*names.begin());
      const Tensor &b = find(*(names.begin() + 1));
      if (a.ggml_type >= 0 && b.ggml_type >= 0 && a.cols == b.cols &&
          kernels::dense_tc_pair_supported(a.ggml_type, b.ggml_type, n_in)) {
        kernels::dense_tc_pair(a.ggml_type, blob_ + a.off,
                               *out + (size_t)col * a.rows, (int)a.rows,
                               b.ggml_type, blob_ + b.off,
                               *(out + 1) + (size_t)col * b.rows, (int)b.rows,
                               scratch, n_in, ncols, cs);
        continue;
      }
    }
    for (const char *name : names) {
      const Tensor &t = find(name);
      if ((t.dtype != "bf16" && t.ggml_type < 0) || t.rows <= 0 ||
          t.cols != n_in)
        throw std::runtime_error("dflash2: invalid projection");
      if (t.ggml_type >= 0) {
        float *y = *out++ + (size_t)col * t.rows;
        if (kernels::dense_tc_supported(t.ggml_type, n_in, (int)t.rows,
                                        blob_ + t.off))
          kernels::dense_tc(t.ggml_type, blob_ + t.off, scratch, y, n_in,
                            (int)t.rows, ncols, cs);
        else
          kernels::native_mmvq(t.ggml_type, blob_ + t.off, scratch, y, n_in,
                               (int)t.rows, ncols, cs);
        continue;
      }
      kernels::bf16_gemv_fp32_mmvf_multi(
          input, t.cols, reinterpret_cast<const uint16_t *>(blob_ + t.off),
          *out++ + (size_t)col * t.rows, t.rows, t.cols, t.rows, ncols, cs);
    }
  }
}

bool Dflash2Drafter::record_context(const float *taps, int n, cudaStream_t cs,
                                    std::string &err) {
  using namespace kernels;
  try {
    const int tap_n = (int)taps_.size() * N_;
    df2_rope_table(d_meta_ + 0, n, theta_, rt_ctx_, cs);
    df2_copy_input(taps, tap_n, n, context_input_, cs);
    gemm({"fc.weight"}, {fcy_}, n, cs, context_input_);
    df2_rms_input(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_,
                  context_input_, cs);
    for (int l = 0; l < L_; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      const std::string kp = p + "self_attn.k_proj.weight",
                        vp = p + "self_attn.v_proj.weight";
      gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, cs, context_input_);
      df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), d_meta_ + 0,
                  rt_ctx_, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs);
      if (fork_)
        cudaEventRecord(ev_ctx_[l], cs);
    }
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
  return true;
}

bool Dflash2Drafter::record_block(cudaStream_t cs, std::string &err) {
  using namespace kernels;
  try {
    const int B = block_rows_;
    const int drafts = B - 1;
    df2_rope_table(d_meta_ + kBlockPos, B, theta_, rt_blk_, cs);
    embed_(d_meta_ + kTokens, B, h_, cs);
    df2_rms_input(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_,
                  block_input_, cs);
    for (int l = 0; l < L_; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      const float *base_a = f32(p + "attention_conv.base_kernel");
      const float *base_m = f32(p + "mlp_conv.base_kernel");
      const std::string akp = p + "attention_conv.kernel_projection.weight",
                        mkp = p + "mlp_conv.kernel_projection.weight";
      const std::string qp = p + "self_attn.q_proj.weight",
                        kp = p + "self_attn.k_proj.weight",
                        vp = p + "self_attn.v_proj.weight";
      const std::string gp = p + "mlp.gate_proj.weight",
                        up = p + "mlp.up_proj.weight";
      gemm({akp.c_str()}, {dyn_}, B, cs, block_input_);
      df2_conv_input(xn_, dyn_, 640, base_a, B, N_, group_, block_input_, cs);
      gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs,
           block_input_);
      df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"),
                       f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_,
                       B, NH_, NKV_, eps_, cs);
      if (fork_)
        cudaStreamWaitEvent(cs, ev_ctx_[l], 0);
      df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + kAnchorPos,
                    kRing, window_, B, part_, block_input_, cs);
      gemm({(p + "self_attn.o_proj.weight").c_str()}, {o_}, B, cs,
           block_input_);
      df2_conv_rms_input(o_, dyn_ + 320, 640, base_a + 2 * N_, h_,
                         f32(p + "post_attention_layernorm.weight"), xn_, B, N_,
                         group_, eps_, block_input_, cs);
      gemm({mkp.c_str()}, {dyn_}, B, cs, block_input_);
      df2_conv_input(xn_, dyn_, 640, base_m, B, N_, group_, block_input_, cs);
      gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, block_input_);
      df2_swiglu_input(gate_, up_, B, I_, block_input_, cs);
      gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, block_input_);

      const std::string next_gamma =
          l + 1 < L_
              ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight"
              : std::string("norm.weight");
      df2_conv_rms_input(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_,
                         f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_,
                         eps_, block_input_, cs);
    }
    df2_copy_input(fn_ + N_, N_, drafts, block_input_, cs);
    for (int col = 0; col < drafts; col += kRows) {
      const int count = std::min(kRows, drafts - col);
      native_quantize_q8_1(fn_ + (col + 1) * N_, xq_, N_, count, cs);
      float *logits = logits_ + (size_t)col * (size_t)n_dvocab_;
      if (head_type_ == 8 && q80_head_supported(N_, count))
        q80_head_mmvq(head_w_, xq_, logits, N_, (int)n_dvocab_, count, cs);
      else
        native_mmvq(head_type_, head_w_, xq_, logits, N_, (int)n_dvocab_, count,
                    cs);
    }
    df2_topk(logits_, (int)n_dvocab_, dvocab_, drafts, topscratch_, topval_,
             toptok_, cs);
    gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, drafts, cs,
         block_input_);
    df2_select(hsel_, topval_, toptok_,
               reinterpret_cast<const uint16_t *>(
                   blob_ + find("candidate_selector.predecessor_codebook").off),
               reinterpret_cast<const uint16_t *>(
                   blob_ + find("candidate_selector.successor_codebook").off),
               rank_, drafts, d_meta_ + kAnchor, out_tok_, out_prob_, cs);
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
  return true;
}

bool Dflash2Drafter::capture(int n, const float *taps, std::string &err) {
  if (exec_[n])
    return true;
  if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess) {
    err = "dflash2: begin capture";
    return false;
  }
  kernels::copy_i32_from_mapped(d_meta_, m_meta_, kMetaWords, cs_);

  fork_ = cs2_ != nullptr;
  if (fork_) {
    cudaEventRecord(ev_fork_, cs_);
    cudaStreamWaitEvent(cs2_, ev_fork_, 0);
  }
  bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err);
  if (fork_)
    cudaEventRecord(ev_join_, cs2_);
  if (ok)
    ok = record_block(cs_, err);
  if (fork_)
    cudaStreamWaitEvent(cs_, ev_join_, 0);
  fork_ = false;
  cudaGraph_t graph = nullptr;
  const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
  if (!ok) {
    if (graph)
      cudaGraphDestroy(graph);
    return false;
  }
  if (ce != cudaSuccess ||
      cudaGraphInstantiate(&exec_[n], graph, 0) != cudaSuccess) {
    if (graph)
      cudaGraphDestroy(graph);
    err = std::string("dflash2: capture: ") + cudaGetErrorString(ce);
    return false;
  }
  cudaGraphDestroy(graph);
  cudaGraphUpload(exec_[n], cs_);
  cudaStreamSynchronize(cs_);
  return true;
}

bool Dflash2Drafter::prepare(const float *taps, std::string &err) {
  if (!bound_) {
    err = "dflash2: not bound";
    return false;
  }

  const OnDevice on_device(device_);
  const Clock::time_point t0 = Clock::now();
  timing_ = ev_a_ != nullptr && ev_b_ != nullptr;
  for (int i = 0; i < 8; ++i)
    h_meta_[i] = i;
  for (int n = 1; n <= kRows; ++n)
    if (!capture(n, taps, err))
      return false;
  graph_taps_ = taps;
  ms_capture = ms_since(t0);
  return true;
}

bool Dflash2Drafter::context(const float *taps, int64_t n, int64_t pos0,
                             std::string &err) {
  const OnDevice on_device(device_);
  const int tap_n = (int)taps_.size() * N_;
  for (int64_t r0 = 0; r0 < n; r0 += kRows) {
    const int m = (int)std::min<int64_t>(kRows, n - r0);
    for (int i = 0; i < m; ++i)
      h_meta_[i] = (int32_t)(pos0 + r0 + i);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, kMetaWords, cs_);
    if (!record_context(taps + (size_t)r0 * (size_t)tap_n, m, cs_, err))
      return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
      err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError());
      return false;
    }
  }
  have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
  return true;
}

bool Dflash2Drafter::launch(const float *taps, int n_new, int64_t pos0,
                            int32_t anchor, int32_t *drafts, float *probs,
                            bool reuse_chain, std::string &err) {
  if (!bound_) {
    err = "dflash2: not bound";
    return false;
  }
  if (n_new < 1 || n_new > kRows) {
    err = "dflash2: draft rows out of range";
    return false;
  }
  if (have_ctx_ != pos0) {
    err = "dflash2: target features do not cover the draft prefix";
    return false;
  }
  if (n_launch_ > 0) {
    err = "dflash2: a block is already in flight";
    return false;
  }
  const OnDevice on_device(device_);
  t_launch_ = Clock::now();
  if (block_rows_ > kRows && reuse_chain && remaining_ > n_new &&
      drafts[n_new - 1] == anchor) {
    // the accepted window ended on the chain's next proposal: the rest of the chain is still the target's text, so
    // only the rows it just wrote enter the ring and the remaining drafts are kept
    if (!context(taps, n_new, pos0, err))
      return false;
    remaining_ -= n_new;
    std::memmove(drafts, drafts + n_new, (size_t)remaining_ * sizeof(*drafts));
    if (probs)
      std::memmove(probs, probs + n_new, (size_t)remaining_ * sizeof(*probs));
    ++reused;
    n_launch_ = 0;
    return true;
  }
  if (graph_taps_ != nullptr && graph_taps_ != taps) {
    err = "dflash2: the taps buffer moved after capture";
    return false;
  }
  if (!exec_[n_new]) {

    for (int i = 0; i < 8; ++i)
      h_meta_[i] = (int32_t)(pos0 + i);
    if (!capture(n_new, taps, err))
      return false;
    graph_taps_ = taps;
  }
  const int64_t s = pos0 + n_new;
  for (int i = 0; i < kRows; ++i)
    h_meta_[i] = (int32_t)(pos0 + std::min(i, n_new - 1));
  for (int i = 0; i < block_rows_; ++i)
    h_meta_[kBlockPos + i] = (int32_t)(s + i);
  h_meta_[kAnchor] = anchor;
  h_meta_[kAnchorPos] = (int32_t)s;
  h_meta_[kTokens] = anchor;
  for (int i = 1; i < block_rows_; ++i)
    h_meta_[kTokens + i] = mask_token_;
  for (int i = 0; i < block_rows_ - 1; ++i) {
    ((volatile int32_t *)h_out_)[i] = -1;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  remaining_ = block_rows_ - 1;
  if (timing_)
    cudaEventRecord(ev_a_, cs_);
  if (cudaGraphLaunch(exec_[n_new], cs_) != cudaSuccess) {
    err =
        std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError());
    return false;
  }
  if (timing_)
    cudaEventRecord(ev_b_, cs_);
  n_launch_ = n_new;
  have_ctx_ = std::max<int64_t>(have_ctx_, s);
  return true;
}

bool Dflash2Drafter::collect(int32_t *drafts, float *probs,
                             std::string &err) {
  if (n_launch_ <= 0)
    return true;
  const OnDevice on_device(device_);
  const int n = remaining_;
  // the selector writes the block's last position and closes the graph, so when it lands every draft is readable; the
  // stream query is the way out if the graph ended early
  uint32_t spins = 0;
  while (((volatile int32_t *)h_out_)[n - 1] < 0) {
    basalt::platform::cpu_relax();
    if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady)
      break;
  }
  if (((volatile int32_t *)h_out_)[n - 1] < 0) {
    (void)cudaStreamSynchronize(cs_);
    err =
        std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError());
    n_launch_ = 0;
    return false;
  }
  for (int i = 0; i < n; ++i) {
    drafts[i] = ((volatile int32_t *)h_out_)[i];
    if (probs)
      probs[i] = ((volatile float *)h_out_)[16 + i];
  }
  if (timing_) {
    float ms = 0;
    if (cudaEventSynchronize(ev_b_) == cudaSuccess &&
        cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess)
      ms_gpu += ms;
  }
  n_launch_ = 0;
  ms_draft += ms_since(t_launch_);
  ++rounds;
  return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float> &rows, std::string &err) {
  rows.resize((size_t)block_rows_ * (size_t)N_);
  if (cudaMemcpy(rows.data(), fn_, rows.size() * 4, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    err = "dflash2: debug copy";
    return false;
  }
  return true;
}

} // namespace basalt::core
