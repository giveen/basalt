// Copyright (c) 2026 Niko1221 and the Strata contributors. MIT licensed.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/core/session.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace basalt::core {

class NativeHead;

class Dflash2Drafter {
public:
  static constexpr int kRows = 8;
  static constexpr int kMaxRows = 16;
  static constexpr int kMaxDrafts = kMaxRows - 1;
  static constexpr int kRing = 2064;
  Dflash2Drafter() = default;
  ~Dflash2Drafter();
  Dflash2Drafter(const Dflash2Drafter &) = delete;
  Dflash2Drafter &operator=(const Dflash2Drafter &) = delete;

  /// Read drafter tensors and configuration from the model file's sections.
  bool load(const FileRange &model, const FileRange &config, int device,
            int block_rows, std::string &err);
  /// The rows of the target head the drafter scores: the config's `proposal_ids` shortlist when it has one, else the
  /// whole head.  Set by `load`; `bind` gathers them.
  int64_t proposal_rows() const { return (int64_t)proposal_ids_.size(); }
  bool bind(const WeightTable &wt, const NativeHead *head, std::string &err);
  using EmbedFn = std::function<void(const int32_t *tokens_dev, int n,
                                     float *out, cudaStream_t)>;
  bool bind_raw(EmbedFn embed, const void *head_w, int head_type,
                int64_t head_rows, const std::vector<int32_t> &vocab,
                std::string &err);

  void reset() {
    have_ctx_ = 0;
    remaining_ = 0;
  }
  int draft_count() const { return remaining_; }
  /// Append committed target features, laid out as [rows][layers][hidden].
  bool context(const float *taps_dev, int64_t n, int64_t pos0,
               std::string &err);
  /// One round in two halves: `launch` stages the block and starts it on the drafter's stream, `collect` reads the
  /// proposal.  Between them the host may run anything that touches neither the drafter nor the window's rows (a
  /// window's commit lands there), so the block does not queue behind it.  `draft` runs both.
  bool launch(const float *taps_dev, int n_new, int64_t pos0, int32_t anchor,
              int32_t *drafts, float *probs, bool reuse_chain, std::string &err);
  bool collect(int32_t *drafts, float *probs, std::string &err);
  bool draft(const float *taps_dev, int n_new, int64_t pos0, int32_t anchor,
             int32_t *drafts, float *probs, bool reuse_chain, std::string &err) {
    return launch(taps_dev, n_new, pos0, anchor, drafts, probs, reuse_chain,
                  err) &&
           collect(drafts, probs, err);
  }

  bool prepare(const float *taps_dev, std::string &err);

  uint64_t vram_bytes() const { return vram_; }
  int64_t n_draft_vocab() const { return n_dvocab_; }
  int n_embd() const { return N_; }
  const std::vector<int> &tap_layers() const { return taps_; }
  cudaStream_t stream() const { return cs_; }
  bool debug_rows(std::vector<float> &rows, std::string &err);

  double ms_draft = 0, ms_gpu = 0, ms_ctx = 0, ms_capture = 0;
  int64_t rounds = 0;   ///< blocks the drafter computed
  int64_t reused = 0;   ///< rounds served by the accepted window's remaining chain instead

private:
  static constexpr int kMetaWords = 64;
  static constexpr int kBlockPos = 8;
  static constexpr int kAnchor = 24;
  static constexpr int kAnchorPos = 25;
  static constexpr int kTokens = 32;
  int block_rows_ = kRows;
  int remaining_ = 0;
  struct Tensor {
    std::string name, dtype;
    int ggml_type = -1;
    int64_t rows = 0, cols = 0;
    uint64_t off = 0, bytes = 0;
  };
  const Tensor &find(const std::string &name) const;
  const float *f32(const std::string &name) const {
    return reinterpret_cast<const float *>(blob_ + find(name).off);
  }
  void gemm(std::initializer_list<const char *> names,
            std::initializer_list<float *> outs, int ncols, cudaStream_t cs,
            const float *input);
  bool record_context(const float *taps, int n, cudaStream_t cs,
                      std::string &err);
  bool record_block(cudaStream_t cs, std::string &err);
  bool capture(int n, const float *taps, std::string &err);
  bool alloc(std::string &err);

  int device_ = -1;
  cudaStream_t cs_ = nullptr;
  std::vector<Tensor> tensors_;
  uint8_t *blob_ = nullptr;
  uint64_t vram_ = 0;
  bool quantized_ = false;
  uint8_t *block_q8_ = nullptr;
  uint8_t *context_q8_ = nullptr;
  int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680,
      window_ = 2048, group_ = 16, rank_ = 256, mask_token_ = 0;
  float theta_ = 10000.0f, eps_ = 1e-6f;
  std::vector<int> taps_;
  int64_t vocab_ = 0;
  std::vector<int32_t> proposal_ids_;   ///< the head rows the drafter scores, in rank order (empty: all of them)
  void *shortlist_ = nullptr;           ///< those rows, gathered from the target head

  EmbedFn embed_;
  const void *head_w_ = nullptr;
  int head_type_ = -1;
  int64_t n_dvocab_ = 0;
  int32_t *dvocab_ = nullptr;
  uint8_t *dhead_ = nullptr;
  bool bound_ = false;

  int32_t *d_meta_ = nullptr;
  int32_t *h_meta_ = nullptr;
  int32_t *m_meta_ = nullptr;
  int32_t *h_out_ = nullptr;
  float *hp_out_ = nullptr;
  float *kring_[8] = {}, *vring_[8] = {};
  float *rt_ctx_ = nullptr, *rt_blk_ = nullptr;
  uint8_t *xq_ = nullptr;
  float *block_input_ = nullptr;
  float *context_input_ = nullptr;
  cudaStream_t cs2_ = nullptr;
  cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr, ev_ctx_[8] = {};
  bool fork_ = false;
  float *fcy_ = nullptr, *hctx_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
  float *h_ = nullptr, *xn_ = nullptr, *xc_ = nullptr, *dyn_ = nullptr,
        *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr,
        *kblk_ = nullptr;
  float *ao_ = nullptr, *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr,
        *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
  float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr,
        *out_prob_ = nullptr;
  int32_t *toptok_ = nullptr, *out_tok_ = nullptr;
  uint32_t *topscratch_ = nullptr;
  cudaGraphExec_t exec_[kRows + 1] = {};
  const float *graph_taps_ = nullptr;
  int64_t have_ctx_ = 0;
  cudaEvent_t ev_a_ = nullptr, ev_b_ = nullptr;
  bool timing_ = false;
  int n_launch_ = 0;   ///< context rows of the block in flight (0: nothing pending)
  std::chrono::steady_clock::time_point t_launch_{};
};

} // namespace basalt::core
