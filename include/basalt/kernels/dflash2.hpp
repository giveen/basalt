// Copyright (c) 2026 Niko1221 and the Strata contributors. MIT licensed.
#pragma once

#include <cstdint>

namespace basalt::kernels {

constexpr int kDf2Rows = 8;
constexpr int kDf2Kv = 8;
constexpr int kDf2Gqa = 4;
constexpr int kDf2Hd = 128;
constexpr int kDf2Chunk = 128;
constexpr int kDf2MaxChunks = 17;
constexpr int kDf2TopK = 16;
constexpr int kDf2TopBlocks = 64;
void df2_conv(const float *x, const float *dyn, int dyn_stride, const float *base, const float *resid, float *out, int rows, int H,
              int group, void *stream);

void df2_rope_table(const int32_t *pos, int rows, float theta, float *table, void *stream);

void df2_norm_rope(const float *in, int in_stride, const float *gamma, const int32_t *pos, const float *rope_table, float *out,
                   int slot_mod, int rows, int heads, float eps, void *stream);

void df2_copy_rows(const float *in, float *out, const int32_t *pos, int slot_mod, int rows, int cols, void *stream);

void df2_attention(const float *q, const float *kring, const float *vring, const float *kblk, const float *vblk, const int32_t *anchor_pos,
                   int ring_slots, int window, int rows, float *part, float *out, void *stream);

void df2_copy_input(const float *x, int K, int ncols, float *out, void *stream);
void df2_rms_input(const float *h, const float *gamma, float *xn, int rows, int N, float eps, float *out, void *stream);
void df2_conv_input(const float *x, const float *dyn, int dyn_stride, const float *base, int rows, int H, int group, float *out,
                    void *stream);
void df2_swiglu_input(const float *gate, const float *up, int rows, int I, float *out, void *stream);

void df2_conv_rms_input(const float *x, const float *dyn, int dyn_stride, const float *base, float *h, const float *gamma, float *xn,
                        int rows, int H, int group, float eps, float *out, void *stream);
void df2_norm_rope_qk(const float *q, const float *k, const float *gq, const float *gk, const float *table, float *qo, float *ko, int rows,
                      int q_heads, int k_heads, float eps, void *stream);
void df2_kv_ring(const float *kin, const float *vin, const float *gk, const int32_t *pos, const float *table, float *kring, float *vring,
                 int slots, int rows, int heads, float eps, void *stream);

void df2_topk(const float *logits, int n, const int32_t *idx_map, int rows, uint32_t *scratch, float *val, int32_t *tok, void *stream);

void df2_select(const float *hp, const float *unary, const int32_t *tok, const uint16_t *pred_cb, const uint16_t *succ_cb, int rank,
                int rows, const int32_t *anchor_tok, int32_t *out_tok, float *out_prob, void *stream);

void df2_tap_gr(const float *R, int64_t R_stride, const float *bo, int64_t bo_stride, const float *inject, int64_t inj_stride, float *out,
                int64_t out_stride, int rows, int64_t n_embd, int hc, void *stream);

void df2_tap_rows(const float *R, float *out, int64_t out_stride, int64_t n_embd, int hc, int rows, void *stream);

} // namespace basalt::kernels
