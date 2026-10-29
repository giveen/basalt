// src/program/serve/runtime.cpp - the persistent request driver.
#include "runtime.hpp"
#include "../cli/tokens.hpp"
#include "loans.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/core/logit_bias.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/pinned.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/core/remote_experts.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/program/image_loan.hpp"
#include "basalt/program/multi_gpu_plan.hpp"
#include "basalt/program/tokenizer.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/mrope.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/native_dense.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/ppl_gate.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/spec/draft_policy.hpp"
#include "basalt/spec/suffix_drafter.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"
#include <cerrno>
#include <cstddef>   // std::ptrdiff_t

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <set>
#include <tuple>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::program::serve {
namespace {

namespace cli = basalt::program::cli;
namespace serve_loans = basalt::program::serve;
using engine_local::Drive;
using engine_local::GpuStage;
using engine_local::JobRunner;
using engine_local::PromptStage;
using engine_local::SplitDrive;
using engine_local::biggest_chunk;
using engine_local::drive_pool_multi;
using engine_local::drive_pool_split;
using engine_local::kSatBoost;
using engine_local::peer_hot_share_order;
using Clock = std::chrono::steady_clock;

// A saturated adaptation round may fill more of the tier on its next pass.
constexpr int kSatRatio = 2;

// Seeds the adaptive tier's usage counts from the routing of the prompt just read, so the first adaptation rounds
// of a request already move the experts the new topic uses toward the GPUs.  The earlier usage decays to kSeedKeep and
// the prompt adds its routed-entry counts scaled to kSeedTokens tokens of usage.  Resident experts are always seeded;
// of the rest only the kSeedTop with the most prompt entries are (a peer-held one weighs kSeedPeerWeight, since it
// costs less to leave where it is than a CPU one).
void seed_usage_from_prompt(std::vector<float>& usage, const std::vector<uint32_t>& hist, int64_t tokens,
                            const std::vector<int32_t>& host_res, int64_t n_expert, const basalt::core::PeerExperts& peer) {
    constexpr double kSeedTokens = 64.0;
    constexpr float kSeedKeep = 0.3f, kSeedPeerWeight = 0.35f;
    constexpr size_t kSeedTop = 32;
    if (tokens <= 0 || usage.empty() || hist.size() != usage.size() || host_res.size() != usage.size()) return;
    std::vector<uint8_t> allow(hist.size(), 1);
    std::vector<std::pair<float, size_t>> rank;
    for (size_t i = 0; i < hist.size(); ++i) {
        if (hist[i] == 0 || host_res[i] >= 0) continue;
        allow[i] = 0;
        const bool on_peer = peer.valid() && peer.has((int64_t) i / n_expert, (int64_t) i % n_expert);
        rank.emplace_back((float) hist[i] * (on_peer ? kSeedPeerWeight : 1.0f), i);
    }
    const size_t k = std::min(rank.size(), kSeedTop);
    std::partial_sort(rank.begin(), rank.begin() + (ptrdiff_t) k, rank.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (size_t j = 0; j < k; ++j) allow[rank[j].second] = 1;
    const float scale = (float) (kSeedTokens / (double) tokens);
    for (size_t i = 0; i < hist.size(); ++i) usage[i] = usage[i] * kSeedKeep + (allow[i] ? scale * (float) hist[i] : 0.0f);
}


// The same with free places of the RAM copy instead of exchange buffers (see FileExpertSource::set_spares): each
// evicted expert's slot is copied into a free place, which becomes its home once the copies have landed.  A swap with
// no free place waits for a later round.  `released` gets the (layer, in) pairs whose places are free once the incoming
// experts are resident on the GPU.
template <class Swap, class CopyBack, class Wait>
bool resident_spare_swaps(basalt::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, CopyBack&& copy_back, Wait&& wait,
                          std::vector<std::pair<int32_t, int32_t>>& released) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Home { int32_t layer, out; uint64_t off; };
    std::vector<Home> homes;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        uint64_t off = 0;
        uint8_t* dst = src.spare_take(s.layer, off);
        if (dst == nullptr) continue;
        if (!copy_back(s.layer, slot, dst)) {
            src.spare_return(s.layer, off);
            return false;
        }
        homes.push_back({s.layer, s.out, off});
        released.emplace_back(s.layer, s.in);
        kept.push_back(s);
    }
    if (!homes.empty()) {
        if (!wait()) return false;
        for (const Home& h : homes)
            if (!src.spare_land(h.layer, h.out, h.off)) return false;
    }
    swaps.swap(kept);
    return true;
}



/// Pipelined windows: the least chance (the cumulative draft rule's product up to the guess) a window's successor is
/// launched on, and how far below the window threshold the chain runs on past its window so there are drafts for it.
constexpr float kPairTheta = 0.50f;
constexpr float kPairChainScale = 0.25f;
constexpr double kPairFull = 0.70;   ///< the share of windows accepted whole from which the chain runs on


// ---- what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard
};
MemSample mem_sample() {
    const basalt::platform::ProcessMemory p = basalt::platform::process_memory();
    MemSample m;
    m.faults = p.faults;
    m.rss_mib = p.rss_mib;
    m.avail_mib = p.avail_mib;
    m.commit_mib = p.commit_mib;
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in
std::string stage_text() {
    const basalt::core::Progress& p = basalt::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

// the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): many threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    const basalt::platform::ThreadWaits w = basalt::platform::threads_in_disk_wait();
    if (w.threads == 0) return;
    std::string list;
    for (const auto& [name, n] : w.where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", w.waiting, w.threads, list.c_str(),
                 w.waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (--ple-io ram), or a drive too slow for the reads asked of it" : "");
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    basalt::core::Progress& p = basalt::core::progress();
    std::fprintf(f, "basalt serve: error: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", BASALT_VERSION, stage_text().c_str(),
                 (long long) ((basalt::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = basalt::core::diag_pool_fn().load()) fn(f);
        if (auto fn = basalt::core::diag_verify_fn().load()) fn(f);
        disk_wait_threads(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
        );
        std::fflush(f);
    }
}




// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again would make a long session wait for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = basalt::core::ConversationImageKey;
using ConvCheckpoint = basalt::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = basalt::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const basalt::core::ModelGeometry& g, const basalt::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    basalt::core::conversation_session_sizes(g, ss, z, error);
    return z;
}


/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint save: %s", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint restore: %s", error.c_str());
    return false;
}

/// The identity parked conversations on disk are written under (conversation_store.hpp): the model file
/// (its size and samples of its bytes) and the settings that change a state's bytes - the K/V type, the rope, the draft
/// layer, the image encoder, control vectors.  A file of another identity is never restored.
uint64_t conversation_identity(const Options& o, const basalt::core::ModelGeometry& g) {
    using basalt::core::conversation_store_hash;
    char num[512];
    std::snprintf(num, sizeof num, " rope %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g cvec %d %d %d %d", o.rope_scale,
                  o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor, o.yarn_attn_factor,
                  o.yarn_beta_fast, o.yarn_beta_slow, o.cvec_first, o.cvec_last, o.cvec_mode, o.cvec_single);
    std::string text = std::string("basalt conversation state kv ") + o.kv + " rope " +
                       o.rope_scaling + num + " mtp " + o.mtp + " mmproj " + o.vision_mmproj;
    for (const auto& [path, scale] : o.cvec_files) text += " cvec " + path + " " + std::to_string(scale);
    uint64_t h = conversation_store_hash(text.data(), text.size(), 1);
    const int64_t geo[] = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_v_heads, g.n_head_kv,
                           g.head_dim, g.idx_key_dim, g.n_expert, g.n_ff};
    h = conversation_store_hash(geo, sizeof geo, h);
    uint64_t size = 0;
    basalt::platform::file_size(o.model, size);
    h = conversation_store_hash(&size, sizeof size, h);
    const int fd = basalt::platform::open_fd_utf8(o.model);
    std::vector<uint8_t> buf((size_t) 64 << 10);
    const uint64_t n = std::min<uint64_t>(buf.size(), size);
    for (int i = 0; i < 16 && fd >= 0 && n > 0; ++i) {
        const uint64_t off = (size - n) / 15 * (uint64_t) i;
        if (!basalt::platform::pread_all(fd, buf.data(), n, off)) break;
        h = conversation_store_hash(buf.data(), (size_t) n, h);
    }
    basalt::platform::close_fd(fd);
    return h;
}


}  // namespace

/// Host RAM a server takes while it serves, beyond what it holds once ready: the conversation checkpoints (the solo
/// session's and those a batch slot keeps, each the whole running state), the parked conversations, one image's
/// host buffers at the largest canvas and a running request's own buffers.  The RAM copy of the experts leaves this
/// free beside the margin, so serving stays above the RAM guard.
uint64_t serve_growth_bytes(const Options& o, const basalt::core::ModelGeometry& g) {
    if (!o.serve) return 0;
    uint64_t bytes = basalt::platform::kRequestHostBytes;
    basalt::core::ConversationStateSizes z;
    std::string error;
    if (o.prompt_cache > 0 && basalt::core::conversation_state_sizes(g, z, error)) {
        const uint64_t one = z.gdn + z.ple + (uint64_t) g.n_qsa_layers() * (z.tail + z.dead + z.block_pos);
        bytes += (uint64_t) (o.prompt_cache + 1) * (uint64_t) std::max(o.batch, 1) * one;
        bytes += (uint64_t) o.conversation_cache_mib << 20;
    }
    if (!o.vision_mmproj.empty()) {
        // the canvas (8-bit RGB), its normalized planar floats and the patches
        const basalt::vision::Geometry vg;
        const uint64_t pixels = (uint64_t) vg.max_tokens * (uint64_t) (vg.patch * vg.merge) * (uint64_t) (vg.patch * vg.merge);
        bytes += pixels * 3 * (1 + 2 * sizeof(float));
    }
    return bytes;
}

int run(Context& context) {
    auto& [
        o, ppl_base, ppl_out, model_file, peer_streamed_share, split_at,
        split_devs, split_same, multi_gpu, pipe_want, no_spec, err,
        main_link_gbps, g, K, profile, peer_hot_share, profile_loaded,
        pf_borrow, wt, d_mrope, mrope_host, cvec_summary, ss,
        main_stream, main_cs, ple_table, vis_enc, stages, last_st,
        stage_of, profile_all, kv_floor, bslot_ss, slot_cap, slot0_is_solo,
        mtp, mtp_on, mtp_kv, n_vocab, grammar_vocab, native_head,
        mtp_head, src, srcp, xcache, remote_experts, remote_opt,
        peer, xpeers, drive, host_res, res_put, d_res,
        thits, graph_hits, auto_lend_pct, auto_ceiling, prompt_file_bytes, pool,
        ram_floor_bytes, pstages, read_cap, kSplitRefCells, split_hand_rows, split_peer_rows,
        split_price, split_layers_text, split_stages_text, split_stages, tier_file_bytes, report_expert_tiers
    ] = context;
    if ((!no_spec && (o.spec < 2 || o.mtp.empty())) || o.prefill_chunk <= 0 ||
        (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
        basalt::log::error("basalt serve", "needs --spec T (0 for none; T >= 2 also needs the model's draft layer) "
                             "and --prefill CHUNK (and a fillable "
                             "--expert-cache; the graphed hit path additionally needs an expert profile)");
        return 2;
    }
    basalt::prefill::Prefill sp;
    void* borrow = nullptr;
    uint64_t borrow_bytes = 0;
    int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
    // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]), which is
    // the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage borrowing the tail
    // of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the chunk, is laid out by
    // `Prefill::relayout`, and is refilled before any window reads - so outside the prompt the whole cache is
    // expert cache.  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose slot values
    // are indices into whichever cache owns the layer, so a loan that marked rows by slot number alone would hand
    // CUDA0 a slot belonging to another stage's cache.
    struct PfPart {
        basalt::core::ExpertCache* cache = nullptr;
        const basalt::core::SessionState* ses = nullptr;
        basalt::prefill::Prefill* sp = nullptr;
        int dev = -1;                  // -1: leave the device alone (CUDA0)
        int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
        int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
        int32_t first_now = -1;        // where its buffers are laid out now
        int64_t lent_chunk = 0;
        std::vector<std::pair<int32_t, int32_t>> lent;
    };
    auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
        return serve_loans::cache_slots_for(*p.cache, basalt::prefill::Prefill::bytes_needed(g, *p.ses, c));
    };
    auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
        basalt::core::ExpertCache& xc = *p.cache;
        return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                 : (uint64_t) (xc.slots() - first) *
                                       (uint64_t) basalt::kernels::cpu::expert_layout().max_blob;
    };
    std::vector<PfPart> pf_parts;
    // A prompt read beside decoding slots lends at most this many cache slots, so their experts stay in the
    // cache, and its streamed ring is this deep (the loan shrinks with it); otherwise the ring is full depth.
    constexpr int64_t kLendCapDecoding = 1216;
    constexpr int kRingDecoding = 64;
    // A read of at most this many tokens runs alone even while slots decode: they wait for it (a few seconds at
    // most) and it lends and reads as on an idle engine.  A longer read goes beside them, within the limits above,
    // and so does the rest of one that gave way to a shorter request (in its chunk).
    constexpr int64_t kReadAlone = 16384;
    // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
    // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
    // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
    if (pf_borrow && d_res != nullptr) {
        pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
        for (auto& st : stages)
            pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
        // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
        // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
        // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
        // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
        auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
            const int64_t k = part_slots(p, c);
            if (k <= 0 || k + 128 > p.cache->slots()) return false;
            return !(cap && k * 100 > auto_lend_pct * p.cache->slots());
        };
        // `only`: CUDA0's cache alone (what one GPU would choose, for the log below); null: every one
        auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
            if (only != nullptr) return fits_one(*only, c, cap);
            for (const PfPart& p : pf_parts)
                if (!fits_one(p, c, cap)) return false;
            return true;
        };
        auto scan = [&](const PfPart* only) -> int64_t {
            if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
                for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                    if (fits(c, false, only)) return c;
                return 0;
            }
            // the largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend, under
            // the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192)
            return engine_local::biggest_chunk(auto_ceiling, [&](int64_t t) -> bool { return fits(t, true, only); });
        };
        const int64_t chunk = scan(nullptr);
        // A small card in a layer split caps every stage's chunk.  Named when it bites, so the regression is one
        // log line: each stage that cannot fund the chunk CUDA0 alone would read in.
        const int64_t alone = pf_parts.size() > 1 ? scan(&pf_parts[0]) : chunk;
        if (chunk > 0 && alone > chunk) {
            for (size_t i = 1; i < pf_parts.size(); ++i) {
                const PfPart& p = pf_parts[i];
                if (fits_one(p, alone, o.prefill_auto)) continue;
                const int dev = p.dev < 0 ? 0 : p.dev;
                cudaDeviceProp prop{};
                if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
                    (void) cudaGetLastError();
                    prop.name[0] = 0;
                }
                const std::string pct =
                    o.prefill_auto ? ", and lend at most " + std::to_string(auto_lend_pct) + "%" : "";
                basalt::log::warn("basalt serve", "prompt chunk %lld tokens, not %lld: CUDA%d (%s) "
                                     "has %lld expert-cache slots, and a %lld-token chunk borrows %lld of them "
                                     "(it must keep 128%s) - prompts read slower than on CUDA0 alone",
                             (long long) chunk, (long long) alone, dev, prop.name,
                             (long long) p.cache->slots(), (long long) alone,
                             (long long) part_slots(p, alone), pct.c_str());
                // the helper tiers start at CUDA1 without a split (and are enabled in order)
                basalt::log::info("basalt serve", "  a card this small can serve as a helper expert cache "
                                     "instead of a split stage: without --layer-split, with %s",
                             dev == 1 ? "--expert-cache-device1 N" : "--expert-cache-device1..3 N, in order");
            }
        }
        if (chunk > 0) {
            if (o.prefill_auto)
                // the RESOLVED ring: a chunk under stream_all_min() gets the staging slots and those for copies
                // ahead of its routing.  This is what `init` lays out, and what the INFO line reports to the Monitor tab.
                basalt::log::info("basalt serve", "prompt chunk auto: %lld tokens, a %lld-slot ring",
                             (long long) chunk,
                             (long long) basalt::prefill::Prefill::ring_slots_for(chunk));
            else if (chunk != o.prefill_chunk)
                basalt::log::info("basalt serve", "prompt chunk %lld -> %lld tokens so its buffers fit in "
                                     "every expert cache", (long long) o.prefill_chunk, (long long) chunk);
            o.prefill_chunk = chunk;
            for (PfPart& p : pf_parts) {
                p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                p.first_now = p.first;
            }
            // a split stage whose card still has room for the chunk's buffers (its cache already holds every
            // expert of its layers, so auto stopped short of its VRAM) keeps them as its own instead of borrowing
            // cache slots: nothing of it is lent, streamed during the prompt or refilled after it.  Room = the
            // buffers + 1.5 GiB (the verify windows, the draft head, cuBLAS workspaces made after this).  Layer
            // splits only - one GPU keeps its loan.
            if (pf_parts.size() > 1) {
                for (PfPart& p : pf_parts) {
                    const basalt::core::OnDevice on(p.dev);
                    size_t fb = 0, tb = 0;
                    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                    const uint64_t need = basalt::prefill::Prefill::bytes_needed(g, *p.ses, chunk);
                    if ((uint64_t) fb >= need + (3ull << 29)) {
                        basalt::log::info("basalt serve", "  CUDA%d keeps its own prompt buffers (%.2f GiB of "
                                             "%.2f GiB free): no loan", p.dev < 0 ? 0 : p.dev,
                                     (double) need / 1073741824.0, (double) fb / 1073741824.0);
                        p.first = -1;
                        p.first_now = -1;
                    }
                }
            }
            lend_first = pf_parts[0].first;
            borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
            borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
        } else if (o.prefill_auto) {
            o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
        } else if (pf_parts.size() > 1) {
            // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
            // here and so do we, rather than refusing to start - but say what every stage has, because a split
            // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
            // and `init` would otherwise report only that the buffers do not fit.
            basalt::log::info("basalt serve", "no stage can lend the prompt path its %lld-token buffers, so "
                                 "each stage allocates its own:", (long long) o.prefill_chunk);
            for (const PfPart& p : pf_parts)
                basalt::log::info("basalt serve", "  CUDA%d has %lld slots, and a %lld-token chunk needs "
                                     "the last %lld of them", p.dev < 0 ? 0 : p.dev,
                             (long long) p.cache->slots(), (long long) o.prefill_chunk,
                             (long long) part_slots(p, o.prefill_chunk));
        }
    } else if (o.prefill_auto && d_res == nullptr) {
        o.prefill_chunk = 1024;   // no expert cache at all: small buffers of its own
    }
    bool any_loan = borrow != nullptr;
    for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
    if (any_loan) {
        if (borrow != nullptr)
            basalt::log::info("basalt serve", "the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
        for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
            if (pf_parts[i].first >= 0) basalt::log::info("basalt serve", "  CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)",
                         pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                         (long long) pf_parts[i].cache->slots(),
                         (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
    } else {
        basalt::log::info("basalt serve", "the prompt path allocates its own buffers (too few cache slots to borrow)");
    }
    // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next.
    // THE CHUNK STEPS DOWN INSTEAD OF EXITING.  A split's stage caches are sized after the arena is registered, and
    // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned a
    // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
    // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
    auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
            const basalt::core::OnDevice on(st.dev);
            void* sb = nullptr;              // this stage's own loan, out of its own cache
            uint64_t sbb = 0;
            // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
            if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
                sb = st.cache.device_slot(pf_parts[i + 1].first);
                sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
            }
            if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream,
                            err, sb, sbb)) {
                err = "layer split, CUDA" + std::to_string(st.dev) + " prompt path: " + err;
                return err.find("do not fit") != std::string::npos ? 2 : 1;
            }
        }
        if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
        if (!multi_gpu && o.batch <= 0 && !o.no_pool) { sp.set_cpu_pool(&pool); sp.set_cpu_share(o.prefill_cpu_share); }   // the pool is idle while a prompt is read
        if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))
            return err.find("do not fit") != std::string::npos ? 2 : 1;
        return 0;
    };
    static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
    {
        // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave headroom
        // on that device (a chunk that fits to the last MiB leaves the GEMM library nothing: its GEMMs would then
        // fail to launch).  `bytes_needed` is the same count `init` makes.
        const int64_t kHeadroom = 512ll << 20;
        auto own_fits = [&](int64_t c, int& dev_out, int64_t& need_out, int64_t& free_out) -> bool {
            for (size_t i = 0; i <= stages.size(); ++i) {
                const bool loan = i == 0 ? borrow != nullptr : (i < pf_parts.size() && pf_parts[i].first >= 0);
                if (loan) continue;
                const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                const basalt::core::OnDevice on(dev);
                size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                const int64_t need = (int64_t) basalt::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[i - 1]->ss, c);
                if (need + kHeadroom > (int64_t) fb) {
                    dev_out = dev < 0 ? 0 : dev; need_out = need; free_out = (int64_t) fb;
                    return false;
                }
            }
            return true;
        };
        int dev = 0;
        int64_t need = 0, fb = 0;
        if (!own_fits(o.prefill_chunk, dev, need, fb)) {
            int64_t c = 0;
            for (const int64_t s : kStepChunks) {
                int d2 = 0;
                int64_t n2 = 0, f2 = 0;
                if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
            }
            if (c > 0) {
                basalt::log::warn("basalt serve", "a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                     "%lld MiB free: %lld-token chunks", (long long) o.prefill_chunk,
                             (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                o.prefill_chunk = c;
            }
        }
    }
    for (;;) {
        const int r = init_prompt_paths();
        if (r == 0) break;
        int64_t next = 0;
        for (const int64_t c : kStepChunks)
            if (c < o.prefill_chunk) { next = c; break; }
        if (r == 2 && next > 0) {
            basalt::log::warn("basalt serve", "%s: trying a %lld-token chunk", err.c_str(), (long long) next);
            // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
            // error: it is sticky, and the next launch check would report "out of memory" for a kernel
            for (auto& stp : stages) {
                const basalt::core::OnDevice on(stp->dev);
                stp->sp.reset();
                (void) cudaGetLastError();
            }
            sp.reset();
            (void) cudaGetLastError();
            o.prefill_chunk = next;
            if (any_loan) {                  // smaller loans for the smaller chunk
                for (PfPart& p : pf_parts)
                    if (p.first >= 0) {
                        p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                        p.first_now = p.first;
                    }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            }
            err.clear();
            continue;
        }
        basalt::log::error("basalt serve", "%s", err.c_str());
        if (err.find("fit") != std::string::npos)   // say what frees VRAM
            basalt::log::error("basalt serve", "the GPU has too little free VRAM for the prompt path: move images "
                                 "to the CPU (--vision cpu) or off, close other programs using the GPU, use a shorter "
                                 "context, or read prompts in smaller chunks (--prefill 512)");
        return 1;
    }
    if (peer.valid() && o.peer_exclusive) sp.set_ring_peer(&peer);
    if (peer.valid() && o.peer_prefill_rows != 0) {
        const int64_t rows = o.peer_prefill_rows > 0 ? o.peer_prefill_rows : o.prefill_chunk * K / 2;
        if (!sp.set_peer(&peer, rows, peer_streamed_share, err)) {
            basalt::log::warn("basalt serve", "%s - the prompt path stays on the primary GPU", err.c_str());
            err.clear();
        }
        split_peer_rows = sp.peer_routed_rows_max();
    }
    // ---- the prompt-time split's stages on the tier cards.  A stage's place, from the end of its tier's slots: the
    // weights, the session (for the whole context here; for its prompt's cells and layers at each read), then the
    // prompt buffers.  The layout is made over live slots and writes nothing until a read lends them.
    auto split_region = [&](PromptStage& st, int64_t cells, int64_t chunk, int64_t l0, int64_t l1, uint8_t*& w_at,
                            uint8_t*& s_at, uint8_t*& b_at) -> bool {
        const basalt::core::ExpertCache& pc = st.tier->cache();
        uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
        auto down = [](uint8_t* p) { return (uint8_t*) ((uintptr_t) p & ~(uintptr_t) 255); };
        w_at = down(base + pc.bytes() - st.image_bytes);
        s_at = down(w_at - basalt::core::prompt_session_bytes(g, cells, l0, l1));
        basalt::core::prompt_session_carve(g, cells, K, s_at, st.ss, l0, l1);
        b_at = down(s_at - basalt::prefill::Prefill::bytes_needed(g, st.ss, chunk));
        // the loan leaves the tier a tenth of its slots, as the main card's
        return b_at >= base + pc.bytes() / 10;
    };
    // its buffers' and the hand-off buffers' size; a larger chunk reads on CUDA0
    const int64_t split_chunk_max = read_cap();
    bool split_ready = split_stages >= 2 && !pstages.empty();
    for (size_t j = 0; split_ready && j < pstages.size(); ++j) {
        PromptStage& st = *pstages[j];
        const basalt::core::OnDevice on(st.dev);
        size_t fb0 = 0, tb0 = 0;
        cudaMemGetInfo(&fb0, &tb0);
        uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
        // (a read of the longest prompt the start priced must fit; a longer one checks its own fit)
        bool ok = split_chunk_max >= 1024 &&
                  split_region(st, std::min<int64_t>(o.max_context, kSplitRefCells), split_chunk_max, st.lo, st.hi,
                               w_at, s_at, b_at);
        ok = ok && cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) == cudaSuccess &&
             st.state_in.init(0, st.dev, 8u << 20, err) && st.state_back.init(st.dev, 0, 8u << 20, err);
        if (ok) {
            st.res.assign((size_t) (g.n_layers * g.n_expert), basalt::core::kNotResident);
            st.sp.set_stage(st.lo, -1, nullptr);
            ok = st.sp.init(st.wt, g, st.ss, srcp, &st.tier->cache(), st.res.data(), split_chunk_max,
                            (void*) st.stream, err, b_at, (uint64_t) (s_at - b_at));
        }
        // a stage that may hand its chunks on (every card but the one that reads the last layers) holds two pinned
        // buffers of the stage's chunk, as the main card does
        ok = ok && (j == 0 || st.sp.reserve_hand_off(split_chunk_max, err));
        // a read's kernels and small buffers load on the card outside the lent slots: a card without room for them
        // beside its stage declines here, before a read could fail on them
        constexpr size_t kStageFreeMin = 256ull << 20;
        size_t fb = 0, tb = 0;
        if (ok && cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb < kStageFreeMin) {
            ok = false;
            err = std::to_string(fb >> 20) + " MiB free beside its stage, a read needs " +
                  std::to_string(kStageFreeMin >> 20) + "; --peer-reserve-mib";
        }
        if (ok) {
            st.sp.set_ring_remote(host_res.data(), &xcache, 0);
            basalt::log::info("basalt serve", "prompt split: CUDA%d can take layers %lld-%lld of a long prompt (chunks up "
                                 "to %lld tokens, links %.1f/%.1f GB/s); %zu MiB free on it (%zu before)", st.dev,
                              (long long) st.lo, (long long) st.hi - 1, (long long) split_chunk_max,
                              main_link_gbps, st.link_gbps, fb >> 20, fb0 >> 20);
        } else {
            basalt::log::warn("basalt serve", "prompt split: off (CUDA%d: %s)", st.dev,
                              err.empty() ? "too few tier slots" : err.c_str());
            err.clear();
            (void) cudaGetLastError();
            split_ready = false;
        }
    }
    // the main card hands its chunks on through two pinned buffers of the stage's chunk
    if (split_ready) split_hand_rows = split_chunk_max;
    if (split_ready && !sp.reserve_hand_off(split_chunk_max, err)) {
        basalt::log::warn("basalt serve", "prompt split: off (%s)", err.c_str());
        err.clear();
        split_ready = false;
    }
    std::vector<uint32_t> split_hist;   // the routed entries the stages counted in the last split read
    int64_t split_hist_tokens = 0;
    if (split_ready && pstages.size() >= 2)
        basalt::log::info("basalt serve", "prompt split: up to %d stages over CUDA0 and %zu tier cards", split_stages,
                          pstages.size());
    // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last `penalty_last_n`
    // tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for the widest window; a
    // request without penalties gets a null buffer and takes the byte-for-byte neutral path (no upload, no buffer
    // handed to the sampler).
    constexpr int kPenaltyWindowCap = 4096;
    constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) basalt::kernels::kVerifyMaxT;
    int32_t* d_hist = nullptr;
    std::vector<int32_t> hist_stage(kHistSlots, -1);
    const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
    if (const basalt::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
        basalt::log::error("basalt serve", "the penalty-history allocation failed");
        return 1;
    }
    basalt::core::Verifier ver;
    basalt::core::VerifyHits vh;
    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
    vh.n_slots = xcache.slots();
    // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
    // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
    // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
    basalt::core::Verifier ver_same;
    SplitDrive split_drive, split_drive_b;
    basalt::core::Verifier ver_b;   // pipelined windows: the second chain over the same sessions (stage 0)
    void* pair_snap = nullptr;
    auto stage_ver = [&](int st) -> basalt::core::Verifier& {
        return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
    };
    auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
    const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
    // two stages, the drafter and one conversation at a time: windows are pipelined over two chains of verifiers
    const bool pipe_on = pipe_want && n_stages == 2;
    auto stage_verb = [&](int st) -> basalt::core::Verifier& { return st == 0 ? ver_b : stages[(size_t) st - 1]->ver_b; };
    if (n_stages > 1) {
        const size_t hb = (size_t) basalt::kernels::kVerifyMaxT *
                          (size_t) basalt::core::Verifier::handoff_floats(g) * sizeof(float);
        std::vector<float*> hand((size_t) n_stages - 1, nullptr);
        for (float*& h : hand) {
            float* hh = nullptr;
            if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                basalt::log::error("basalt serve", "the layer-split hand-off allocation failed");
                return 1;
            }
            std::memset(hh, 0, hb);
        }
        split_drive.base = &drive;
        split_drive.n = n_stages;
        for (int st = 0; st < n_stages; ++st) {
            stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                    st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
            split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
            split_drive.cache_base[st] = drive.d.cache_base;
            split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
            split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
        }
        for (int st = 1; st < n_stages; ++st) {
            bool ok_s = false;
            if (split_same) {
                ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err);
            } else {
                GpuStage& gs = *stages[(size_t) st - 1];
                const basalt::core::OnDevice on(gs.dev);
                basalt::core::VerifyHits vs;
                vs.d_res = gs.d_res;
                vs.h_res = host_res.empty() ? nullptr : host_res.data();
                vs.cache_base = gs.cache.device_slot(0);
                vs.blob = thits.blob;
                vs.slot_off = gs.cache.slot_offsets();
                vs.n_slots = gs.cache.slots();
                gs.ver.set_remote_expert_opt(remote_opt.get());
                ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err);
                split_drive.cache_base[st] = gs.cache.device_slot(0);
                split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
            }
            if (!ok_s) {
                basalt::log::error("basalt serve", "layer split, stage %d: %s", st + 1, err.c_str());
                return 1;
            }
        }
        for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
        if (pipe_on) {
            GpuStage& gs = *stages[0];
            float* hb_host = nullptr;
            float* hand_b = nullptr;
            if (cudaHostAlloc((void**) &hb_host, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &hand_b, hb_host, 0) != cudaSuccess ||
                cudaMalloc(&pair_snap, basalt::core::Verifier::pair_snapshot_bytes(g, ss, 0, split_at[0])) != cudaSuccess) {
                basalt::log::error("basalt serve", "the pipelined windows' buffers failed");
                return 1;
            }
            std::memset(hb_host, 0, hb);
            ver_b.set_stage(0, split_at[0], nullptr, hand_b);
            gs.ver_b.set_stage(split_at[0], -1, hand_b, nullptr);
            ver.set_pipelined(pair_snap);
            ver_b.set_pipelined(pair_snap);
            const basalt::core::OnDevice on(gs.dev);
            basalt::core::VerifyHits vs;
            vs.d_res = gs.d_res;
            vs.h_res = host_res.empty() ? nullptr : host_res.data();
            vs.cache_base = gs.cache.device_slot(0);
            vs.blob = thits.blob;
            vs.slot_off = gs.cache.slot_offsets();
            vs.n_slots = gs.cache.slots();
            gs.ver_b.set_remote_expert_opt(remote_opt.get());
            if (!gs.ver_b.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
                basalt::log::error("basalt serve", "pipelined windows, stage 2: %s", err.c_str());
                return 1;
            }
            ver_b.set_next(&gs.ver_b, &split_drive_b);
            split_drive_b = split_drive;
            basalt::log::info("basalt serve", "pipelined windows: on");
        }
        std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
        for (int st = 1; st < n_stages; ++st)
            plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                      " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
        basalt::log::info("basalt serve", "layer split: layers %s, one hand-off per window", plan_s.c_str());
    }
    ver.set_remote_expert_opt(remote_opt.get());
    ver.set_main_priority(o.batch > 0);   // beside a prompt read the slots' windows must win the SMs
    if (peer.valid()) ver.set_peer(&peer);   // the window drives the peer tier itself
    for (const auto& xpe : xpeers)
        if (xpe->valid()) ver.add_peer(xpe.get());
    ver.set_cpu_warm(drive.d.pool, drive.d.src);
    if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err) ||
        (mtp_on && !mtp.bind(wt, multi_gpu ? &mtp_head : &native_head, ver.final_R_all(), err))) {
        basalt::log::error("basalt serve", "%s", err.c_str());
        return 1;
    }
    if (pipe_on && !ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
        basalt::log::error("basalt serve", "pipelined windows: %s", err.c_str());
        return 1;
    }
    // each stage's verifier gets its stage's slot sessions
    if (o.batch > 0) {
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            std::vector<basalt::core::SessionState*> ptrs;
            for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
            basalt::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            if (!vk.init_slots(ptrs, err)) {
                basalt::log::error("basalt serve", "--batch, stage %zu: %s", k + 1, err.c_str());
                return 1;
            }
        }
    }
    for (int st = 0; st < n_stages && n_stages > 1; ++st) {
        split_drive.plan[st] = stage_ver(st).plan_sink();
        if (pipe_on) split_drive_b.plan[st] = stage_verb(st).plan_sink();
    }
    // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
    const basalt::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
    void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
    std::vector<int64_t> cur;
    // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens it
    // has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the saved
    // points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain - the radix
    // cache's tree collapsed onto the one branch of history whose cells the session holds.  The chain's root is the
    // deepest point every request so far shared (the end of the system prompt, in practice); the retention policy
    // pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that shares that prefix mounts through it
    // instead of reading it again.
    std::vector<int32_t> live;
    std::vector<ImgKey> live_imgs, req_imgs;
    bool live_ok = false;
    std::vector<ConvCheckpoint> checks;
    uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
    // A turn-boundary checkpoint's GDN bytes reach the host after the first token (see the deferral class); this
    // is declared after `checks` so it is destroyed, and its copy finished, before the checkpoints are.
    basalt::core::ConversationCheckpointDeferral ckd;
    {
        ConvCheckpoint warm;   // one set of pooled blocks, page-faulted now instead of on the first request
        const ConvStateSizes zw = conv_state_sizes(g, ss);
        if (o.prompt_cache > 0) {
            warm.gdn.resize(zw.gdn);
            std::memset(warm.gdn.data(), 0, warm.gdn.size());
            if (ss.ple_hist) { warm.ple.resize(zw.ple); std::memset(warm.ple.data(), 0, warm.ple.size()); }
        }
        if (o.prompt_cache > 0 && !multi_gpu) {
            std::string ce;
            if (!ckd.reserve(zw.gdn, ce))
                basalt::log::warn("basalt serve", "deferred checkpoint copy off: %s", ce.c_str());
        }
    }
    // finishes the in-flight host copy; a failed one drops its checkpoint
    auto ckd_finish = [&]() {
        const void* dst = ckd.pending();
        if (dst == nullptr) return;
        std::string ce;
        if (ckd.finish(ce)) return;
        basalt::log::error("basalt serve", "%s", ce.c_str());
        checks.erase(std::remove_if(checks.begin(), checks.end(),
                                    [&](const ConvCheckpoint& k) { return k.gdn.data() == dst; }), checks.end());
    };
    bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
    // the session the request path reads, parks and restores: the solo one, or (a --batch request for a slot other
    // than 0) a view of it carrying that slot's state, so the prompt is read in the slot itself
    // (one view per slot, each at a fixed address: the verifier keeps solo window graphs per view)
    std::vector<basalt::core::SessionState> rviews((size_t) std::max(o.batch, 1));
    basalt::core::SessionState* rsp = &ss;
    basalt::core::ConversationCache conversations(
        o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
        (size_t) o.conversation_cache_slots);
    // the disk tier under it (--conversation-cache-dir): what leaves the session or the RAM cache is written there,
    // and a prompt that starts with a stored conversation reads it back, after a restart too
    basalt::core::ConversationStore disk;
    constexpr int64_t kDiskMinTokens = 8192;              // shorter ones are cheaper to read again
    constexpr int64_t kDiskStaleTokens = 4096;            // a stored copy this close to the chain's end is kept
    constexpr uint64_t kDiskQueueCap = 4ull << 30;        // images waiting for the writer, in RAM
    uint64_t disk_floor = 0;                              // the RAM a capture or a read leaves free (set below)
    // Whether the disk tier needs no new copy of a conversation: it is short, or a stored prefix of it reaches
    // within kDiskStaleTokens of the chain's deepest checkpoint (the point its next turn goes on from; a restore
    // from that copy reads the rest again).  Each turn of a growing conversation is not written again in full.
    auto disk_has = [&](const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                        const std::vector<ConvCheckpoint>& cks, bool cvec) -> bool {
        if ((int64_t) ids.size() < kDiskMinTokens) return true;
        int64_t deepest = 0;
        for (const ConvCheckpoint& c : cks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
        if (deepest <= 0) deepest = (int64_t) ids.size();
        const auto m = disk.best(ids.data(), ids.size(), imgs, cvec);
        return m.tokens > 0 && m.tokens + kDiskStaleTokens >= deepest;
    };
    // A capture of a conversation's state, or a read of one from disk, takes `need` bytes of RAM: only while the RAM
    // guard reads its level as ok, and with `floor` left free after it (the disk tier's floor sits above the guard's).
    auto capture_admit = [&](uint64_t need, uint64_t floor) -> bool {
        return basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok &&
               basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), need, floor);
    };
    if (o.conversation_cache_disk_mib > 0) {
        std::string de;
        const auto t0 = Clock::now();
        if (o.prompt_cache <= 0 || !stages.empty()) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s",
                              o.prompt_cache <= 0 ? "--prompt-cache 0" : "a layer split");
        } else if (!disk.open(o.conversation_cache_dir, conversation_identity(o, g),
                              (uint64_t) o.conversation_cache_disk_mib << 20, de)) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s", de.c_str());
        } else {
            conversations.set_spill(true);
            const auto st = disk.stats();
            basalt::log::info("basalt serve", "conversation cache on disk: %s, %zu conversations of this model "
                              "(%.2f GiB), budget %lld MiB, indexed in %.0f ms", o.conversation_cache_dir.c_str(),
                              st.files, (double) st.bytes / 1073741824.0, (long long) o.conversation_cache_disk_mib,
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
    }
    // Save only on a switch/rewind, not on each continuing request. No graph
    // addresses change: all parked images live in ordinary host vectors.
    // `kept`: the tokens of the outgoing conversation the session goes on holding (the request continues from its
    // own chain), 0 when another conversation takes the session
    auto park_current = [&](size_t held, int64_t kept) -> bool {
        ckd_finish();
        if ((!conversations.enabled() && !disk.enabled()) || !live_ok || live.empty()) return true;
        // the disk tier takes the image when there is no RAM tier or it cannot make room, unless the conversation is
        // short or stored already, or the request goes on from the chain's deepest checkpoint (the same
        // conversation's next turn: all it drops is the last reply)
        bool to_disk = !conversations.enabled();
        int64_t deepest = 0;
        for (const ConvCheckpoint& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
        const bool disk_wants = disk.enabled() && !(kept > 0 && kept >= deepest) &&
                                !disk_has(live, live_imgs, checks, cvec_cached);
        if (to_disk && !disk_wants) return true;
        // before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold nothing
        // the outgoing chain does not, apart from the tail this conversation rewrote)
        if (const size_t dropped = conversations.drop_superseded(live, live_imgs, checks, cvec_cached))
            basalt::log::info("basalt serve", "conversation cache: dropped %zu superseded cop%s of this "
                         "conversation; parked=%zu", dropped, dropped == 1 ? "y" : "ies", conversations.size());
        // A layer split parks one image per stage: the first stage's (with the draft layer's K/V, the drafter
        // runs on CUDA0) and one per later stage (without it).  The checkpoints are MOVED apart into their
        // stage parts for the capture and put back together whichever way this ends (no running state copied).
        const size_t n_st = stages.size();
        const basalt::core::QsaState* draft0 = mtp_kv;
        auto draft_of = [&](size_t) -> const basalt::core::QsaState* { return nullptr; };
        basalt::core::ConversationCheckpointSplit cs;
        if (n_st > 0) cs = basalt::core::conversation_checkpoints_split(std::move(checks), n_st);
        struct MergeBack {
            basalt::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
            ~MergeBack() { if (on && !basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
        } merge_back{cs, checks, n_st > 0};
        const basalt::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
        auto reuse = conversations.take_reuse(rsp);
        std::vector<basalt::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
        stage_reuse.resize(n_st);
        reuse.stages.clear();
        // the later stages' estimates, each with its own retained K/V when it still fits
        size_t stage_estimate = 0, stage_retained = 0;
        for (size_t k = 0; k < n_st; ++k) {
            auto& st = stages[k];
            const basalt::core::OnDevice on(st->dev);
            const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
            size_t b = 0;
            if (!basalt::core::conversation_snapshot_bytes(view_k, st->ss, g, draft_of(k), b, err)) {
                basalt::log::warn("basalt serve", "conversation cache: skip parking (stage CUDA%d: %s)",
                             st->dev, err.c_str());
                err.clear();
                return true;
            }
            if (!stage_reuse[k].kv.empty()) {
                size_t r = 0;
                if (basalt::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, st->ss, g, draft_of(k),
                                                                      r, err)) {
                    b = r;
                    stage_retained += stage_reuse[k].bytes();
                } else {
                    stage_reuse[k] = {};
                    err.clear();
                }
            }
            stage_estimate += b;
        }
        size_t estimate = 0;
        if (!basalt::core::conversation_snapshot_bytes(view, *rsp, g, draft0, estimate, err)) {
            basalt::log::warn("basalt serve", "conversation cache: skip parking (%s)", err.c_str());
            err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
            return true;
        }
        const size_t fresh_estimate = estimate;
        if (!reuse.kv.empty() && !basalt::core::conversation_snapshot_capture_bytes(
                reuse, view, *rsp, g, draft0, estimate, err)) {
            reuse = {};
            estimate = fresh_estimate;
            err.clear();
        }
        estimate += stage_estimate;
        // A park carrying its own retained K/V replaces memory the cache
        // already held, so capacity is make_room's call - it runs next either
        // way, and put()'s accounting still bounds the budget. The with-reuse
        // estimate must stay uncapped: it counts the retained buffers'
        // capacity and directories, and put() charges that same true size -
        // a capped figure would under-evict and overfill the budget.
        const bool ram_room = !to_disk && conversations.make_room(estimate, held);
        if (!to_disk && !ram_room && disk_wants) to_disk = true;   // larger than the RAM tier: the disk tier takes it
        if (to_disk ? disk.queued_bytes() + estimate > kDiskQueueCap : !ram_room) {
            basalt::log::info("basalt serve", "conversation cache: skip parking (snapshot %zu MiB exceeds available budget)",
                         estimate >> 20);
            return true;
        }
        const auto t0 = Clock::now();
        try {
            const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024,
                                                      disk.enabled() ? disk_floor : 0);
            const size_t retained = reuse.bytes() + stage_retained;
            const size_t additional = estimate > retained ? estimate - retained : 0;
            if (!capture_admit(additional, floor)) {
                basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable, or RAM below the guard)",
                             additional >> 20, (long long) o.conversation_cache_min_free_mib);
                return true;
            }
            basalt::core::SavedConversation image;
            size_t reused_bytes = 0;
            if (!basalt::core::conversation_snapshot_save(image, view, *rsp, g, draft0, err,
                    std::move(reuse), &reused_bytes)) return false;
            for (size_t k = 0; k < n_st; ++k) {   // the later stages
                auto& st = stages[k];
                const basalt::core::OnDevice on(st->dev);
                if (cudaDeviceSynchronize() != cudaSuccess) { err = "stage sync failed"; return false; }
                const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
                basalt::core::SavedConversation part;
                if (!basalt::core::conversation_snapshot_save(part, view_k, st->ss, g, draft_of(k), err,
                        std::move(stage_reuse[k]), &reused_bytes))
                    return false;
                image.stage_images.push_back(std::move(part));
            }
            if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
                basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)");
                return true;
            }
            const size_t snapshot_bytes = image.bytes();
            const bool stored = to_disk ? disk.put(std::move(image), kDiskQueueCap)
                                        : conversations.put(std::move(image), held);
            basalt::log::info("basalt serve", "conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu%s",
                         stored ? "parked" : "skipped", live.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                         conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes, reused_bytes,
                         to_disk ? " (to disk)" : "");
            // what the RAM tier evicted to make room goes to the disk tier
            for (auto& s : conversations.take_spilled())
                if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec)) disk.put(std::move(s), kDiskQueueCap);
        } catch (const std::bad_alloc&) {
            // The active state has not been touched. Continue with normal
            // prompt processing rather than killing a serving process.
            basalt::log::warn("basalt serve", "conversation cache: allocation failed; skip parking");
        }
        return true;
    };
    int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
    // the end of the part being read when the part loop saves a checkpoint there (else -1): the periodic
    // checkpoint skips that cell, the loop's copy runs beside what follows
    int64_t pp_part_ckpt = -1;
    // the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled mid-read
    // reports as read, instead of the whole prompt
    int64_t pp_reached = 0;
    Clock::time_point pp_t0 = Clock::now();
    auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
        std::vector<ImgKey> v;
        for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
        return v;
    };
    // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
    // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
    // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
    // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
    std::mutex part_mu;
    std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
    std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
    // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
    // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
    // why the last checkpoint_at failed - a failed device sync is the GPU itself, not the checkpoint, and the error
    // says so
    std::string ckpt_why;
    auto gpu_sync_ok = [&]() -> bool {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e == cudaSuccess) return true;
        ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                   ") - a GPU hang";
        return false;
    };
    auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool defer = false) -> bool {
        ckpt_why.clear();
        ckd_finish();
        if (o.prompt_cache <= 0 || L < 1) return true;
        for (ConvCheckpoint& c : checks)
            if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
        ConvCheckpoint c;
        c.ids.assign(cur.begin(), cur.begin() + L);
        c.imgs = imgs_below(req_imgs, L);
        if (parts != nullptr) {
            if (parts->size() != stages.size() + 1) return false;
            c.gdn = std::move((*parts)[0].gdn);
            c.ple = std::move((*parts)[0].ple);
            c.tails = std::move((*parts)[0].tails);
            c.dead = std::move((*parts)[0].dead);
            c.block_pos = std::move((*parts)[0].block_pos);
            for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
        } else {
            if (!gpu_sync_ok()) return false;
            std::string ce;
            if (defer && stages.empty()) {
                if (!ckd.save(c, *rsp, g, ce)) { basalt::log::error("basalt serve", "checkpoint save: %s", ce.c_str()); return false; }
            } else if (!checkpoint_save(c, *rsp, g)) return false;
            for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                const basalt::core::OnDevice on(st->dev);
                ConvCheckpoint part;
                part.ids = c.ids;
                if (!gpu_sync_ok() || !checkpoint_save(part, st->ss, g)) return false;
                c.stage_parts.push_back(std::move(part));
            }
        }
        c.used = ++check_clock;
        checks.push_back(std::move(c));
        while ((int) checks.size() > o.prompt_cache) {
            std::vector<uint64_t> stamps;
            stamps.reserve(checks.size());
            for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
            const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                               o.prompt_cache);
            if (checks[victim].gdn.data() == ckd.pending()) ckd_finish();
            checks.erase(checks.begin() + (std::ptrdiff_t) victim);
        }
        return true;
    };
    // live progress inside a chunk: PL <estimated position> <prompt tokens> <ms> <tokens/s>, at most twice a second
    sp.on_layer = [&, last_ms = 0.0](int64_t layers_done, int64_t n_layers, int64_t T, int64_t p0) mutable {
        if (pp_total <= 0 || layers_done <= 0 || n_layers <= 0) return;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        if (ms - last_ms < 500.0 && ms >= last_ms) return;
        last_ms = ms;
        const int64_t at = p0 + T * layers_done / n_layers;
        std::printf("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (at - pp_from) / ms : 0.0);
        std::fflush(stdout);
    };
    sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
        std::vector<int32_t> nxt((size_t) T);
        for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
        // batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
        const bool batched = mtp_on && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
        if (!e.empty() || (mtp_on && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
        // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
        const int64_t done = p0 + T;
        pp_reached = done;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
        basalt::core::progress_at("reading the prompt (batched), done up to token", done);
        basalt::core::progress_beat();
        std::fflush(stdout);
        if (o.prompt_cache_every > 0 && done >= pp_next_check) {
            if (done == pp_part_ckpt && !multi_gpu) {
                pp_next_check = done + o.prompt_cache_every;
                return true;
            }
            bool saved = false;
            if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                std::vector<ConvCheckpoint> parts;
                {
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto it = part_at.find(done);
                    if (it != part_at.end()) parts = std::move(it->second);
                    part_at.erase(part_at.begin(), part_at.upper_bound(done));
                }
                bool complete = parts.size() == stages.size() + 1;
                for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
            } else {
                saved = checkpoint_at(done);
            }
            if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
            pp_next_check = done + o.prompt_cache_every;
        }
        return true;
    };
    if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
        stages.back()->sp.on_chunk = std::move(sp.on_chunk);
        sp.on_chunk = nullptr;
        for (size_t i = 0; i <= stages.size(); ++i) {
            basalt::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
            basalt::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
            stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                part_next[i] = done + o.prompt_cache_every;
                ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                part.ids.assign(cur.begin(), cur.begin() + done);
                if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                std::lock_guard<std::mutex> lk(part_mu);
                auto& v = part_at[done];
                v.resize(stages.size() + 1);
                v[i] = std::move(part);
                return true;
            };
        }
    }
    drive.d.plan = ver.plan_sink();
    drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
    const bool all_experts_resident = !host_res.empty() &&
        std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
    if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
        drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
    // --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays empty
    // and nothing below runs).  It needs the adaptive tier's counts and the residency table.
    std::vector<double> heat;
    if (!o.expert_profile_save.empty() && !all_experts_resident) {
        if (drive.d.usage.empty() || host_res.empty())
            basalt::log::warn("basalt serve", "--expert-profile-save needs the adaptive tier (--adapt-every and "
                                 "--adapt-swaps above 0) and an expert profile: nothing will be saved");
        else
            heat.assign(drive.d.usage.size(), 0.0);
    }
    Clock::time_point profile_saved_at = Clock::now();
    cudaStream_t adapt_stream = nullptr;
    if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
        basalt::log::error("basalt serve", "cannot create the refill stream");
        return 1;
    }
    // swaps in flight - (residency index, slot) admitted when adapt_ev has completed
    std::vector<std::pair<int32_t, int32_t>> pending;
    std::vector<std::array<int32_t, 3>> promoted;   // (layer, in, out) of swaps from the peer
    cudaEvent_t adapt_ev = nullptr;
    cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
    // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
    auto res_upload = [&]() {
        if (d_res != nullptr)
            res_put(d_res);
        for (auto& st : stages) {
            const basalt::core::OnDevice on(st->dev);
            res_put(st->d_res);
        }
    };
    // the peer's swaps land a window after their copies are queued, so no window waits for a copy
    bool peer_landing = false;
    std::vector<uint8_t> xlanding(xpeers.size(), 0);
    // the peer tiers take the next most-routed CPU misses: each passes over what the primary (r0) or another tier
    // holds or takes in this round; with one tier r0 is its whole view
    auto adapt_peers = [&](const std::vector<int32_t>& r0, std::string& perr) -> bool {
        const int swaps_each = o.peer_adapt_swaps >= 0 ? o.peer_adapt_swaps : o.adapt_swaps;
        if (xpeers.empty()) return peer.adapt(drive.d.usage.data(), r0.data(), swaps_each, perr);
        const size_t nt = 1 + xpeers.size();
        auto tier = [&](size_t i) -> basalt::core::PeerExperts& { return i == 0 ? peer : *xpeers[i - 1]; };
        for (size_t t = 0; t < nt; ++t) {
            if (!tier(t).valid()) continue;
            std::vector<int32_t> seen = r0;
            for (size_t u = 0; u < nt; ++u) {
                if (u == t || !tier(u).valid()) continue;
                const std::vector<int32_t>& ru = tier(u).residency();
                for (size_t i = 0; i < seen.size() && i < ru.size(); ++i)
                    if (ru[i] >= 0) seen[i] = 0;
                for (size_t i : tier(u).incoming())
                    if (i < seen.size()) seen[i] = 0;
            }
            if (!tier(t).adapt(drive.d.usage.data(), seen.data(), swaps_each, perr)) return false;
        }
        return true;
    };
    std::vector<std::pair<int32_t, int32_t>> pending_spare;   // the RAM places the incoming experts free once resident
    auto apply_pending = [&](bool wait, bool exchanges = true) {
        if (peer.valid()) {
            if (!wait) peer.apply_pending(wait);
            else if (peer_landing) { peer.apply_pending(true); peer_landing = false; }
            else if (peer.has_pending()) { peer.issue_pending(true); peer_landing = true; }
        }
        for (size_t q = 0; q < xpeers.size(); ++q) {
            basalt::core::PeerExperts& x = *xpeers[q];
            if (!x.valid()) continue;
            if (!wait) x.apply_pending(wait);
            else if (xlanding[q]) { x.apply_pending(true); xlanding[q] = 0; }
            else if (x.has_pending()) { x.issue_pending(true); xlanding[q] = 1; }
        }
        if (pending.empty()) return;
        if (wait) cudaEventSynchronize(adapt_ev);
        else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
        for (auto& st : stages)
            if (st->adapt_live) {
                if (wait) cudaEventSynchronize(st->adapt_ev);
                else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
            }
        for (auto& st : stages) st->adapt_live = false;
        if (exchanges) src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
        for (const auto& [i, slot] : pending) {
            host_res[(size_t) i] = slot;
            srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
        }
        for (const auto& [l, e] : pending_spare) src.spare_release(l, e);
        pending_spare.clear();
        pending.clear();
        res_upload();
        // a promoted expert is CUDA0's now: its peer slot takes the expert CUDA0 gave up (in RAM since the commit)
        for (const auto& p : promoted) {
            std::string perr;
            if (!peer.reassign(p[0], p[1], p[2], srcp->blob(p[0], p[2]), perr))
                basalt::log::warn("basalt serve", "%s", perr.c_str());
        }
        promoted.clear();
    };
    // the VRAM tier follows the conversation (the same rule as the speculative loop below)
    // the peer's experts are candidates too, so the ones a workload routes most move to
    // CUDA0 (from their RAM copy, as any other swap) and its coldest go to the CPU
    // routings a peer expert needs above CUDA0's victim before it moves
    constexpr float promote_margin = 10.0f;
    // the round's swaps, best first: the most-routed missing experts against the least-routed resident ones, per
    // layer, or over all layers (`across`: the incoming expert takes a slot of the outgoing one's layer `lout`)
    struct Swap { float gain; int32_t layer, in, out, lout; };
    // a layer whose peer experts take more than this share of its 10 routed entries per token promotes them without
    // the margin (a layer's peer share, not the step's, sets how late the peer is)
    constexpr float promote_load = 0.0f;
    // the profile's rank of each pair: of equally routed experts the better-ranked one comes in and the worse-ranked
    // one leaves first
    std::vector<int32_t> prof_rank((size_t) (g.n_layers * g.n_expert), (int32_t) profile.size());
    for (size_t i = 0; i < profile.size(); ++i) {
        const int64_t pl = profile[i].first, pe = profile[i].second;
        if (pl >= 0 && pl < g.n_layers && pe >= 0 && pe < g.n_expert && prof_rank[(size_t) (pl * g.n_expert + pe)] == (int32_t) profile.size())
            prof_rank[(size_t) (pl * g.n_expert + pe)] = (int32_t) i;
    }
    auto rank_swaps = [&](std::vector<Swap>& swaps, bool across) {
        swaps.clear();
        if (across) {
            // one ranking over every layer: a layer that routes more experts than its slots hold takes the slots of
            // the least-routed experts of any layer, so the slots follow the request's layers; with every layer's
            // residents to choose from, a missing expert is a candidate at a lower count
            struct Ent { float u; int32_t rank, layer, e; };
            std::vector<Ent> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    const int32_t rk = prof_rank[(size_t) (l * g.n_expert + e)];
                    if (r[e] >= 0) vict.push_back({u[e], rk, (int32_t) l, e});
                    else if (u[e] >= 1.5f) cand.push_back({u[e], rk, (int32_t) l, e});
                }
            }
            if (cand.empty() || vict.empty()) return;
            auto key = [](const Ent& x) { return std::make_tuple(x.rank, x.layer, x.e); };
            std::sort(cand.begin(), cand.end(), [&](const Ent& a, const Ent& b) {
                return a.u != b.u ? a.u > b.u : key(a) < key(b);
            });
            const size_t nc = std::min(cand.size(), vict.size());
            std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(), [&](const Ent& a, const Ent& b) {
                return a.u != b.u ? a.u < b.u : key(a) > key(b);
            });
            for (size_t i = 0; i < nc; ++i) {
                if (cand[i].u < vict[i].u + 1.5f) break;
                swaps.push_back({cand[i].u - vict[i].u, cand[i].layer, cand[i].e, vict[i].e, vict[i].layer});
            }
            std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            return;
        }
        std::vector<std::pair<float, int32_t>> cand, vict;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            cand.clear();
            vict.clear();
            const float* u = drive.d.usage.data() + l * g.n_expert;
            const int32_t* r = host_res.data() + l * g.n_expert;
            float margin = promote_margin;
            if (promote_load > 0.0f && peer.valid()) {
                float all = 0.0f, on = 0.0f;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    all += u[e];
                    if (r[e] < 0 && peer.has(l, e)) on += u[e];
                }
                if (all > 0.0f && on * 10.0f > promote_load * all) margin = 0.0f;
            }
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                if (r[e] < 0) {
                    const bool on_peer = peer.valid() && peer.has(l, e);
                    if (u[e] >= 2.0f && !(remote_opt && remote_opt->owns(l, e)))
                        cand.emplace_back(on_peer ? u[e] - margin : u[e], e);
                }
                else vict.emplace_back(u[e], e);
            }
            if (cand.empty() || vict.empty()) continue;
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
            const size_t nc = std::min(cand.size(), vict.size());
            std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < nc; ++i) {
                if (cand[i].first < vict[i].first + 1.5f) break;
                swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second, (int32_t) l});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    };
    auto adapt = [&]() -> bool {
        if (!pending.empty()) return true;   // the previous swaps are still in flight
        const auto at0 = Clock::now();
        const int64_t peer_sw0 = peer.valid() ? peer.swaps() : 0;
        size_t n_ranked = 0, n_kept = 0, n_promoted = 0;
        if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
        std::vector<Swap> swaps;
        rank_swaps(swaps, false);
        n_ranked = swaps.size();
        if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
        std::vector<int> staged_on;   // the devices whose adapt streams hold copies back (-1: CUDA0)
        const auto copy_back = [&](int32_t layer, int32_t slot, uint8_t* dst) -> bool {
            const int stn = multi_gpu ? stage_of(layer) : 0;
            GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
            const basalt::core::OnDevice on(gs ? gs->dev : -1);
            staged_on.push_back(gs ? gs->dev : -1);
            return cudaMemcpyAsync(dst, gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot),
                                   (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                                   cudaMemcpyDeviceToHost, gs ? gs->adapt_stream : adapt_stream) == cudaSuccess;
        };
        const auto wait_copies = [&]() -> bool {
            bool ok = true;
            for (const auto& st : stages)
                if (std::find(staged_on.begin(), staged_on.end(), st->dev) != staged_on.end()) {
                    const basalt::core::OnDevice on(st->dev);
                    ok = cudaStreamSynchronize(st->adapt_stream) == cudaSuccess && ok;
                }
            if (std::find(staged_on.begin(), staged_on.end(), -1) != staged_on.end())
                ok = cudaStreamSynchronize(adapt_stream) == cudaSuccess && ok;
            return ok;
        };
        if (!resident_spare_swaps(src, host_res, g.n_expert, swaps, copy_back, wait_copies, pending_spare)) return false;
        n_kept = swaps.size();
        const auto at1 = Clock::now();
        bool main_live = false;
        for (const Swap& s : swaps) {
            const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
            const int32_t slot = host_res[out];
            const uint8_t* b = srcp->blob(s.layer, s.in);
            const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
            GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
            const basalt::core::OnDevice on(gs ? gs->dev : -1);
            if (slot < 0 || b == nullptr ||
                cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                                (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                cudaMemcpyHostToDevice, gs ? gs->adapt_stream : adapt_stream) != cudaSuccess) {
                basalt::log::error("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                             (int) s.layer, (int) slot, cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            if (gs) gs->adapt_live = true;
            else main_live = true;
            host_res[out] = basalt::core::kNotResident;   // evicted now: the CPU computes it meanwhile
            srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
            pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            if (peer.valid() && peer.has(s.layer, s.in)) promoted.push_back({s.layer, s.in, s.out});
        }
        if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
        (void) main_live;
        for (auto& st : stages)
            if (st->adapt_live) {
                const basalt::core::OnDevice on(st->dev);
                cudaEventRecord(st->adapt_ev, st->adapt_stream);
            }
        // the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its decays: the
        // sum is proportional to the routing itself) - only with --expert-profile-save, else `heat` is empty
        for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
        if (peer.valid()) {                   // multi-GPU: the peer tiers take the next most-routed CPU misses
            std::vector<int32_t> r0 = host_res;
            for (const auto& [i, slot] : pending) r0[(size_t) i] = slot;   // swapped into the primary already
            std::string perr;
            if (!adapt_peers(r0, perr)) {
                basalt::log::error("basalt serve", "%s", perr.c_str());
                return false;
            }
        }
        for (float& v : drive.d.usage) v *= o.adapt_decay;
        return true;
    };
    // The resident RAM mode on one GPU, with or without the peer: a round's swaps move in three steps at window
    // boundaries, with no host copy and no blocking copy on the decode path.  (1) The evicted experts' slots are
    // copied by DMA into free places of the RAM copy (they stay resident meanwhile).  (2) Those places become
    // theirs, the CPU computes them, and the incoming experts are copied into the slots.  (3) The incoming experts
    // are resident and their RAM places free.  Step (2) waits for the eviction copies; at step (3) the next
    // window's gate/up launches wait for their own layer's incoming copies (the landed words below), so the loop
    // does not, and where an expert runs depends only on the window count, as with the plain rounds.
    const bool adapt_fast =
                            !multi_gpu && !remote_opt;
    // Its rounds come every --adapt-every windows, so the counts decay less per round than the plain default
    // (--adapt-decay when given, else 0.85): the routing counts stay large enough for the thresholds and the peer
    // margin.
    float fast_decay = o.adapt_decay_set ? o.adapt_decay : 0.85f;   // PROBE adapt sets it
    struct XSwap { int32_t layer, in, out, lout, slot; uint64_t off; bool xchg, promo, p2p; };
    const bool peer_excl = o.peer_exclusive && peer.valid() && src.complement_ready();
    // the rounds rank over all layers when no peer tier holds experts and every layer's blobs are one size (a slot
    // then takes any layer's expert)
    bool xq_across = !peer.valid() && xpeers.empty();
    for (int64_t l = 1; xq_across && l < g.n_layers; ++l)
        xq_across = basalt::kernels::cpu::expert_layout().blob_bytes(l) == basalt::kernels::cpu::expert_layout().blob_bytes(0);
    std::vector<XSwap> xq;
    int xq_stage = 0;   // 0 idle, 1 evictions copying to RAM, 2 incoming experts copying to their slots
    bool xq_saturated = false;   // the last round had at least kSatRatio times --adapt-swaps misses to swap
    // The landed words (Verifier::landed_words): a round's incoming copies are queued in layer order, each layer's
    // followed by a 4-byte copy of the round's number into the layer's word.  The copies' sources are a pinned
    // ring of rounds (every layer's word, then the round's number).
    constexpr int kXqPinRounds = 4;
    uint32_t xq_no = 0;
    bool xq_words = false;   // the round in flight writes the landed words
    uint32_t* xq_pin = nullptr;
    if (ver.landed_words() != nullptr &&
        cudaHostAlloc((void**) &xq_pin, (size_t) kXqPinRounds * (size_t) (g.n_layers + 1) * 4, cudaHostAllocPortable) !=
            cudaSuccess) {
        cudaGetLastError();
        xq_pin = nullptr;
    }
    cudaEvent_t p2p_ev = nullptr;   // the copies out of the peer's slots, which a promotion's reassign refills
    cudaEventCreateWithFlags(&p2p_ev, cudaEventDisableTiming);
    // a round made resident before its copies were done: the bookkeeping that waits for them (the next round or
    // a drain), and the RAM pages of its experts
    bool xq_unsettled = false;
    std::vector<std::pair<int32_t, int32_t>> xq_released;
    auto xq_settle = [&]() {
        if (!xq_unsettled) return;
        cudaEventSynchronize(adapt_ev);
        for (const auto& [l, e] : xq_released) srcp->release(l, e);
        xq_released.clear();
        res_upload();
        xq_unsettled = false;
    };
    auto xq_round = [&]() -> bool {
        xq_settle();
        const auto at0 = Clock::now();
        const int64_t peer_sw0 = peer.valid() ? peer.swaps() : 0;
        std::vector<Swap> swaps;
        rank_swaps(swaps, xq_across);
        const size_t n_ranked = swaps.size();
        // the base share goes first, then the peer takes its own from the rest, then a saturated pipeline's extra
        // swaps take what the peer left: a large round never takes the peer's candidates
        const size_t base = (size_t) o.adapt_swaps;
        const size_t cap = xq_saturated ? base * (size_t) kSatBoost : base;
        xq_saturated = n_ranked >= base * (size_t) kSatRatio;
        if (swaps.size() > cap) swaps.resize(cap);
        xq.clear();
        size_t n_noplace = 0;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        std::set<size_t> taken;   // the round's victims
        for (const Swap& s : swaps) taken.insert((size_t) s.lout * g.n_expert + s.out);
        auto plan = [&](Swap s) -> bool {
            // an expert only the peer holds is copied from its VRAM; it needs a RAM place for the one it replaces
            const bool p2p = peer_excl && peer.has(s.layer, s.in) && !src.has_resident(s.layer, s.in);
            if (p2p && src.has_resident(s.layer, s.out)) {
                // a lent slot's expert keeps its RAM place, which the incoming one would not have: the layer's
                // least-routed expert without one goes instead, if that is still a gain
                const float* u = drive.d.usage.data() + (size_t) s.layer * g.n_expert;
                int32_t best = -1;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    const size_t k = (size_t) s.layer * g.n_expert + e;
                    if (host_res[k] < 0 || src.has_resident(s.layer, e) || taken.count(k)) continue;
                    if (best < 0 || u[e] < u[best]) best = e;
                }
                if (best < 0 || u[s.in] - promote_margin < u[best] + 1.5f) return true;
                taken.insert((size_t) s.layer * g.n_expert + best);
                s.out = best;
            }
            const int32_t slot = host_res[(size_t) s.lout * g.n_expert + s.out];
            if (slot < 0) return true;
            const bool xchg = src.complement_ready() && (p2p || src.has_resident(s.layer, s.in)) &&
                              !src.has_resident(s.lout, s.out);
            uint64_t off = 0;
            if (xchg) {
                uint8_t* dst = src.spare_take(s.lout, off);
                if (dst == nullptr) { ++n_noplace; return true; }   // no free place: a later round
                if (cudaMemcpyAsync(dst, xcache.device_slot(slot), (size_t) lay.blob_bytes(s.lout),
                                    cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                    basalt::log::error("basalt serve", "an eviction copy failed (layer %d, slot %d): %s",
                                 (int) s.lout, (int) slot, cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
            }
            if (p2p) peer.hold(s.layer, s.in);
            xq.push_back({s.layer, s.in, s.out, s.lout, slot, off, xchg,
                          peer.valid() && peer.has(s.layer, s.in), p2p});
            return true;
        };
        const size_t n_base = std::min(swaps.size(), base);
        for (size_t i = 0; i < n_base; ++i)
            if (!plan(swaps[i])) return false;
        for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
        std::vector<size_t> peer_takes;
        if (peer.valid()) {   // the peer tiers take the next most-routed CPU misses (not the ones moving to CUDA0)
            std::vector<int32_t> r0 = host_res;
            for (const XSwap& x : xq) r0[(size_t) x.layer * g.n_expert + x.in] = x.slot;
            std::string perr;
            if (!adapt_peers(r0, perr)) {
                basalt::log::error("basalt serve", "%s", perr.c_str());
                return false;
            }
            if (swaps.size() > n_base) {
                peer_takes = peer.incoming();
                for (const auto& xpe : xpeers)
                    if (xpe->valid())
                        for (size_t i : xpe->incoming()) peer_takes.push_back(i);
            }
            std::sort(peer_takes.begin(), peer_takes.end());
        }
        for (size_t i = n_base; i < swaps.size(); ++i) {
            if (std::binary_search(peer_takes.begin(), peer_takes.end(),
                                   (size_t) swaps[i].layer * g.n_expert + swaps[i].in)) continue;
            if (!plan(swaps[i])) return false;
        }
        if (!xq.empty()) {
            cudaEventRecord(adapt_ev, adapt_stream);
            xq_stage = 1;
        }
        for (float& v : drive.d.usage) v *= fast_decay;
        return true;
    };
    // `wait`: step (3) waits for the copies even with the landed words (the batch windows, a drain)
    auto xq_advance = [&](bool wait) -> bool {
        if (xq_stage == 0) return true;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        if (xq_stage == 2) ver.flush_post_launch();
        const bool nowait = xq_stage == 2 && xq_words && !wait;
        if (!nowait) cudaEventSynchronize(adapt_ev);
        if (xq_stage == 1) {
            for (const XSwap& x : xq) {
                if (x.xchg && !src.spare_land(x.lout, x.out, x.off)) {
                    basalt::log::error("basalt serve", "an evicted expert has no RAM place (layer %d, expert %d)",
                                 (int) x.lout, (int) x.out);
                    return false;
                }
                host_res[(size_t) x.lout * g.n_expert + x.out] = basalt::core::kNotResident;
            }
            {   // the residency uploads go ahead of the incoming copies in the copy queue
                std::string perr;
                if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                res_upload();
                ver.dm_fence(adapt_stream);
            }
            bool deferred = false, p2p = false;
            for (const XSwap& x : xq) {   // stream order keeps each behind its slot's eviction copy
                std::string perr;
                if (x.p2p) {
                    if (!peer.copy_to(x.layer, x.in, xcache.device_slot(x.slot), adapt_stream, perr)) {
                        basalt::log::error("basalt serve", "%s", perr.c_str());
                        return false;
                    }
                    p2p = true;
                    continue;
                }
                deferred = true;
            }
            if (p2p) cudaEventRecord(p2p_ev, adapt_stream);
            const uint32_t* pw = nullptr;   // this round's landed words and number
            xq_words = xq_pin != nullptr && ver.landed_ok();
            if (xq_words) {
                ++xq_no;
                uint32_t* w = xq_pin + (size_t) (xq_no % kXqPinRounds) * (size_t) (g.n_layers + 1);
                // a layer without an incoming RAM copy is in now (its copies out of the peer are ahead), the others
                // once their own copies are
                for (int64_t l = 0; l < g.n_layers; ++l) w[l] = xq_no;
                for (const XSwap& x : xq)
                    if (!x.p2p) w[x.layer] = xq_no - 1;
                w[g.n_layers] = xq_no;
                if (cudaMemcpyAsync(ver.landed_words(), w, (size_t) g.n_layers * 4, cudaMemcpyHostToDevice, adapt_stream) !=
                    cudaSuccess) {
                    basalt::log::error("basalt serve", "the landed words' copy failed: %s", cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
                pw = w;
            }
            if (deferred) {   // the RAM copies are queued once the next window is launched, and run beside it
                ver.set_post_launch([&, pw] {
                    const auto& lay2 = basalt::kernels::cpu::expert_layout();
                    std::vector<uint32_t> ord;   // layer order: a layer's word goes up right after its own copies
                    for (uint32_t i = 0; i < (uint32_t) xq.size(); ++i)
                        if (!xq[i].p2p) ord.push_back(i);
                    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return xq[a].layer < xq[b].layer; });
                    for (size_t k = 0; k < ord.size(); ++k) {
                        const XSwap& x = xq[ord[k]];
                        const uint8_t* b = srcp->blob(x.layer, x.in);
                        if (b == nullptr || cudaMemcpyAsync(xcache.device_slot(x.slot), b, (size_t) lay2.blob_bytes(x.layer),
                                                            cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess)
                            basalt::log::warn("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                                         (int) x.layer, (int) x.slot, cudaGetErrorString(cudaGetLastError()));
                        if (pw != nullptr && (k + 1 == ord.size() || xq[ord[k + 1]].layer != x.layer) &&
                            cudaMemcpyAsync(ver.landed_words() + x.layer, pw + g.n_layers, 4, cudaMemcpyHostToDevice,
                                            adapt_stream) != cudaSuccess)
                            basalt::log::warn("basalt serve", "the landed word's copy failed (layer %d): %s", (int) x.layer,
                                         cudaGetErrorString(cudaGetLastError()));
                    }
                    cudaEventRecord(adapt_ev, adapt_stream);
                });
            } else {
                cudaEventRecord(adapt_ev, adapt_stream);
            }
            xq_stage = 2;
            return true;
        }
        // with the landed words the incoming experts are resident from the next window on while their copies land:
        // its gate/up launches wait for them, and what needs them done waits in xq_settle
        if (nowait && std::any_of(xq.begin(), xq.end(), [](const XSwap& x) { return x.p2p; }))
            cudaEventSynchronize(p2p_ev);   // a promotion's peer slot is refilled below
        for (const XSwap& x : xq) {
            host_res[(size_t) x.layer * g.n_expert + x.in] = x.slot;
            if (x.xchg && !x.p2p) src.spare_release(x.layer, x.in);
            if (nowait) xq_released.emplace_back(x.layer, x.in);
            else srcp->release(x.layer, x.in);
        }
        if (nowait) {
            std::string perr;
            if (!ver.landed_need(xq_no, perr)) {
                basalt::log::error("basalt serve", "%s", perr.c_str());
                return false;
            }
            xq_unsettled = true;
        } else {
            res_upload();
        }
        for (const XSwap& x : xq) {   // a promoted expert's peer slot takes the one CUDA0 gave up
            if (!x.promo) continue;
            std::string perr;
            if (!peer.reassign(x.layer, x.in, x.out, srcp->blob(x.layer, x.out), perr))
                basalt::log::warn("basalt serve", "%s", perr.c_str());
        }
        xq.clear();
        xq_stage = 0;
        return true;
    };
    auto xq_drain = [&]() -> bool {
        while (xq_stage != 0)
            if (!xq_advance(true)) return false;
        xq_settle();
        return true;
    };
    // write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).  A swap
    // still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
    auto save_profile = [&](const char* why) {
        if (heat.empty()) return;
        std::vector<uint8_t> resident(host_res.size(), 0);
        for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
        for (const auto& p : pending) resident[(size_t) p.first] = 1;
        std::string e;
        const auto ranked = basalt::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                               profile_loaded);
        if (basalt::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
            basalt::log::info("basalt serve", "expert profile saved to %s (%s)", o.expert_profile_save.c_str(),
                         why);
        else
            basalt::log::warn("basalt serve", "the expert profile was not saved: %s", e.c_str());
        profile_saved_at = Clock::now();
    };
    // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
    // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
    // GENT: the request text arrives as bytes and the reader thread turns it into ids, from the pack's tokenizer tables
    basalt::program::Tokenizer text_tokenizer;
    std::string text_tokenizer_note = "this pack has no qwen35 tokenizer tables";
    {
        std::vector<uint8_t> vj, mj, tj, cj;
        std::string terr;
        if (basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/merges.txt"), mj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/tokenizer.json"), cj, terr) &&
            std::string_view((const char*) cj.data(), cj.size()).find("\"pre\": \"qwen35\"") != std::string_view::npos) {
            if (text_tokenizer.load(std::string_view((const char*) vj.data(), vj.size()),
                                    std::string_view((const char*) mj.data(), mj.size()),
                                    std::string_view((const char*) tj.data(), tj.size()), terr))
                text_tokenizer_note.clear();
            else
                text_tokenizer_note = terr;
        }
    }
    std::mutex gent_mu;
    std::map<uint64_t, std::vector<int32_t>> gent_ids;   // tokenized prompts waiting for their request line, by sequence
    uint64_t gent_seq = 0;
    std::atomic<bool> stop_req{false};
    std::atomic<bool> stdin_lost{false};   // set once at EOF: the peer is gone, so no request keeps running
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    std::thread([&] {
        // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
        // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
        // std::exit) would hang in exit(), and the server would wait for it forever
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
                const std::ptrdiff_t n = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        // exactly n raw bytes: what the line reader already holds, then the pipe
        auto read_exact = [&](size_t n, std::string& out) -> bool {
            while (buf.size() < n) {
                const std::ptrdiff_t got = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                if (got < 0 && errno == EINTR) continue;
                if (got <= 0) return false;
                buf.append(chunk, (size_t) got);
            }
            out.assign(buf, 0, n);
            buf.erase(0, n);
            return true;
        };
        // GENT/SGENT/BGENT <...> <max_new> [k=v ...] <nbytes> [plain=a:b,...] + nbytes of UTF-8: tokenize here and hand
        // the main loop the matching GEN/SGEN/BGEN line, with "@<seq>" where the ids would be.  False: the input ends
        // inside the payload.
        auto take_text = [&](const std::string& head_line, std::string& line_out) -> bool {
            std::vector<std::string> tk;
            {
                std::istringstream is(head_line);
                for (std::string w; is >> w;) tk.push_back(w);
            }
            const size_t lead = tk[0] == "GENT" ? 2 : 3;   // the command, [the slot,] max_new
            std::string head = tk[0].substr(0, tk[0].size() - 1);
            long long nbytes = -1;
            std::string plain_text;
            for (size_t i = 1; i < tk.size(); ++i) {
                if (i < lead) head += " " + tk[i];
                else if (tk[i].rfind("plain=", 0) == 0) plain_text = tk[i].substr(6);
                else if (tk[i].find('=') != std::string::npos) head += " " + tk[i];
                else if (nbytes < 0) nbytes = std::atoll(tk[i].c_str());
            }
            if (tk.size() < lead || nbytes < 0 || nbytes > (1LL << 31)) {
                line_out = "ERR_GENT bad header: GENT <max_new> [k=v ...] <nbytes> [plain=a:b,...]";
                return true;
            }
            std::string payload;
            if (!read_exact((size_t) nbytes, payload)) { line_out = "ERR_GENT the prompt text was cut short"; return false; }
            if (!text_tokenizer_note.empty()) { line_out = "ERR_GENT " + text_tokenizer_note; return true; }
            std::vector<basalt::program::Tokenizer::Span> spans;
            for (size_t p = 0; p < plain_text.size();) {
                char* e1 = nullptr;
                const unsigned long long a = std::strtoull(plain_text.c_str() + p, &e1, 10);
                if (*e1 != ':') break;
                char* e2 = nullptr;
                const unsigned long long b = std::strtoull(e1 + 1, &e2, 10);
                spans.push_back({(size_t) a, (size_t) b});
                p = (size_t) (e2 - plain_text.c_str());
                if (p < plain_text.size() && plain_text[p] == ',') ++p;
            }
            std::vector<int32_t> tids;
            std::string terr;
            if (!text_tokenizer.encode(payload, true, spans, tids, terr)) { line_out = "ERR_GENT tokenizer: " + terr; return true; }
            std::lock_guard<std::mutex> lk(gent_mu);
            const uint64_t seq = ++gent_seq;
            gent_ids[seq] = std::move(tids);
            while (gent_ids.size() > 8) gent_ids.erase(gent_ids.begin());   // a request that never claimed its ids
            line_out = head + " @" + std::to_string(seq);
            return true;
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            if (l.rfind("GENT ", 0) == 0 || l.rfind("SGENT ", 0) == 0 || l.rfind("BGENT ", 0) == 0) {
                std::string rewritten;
                const bool framed = take_text(l, rewritten);
                l = rewritten;
                if (!framed) {   // the input ended inside the payload: report it
                    std::lock_guard<std::mutex> lk(in_mu);
                    in_lines.push_back(l);
                    in_cv.notify_one();
                    break;
                }
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_lines.push_back(l);
            in_cv.notify_one();
        }
        stdin_lost.store(true);
        stop_req.store(true);
        std::lock_guard<std::mutex> lk(in_mu);
        in_eof = true;
        in_cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) -> bool {
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };
    // the same, waiting at most `ms`: 1 a line, 0 none came, -1 the input ended
    auto next_line_within = [&](std::string& out, int ms) -> int {
        std::unique_lock<std::mutex> lk(in_mu);
        if (!in_cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return !in_lines.empty() || in_eof; }))
            return 0;
        if (in_lines.empty()) return -1;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return 1;
    };
    // a request also stops when the RAM guard judges the RAM critical; `ram_stop` keeps that it did, so the request
    // ends on an error instead of a cancel
    std::atomic<bool> ram_stop{false};
    auto halted = [&]() -> bool {
        if (stop_req.load() || stdin_lost.load()) return true;
        if (basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Critical) return false;
        ram_stop.store(true);
        return true;
    };
    sp.should_stop = [&] { return halted(); };
    {
        // what is left once everything is allocated: a GPU filled to the brim may not fail but page, and a page-in
        // while the verify graph spins on a host flag stalls the request for good
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t free_mib = (int64_t) (free_b >> 20);
        if (free_mib >= kVramStallMib) {
            basalt::log::info("basalt serve", "%lld MiB of VRAM free with everything loaded", (long long) free_mib);
        } else {
            basalt::log::warn("basalt", "only %lld MiB of VRAM free, requests may stall; close other "
                                 "programs that use the GPU", (long long) free_mib);
        }
    }
    // what the server's Monitor tab shows (the server skips unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // "Experts in VRAM" is every tier, not one card's.  The tiers are disjoint by construction
        // (remote_experts.cpp skips any pair a stage already claimed), so they add.
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        int64_t slots_all = slots_primary, mib_all = mib_primary;
        for (const auto& st : stages) {
            slots_all += (int64_t) st->cache.slots();
            mib_all += (int64_t) (st->cache.bytes() >> 20);
        }
        for (int r = 0; r < 3; ++r)
            if (o.expert_cache_remote[(size_t) r] > 0) {
                slots_all += remote_experts[(size_t) r].resident();
                mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
            }
        // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
        // bus ids of the cards this engine computes on, primary first: what a benchmark row records
        std::string quant_label, gpu_ids;
        {
            const basalt::core::Json* nm = model_file.meta().get("name");
            std::string nmv = nm != nullptr && nm->is_string() ? nm->s : std::string();
            quant_label = nmv.substr(nmv.rfind(' ') == std::string::npos ? 0 : nmv.rfind(' ') + 1);
            for (const char* pre : {"NVIDIA-", "UD-"})
                if (quant_label.rfind(pre, 0) == 0) quant_label.erase(0, std::strlen(pre));
            if (quant_label.empty()) quant_label = "unknown";
            for (int d : basalt::program::engine_gpu_order(o.peer_device, o.peer_devices,
                                                           multi_gpu ? split_devs : std::vector<int>())) {
                char bus[32] = {};
                if (cudaDeviceGetPCIBusId(bus, sizeof bus, d) == cudaSuccess)
                    gpu_ids += (gpu_ids.empty() ? "" : ",") + std::string(bus);
            }
        }
        std::printf("INFO logit_bias=%lld context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                    "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                    "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                    "conversation_cache_mib=%lld conversation_cache_slots=%d "
                    "conversation_cache_min_free_mib=%lld tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" BASALT_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context, o.kv.c_str(),
                    (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                     ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                    (long long) slots_all, (long long) mib_all,
                    (long long) slots_primary, (long long) mib_primary,
                    o.spec, o.mtp_max_t,
                    o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                    (long long) ((o.mmap_experts ? src.resident_bytes() : basalt::kernels::cpu::expert_layout().total) >> 20),
                    pool.workers(), o.pcie_frac,
                    (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                    (long long) o.conversation_cache_min_free_mib, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                    o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                   " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                   " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                   " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                    quant_label.c_str(), gpu_ids.c_str());
    }
    if (text_tokenizer_note.empty()) std::printf("INFO text_prompt=1\n");
    // a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on a flag nobody
    // will raise - end the engine with where it was, so the server starts it again instead of the GPU spinning
    // forever.  Before it does, it reports what every part was doing (stall_report), so one occurrence says where
    // the wait is.
    {
        const int limit = 60;   // one step (a prompt layer, a verify window) takes seconds
        if (limit > 0)
            std::thread([limit] {
                basalt::core::Progress& p = basalt::core::progress();
                uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                auto since = std::chrono::steady_clock::now();
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    const auto now = std::chrono::steady_clock::now();
                    const uint64_t b = p.beats.load();
                    if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                    if (now - since < std::chrono::seconds(limit)) continue;
                    basalt::log::error("basalt serve", "no progress for %d s during a request (%s) - stopping "
                                         "the engine so the server starts it again",
                                 limit, stage_text().c_str());
                    stall_report(basalt::log::stream(), p.ticks.load() - ticks_at);
                    basalt::core::release_gpu_waits(basalt::log::stream());   // no spin kernel outlives the process
                    std::abort();
                }
            }).detach();
    }
    // the load's own buffers are gone: RAM still free keeps the next-hottest experts the first sizing left out
    if (srcp == &src && src.complement_ready() && o.resident_budget == 0) {
        int64_t n_top = 0;
        // what the engine still takes once serving (see serve_growth_bytes) and the file cache's floor while
        // experts are read from the SSD (none when every expert has a tier)
        const uint64_t later = serve_growth_bytes(o, g) +
                               (ram_floor_bytes - std::min(ram_floor_bytes, basalt::platform::ram_margin()));
        const uint64_t add = src.top_up(stages.empty() ? profile : profile_all, [&](int32_t l, int32_t e) {
            return host_res[(size_t) l * g.n_expert + (size_t) e] >= 0 || (peer.valid() && peer.has(l, e));
        }, o.resident_pin, n_top, later);
        if (add > 0) {
            basalt::platform::watch_resident_ram(true, src.resident_bytes() + add);
            basalt::log::info("basalt generate", "RAM top-up: +%.2f GiB, %lld experts (total %.2f GiB)",
                         (double) add / 1073741824.0, (long long) n_top,
                         (double) (src.resident_bytes() + add) / 1073741824.0);
        }
    }
    report_expert_tiers();
    basalt::platform::stop_memory_watchdog();
    // from here the RAM guard watches: below its floor the engine sheds and new requests wait, and a running
    // request stops only once the RAM is critical
    const uint64_t guard_floor = basalt::platform::ram_guard_floor(basalt::platform::ram_margin(), tier_file_bytes, prompt_file_bytes);
    basalt::platform::start_ram_guard(guard_floor);
    disk_floor = guard_floor + (1ull << 30);   // a disk-tier capture or read stays a GiB above the guard
    // what a stop's error compares against, to tell the engine's own growth from other programs'
    const uint64_t ready_avail = basalt::platform::available_memory();
    const uint64_t ready_rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
    // once per drop below the guard's floor, on the serve thread between steps: the parked conversations leave RAM
    // (to the disk tier when it is on and holds no copy of them yet: queued for its writer, whose writes free them;
    // the others are dropped), the checkpoint blocks kept for reuse are freed and the heap's free pages return to
    // the OS.  The checkpoints stay: a live request's rewind and the deferred copy in flight use them.
    uint64_t shed_drop = 0;
    auto shed_if_low = [&]() {
        if (basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok) return;
        const uint64_t drop = basalt::platform::ram_guard_drops();
        if (drop == shed_drop) return;
        shed_drop = drop;
        const size_t parked = conversations.size();
        const size_t freed = conversations.clear();
        size_t to_disk = 0;
        for (auto& sc : conversations.take_spilled())
            if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
                disk.put(std::move(sc), kDiskQueueCap))
                ++to_disk;
        if (freed > 0)
            basalt::log::info("basalt serve", "RAM guard: %zu parked conversations (%zu MiB) leave RAM, %zu of them "
                              "to disk", parked, freed >> 20, to_disk);
        if (const size_t pooled = basalt::core::CheckpointBytes::allocator_type::release_pooled(); pooled > 0)
            basalt::log::info("basalt serve", "RAM guard: freed %zu MiB of checkpoint blocks kept for reuse",
                              pooled >> 20);
        const uint64_t before = basalt::platform::available_memory();
        basalt::platform::release_free_heap();
        const uint64_t after = basalt::platform::available_memory();
        basalt::log::info("basalt serve", "RAM guard: free heap returned to the OS, %lld MiB free now (%+lld MiB)",
                          (long long) (after >> 20), ((long long) after - (long long) before) / (1ll << 20));
    };
    // a VRAM shrink's RAM homes leave the guard's floor above the margin and the top-up's own slack free
    src.set_homes_reserve(guard_floor - std::min(guard_floor, basalt::platform::ram_margin()) + (256ull << 20));
    if (const uint64_t avail = basalt::platform::available_memory(); avail > 0)
        basalt::log::info("basalt serve", "RAM %.2f GiB free, guard %.2f GiB, %.2f GiB kept for serving",
                     (double) avail / 1073741824.0, (double) guard_floor / 1073741824.0,
                     (double) serve_growth_bytes(o, g) / 1073741824.0);
    // a view of the solo session carrying slot b's state: what the request path and the solo verifier work on for a
    // request that runs in that slot
    auto make_view = [&](basalt::core::SessionState& v, int b) {
        const basalt::core::SessionState& t = *bslot_ss[0][(size_t) b];
        v = ss;
        v.gdn_state = t.gdn_state;
        v.qsa_states = t.qsa_states;
        v.ple_hist = t.ple_hist;
        v.ple.hist = t.ple_hist;
        v.ple_prev[0] = t.ple_prev[0];
        v.ple_prev[1] = t.ple_prev[1];
        v.ple_token = t.ple_token;
        v.ple.token = &v.ple_token;
        v.ple.prev = v.ple_prev;
    };
    // every window width's graph is captured before READY, not by the first request that needs it
    if (o.serve && stages.empty() && !ver.warm_graphs(err)) {
        basalt::log::error("basalt serve", "capturing the verify windows failed: %s", err.c_str());
        return 1;
    }
    // the solo windows of every slot session, so a lone request in any slot runs the solo path in place
    if (o.serve && slot0_is_solo && o.batch > 1) {
        const auto tw = Clock::now();
        size_t free0 = 0, total0 = 0;
        cudaMemGetInfo(&free0, &total0);
        for (int b = 1; b < o.batch; ++b) {
            make_view(rviews[(size_t) b], b);
            if (!ver.warm_session(&rviews[(size_t) b], err)) {
                basalt::log::error("basalt serve", "capturing slot %d's solo windows failed: %s", b, err.c_str());
                return 1;
            }
        }
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total0);
        basalt::log::info("basalt serve", "solo windows of %d more slot sessions captured in %.0f ms (%.1f MiB of VRAM)",
                     o.batch - 1, std::chrono::duration<double, std::milli>(Clock::now() - tw).count(),
                     (double) ((int64_t) free0 - (int64_t) free1) / 1048576.0);
    }
    const int S = o.spec;
    const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
    if (mtp_on && (pipe_want || S_mtp < S)) mtp.set_max_drafts((pipe_on ? basalt::core::kMtpChainMax : S_mtp) - 1);
    if (mtp_on && !mtp.prewarm(err)) {
        basalt::log::error("basalt serve", "capturing the draft graphs failed: %s", err.c_str());
        return 1;
    }
    ple_table.rearm_keepalive();   // a long load must not lapse the SSD keep-alive window before the first request
    // the tiers' placement at READY, which PROBE adapt puts back
    const std::vector<int32_t> res_at_start = host_res;
    const std::vector<int32_t> peer_at_start = peer.valid() ? peer.residency() : std::vector<int32_t>();
    const std::vector<int32_t> prof_rank_start = prof_rank;
    std::printf("INFO file_experts_bytes=%llu\n", (unsigned long long) tier_file_bytes);   // experts only the file holds
    std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
    std::fflush(stdout);
    std::string line;
    int64_t rounds = 0;
    basalt::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
    basalt::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
    // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
    // or more basalt-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
    // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
    // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
    constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
    bool mrope_identity = true;
    std::vector<float> img_rows;
    std::vector<const float*> row_ptr;
    // ---- VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache until that
    // much VRAM is free for other programs, or grows it back towards its full size when more than that is free.
    // Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back the cache's
    // LAST segments: the experts in their slots become CPU misses, as any expert outside the cache is (no output
    // changes beyond what a smaller cache decodes; in the resident RAM mode each is first copied to a RAM home),
    // the prompt path's loan moves down to the end of what is left
    // (a smaller chunk when it no longer fits).  A grow maps them again and puts each slot's expert back (or, when
    // the adaptive tier already brought that one back, the most-routed missing expert of the same layer); the
    // adaptive tier carries on from there.
    std::vector<std::pair<int32_t, int32_t>> vram_evicted;   // (residency index, slot) the shrinks took
    const int64_t chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache
    // the prompt path's loan with `live` slots: its chunk as large as fits, and the first slot it covers
    auto loan_plan = [&](int64_t live, int64_t& chunk) -> int32_t {
        PfPart& p = pf_parts[0];
        auto fits = [&](int64_t c) {
            const int64_t k = part_slots(p, c);
            return k > 0 && k + 128 <= live && !(o.prefill_auto && k * 100 > auto_lend_pct * live);
        };
        // auto: the largest chunk on the start-up's 256-token grid, so a few slots lent away cost a few tokens
        // of chunk, not half of it
        int64_t c = chunk_full;
        if (o.prefill_auto && !fits(c)) {
            c = engine_local::biggest_chunk(chunk_full, fits);
            if (c == 0) c = 256;
        }
        while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
        chunk = c;
        return (int32_t) std::max<int64_t>(0, live - part_slots(p, c));
    };
    auto relend = [&]() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
        if (pf_parts.empty() || pf_parts[0].first < 0) return;
        PfPart& p = pf_parts[0];
        int64_t c = 0;
        p.first = loan_plan(xcache.slots(), c);
        o.prefill_chunk = c;
        p.first_now = -1;   // laid out again at the next loan
    };
    // The resident RAM mode holds only the experts no slot holds, so an expert in a slot about to go has no RAM
    // copy: it is copied from its slot into a RAM home first and read from there like the compact copy's.  Refused
    // (nothing copied) when the RAM above the floor cannot take them.  `live`: the slots that stay.
    auto home_given_up = [&](int64_t live, std::string& e) -> bool {
        if (!src.complement_ready()) return true;
        // the slots the loan will overwrite keep a RAM copy too, so a prompt never costs a file read
        int64_t from = live;
        if (!pf_parts.empty() && pf_parts[0].first >= 0) {
            int64_t c = 0;
            from = std::min<int64_t>(live, loan_plan(live, c));
        }
        std::vector<std::pair<int32_t, int32_t>> give;
        std::vector<int32_t> slots;
        for (size_t i = 0; i < host_res.size(); ++i) {
            const int64_t layer = (int64_t) i / g.n_expert, ex = (int64_t) i % g.n_expert;
            if (host_res[i] >= from && !src.has_resident(layer, ex)) {
                give.emplace_back((int32_t) layer, (int32_t) ex);
                slots.push_back(host_res[i]);
            }
        }
        if (give.empty()) return true;
        std::vector<uint8_t*> dst;
        if (!src.homes_take(give, e, dst)) return false;
        for (size_t k = 0; k < give.size(); ++k) {
            const size_t bytes = (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(give[k].first);
            if (cudaMemcpyAsync(dst[k], xcache.device_slot(slots[k]), bytes, cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
                e = std::string("copying an expert to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
                (void) cudaDeviceSynchronize();   // queued copies stop writing before their regions are freed
                (void) cudaGetLastError();
                src.homes_abort();
                return false;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            e = std::string("copying the experts to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
            (void) cudaDeviceSynchronize();
            (void) cudaGetLastError();
            src.homes_abort();
            return false;
        }
        src.homes_commit();
        basalt::log::info("basalt serve", "VRAM: %zu experts moved from the cache to RAM (%.0f MiB held)", give.size(),
                     (double) src.homes_bytes() / 1048576.0);
        return true;
    };
    // `for_sleep` (SLEEP and WAKE): the cache goes to its smallest size and comes back to the same placement, with no
    // request between the two, so neither --vram-elastic (any segmented cache does) nor a peer card's tier stands in
    // the way
    auto vram_command = [&](const std::string& cmd, std::string& e, bool for_sleep,
                            std::string* reply = nullptr) -> bool {
        // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
        char* end = nullptr;
        const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
        const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
        if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
        if (reserve < kVramStallMib && !for_sleep) {
            e = "VRAM: a reserve below " + std::to_string(kVramStallMib) + " MiB lets requests stall; the smallest "
                "safe value is " + std::to_string(kVramStallMib);
            return false;
        }
        if ((!o.vram_elastic && !for_sleep) || !xcache.segmented()) {
            e = for_sleep ? "SLEEP needs a segmented expert cache (an elastic KV cache or --vram-elastic)"
                          : "VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)";
            return false;
        }
        if (peer.valid() && !for_sleep) { e = "VRAM: not with a second GPU as expert tier"; return false; }
        for (const PfPart& p : pf_parts)
            if (!p.lent.empty()) { e = "VRAM: a prompt's loan is still out"; return false; }
        if (!ver.wait_commit(e)) return false;
        apply_pending(true);                     // the adaptive tier's swaps in flight land first
        if (!xq_drain()) { e = "VRAM: an adaptive swap failed"; return false; }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            e = std::string("VRAM: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const auto t0 = Clock::now();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
        const int64_t before = xcache.slots();
        // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
        const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
            128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
        std::string note;
        if ((int64_t) free_b < want_free) {
            const int64_t floor_b = xcache.bytes_of(floor_slots);
            int64_t keep = mapped - (want_free - (int64_t) free_b);
            if (keep < floor_b) {
                keep = floor_b;
                note = " (the cache keeps its smallest size: the prompt path's buffers)";
            }
            if (!home_given_up(xcache.slots_after_shrink(keep), e)) { e = "VRAM: " + e + " (nothing changed)"; return false; }
            // the homes are committed by now: whatever the shrink left or took is followed from the cache itself
            const bool shrunk = xcache.shrink(keep, e);
            const int64_t live = xcache.slots();
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] >= live) {
                    vram_evicted.emplace_back((int32_t) i, host_res[i]);
                    host_res[i] = basalt::core::kNotResident;
                }
            res_upload();
            if (!shrunk) return false;
        } else if (mapped < xcache.full_bytes()) {
            std::string gerr;
            if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
            const int64_t live = xcache.slots();
            std::string ferr;
            std::vector<std::pair<int32_t, int32_t>> keep_out;
            struct Fill { int32_t slot; const uint8_t* b; int64_t bytes; };
            std::vector<Fill> fills;
            for (const auto& [i, slot] : vram_evicted) {
                if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                const int64_t layer = i / g.n_expert;
                int64_t pick = i;
                if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                    pick = -1;
                    float best = -1.0f;
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        const size_t j = (size_t) (layer * g.n_expert + ex);
                        if (host_res[j] >= 0 || (src.complement_ready() && !src.has_resident(layer, ex))) continue;
                        const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                        if (u > best) { best = u; pick = (int64_t) j; }
                    }
                }
                if (pick < 0) continue;
                const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
                if (b == nullptr) {
                    e = "VRAM: refilling the cache failed: an expert has no RAM copy";
                    return false;
                }
                fills.push_back({slot, b, (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer)});
                host_res[(size_t) pick] = slot;
            }
            // after a SLEEP the RAM copy is registered again in address order (WAKE): the copies follow it
            const bool repinning = src.complement_unpinned();
            if (repinning)
                std::sort(fills.begin(), fills.end(),
                          [](const Fill& x, const Fill& y) { return std::less<const uint8_t*>()(x.b, y.b); });
            for (const Fill& f : fills) {
                if (repinning && !src.repin_wait(f.b, (uint64_t) f.bytes)) {
                    e = "VRAM: the RAM copy could not be registered again";
                    return false;
                }
                if (!xcache.fill_slot_queued(f.slot, f.b, ferr, f.bytes)) {
                    e = "VRAM: refilling the cache failed: " + ferr;
                    return false;
                }
            }
            vram_evicted.swap(keep_out);
            if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
            res_upload();
            relend();
            // the slots hold the experts again: every expert in a slot below the loan has its RAM home freed (the
            // copy is the slots' now: those given up, those the loan range kept, and those the adaptive tier moved
            // back).  The loan's own slots keep theirs: a prompt overwrites them and reads those experts from RAM.
            if (src.complement_ready()) {
                const int64_t loan_from = pf_parts.empty() || pf_parts[0].first < 0 ? INT64_MAX : pf_parts[0].first;
                size_t released = 0;
                for (size_t j = 0; j < host_res.size(); ++j)
                    if (host_res[j] >= 0 && host_res[j] < loan_from &&
                        src.home_release((int64_t) j / g.n_expert, (int64_t) j % g.n_expert))
                        ++released;
                const uint64_t freed = src.homes_reclaim() + src.homes_compact();
                basalt::log::info("basalt serve", "VRAM: %zu RAM homes released, %.0f MiB freed, %.0f MiB kept for the "
                                  "prompt loan (slots from %lld)", released, (double) freed / 1048576.0,
                                  (double) src.homes_bytes() / 1048576.0, (long long) std::min<int64_t>(loan_from, live));
            }
        }
        relend();
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::info("basalt serve", "VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                             "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s",
                     reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                     (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                     (long long) (free_b >> 20), (long long) o.prefill_chunk,
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
        char line[400];
        std::snprintf(line, sizeof line, "VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld "
                      "expert_cache_mib=%lld expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld "
                      "ram_homes_mib=%lld\n", reserve, (long long) xcache.slots(), (long long) xcache.full_slots(),
                      (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                      (long long) (free_b >> 20), (long long) o.prefill_chunk, (long long) (src.homes_bytes() >> 20));
        if (reply != nullptr) *reply = line;
        else std::fputs(line, stdout);
        return true;
    };
    // ---- the elastic KV (kKvElasticFloor): a session's K/V pools back their cells on demand, out of the expert
    // cache's tail.  `kv_fit` backs `cells` of a session before anything writes them: the work in flight lands,
    // the cache gives back its last segments (their experts become CPU misses, read from their RAM copies), and the
    // pools map that VRAM, zeroed.  `kv_release` gives a session's cells past `cells` back and the cache maps its
    // tail again, each returned slot refilled from RAM.  Both run between windows, never inside one.
    // a request's room past its prompt, and each later step (batch slots: a smaller step, as their base)
    const int64_t kKvGrowCells = o.batch > 0 ? 4096 : 16384;
    constexpr int64_t kKvWindowCells = 2 * basalt::kernels::kVerifyMaxT + 64;   // a decoding session's next window
    int64_t kv_moves = 0;
    // the expert cache never shrinks below the slots the prompt path's smallest loan needs (a 256-token chunk, and
    // the 128 slots a loan leaves), so the next prompt can always be read
    auto cache_floor_slots = [&]() -> int64_t {
        if (pf_parts.empty() || pf_parts[0].first < 0) return 0;
        const int64_t k = part_slots(pf_parts[0], 256);
        int64_t f = k + 128;
        if (o.prefill_auto) f = std::max<int64_t>(f, (k * 100 + auto_lend_pct - 1) / auto_lend_pct);
        return std::min<int64_t>(f, xcache.full_slots());
    };
    bool kv_short = false;   // the last kv_fit refused: the cells would take the cache below its floor
    bool kv_keep_routed = false;   // an admission while slots decode: a shrink keeps their routed experts
    // The experts the decoding slots routed lately leave CUDA0's slots from `first` on, which a prompt's loan or the
    // KV is about to take: each trades places with a less-routed expert of its layer below `first` that has a RAM
    // copy (a slot keeps its layer's size; the experts from `first` on come back from RAM).  One blob copy per
    // trade; the table is uploaded by the caller.  `why` names the taker for the log.
    auto move_routed_below = [&](int64_t first, const char* why, std::string& e) -> bool {
        if (drive.d.usage.empty() || srcp != &src || !src.complement_ready() || multi_gpu || first <= 0 ||
            first >= xcache.slots())
            return true;
        const auto t0 = Clock::now();
        const auto& lay = basalt::kernels::cpu::expert_layout();
        int64_t moved = 0, routed = 0;
        std::vector<int32_t> hot, cold;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const float* u = drive.d.usage.data() + (size_t) (l * g.n_expert);
            int32_t* r = host_res.data() + (size_t) (l * g.n_expert);
            hot.clear();
            cold.clear();
            for (int32_t ex = 0; ex < (int32_t) g.n_expert; ++ex) {
                if (r[ex] >= first) {
                    if (u[ex] > 0.0f) hot.push_back(ex);
                } else if (r[ex] >= 0 && src.has_resident(l, ex)) {
                    cold.push_back(ex);
                }
            }
            routed += (int64_t) hot.size();
            if (hot.empty() || cold.empty()) continue;
            std::sort(hot.begin(), hot.end(), [&](int32_t a, int32_t b) { return u[a] > u[b]; });
            std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) {
                return u[a] != u[b] ? u[a] < u[b] : r[a] > r[b];   // then the profile's coldest
            });
            for (size_t k = 0; k < hot.size() && k < cold.size() && u[hot[k]] > u[cold[k]]; ++k) {
                const int32_t sh = r[hot[k]], sc = r[cold[k]];
                if (cudaMemcpyAsync(xcache.device_slot(sc), xcache.device_slot(sh), (size_t) lay.blob_bytes(l),
                                    cudaMemcpyDeviceToDevice, adapt_stream) != cudaSuccess) {
                    e = std::string("moving an expert out of the cache's tail: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                r[hot[k]] = sc;
                r[cold[k]] = sh;
                ++moved;
            }
        }
        if (moved > 0 && cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
            e = std::string("moving experts out of the cache's tail: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        basalt::log::info("basalt batch", "%s: %lld of the %lld routed experts in the cache's last %lld slots moved "
                             "below them (%.1f ms)", why, (long long) moved, (long long) routed,
                     (long long) (xcache.slots() - first),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        return true;
    };
    auto kv_drain = [&](std::string& e) -> bool {
        if (!ver.wait_commit(e)) return false;
        apply_pending(true);
        if (!xq_drain()) { e = "elastic KV: an adaptive swap failed"; return false; }
        for (const PfPart& p : pf_parts)
            if (!p.lent.empty()) { e = "elastic KV: a prompt's loan is still out"; return false; }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            e = std::string("elastic KV: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        return true;
    };
    // maps the cache's tail again as far as the free VRAM above the reserve allows, and refills the slots it gets
    // back: each with the expert it held, or (when that one is resident again, or has no RAM copy any more) the
    // layer's most-routed expert that no GPU holds and RAM does
    auto cache_regrow = [&](std::string& e) -> bool {
        if (!xcache.segmented() || xcache.mapped_bytes() >= xcache.full_bytes()) return true;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t room = (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20);
        if (room <= 0) return true;
        std::string gerr;
        (void) xcache.grow(xcache.mapped_bytes() + room, gerr);   // what it could not map stays with the driver
        const int64_t live = xcache.slots();
        auto usable = [&](int64_t j) {
            const int64_t l = j / g.n_expert, ex = j % g.n_expert;
            return host_res[(size_t) j] < 0 && (!src.complement_ready() || src.has_resident(l, ex)) &&
                   !(peer.valid() && peer.has(l, ex));
        };
        std::vector<std::pair<int32_t, int32_t>> keep_out;
        std::string ferr;
        for (const auto& [i, slot] : vram_evicted) {
            if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
            const int64_t layer = i / g.n_expert;
            int64_t pick = usable(i) ? i : -1;
            if (pick < 0) {
                float best = -1.0f;
                for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                    const int64_t j = layer * g.n_expert + ex;
                    if (!usable(j)) continue;
                    const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) j];
                    if (u > best) { best = u; pick = j; }
                }
            }
            if (pick < 0) continue;
            const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
            if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                    (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer))) {
                e = "elastic KV: refilling the cache failed: " + ferr;
                return false;
            }
            host_res[(size_t) pick] = slot;
        }
        vram_evicted.swap(keep_out);
        if (!xcache.sync_queued(ferr)) { e = "elastic KV: " + ferr; return false; }
        res_upload();
        relend();
        return true;
    };
    // `shrink` false: only VRAM that is free now (a prompt read is running beside the windows, its loan in the
    // cache's last slots) - false with `e` empty when that is not enough
    // `min_cells`: when `cells` would take the cache below its floor, as many as leave it, down to these
    auto kv_fit = [&](basalt::core::SessionState& s, int64_t cells, std::string& e, bool shrink = true,
                      int64_t min_cells = -1) -> bool {
        kv_short = false;
        if (kv_floor <= 0) return true;
        cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
        int64_t add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
        if (add <= 0) return true;
        const auto t0 = Clock::now();
        if (shrink && !kv_drain(e)) return false;
        const int64_t before = xcache.slots();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t room = std::max<int64_t>(0, (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20));
        if (!shrink && add > room) { e.clear(); return false; }
        if (add > room && xcache.segmented()) {
            const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
            auto keep_for = [&](int64_t a) {
                return std::max<int64_t>(0, (xcache.mapped_bytes() - (a - room)) / gran * gran);
            };
            const int64_t floor = cache_floor_slots();
            auto leaves_floor = [&](int64_t c) {
                const int64_t a = (int64_t) basalt::core::session_kv_grow_bytes(s, c);
                return a <= room || xcache.slots_after_shrink(keep_for(a)) >= floor;
            };
            if (!leaves_floor(cells)) {
                // the most cells on the 256-cell grid that leave it, down to `min_cells`
                int64_t best = -1;
                if (min_cells > 0 && min_cells < cells && leaves_floor(min_cells)) {
                    best = min_cells;
                    for (int64_t lo = min_cells / 256 + 1, hi = cells / 256; lo <= hi;) {
                        const int64_t mid = lo + (hi - lo) / 2;
                        if (mid * 256 > min_cells && leaves_floor(mid * 256)) { best = mid * 256; lo = mid + 1; }
                        else hi = mid - 1;
                    }
                }
                if (best < 0) {
                    kv_short = true;
                    const int64_t want = min_cells > 0 && min_cells < cells ? min_cells : cells;
                    e = "elastic KV: " + std::to_string(want) + " cells of KV (+" +
                        std::to_string((int64_t) basalt::core::session_kv_grow_bytes(s, want) >> 20) +
                        " MiB) would take the expert cache below the " + std::to_string(floor) +
                        " slots the next prompt chunk needs (" + std::to_string(xcache.slots()) + " now, " +
                        std::to_string(room >> 20) + " MiB free above the reserve)";
                    return false;
                }
                cells = best;
                add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
            }
        }
        if (add > room && xcache.segmented()) {
            // whole tail segments: everything from the first one at or past `keep` goes
            const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
            const int64_t keep = std::max<int64_t>(0, (xcache.mapped_bytes() - (add - room)) / gran * gran);
            if (kv_keep_routed && !move_routed_below(xcache.slots_after_shrink(keep), "elastic KV", e)) return false;
            if (!xcache.shrink(keep, e)) return false;
            const int64_t live = xcache.slots();
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] >= live) {
                    vram_evicted.emplace_back((int32_t) i, host_res[i]);
                    host_res[i] = basalt::core::kNotResident;
                }
            res_upload();
            relend();
        }
        if (!basalt::core::session_kv_map(s, cells, e)) { e = "elastic KV: " + e; return false; }
        ++kv_moves;
        basalt::log::info("basalt serve", "elastic KV: %lld cells backed (+%.0f MiB), the expert cache %lld -> %lld "
                             "slots, in %.1f ms", (long long) basalt::core::session_kv_cells(s),
                     (double) add / 1048576.0, (long long) before, (long long) xcache.slots(),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        return true;
    };
    auto kv_release = [&](basalt::core::SessionState& s, int64_t cells, std::string& e) -> bool {
        if (kv_floor <= 0) return true;
        cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
        if (basalt::core::session_kv_bytes_for(s, basalt::core::session_kv_cells(s)) <=
            basalt::core::session_kv_bytes_for(s, cells)) return true;
        const auto t0 = Clock::now();
        if (!kv_drain(e)) return false;
        const int64_t before = xcache.slots(), had = basalt::core::session_kv_cells(s);
        if (!basalt::core::session_kv_unmap(s, cells, e)) { e = "elastic KV: " + e; return false; }
        if (!cache_regrow(e)) return false;
        ++kv_moves;
        basalt::log::info("basalt serve", "elastic KV: %lld -> %lld cells backed, the expert cache %lld -> %lld "
                             "slots, in %.1f ms", (long long) had, (long long) basalt::core::session_kv_cells(s),
                     (long long) before, (long long) xcache.slots(),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        return true;
    };
    // ---- --batch: the slots of the batch windows
    struct BSlot {
        bool active = false, stop = false;
        int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
        int64_t p = 0;                 ///< its position
        int64_t produced = 0, max_new = 0;
        Clock::time_point t0;
        // the slot's own conversation cache: the tokens its sessions hold (the prompt, then every token a window
        // fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from it
        // instead of reading its history again (see `slot_source` in the request path).
        std::vector<int32_t> ids;
        bool cached = false;           ///< idle, and its sessions still hold `ids`
        bool cvec = true;              ///< the control vector setting `ids` were read with
        std::vector<ImgKey> imgs;      ///< the pictures in `ids`: a later request reuses them only with the same ones
        // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's header): a
        // client's next turn renders the history again WITHOUT this reply's thinking, so it matches the slot's
        // tokens only up to there - the slot's K/V up to it plus this state continue from it
        std::vector<ConvCheckpoint> checks;
        // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
        // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0;
        // `chunk`: the chunk of a read beside decoding slots, whose rest is read beside them in it, else 0)
        bool partial = false, partial_from0 = false;
        int64_t partial_chunk = 0;
        // BSCRIPT (a test drive of the multi-token segments): the tokens this slot's request is known to produce
        // (script[0] is its first token); the windows draft from it, some drafts wrong, and keep what matches
        std::vector<int32_t> script;
        std::mt19937_64 rng;
        // the request's grammar (the batch windows' form of the solo loop's `gguard`): the window's kept picks are
        // scanned, the first refused one is resampled under the mask, and the segment ends there
        std::unique_ptr<basalt::program::grammar::Guard> guard;
        // the MTP drafter's side of the slot (the draft K/V is the drafter's store of the same index): the drafts and
        // their probabilities for the slot's next window, the request's draft-length rule state, its prompt-lookup
        // history and last proposal
        int32_t dr[16] = {};
        float dprob[16] = {};
        int n_dr = 0;
        basalt::core::DraftCumAdapt cum;
        std::unique_ptr<basalt::spec::SuffixDrafter> sfx;
        int32_t sbuf[16] = {};
        int sk = 0, smatch = 0;
        bool from_sfx = false;
        double acc = 0.6;          ///< drafts accepted per offered, smoothed: who gets the free rows
        int64_t since_draft = 0;   ///< windows since the slot last drafted
        int64_t windows = 0;
        int64_t drafts_accepted = 0, drafts_offered = 0;   ///< the request's draft tokens, for BDONE
    };
    std::vector<BSlot> bs((size_t) std::max(o.batch, 0));
    // the real MTP drafts in the batch windows (a draft K/V per slot); the scripted test drive keeps its own
    const bool batch_spec = mtp_on && mtp.main_store() > 0 && o.batch > 0;
    const int S_all = S;
    basalt::spec::DraftPolicy bpolicy(S);   // the lookup-or-MTP choice, costs per row
    double bt_draft = 0, bt_draft_last = 0;
    int64_t bt_slotwin = 0, bt_dr_off = 0, bt_dr_acc = 0;
    std::vector<std::pair<std::vector<int32_t>, uint64_t>> bscript((size_t) std::max(o.batch, 0));
    const bool bdigest = basalt::core::exp_flag("concur5_digest", 0) != 0;
    // concur9_logit_hash: a hash of the head logits of every accepted row, on stderr (solo and batch alike)
    const bool logit_hash = basalt::core::exp_flag("concur9_logit_hash", 0) != 0;
    std::vector<float> lh_buf;
    auto row_hash = [&](int t) -> uint64_t {
        lh_buf.resize((size_t) n_vocab);
        if (!ver.copy_logits(t, lh_buf.data())) return 0;
        uint64_t h = 1469598103934665603ull;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(lh_buf.data());
        for (size_t i = 0; i < lh_buf.size(); ++i) { h ^= w[i]; h *= 1099511628211ull; }
        return h;
    };
    // adaptive tier rounds in the batch windows (routing counts from every row of every slot)
    const bool batch_adapt = adapt_fast && !drive.d.usage.empty();
    int64_t ba_win = 0, ba_swaps = 0, ba_rounds = 0, ba_swaps_l = 0, ba_rounds_l = 0, ba_h_l = 0, ba_m_l = 0;
    double ba_ms = 0;
    JobRunner ba_thr;       // queues the adaptive copies beside the windows
    bool ba_job = false;
    // timing of the batch windows since the slots were last all idle (one stderr line then)
    double bt_run = 0, bt_commit = 0, bt_emit = 0;
    double bt_wait0 = 0, bt_pool0 = 0;
    int64_t bt_miss0 = 0, bt_hits0 = 0, bt_pcie0 = 0;
    int64_t bt_windows = 0, bt_rows = 0, bt_tokens = 0;
    Clock::time_point bt_start = Clock::now();
    int admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
    int solo_slot = -1;                ///< SGEN: the slot a solo request runs in, on that slot's own state
    long long admit_max_new = 0;
    auto batch_on = [&] { for (const BSlot& b : bs) if (b.active) return true; return false; };
    // a prompt read on its own thread beside the slots' windows: the tier stays as it is meanwhile
    bool ov_active = false;
    std::function<bool()> ov_post;   // what the overlapped read's thread does right after its last chunk
    bool ov_post_done = false;
    auto try_next_line = [&](std::string& out) -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };
    // the elastic KV's spare cells back to the expert cache: every session keeps what it holds (the solo session
    // `ss_keep` cells, a decoding slot its next window too), idle ones first; `skip` and slot `skip_slot` are left
    // alone, and it stops once `enough` (if set) is true
    auto kv_give_back = [&](int64_t ss_keep, const basalt::core::SessionState* skip, int skip_slot,
                            const std::function<bool()>& enough, int64_t& given, std::string& e) -> bool {
        std::vector<basalt::core::SessionState*> sess;
        std::vector<basalt::program::KvHold> holds;
        auto add = [&](basalt::core::SessionState& s, int64_t keep, bool active) {
            if (&s == skip) return;
            for (size_t i = 0; i < sess.size(); ++i)
                if (sess[i] == &s) {
                    holds[i].keep = std::max(holds[i].keep, keep);
                    holds[i].active = holds[i].active || active;
                    return;
                }
            sess.push_back(&s);
            holds.push_back({basalt::core::session_kv_cells(s), keep, active});
        };
        add(ss, ss_keep, false);
        for (size_t b = 0; !bslot_ss.empty() && b < bslot_ss[0].size() && b < bs.size(); ++b) {
            if ((int) b == skip_slot) continue;
            const BSlot& sb = bs[b];
            add(*bslot_ss[0][b], std::max<int64_t>((int64_t) sb.ids.size(), sb.active ? sb.p + kKvWindowCells : 0),
                sb.active);
        }
        for (const auto& [i, keep] : basalt::program::kv_giveback_order(holds, kv_floor)) {
            if (holds[(size_t) i].backed < 0) continue;
            if (!kv_release(*sess[(size_t) i], keep, e)) return false;
            given += holds[(size_t) i].backed - basalt::core::session_kv_cells(*sess[(size_t) i]);
            if (enough && enough()) break;
        }
        return true;
    };
    // ---- ENC <image> <out> [<image> <out> ...] (--vision-mmproj), between requests and between batch windows.  The
    // images are decoded on the host first; then the encoder borrows the last expert-cache slots (the prompt path's
    // way: their rows are marked missing for the time), uploads its weights from the pinned copy, encodes each
    // image on its own stream, and the slots get their experts back from RAM before anything else reads the cache.
    // One reply line per image: `ENC OK <n> <nx> <ny> <ms>`, or a single `ENC ERR <message>`.
    auto enc_command = [&](const std::string& cmd) {
        auto reply_err = [&](const std::string& m, bool logged = false) {
            if (!logged) basalt::log::warn("basalt serve", "image encoder: refused: %s", m.c_str());
            std::printf("ENC ERR %s\n", m.c_str());
            std::fflush(stdout);
        };
        if (o.vision_mmproj.empty()) { reply_err("this engine was started without --vision-mmproj"); return; }
        if (pf_parts.size() != 1 || pf_parts[0].first < 0 || !stages.empty()) {
            reply_err("the in-engine image encoder needs one GPU's expert cache with a prompt lend region");
            return;
        }
        std::vector<std::string> words;
        {
            size_t pos = 3;
            for (;;) {
                pos = cmd.find_first_not_of(' ', pos);
                if (pos == std::string::npos) break;
                const size_t end = cmd.find(' ', pos);
                words.push_back(cmd.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
                if (end == std::string::npos) break;
                pos = end;
            }
        }
        if (words.empty() || words.size() % 2 != 0) { reply_err("expected: ENC <image> <output> [<image> <output> ...]"); return; }
        const auto t0 = Clock::now();
        struct Job { std::string out; basalt::vision::Size sz; std::vector<float> patches; };
        std::vector<Job> jobs;
        const basalt::vision::Geometry geo;
        int max_patches = 0;
        for (size_t i = 0; i < words.size(); i += 2) {
            std::ifstream f(words[i], std::ios::binary);
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            basalt::vision::Image img;
            std::string ierr;
            if (!f || bytes.empty() || !basalt::vision::decode_image(bytes.data(), bytes.size(), img, ierr)) {
                reply_err("cannot read the image " + words[i]);
                return;
            }
            Job j;
            j.out = words[i + 1];
            j.sz = basalt::vision::fit_size(img.w, img.h, geo);
            std::vector<float> planar;
            basalt::vision::normalize_planar(basalt::vision::to_canvas(img, j.sz), planar);
            basalt::vision::patchify(planar, j.sz, geo, j.patches);
            max_patches = std::max(max_patches, j.sz.w / 16 * (j.sz.h / 16));
            jobs.push_back(std::move(j));
        }
        const int tokens = std::max(max_patches / 4, geo.min_tokens);
        const auto t_dec = Clock::now();
        // the work that touches the cache starts from a quiet cache and device
        std::string e;
        if (ba_job) { ba_thr.join(); ba_job = false; }
        if (!ver.wait_commit(e)) { reply_err(e); return; }
        apply_pending(true);
        if (!xq_drain()) { reply_err("an adaptive swap failed"); return; }
        if (cudaDeviceSynchronize() != cudaSuccess) { reply_err(std::string("device: ") + cudaGetErrorString(cudaGetLastError())); return; }
        PfPart& p = pf_parts[0];
        if (!p.lent.empty()) { reply_err("a prompt's loan is still out"); return; }
        const int64_t k = serve_loans::cache_slots_for(*p.cache, vis_enc.arena_bytes(tokens));
        // the room: the cache's tail down to the first slot whose expert has no RAM copy to refill it from (never
        // less than the prompt path's loan region, which always has one)
        auto lendable = [&]() -> int64_t {
            std::vector<uint8_t> in_ram;
            if (srcp == &src && src.complement_ready()) {
                in_ram.resize(host_res.size());
                for (size_t i = 0; i < host_res.size(); ++i)
                    in_ram[i] = src.resident_blob((int64_t) i / g.n_expert, (int64_t) i % g.n_expert) != nullptr;
            }
            const int64_t floor = basalt::program::image_lend_floor(host_res, in_ram, p.cache->slots());
            int64_t r = p.cache->slots() - std::min<int64_t>(floor, p.first);
            if (batch_on()) r = std::min<int64_t>(r, kLendCapDecoding);
            return r;
        };
        int64_t room = lendable();
        // short of room: the elastic KV gives back the cells no session holds (idle slots first), and the cache
        // maps that VRAM again; the sessions take cells back on demand after the encode
        int64_t kv_given = 0;
        if (k > room && kv_floor > 0 &&
            !kv_give_back((int64_t) live.size(), nullptr, -1, [&] { room = lendable(); return k <= room; }, kv_given, e)) {
            reply_err(e);
            return;
        }
        if (k <= 0 || k > room) {
            basalt::log::warn("basalt serve", "image encoder: refused, %lld cache slots needed, %lld lendable "
                                 "(cache %lld slots, %lld KV cells given back, the rest held by the sessions)",
                         (long long) k, (long long) room, (long long) p.cache->slots(), (long long) kv_given);
            reply_err("the image needs " + std::to_string((long long) k) + " expert-cache slots and " +
                      std::to_string((long long) room) + " can be lent" +
                      (kv_floor > 0 ? ", with every spare KV cell given back" : ""), true);
            return;
        }
        if (kv_given > 0)
            basalt::log::info("basalt serve", "image encoder: %lld KV cells given back for %lld cache slots",
                         (long long) kv_given, (long long) k);
        const int32_t first = (int32_t) (p.cache->slots() - k);
        for (int64_t l = p.lb; l < p.le; ++l)
            for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                const size_t i = (size_t) (l * g.n_expert + ex);
                if (host_res[i] >= first) {
                    p.lent.emplace_back((int32_t) i, host_res[i]);
                    host_res[i] = basalt::core::kNotResident;
                }
            }
        res_upload();
        const auto t_lent = Clock::now();
        std::vector<std::array<double, 4>> done;   // n, nx, ny, ms
        std::string eerr;
        bool ok = vis_enc.attach(p.cache->device_slot(first), part_bytes(p, first), tokens, eerr);
        if (!ok && eerr.empty()) eerr = "the encoder could not be placed";
        const auto t_up = Clock::now();
        for (size_t i = 0; ok && i < jobs.size(); ++i) {
            const Job& j = jobs[i];
            const auto ti = Clock::now();
            std::vector<float> out;
            if (!vis_enc.encode(j.patches, j.sz, out)) { ok = false; eerr = "the vision encoder failed: " + vis_enc.last_error(); break; }
            const int nx = j.sz.w / 32, ny = j.sz.h / 32, n = nx * ny, n_embd = (int) g.n_embd;
            std::FILE* fo = std::fopen(j.out.c_str(), "wb");
            if (!fo) { ok = false; eerr = "cannot write " + j.out; break; }
            const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
            const bool wrote = (int) (out.size() / (size_t) n_embd) == n && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                               std::fwrite(out.data(), sizeof(float) * (size_t) n_embd, (size_t) n, fo) == (size_t) n;
            std::fclose(fo);
            if (!wrote) { ok = false; eerr = "short write to " + j.out; break; }
            done.push_back({(double) n, (double) nx, (double) ny,
                            std::chrono::duration<double, std::milli>(Clock::now() - ti).count()});
        }
        vis_enc.detach();
        const auto t_enc = Clock::now();
        // the slots get their experts back from RAM, in the same slots, then the table goes up again
        for (const auto& [i, slot] : p.lent) {
            const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
            const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
            if (b == nullptr || !p.cache->fill_slot_queued(slot, b, e, nb)) {
                basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
                reply_err("refilling the expert cache failed: " + e, true);
                return;
            }
            host_res[(size_t) i] = slot;
        }
        if (!p.cache->sync_queued(e)) {
            basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
            reply_err("refilling the expert cache failed: " + e, true);
            return;
        }
        const int64_t n_lent = (int64_t) p.lent.size();
        p.lent.clear();
        p.lent_chunk = 0;
        p.first_now = -1;   // the prompt path lays its buffers out again at its next loan
        res_upload();
        const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        basalt::log::info("basalt serve", "image encoder: %zu image(s) up to %d tokens in %lld cache slots (%.2f GiB), "
                             "%lld expert rows lent; decode %.0f ms, lend %.0f ms, weights %.0f ms, encode %.0f ms, refill %.0f ms",
                     jobs.size(), tokens, (long long) k, (double) vis_enc.arena_bytes(tokens) / 1073741824.0, (long long) n_lent,
                     ms(t0, t_dec), ms(t_dec, t_lent), ms(t_lent, t_up), ms(t_up, t_enc), ms(t_enc, Clock::now()));
        if (!ok) { reply_err(eerr); return; }
        for (const auto& d : done) std::printf("ENC OK %d %d %d %.0f\n", (int) d[0], (int) d[1], (int) d[2], d[3]);
        std::fflush(stdout);
    };
    // the session a request just left behind (its prompt) -> slot b's sessions, on every stage
    auto copy_to_slot = [&](int b, const std::vector<int32_t>& ids, std::string& e) -> bool {
        const int64_t upto = (int64_t) ids.size();
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            basalt::core::SessionState& from = k == 0 ? *rsp : stages[k - 1]->ss;
            basalt::core::SessionState& to = *bslot_ss[k][(size_t) b];
            if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            // the elastic KV: the slot backs the copied cells and its room past them, and gives back the rest
            if (k == 0 && (!kv_fit(to, upto + kKvGrowCells, e) || !kv_release(to, upto + kKvGrowCells, e)))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
            basalt::core::ConversationCheckpoint ck;
            ck.ids = ids;
            if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
                !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
                return false;
            for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                basalt::core::ConversationKv img;
                if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                    return false;
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
                if (!basalt::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                    return false;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
        }
        if (batch_spec && !mtp.copy_store(mtp.main_store(), b, upto, e)) return false;   // the draft layer's K/V
        return true;
    };
    // slot b's sessions -> the main ones (the reverse of copy_to_slot): a request that continues the
    // conversation an idle slot holds reads on from there.  Every stage, the same calls.
    // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
    // a slot's conversation held in RAM (`hold_slot`) while its KV cells back the request that continues it, when the
    // two do not fit side by side; copy_from_slot restores from it
    struct SlotHeld {
        int b = -1;
        std::vector<basalt::core::ConversationCheckpoint> ck;
        std::vector<std::vector<basalt::core::ConversationKv>> kv;
    };
    SlotHeld slot_held;
    auto hold_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
        slot_held = SlotHeld{};
        const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
        slot_held.ck.resize(bslot_ss.size());
        slot_held.kv.resize(bslot_ss.size());
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
            const basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
            if (from.gdn_state == to.gdn_state) continue;
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
            if (at == nullptr) {
                slot_held.ck[k].ids = bs[(size_t) b].ids;
                if (!basalt::core::conversation_checkpoint_save(slot_held.ck[k], from, g, e)) return false;
            }
            for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                slot_held.kv[k].emplace_back();
                if (!basalt::core::conversation_kv_save(slot_held.kv[k].back(), from.qsa_states[from.qsa_ord0 + j], g,
                                                        upto, true, e))
                    return false;
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
            }
        }
        slot_held.b = b;
        return true;
    };
    auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
        const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
        if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
        const bool held = slot_held.b == b;
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
            basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
            if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
            if (at != nullptr) {
                if (!basalt::core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, g, e))
                    return false;
            } else if (held) {
                if (!basalt::core::conversation_checkpoint_restore(slot_held.ck[k], to, g, e)) return false;
            } else {
                basalt::core::ConversationCheckpoint ck;
                ck.ids = bs[(size_t) b].ids;
                if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
                    return false;
            }
            for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                // the slot's cells straight into the request's session (a held slot's from its RAM copy)
                if (held ? !basalt::core::conversation_kv_restore(slot_held.kv[k][(size_t) j],
                                                                  to.qsa_states[to.qsa_ord0 + j], g, upto, true, e)
                         : !basalt::core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j],
                                                               to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                    return false;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
        }
        if (held) slot_held = SlotHeld{};
        if (batch_spec && !mtp.copy_store(b, mtp.main_store(), upto, e)) return false;
        return true;
    };
    // hashes of everything slot b's sessions hold for its first `upto` tokens, per part (GDN state, PLE history,
    // indexer tails, dead rows, block positions, K, V, scales, pooled rows), for comparing a slot's state window by
    // window against another run of the same request
    auto slot_digest = [&](int b, int64_t upto, uint64_t* part) -> bool {
        for (int i = 0; i < 10; ++i) part[i] = 1469598103934665603ull;
        auto mix = [&](int i, const uint8_t* p, size_t n) {
            uint64_t h = part[i];
            size_t k = 0;
            for (; k + 8 <= n; k += 8) {
                uint64_t w;
                std::memcpy(&w, p + k, 8);
                h = (h ^ w) * 1099511628211ull;
                h ^= h >> 29;
            }
            for (; k < n; ++k) h = (h ^ p[k]) * 1099511628211ull;
            part[i] = h;
        };
        std::string e;
        const basalt::core::SessionState& from = *bslot_ss[0][(size_t) b];
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        basalt::core::ConversationCheckpoint ck;
        ck.ids.assign((size_t) upto, 0);
        if (!basalt::core::conversation_checkpoint_save(ck, from, g, e)) return false;
        const basalt::core::CheckpointBytes* cv[5] = {&ck.gdn, &ck.ple, &ck.tails, &ck.dead, &ck.block_pos};
        for (int i = 0; i < 5; ++i) mix(i, cv[i]->data(), cv[i]->size());
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            basalt::core::ConversationKv img;
            if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto & ~int64_t(3), true, e)) return false;   // whole pages: the cells past `upto` may hold rejected rows
            if (cudaDeviceSynchronize() != cudaSuccess) return false;
            const basalt::core::ConversationBuffer* kb[5] = {&img.k, &img.v, &img.k_scale, &img.v_scale, &img.pooled};
            for (int i = 0; i < 5; ++i) {   // (the pooled rows past the completed blocks are not state)
                const size_t cnt = i == 4 ? std::min(kb[i]->size(), (size_t) (upto >> 2) * (size_t) img.idx_dim * 4) : kb[i]->size();
                kb[i]->visit(0, cnt, [&](const uint8_t* p, size_t n, size_t) { mix(5 + i, p, n); return true; });
            }
        }
        return true;
    };
    // one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
    // so it keeps the conversation it holds
    auto batch_step = [&]() -> bool {
        constexpr int MR = basalt::kernels::kVerifyMaxT;
        const int MB = std::min(MR, ver.max_t());   // the row budget is this verifier capacity; MR only sizes the arrays
        int S = 0, R = 0;
        int rows[MR] = {}, lens[MR] = {}, n_keep[MR] = {}, r0s[MR] = {};
        int32_t tok[MR] = {}, outb[MR] = {};
        int64_t pos[MR] = {}, p0s[MR] = {};
        std::vector<int32_t> drafts[MR];
        // the elastic KV: each slot's next cells backed ahead of its windows; beside a prompt read only from free
        // VRAM, and a slot that needs more than that waits for the read to end
        std::vector<uint8_t> kv_wait(bs.size(), 0);
        if (kv_floor > 0)
            for (int b = 0; b < (int) bs.size(); ++b) {
                if (!bs[(size_t) b].active) continue;
                basalt::core::SessionState& u = *bslot_ss[0][(size_t) b];
                const int64_t backed = basalt::core::session_kv_cells(u), at = bs[(size_t) b].p;
                if (at + 2 * MR + 64 <= backed) continue;
                std::string ke;
                if (!ov_active && ba_job) { ba_thr.join(); ba_job = false; }   // its copies land before the drain
                if (kv_fit(u, at + kKvGrowCells, ke, !ov_active, at + 2 * MR + 64)) continue;
                if (!ke.empty() && kv_short) {   // the expert cache's floor: this slot's reply ends here
                    BSlot& sl = bs[(size_t) b];
                    basalt::log::warn("basalt batch", "slot %d stopped at %lld cells: %s", b, (long long) at, ke.c_str());
                    std::printf("BDONE %d %lld length %.1f %lld %lld\n", b, (long long) sl.produced,
                                std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                                (long long) sl.drafts_accepted, (long long) sl.drafts_offered);
                    sl.active = false;
                    sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids
                    sl.guard.reset();
                    kv_wait[(size_t) b] = 1;
                    continue;
                }
                if (!ke.empty()) { std::printf("ERR %s\n", ke.c_str()); return false; }
                if (at + 2 * MR > backed) kv_wait[(size_t) b] = 1;
            }
        for (int b = 0; b < (int) bs.size() && S < MR; ++b)
            if (bs[(size_t) b].active && !kv_wait[(size_t) b]) {
                rows[S] = b;
                pos[S] = bs[(size_t) b].p;
                lens[S] = 1;
                ++S;
            }
        if (S == 0) return true;
        bool scripted = false;
        for (int sg = 0; sg < S; ++sg) scripted = scripted || !bs[(size_t) rows[sg]].script.empty();
        const bool real = batch_spec && !scripted;
        if (real) {
            // each slot's window as the solo rule makes it (the drafts that clear the cumulative or min-p cut, or a
            // prompt-lookup run the policy prefers); the rows beyond the budget go to the slots whose last kept draft is the
            // least likely to be accepted
            int Tw[MR] = {};
            float val[MR][8] = {};
            for (int sg = 0; sg < S; ++sg) {
                BSlot& sl = bs[(size_t) rows[sg]];
                int T = 1;
                while (T < S_mtp && T - 1 < sl.n_dr && basalt::core::draft_keep(sl.dprob, T - 1, sl.cum.th())) ++T;
                sl.from_sfx = false;
                sl.sk = 0;
                if (sl.sfx && sl.n_dr > 0) {
                    sl.sk = sl.sfx->propose(S_all - 1, sl.sbuf);
                    sl.smatch = sl.sfx->last_match();
                    if (sl.sk > 0 && sl.sbuf[0] == sl.dr[0]) {
                        const basalt::spec::DraftPolicy::Pick pk = bpolicy.choose(T, sl.sk, sl.smatch);
                        if (pk.lookup) { T = pk.t; sl.from_sfx = true; }
                    }
                }
                const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 1;
                T = (int) std::max<int64_t>(1, std::min<int64_t>(T, room));
                Tw[sg] = T;
                float c = 1.0f;
                const double q = sl.from_sfx ? bpolicy.lookup_rate(sl.smatch) : 0.0;
                for (int j = 0; j + 1 < 8; ++j) {   // val[j]: the chance drafts 0..j are all accepted
                    c *= sl.from_sfx ? (float) q : basalt::core::draft_cum_acc(sl.dprob[j]);
                    val[sg][j] = c;
                }
            }
            int sum = 0;
            for (int sg = 0; sg < S; ++sg) sum += Tw[sg];
            if (sum > MB) {
                // every slot keeps one row, and none takes more than an equal share of the window unless the others
                // want less: the slots wanting least are served in full, the rest split what is left
                int ord[MR];
                for (int sg = 0; sg < S; ++sg) ord[sg] = sg;
                std::sort(ord, ord + S, [&](int x1, int x2) { return Tw[x1] < Tw[x2]; });
                int left_rows = MB;
                for (int i = 0; i < S; ++i) {
                    const int share = std::max(1, (left_rows + (S - i) - 1) / (S - i));
                    Tw[ord[i]] = std::min(Tw[ord[i]], share);
                    left_rows -= Tw[ord[i]];
                }
                sum = 0;
                for (int sg = 0; sg < S; ++sg) sum += Tw[sg];
            }
            while (sum > MB) {
                int pick = -1;
                for (int sg = 0; sg < S; ++sg)
                    if (Tw[sg] > 1 && (pick < 0 || val[sg][Tw[sg] - 2] < val[pick][Tw[pick] - 2])) pick = sg;
                --Tw[pick];
                --sum;
            }
            for (int sg = 0; sg < S; ++sg) {
                BSlot& sl = bs[(size_t) rows[sg]];
                lens[sg] = Tw[sg];
                const int32_t* src = sl.from_sfx ? sl.sbuf : sl.dr;
                drafts[sg].assign(src, src + (Tw[sg] - 1));
            }
        } else {   // the rows beyond one per slot go to the scripted slots, as many drafts as the script and the context allow
            int extra = MB - S;
            const int first = (int) (bs[(size_t) rows[0]].rng() % (uint64_t) S);
            for (int i = 0; i < S && extra > 0; ++i) {
                const int sg = (first + i) % S;
                BSlot& sl = bs[(size_t) rows[sg]];
                if (sl.script.empty()) continue;
                const int64_t avail = (int64_t) sl.script.size() - sl.produced;
                const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 2;
                const int want = (int) std::min<int64_t>({(int64_t) (sl.rng() % (uint64_t) (extra + 1)), avail, room});
                if (want <= 0) continue;
                drafts[sg].assign(sl.script.begin() + sl.produced, sl.script.begin() + sl.produced + want);
                if (sl.rng() % 2 == 0) {   // a wrong draft at a random place
                    const size_t c = (size_t) (sl.rng() % (uint64_t) want);
                    drafts[sg][c] = (int32_t) ((drafts[sg][c] + 1 + sl.rng() % 100) % n_vocab);
                }
                lens[sg] += want;
                extra -= want;
            }
        }
        for (int sg = 0; sg < S; ++sg) {
            r0s[sg] = R;
            tok[R] = bs[(size_t) rows[sg]].x;
            for (size_t j = 0; j < drafts[sg].size(); ++j) tok[R + 1 + (int) j] = drafts[sg][j];
            R += lens[sg];
        }
        const bool was_busy = basalt::core::progress().busy.load();
        basalt::core::progress().busy.store(true);
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        if (!ov_active) apply_pending(false);
        if (batch_adapt && !ov_active) {
            // the tier follows what all the slots' rows route: a round every --adapt-every windows.  Its steps
            // are taken once their copies are done; the copies out of the RAM tier (which may read the experts
            // file) are queued from a worker thread beside the windows, so no window waits for one.
            const Clock::time_point ta = Clock::now();
            bool ok = true, acted = false;
            ++ba_win;
            if (ba_job && ba_thr.finished()) { ba_thr.join(); ba_job = false; }
            if (!ba_job) {
                if (xq_stage != 0) {
                    if (cudaEventQuery(adapt_ev) == cudaSuccess) {
                        const int was = xq_stage;
                        ok = xq_advance(true);
                        if (ok && was == 1 && xq_stage == 2) {
                            ba_thr.run([&] { ver.flush_post_launch(); });
                            ba_job = true;
                        }
                        acted = true;
                    }
                } else if (ba_win % o.adapt_every == 0) {
                    ok = xq_round();
                    ba_swaps += (int64_t) xq.size();
                    ++ba_rounds;
                    acted = true;
                }
            }
            std::string perr;
            if (ok && acted && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
            if (!ok) { std::printf("ERR an adaptive refill failed\n"); return false; }
            ba_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            if (ba_win % 2000 == 0 && ba_win > 0) {   // the swap rate and the hit rate over the last 2000 windows
                const int64_t h = drive.d.cache_hits, m = drive.d.multi_misses;
                basalt::log::info("basalt batch adapt", "window %lld: %lld swaps in %lld rounds, VRAM hits %.2f and CPU "
                                     "experts %.2f per layer-window", (long long) ba_win, (long long) (ba_swaps - ba_swaps_l),
                             (long long) (ba_rounds - ba_rounds_l), (double) (h - ba_h_l) / (2000.0 * g.n_layers),
                             (double) (m - ba_m_l) / (2000.0 * g.n_layers));
                ba_swaps_l = ba_swaps; ba_rounds_l = ba_rounds; ba_h_l = h; ba_m_l = m;
            }
        }
        if (bt_windows == 0) {
            bt_start = Clock::now();
            bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
            bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
        }
        const Clock::time_point w0 = Clock::now();
        if (!ver.run_slot_segs(rows, lens, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
            std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
            return false;
        }
        const Clock::time_point w1 = Clock::now();
        // per segment: the drafts the picks confirm, then how far the request goes (stop token, limits) within them
        const char* fin_of[MR] = {};
        auto fin_at = [&](const BSlot& sl, int b, int32_t y, int e) -> const char* {   // how the request ends at pick e
            const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
            return eos ? "stop" : sl.stop ? "cancel" : sl.produced + e + 1 >= sl.max_new ? "length"
                 : sl.p + e + 2 > slot_cap[(size_t) b] ? "length" : nullptr;
        };
        for (int sg = 0; sg < S; ++sg) {
            const BSlot& sl = bs[(size_t) rows[sg]];
            int k = 1;
            while (k < lens[sg] && outb[r0s[sg] + k - 1] == drafts[sg][(size_t) k - 1]) ++k;
            for (int e = 0; e < k; ++e) {
                fin_of[sg] = fin_at(sl, rows[sg], outb[r0s[sg] + e], e);
                if (fin_of[sg] != nullptr) { k = e + 1; break; }
            }
            n_keep[sg] = k;
        }
        // grammar: a slot with a guard scans its kept picks; the first one the grammar refuses is drawn again
        // under the mask (the solo rule) and the segment ends at that row, which stays committed
        std::vector<float> grammar_bias;
        for (int sg = 0; sg < S; ++sg) {
            BSlot& sl = bs[(size_t) rows[sg]];
            if (sl.guard == nullptr || !sl.guard->on()) continue;
            const int miss = sl.guard->scan(outb + r0s[sg], n_keep[sg]);
                if (miss < 0) continue;
            sl.guard->mask_bias({}, grammar_bias);
            int32_t gtok = 0;
            if (!ver.resample_row(r0s[sg] + miss, p0s[sg], grammar_bias, gtok, err)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            outb[r0s[sg] + miss] = gtok;
            if (!sl.guard->take(gtok))
                basalt::log::info("basalt grammar", "slot %d: the masked draw at row %d was token %d", rows[sg], miss, (int) gtok);
            n_keep[sg] = miss + 1;
            fin_of[sg] = fin_at(sl, rows[sg], gtok, miss);
        }
        if (logit_hash)
            for (int sg = 0; sg < S; ++sg)
                for (int e = 0; e < n_keep[sg]; ++e)
                    basalt::log::raw("LH %d %lld %016llx\n", rows[sg], (long long) (bs[(size_t) rows[sg]].p + e),
                                 (unsigned long long) row_hash(r0s[sg] + e));
        // the commit runs beside the drafts when nothing reads the committed state meanwhile
        const bool overlap = real && !bdigest;
        if (!(overlap ? ver.commit_slots_begin(n_keep, err) : ver.commit_slots(n_keep, err))) {
            std::printf("ERR %s\n", err.c_str());
            return false;
        }
        const Clock::time_point w2 = Clock::now();
        auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
        bt_run += msd(w0, w1);
        bt_commit += msd(w1, w2);
        ++bt_windows;
        bt_rows += R;
        for (int sg = 0; sg < S; ++sg) {
            const int b = rows[sg];
            BSlot& sl = bs[(size_t) b];
            const int k = n_keep[sg];
            std::printf("BT %d", b);   // one line per slot and window: every token it kept
            for (int e = 0; e < k; ++e) {
                sl.ids.push_back(tok[r0s[sg] + e]);    // the window fed it: the slot's sessions hold it now
                std::printf(" %d", (int) outb[r0s[sg] + e]);
            }
            std::printf("\n");
            sl.produced += k;
            if (real) {   // the window's verdict on its drafts: the draft-length rule, the lookup history, the policy
                const int a = k - 1, Tw = lens[sg];
                if (Tw > 1 && !sl.from_sfx) sl.cum.observe(sl.dprob, Tw - 1, a);
                if (Tw > 1) sl.acc = 0.8 * sl.acc + 0.2 * ((double) a / (double) (Tw - 1));
                sl.drafts_offered += Tw - 1;
                sl.drafts_accepted += a;
                if (sl.sfx) for (int e = 0; e < k; ++e) sl.sfx->append(outb[r0s[sg] + e]);
                if (sl.windows > 0 && fin_of[sg] == nullptr)
                    bpolicy.observe(sl.from_sfx, Tw, a, sl.smatch, (msd(w0, w2) + bt_draft_last) * (double) Tw / (double) R);
                ++sl.windows;
                bt_slotwin += 1;
                bt_tokens += k;
                bt_dr_off += Tw - 1;
                bt_dr_acc += a;
                p0s[sg] = sl.p;
            }
            if (bdigest) {
                uint64_t part[10] = {};
                slot_digest(b, sl.p + k, part);
                std::printf("BDIG %d %lld", b, (long long) (sl.p + k));
                for (int i = 0; i < 10; ++i) std::printf("%c%llx", i == 0 ? ' ' : ',', (unsigned long long) part[i]);
                std::printf("\n");
            }
            if (fin_of[sg] != nullptr) {
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                std::printf("BDONE %d %lld %s %.1f %lld %lld\n", b, (long long) sl.produced, fin_of[sg], ms,
                            (long long) sl.drafts_accepted, (long long) sl.drafts_offered);
                sl.active = false;
                sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids for the next turn
                sl.guard.reset();
            } else {
                sl.x = outb[r0s[sg] + k - 1];
                sl.p += k;
            }
        }
        std::fflush(stdout);
        const Clock::time_point w3 = Clock::now();
        bt_emit += msd(w2, w3);
        if (real) {
            // the next windows' drafts, one slot at a time on the slot's own draft K/V.  The rows beyond one per
            // slot go to the slots most likely to use them; the others only advance their draft K/V.
            int cont[MR], nc = 0;
            for (int sg = 0; sg < S; ++sg) if (fin_of[sg] == nullptr) cont[nc++] = sg;
            int nfull = std::min(nc, MR - nc);
            std::sort(cont, cont + nc, [&](int x1, int x2) {
                const BSlot& s1 = bs[(size_t) rows[x1]];
                const BSlot& s2 = bs[(size_t) rows[x2]];
                return s1.acc + 0.05 * (double) s1.since_draft > s2.acc + 0.05 * (double) s2.since_draft;
            });
            const float* Rw = mtp.keep_rows(ver.final_R_all(), R, err);
            if (Rw == nullptr) { std::printf("ERR %s\n", err.c_str()); return false; }
            const int64_t HCN = (int64_t) g.hc * g.n_embd;
            for (int ci = 0; ci < nc; ++ci) {
                const int sg = cont[ci];
                const int b = rows[sg];
                BSlot& sl = bs[(size_t) b];
                const int Tw = lens[sg];
                const int a = n_keep[sg] - 1;
                mtp.set_ple_session(bslot_ss[0][(size_t) b].get());
                sl.n_dr = 0;
                if (ci < nfull) {
                    mtp.set_cum_th(sl.cum.th());
                    int nd = 0;
                    if (!mtp.draft_seq(b, Rw + (size_t) r0s[sg] * HCN, Tw, outb + r0s[sg], p0s[sg], a, sl.dr, err, sl.dprob,
                                       &nd)) {
                        std::printf("ERR %s\n", err.c_str());
                        return false;
                    }
                    sl.n_dr = nd;
                    sl.since_draft = 0;
                } else {
                    ++sl.since_draft;
                    if (!mtp.defer_rows(b, Rw + (size_t) r0s[sg] * HCN, outb + r0s[sg], n_keep[sg], p0s[sg], err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return false;
                    }
                }
            }
            mtp.set_ple_session(rsp);
            if (!mtp.idle(err)) { std::printf("ERR %s\n", err.c_str()); return false; }   // (the kept rows are copied)
            bt_draft += msd(w3, Clock::now());
            bt_draft_last = msd(w3, Clock::now());
        }
        if (overlap) {
            const Clock::time_point c0 = Clock::now();
            if (!ver.commit_slots_end(err)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            bt_commit += msd(c0, Clock::now());
        }
        basalt::core::progress().busy.store(was_busy);
        if (!batch_on() && bt_windows > 0) {
            const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
            const double L = (double) g.n_layers;
            basalt::log::info("basalt batch", "%lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach "
                                 "wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f; per layer-window: CPU experts "
                                 "%.2f, VRAM hits %.2f, PCIe %.2f; %.1f rows/s over %.0f ms of wall time (admissions "
                                 "included)",
                         (long long) bt_windows, bt_rows / w, (bt_run + bt_commit + bt_emit + bt_draft) / w, bt_run / w,
                         (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                         (drive.d.multi_misses - bt_miss0) / (w * L), (drive.d.cache_hits - bt_hits0) / (w * L),
                         (drive.d.pcie_experts - bt_pcie0) / (w * L), 1000.0 * bt_rows / std::max(wall, 1e-9), wall);
            if (batch_adapt) {
                basalt::log::info("basalt batch adapt", "%lld rounds, %lld swaps queued (%.1f per round), %.3f ms/window of host "
                                     "time, peer swaps total %lld", (long long) ba_rounds, (long long) ba_swaps,
                             ba_rounds > 0 ? (double) ba_swaps / (double) ba_rounds : 0.0, ba_ms / w,
                             (long long) (peer.valid() ? peer.swaps() : 0));
                ba_swaps = ba_rounds = ba_swaps_l = ba_rounds_l = 0;
                ba_ms = 0;
            }
            if (bt_slotwin > 0)
                basalt::log::info("basalt batch spec", "%.2f tokens/window/slot (%.2f slots/window, %.2f rows/window), "
                                     "drafts %lld accepted of %lld, draft %.2f ms/window (%.2f ms/slot-window), %.1f "
                                     "tokens/s", (double) bt_tokens / (double) bt_slotwin, (double) bt_slotwin / w,
                             bt_rows / w, (long long) bt_dr_acc, (long long) bt_dr_off, bt_draft / w,
                             bt_draft / (double) bt_slotwin, 1000.0 * (double) bt_tokens / std::max(wall, 1e-9));
            for (size_t k = 0; k <= stages.size(); ++k) {
                const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
                if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
            }
            bt_run = bt_commit = bt_emit = bt_draft = 0;
            bt_windows = bt_rows = bt_tokens = bt_slotwin = bt_dr_off = bt_dr_acc = 0;
        }
        return true;
    };
    // ---- --batch-groups: the slot groups pipelined through the stages (--batch-groups G > 1 with a layer split)
    const int n_pipe = (int) stages.size() + 1;
    const bool piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1;
    const int GS = piped ? o.batch / o.batch_groups : o.batch;
    struct PGroup {
        bool inflight = false;
        int stage = 0;                  ///< the stage it runs on or waits for
        int32_t tok[basalt::kernels::kVerifyMaxT] = {};
        int64_t pos[basalt::kernels::kVerifyMaxT] = {};
        int64_t since = 0;              ///< tick it started waiting (fairness)
    };
    std::vector<PGroup> pg((size_t) (piped ? o.batch_groups : 0));
    std::vector<int> stage_group((size_t) n_pipe, -1);
    int64_t pipe_tick = 0, rr = 0;
    auto stage_verifier = [&](int k) -> basalt::core::Verifier& { return k == 0 ? ver : stages[(size_t) k - 1]->ver; };
    auto pipe_inflight = [&] { for (const PGroup& x : pg) if (x.inflight) return true; return false; };
    auto group_active = [&](int gi) {
        for (int t = 0; t < GS; ++t) if (bs[(size_t) (gi * GS + t)].active) return true;
        return false;
    };
    // one turn: serve every running stage, hand finished groups on, start what can start
    auto pump = [&](bool may_start) -> bool {
        ++pipe_tick;
        if (bt_windows == 0 && !pipe_inflight()) {
            bt_start = Clock::now();
            bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
            bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
        }
        for (int k = 0; k < n_pipe; ++k) {
            const int gi = stage_group[(size_t) k];
            if (gi < 0) continue;
            basalt::core::Verifier& vk = stage_verifier(k);
            const int r = vk.batch_poll(win_pool_fn, win_pool_user, err);
            if (r < 0) { std::printf("ERR %s\n", err.c_str()); return false; }
            if (r == 0) continue;
            stage_group[(size_t) k] = -1;
            PGroup& G = pg[(size_t) gi];
            if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
            // the last stage: the group's picks
            const int32_t* outb = vk.batch_out();
            for (int t = 0; t < GS; ++t) {
                BSlot& sl = bs[(size_t) (gi * GS + t)];
                if (!sl.active) continue;
                const int32_t y = outb[t];
                std::printf("BT %d %d\n", gi * GS + t, (int) y);
                ++sl.produced;
                ++bt_rows;
                const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                : sl.p + 2 > slot_cap[(size_t) (gi * GS + t)] ? "length" : nullptr;
                sl.ids.push_back(sl.x);
                if (fin != nullptr) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                    std::printf("BDONE %d %lld %s %.1f 0 0\n", gi * GS + t, (long long) sl.produced, fin, ms);
                    sl.active = false;
                    sl.cached = false;   // the pipeline's pad rows: a pipelined slot is not reused as a cache
                } else {
                    sl.x = y;
                    sl.p += 1;
                }
            }
            std::fflush(stdout);
            ++bt_windows;
            G.inflight = false;
        }
        // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
        for (int k = n_pipe - 1; k >= 0; --k) {
            if (stage_group[(size_t) k] >= 0) continue;
            int pick = -1;
            if (k > 0) {
                for (int gi = 0; gi < (int) pg.size(); ++gi)
                    if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                        (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                        pick = gi;
                if (pick < 0) continue;
            } else {
                if (!may_start) continue;
                for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                    const int gi = (int) ((rr + j) % (int64_t) pg.size());
                    if (!pg[(size_t) gi].inflight && group_active(gi)) pick = gi;
                }
                if (pick < 0) continue;
                rr = pick + 1;
                PGroup& G = pg[(size_t) pick];
                for (int t = 0; t < GS; ++t) {
                    BSlot& sl = bs[(size_t) (pick * GS + t)];
                    G.tok[t] = sl.active ? sl.x : 0;
                    G.pos[t] = sl.active ? sl.p : 0;
                    if (!sl.active) sl.cached = false;   // its pad row writes its state
                }
                G.inflight = true;
                G.stage = 0;
                drive.d.failed = false;
                apply_pending(false);
            }
            PGroup& G = pg[(size_t) pick];
            basalt::core::progress().busy.store(true);
            if (!stage_verifier(k).batch_launch(pick * GS, GS, G.tok, G.pos, err)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            stage_group[(size_t) k] = pick;
        }
        if (drive.d.failed) { std::printf("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed"); return false; }
        if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: the timing line, as batch_step
            const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
            basalt::log::raw("basalt batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = "
                                 "%.1f rows/s (admissions included)\n", o.batch_groups, GS, (long long) bt_windows,
                         (long long) bt_rows, wall, 1000.0 * bt_rows / std::max(wall, 1e-9));
            for (int k = 0; k < n_pipe; ++k) {
                const std::string pr = stage_verifier(k).profile_report();
                if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %d (ms/window):%s\n", k + 1, pr.c_str());
            }
            bt_windows = bt_rows = 0;
            basalt::core::progress().busy.store(false);
        }
        return true;
    };
    auto pipe_drain = [&]() -> bool {
        while (pipe_inflight())
            if (!pump(false)) return false;
        return true;
    };
    // The conversations the engine holds go to the disk tier when no stored copy reaches within kDiskStaleTokens of
    // them: the RAM tier's (at exit), the solo session's and every idle slot's, each with its draft layer's K/V.  Between
    // requests (no slot decoding) this bounds what a crash or a kill loses; a slot's draft K/V is copied into the main
    // store for the capture, which every request refills from its own store first, so between requests that runs only
    // when the solo session holds no conversation of its own there.  At most `limit` images; returns those queued.
    auto spill_to_disk = [&](bool at_exit, size_t limit) -> size_t {
        ckd_finish();   // a turn checkpoint's state may still be on its way to the host
        size_t written = 0;
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
        if (at_exit)
            while (conversations.size() > 0) {
                basalt::core::SavedConversation s = conversations.take(conversations.size() - 1);
                if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec) && disk.put(std::move(s), kDiskQueueCap))
                    ++written;
            }
        auto spill = [&](const basalt::core::SessionState& s, const std::vector<int32_t>& ids,
                         const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
            const basalt::core::ConversationView view{ids, imgs, cks, cvec};
            size_t bytes = 0;
            std::string se;
            if (!basalt::core::conversation_snapshot_bytes(view, s, g, mtp_kv, bytes, se)) {
                basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
                return;
            }
            if (at_exit) disk.flush();   // one image in RAM at a time
            if (!capture_admit(bytes, floor)) {
                basalt::log::info("basalt serve", "conversation cache: %zu tokens not written to disk (physical RAM "
                                  "admission)", ids.size());
                return;
            }
            try {
                const auto t0 = Clock::now();
                basalt::core::SavedConversation image;
                if (!basalt::core::conversation_snapshot_save(image, view, s, g, mtp_kv, se)) {
                    basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
                    return;
                }
                if (disk.put(std::move(image), kDiskQueueCap)) {
                    ++written;
                    if (!at_exit)
                        basalt::log::info("basalt serve", "conversation cache: %zu tokens to disk between requests "
                                          "(captured in %.0f ms)", ids.size(),
                                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
            } catch (const std::bad_alloc&) {
                basalt::log::warn("basalt serve", "conversation cache: allocation failed; not written to disk");
            }
        };
        std::string se;
        if (mtp_on && !mtp.use_main(se)) se.clear();
        const bool slot0_cached = slot0_is_solo && !bs.empty() && bs[0].cached;
        const bool solo_holds = live_ok && !live.empty() && !slot0_cached;
        if (solo_holds && !disk_has(live, live_imgs, checks, cvec_cached)) spill(ss, live, live_imgs, checks, cvec_cached);
        if (solo_holds && !at_exit && batch_spec) return written;   // the main store holds its draft K/V
        for (int b = 0; b < (int) bs.size() && written < limit; ++b) {
            BSlot& sl = bs[(size_t) b];
            if (!sl.cached || sl.active || sl.partial || disk_has(sl.ids, sl.imgs, sl.checks, sl.cvec)) continue;
            if (batch_spec && !mtp.copy_store(b, mtp.main_store(), (int64_t) sl.ids.size(), se)) {
                se.clear();
                continue;
            }
            make_view(rviews[(size_t) b], b);
            spill(rviews[(size_t) b], sl.ids, sl.imgs, sl.checks, sl.cvec);
        }
        if (at_exit) disk.flush();
        return written;
    };
    // PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE], between requests (basalt-probe): the expert tiers
    // go back to their placement at READY, or with FILE to its ranking placed rank for rank into the same slots;
    // the routing counts, the round counter and the draft policy start again; the session's cached prompts are
    // dropped; the adaptive tier's settings apply to the requests that follow.  Only slots whose expert differs
    // are copied, in rounds of as many as the RAM copy has free places for.
    auto probe_adapt = [&](const std::string& cmd, std::string& e) -> bool {
        const auto t0 = Clock::now();
        if (drive.d.usage.empty() || !adapt_fast || host_res.empty() || remote_opt) {
            e = "PROBE adapt: needs the adaptive tier on the main card (--adapt-every and --adapt-swaps above 0, "
                "no layer split)";
            return false;
        }
        if (!xpeers.empty() || (peer.valid() && peer.exclusive())) {
            e = "PROBE adapt: not with more than one peer card or an exclusive peer tier";
            return false;
        }
        int every = o.adapt_every, swaps_n = o.adapt_swaps;
        float decay = fast_decay;
        std::string pfile;
        std::istringstream kvs(cmd.substr(11));
        for (std::string kv; kvs >> kv;) {
            const size_t eq = kv.find('=');
            const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
            const std::string val = eq == std::string::npos ? std::string() : kv.substr(eq + 1);
            if (key == "every" && !val.empty()) every = std::atoi(val.c_str());
            else if (key == "swaps" && !val.empty()) swaps_n = std::atoi(val.c_str());
            else if (key == "decay" && !val.empty()) decay = (float) std::atof(val.c_str());
            else if (key == "profile" && !val.empty()) pfile = val;
            else { e = "PROBE adapt: bad setting " + kv; return false; }
        }
        if (every < 1 || swaps_n < 1 || !(decay > 0.0f && decay < 1.0f)) {
            e = "PROBE adapt: every and swaps must be at least 1, decay between 0 and 1";
            return false;
        }
        for (const PfPart& p : pf_parts)
            if (!p.lent.empty()) { e = "PROBE adapt: a prompt's loan is still out"; return false; }
        // the target: (slot, residency index) per tier
        std::vector<std::pair<int32_t, int32_t>> want_main, want_peer;
        std::vector<int32_t> rank_new = prof_rank_start;
        if (pfile.empty()) {
            for (size_t i = 0; i < res_at_start.size(); ++i)
                if (res_at_start[i] >= 0) want_main.emplace_back(res_at_start[i], (int32_t) i);
            for (size_t i = 0; i < peer_at_start.size(); ++i)
                if (peer_at_start[i] >= 0) want_peer.emplace_back(peer_at_start[i], (int32_t) i);
        } else {
            const auto& lay = basalt::kernels::cpu::expert_layout();
            for (int64_t l = 1; l < g.n_layers; ++l)
                if (lay.blob_bytes(l) != lay.blob_bytes(0)) { e = "PROBE adapt: profile= needs one blob size"; return false; }
            if (xcache.per_layer_admission()) { e = "PROBE adapt: profile= needs a cache shared by all layers"; return false; }
            std::vector<std::pair<int32_t, int32_t>> q;
            int64_t qslots = 0;
            if (!basalt::core::read_expert_profile(basalt::core::FileRange(pfile), g.n_layers, g.n_expert, q, qslots, e)) {
                e = "PROBE adapt: " + e;
                return false;
            }
            if (peer_hot_share) {
                size_t at = 0, fill = 0;
                (void) peer_hot_share_order(q, at, fill);
            }
            // the start's slots in its ranking's order take the file's ranks in order
            std::vector<int32_t> main_slots, peer_slots;
            for (const auto& pr : profile) {
                const size_t i = (size_t) pr.first * g.n_expert + pr.second;
                if (i >= res_at_start.size()) continue;
                if (res_at_start[i] >= 0) main_slots.push_back(res_at_start[i]);
                else if (i < peer_at_start.size() && peer_at_start[i] >= 0) peer_slots.push_back(peer_at_start[i]);
            }
            std::vector<uint8_t> taken(res_at_start.size(), 0);
            size_t k = 0;
            for (const auto& pr : q) {
                const size_t i = (size_t) pr.first * g.n_expert + pr.second;
                if (k >= main_slots.size()) break;
                if (i >= taken.size() || taken[i]) continue;
                taken[i] = 1;
                want_main.emplace_back(main_slots[k++], (int32_t) i);
            }
            k = 0;
            for (const auto& pr : q) {
                const size_t i = (size_t) pr.first * g.n_expert + pr.second;
                if (k >= peer_slots.size()) break;
                if (i >= taken.size() || taken[i]) continue;
                taken[i] = 1;
                want_peer.emplace_back(peer_slots[k++], (int32_t) i);
            }
            rank_new.assign(rank_new.size(), (int32_t) q.size());
            for (size_t r = 0; r < q.size(); ++r) {
                const size_t i = (size_t) q[r].first * g.n_expert + q[r].second;
                if (i < rank_new.size() && rank_new[i] == (int32_t) q.size()) rank_new[i] = (int32_t) r;
            }
        }
        if (!ver.wait_commit(e)) return false;
        apply_pending(true);
        if (!xq_drain()) { e = "PROBE adapt: an adaptive swap failed"; return false; }
        if (peer.valid()) peer.apply_pending(true);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        // the cells a longer prompt backed go back to the expert cache first: the slots it had at READY
        if (!kv_release(ss, 0, e)) { e = "PROBE adapt: " + e; return false; }
        // the main card: an expert leaving a slot without a RAM place takes a free one first (as a round's swap)
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const int64_t live_slots = xcache.slots();
        std::vector<int32_t> holder((size_t) live_slots, -1);   // slot -> residency index
        for (size_t i = 0; i < host_res.size(); ++i)
            if (host_res[i] >= 0 && host_res[i] < live_slots) holder[(size_t) host_res[i]] = (int32_t) i;
        // the target is the set of experts: one already in a slot stays there, each missing one takes the slot of
        // an expert outside the set (its own target slot first)
        std::vector<std::pair<int32_t, int32_t>> todo, later;
        const auto pair_up = [](const std::vector<std::pair<int32_t, int32_t>>& want, const std::vector<int32_t>& res,
                                const std::vector<int32_t>& holder, std::vector<std::pair<int32_t, int32_t>>& out) {
            std::vector<uint8_t> in_set(res.size(), 0);
            for (const auto& w : want)
                if (w.first >= 0 && w.first < (int32_t) holder.size()) in_set[(size_t) w.second] = 1;
            std::vector<uint8_t> free_slot(holder.size(), 0);
            for (size_t sl = 0; sl < holder.size(); ++sl) free_slot[sl] = holder[sl] < 0 || !in_set[(size_t) holder[sl]];
            std::vector<int32_t> missing;
            for (const auto& w : want) {
                if (!in_set[(size_t) w.second] || res[(size_t) w.second] >= 0) continue;
                if (free_slot[(size_t) w.first]) { out.push_back(w); free_slot[(size_t) w.first] = 0; }
                else missing.push_back(w.second);
            }
            size_t sl = 0;
            for (int32_t i : missing) {
                while (sl < free_slot.size() && !free_slot[sl]) ++sl;
                if (sl == free_slot.size()) break;
                out.emplace_back((int32_t) sl, i);
                free_slot[sl] = 0;
            }
        };
        pair_up(want_main, host_res, holder, todo);
        struct Mv { int32_t slot, in, out; uint64_t off; bool xchg; };
        std::vector<Mv> mv;
        int64_t moved = 0, kept = 0;
        for (int pass = 0; pass < 2 && !todo.empty(); ++pass) {   // an expert still resident somewhere waits a pass
            later.clear();
            size_t at = 0;
            while (at < todo.size()) {
                mv.clear();
                for (; at < todo.size(); ++at) {
                    const auto [slot, in] = todo[at];
                    if (host_res[(size_t) in] >= 0) { later.push_back(todo[at]); continue; }
                    const int32_t out = holder[(size_t) slot];
                    const int64_t li = in / g.n_expert, ei = in % g.n_expert;
                    const bool xchg = out >= 0 && src.complement_ready() && src.has_resident(li, ei) &&
                                      !src.has_resident(out / g.n_expert, out % g.n_expert);
                    uint64_t off = 0;
                    if (xchg) {
                        uint8_t* dst = src.spare_take(out / g.n_expert, off);
                        if (dst == nullptr) {
                            if (mv.empty()) { e = "PROBE adapt: no free RAM place for an evicted expert"; return false; }
                            break;
                        }
                        if (cudaMemcpyAsync(dst, xcache.device_slot(slot), (size_t) lay.blob_bytes(out / g.n_expert),
                                            cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                            e = std::string("PROBE adapt: an eviction copy failed: ") + cudaGetErrorString(cudaGetLastError());
                            return false;
                        }
                    }
                    mv.push_back({slot, in, out, off, xchg});
                }
                if (mv.empty()) break;
                if (cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
                    e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                for (const Mv& m : mv) {
                    if (m.out < 0) continue;
                    const int64_t lo = m.out / g.n_expert, eo = m.out % g.n_expert;
                    if (m.xchg && !src.spare_land(lo, eo, m.off)) {
                        e = "PROBE adapt: an evicted expert has no RAM place";
                        return false;
                    }
                    host_res[(size_t) m.out] = basalt::core::kNotResident;
                    holder[(size_t) m.slot] = -1;
                    srcp->prefetch(lo, eo);
                }
                for (const Mv& m : mv) {
                    const int64_t li = m.in / g.n_expert;
                    const uint8_t* b = srcp->blob(li, m.in % g.n_expert);
                    if (b == nullptr || cudaMemcpyAsync(xcache.device_slot(m.slot), b, (size_t) lay.blob_bytes(li),
                                                        cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                        e = "PROBE adapt: an expert's copy into its slot failed";
                        return false;
                    }
                }
                if (cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
                    e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                for (const Mv& m : mv) {
                    const int64_t li = m.in / g.n_expert, ei = m.in % g.n_expert;
                    host_res[(size_t) m.in] = m.slot;
                    holder[(size_t) m.slot] = m.in;
                    if (m.xchg) src.spare_release(li, ei);
                    srcp->release(li, ei);
                    ++moved;
                }
            }
            todo.swap(later);
        }
        kept += (int64_t) todo.size();
        res_upload();
        relend();
        int64_t peer_moved = 0;
        if (peer.valid()) {
            peer.apply_pending(true);
            const std::vector<int32_t>& pres = peer.residency();
            std::vector<int32_t> pholder((size_t) peer.cache().slots(), -1);
            for (size_t i = 0; i < pres.size(); ++i)
                if (pres[i] >= 0 && pres[i] < (int32_t) pholder.size()) pholder[(size_t) pres[i]] = (int32_t) i;
            std::vector<std::pair<int32_t, int32_t>> pwant, pw;
            for (const auto& sw : want_peer)
                if (host_res[(size_t) sw.second] < 0) pwant.push_back(sw);
                else ++kept;   // the main card holds it
            pair_up(pwant, pres, pholder, pw);
            if (!peer.place(pw, peer_moved, e)) { e = "PROBE adapt: " + e; return false; }
        }
        // the adaptation and the draft policy start again, as after a load
        std::fill(drive.d.usage.begin(), drive.d.usage.end(), 0.0f);
        std::fill(heat.begin(), heat.end(), 0.0);
        rounds = 0;
        ba_win = 0;
        xq_saturated = false;
        prof_rank = rank_new;
        policy = basalt::spec::DraftPolicy(S);
        bpolicy = basalt::spec::DraftPolicy(S);
        // a request after this reads its whole prompt
        ckd_finish();
        live.clear();
        live_imgs.clear();
        live_ok = false;
        checks.clear();
        for (BSlot& sl : bs)
            if (!sl.active) sl.cached = false;
        (void) conversations.take_reuse(rsp);
        while (conversations.size() > 0) (void) conversations.take(0);
        o.adapt_every = every;
        o.adapt_swaps = swaps_n;
        o.adapt_decay = decay;
        o.adapt_decay_set = true;
        fast_decay = decay;
        std::printf("PROBE end adapt every=%d swaps=%d decay=%.3f moved=%lld peer_moved=%lld kept=%lld ms=%.0f\n",
                    every, swaps_n, (double) decay, (long long) moved, (long long) peer_moved, (long long) kept,
                    std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        return true;
    };
    for (;;) {
        if (batch_on() || (piped && pipe_inflight())) {
            if (!try_next_line(line)) {
                if (!(piped ? pump(true) : batch_step())) return 1;
                continue;
            }
            // a request reads its prompt through every stage: the groups in flight finish first
            if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return 1;
        } else {
            // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
            // lose goes to disk, one conversation at a time while no line waits
            bool got = false;
            if (disk.enabled()) {
                const int r = next_line_within(line, 1000);
                if (r < 0) break;
                got = r == 1;
                while (!got && !(got = try_next_line(line)) && spill_to_disk(false, 1) > 0) {}
            }
            if (!got && !next_line(line)) break;
        }
        // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
        // request's prompt and first token as a GEN 1, then continues it in that slot
        if (line.rfind("BSTOP ", 0) == 0) {
            const int b = std::atoi(line.c_str() + 6);
            if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
            continue;
        }
        if (line.rfind("BSCRIPT ", 0) == 0) {   // BSCRIPT <slot> <seed> <id,id,...>: before the slot's BGEN
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + 8, &e1, 10);
            char* e2 = nullptr;
            const unsigned long long seed = std::strtoull(e1, &e2, 10);
            std::vector<int64_t> sids;
            std::string se;
            if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
                std::printf("ERR BSCRIPT: bad slot or list\n");
                std::fflush(stdout);
                continue;
            }
            bscript[(size_t) b].first.assign(sids.begin(), sids.end());
            bscript[(size_t) b].second = seed;
            continue;
        }
        if (line.rfind("BYIELD", 0) == 0) continue;   // for a prompt read that has ended meanwhile
        if (line.rfind("ENC ", 0) == 0) {   // the in-engine image encoder, between requests and windows
            enc_command(line);
            continue;
        }
        if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
            std::string perr;
            if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
                if (batch_on()) perr = "PROBE: not while batch slots are decoding";
                else {
                    if (ba_job) { ba_thr.join(); ba_job = false; }
                    std::istringstream kvs(line.substr(10));
                    for (std::string kv; kvs >> kv;) {
                        const size_t eq = kv.find('=');
                        const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                        const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                        if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                        else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                        else perr = "PROBE: bad setting " + kv;
                    }
                    if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
                }
            } else if (line == "PROBE adapt" || line.rfind("PROBE adapt ", 0) == 0) {
                if (batch_on()) perr = "PROBE: not while batch slots are decoding";
                else {
                    if (ba_job) { ba_thr.join(); ba_job = false; }
                    (void) probe_adapt(line, perr);
                }
            } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates, PROBE set or PROBE adapt";
            else if (batch_on()) perr = "PROBE: not while batch slots are decoding";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                basalt::program::ProbeEnv penv;
                penv.pool = &pool;
                penv.src = srcp;
                penv.peer = peer.valid() ? &peer : nullptr;
                penv.n_layers = g.n_layers;
                penv.n_expert = g.n_expert;
                basalt::program::probe_rates(penv, stdout, perr);
            }
            if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
            std::string verr;
            if (batch_on()) verr = "VRAM: not while batch slots are decoding";
            else vram_command(line, verr, false);
            if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
            std::fflush(stdout);
            continue;
        }
        // SLEEP / WAKE, between requests: the expert cache gives its VRAM back down to its smallest size (every
        // expert keeps a RAM copy and the slot it held is remembered), so the process can be suspended with little
        // VRAM left to save; WAKE maps the cache again and puts each expert back in its slot.  Reply: a VRAM line.
        if (line == "SLEEP" || line == "WAKE") {
            std::string verr;
            const bool down = line == "SLEEP";
            if (batch_on()) {
                verr = line + ": not while batch slots are decoding";
            } else {
                // A peer card's tier goes too.  SLEEP then drops the RAM copy's registration with the driver (a process
                // checkpoint would map the copy into every context again on restore); WAKE registers it again while
                // both caches refill from the ranges already back.  A tier or a registration that cannot come back
                // fails WAKE: the server then starts the engine again.
                std::string perr, reply, uerr;
                const auto tp = Clock::now();
                const bool repin = !down && src.complement_unpinned();
                std::function<bool(const void*, uint64_t)> ready;
                if (repin) {
                    src.repin_start(4);
                    ready = [&src](const void* p, uint64_t n) { return src.repin_wait(p, n); };
                }
                const bool peer_ok = !peer.valid() || (down ? peer.sleep(perr) : peer.wake(perr, ready));
                vram_command(down ? "VRAM " + std::to_string(1ll << 30) : std::string("VRAM"), verr, true, &reply);
                if (repin) {
                    if (!src.repin_finish(uerr) && verr.empty()) verr = "WAKE: " + uerr;
                    basalt::log::info("basalt serve", "WAKE: the RAM copy registered again beside the caches' refill "
                                      "in %.0f ms", std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
                }
                if (down && verr.empty() && src.complement_ready()) {
                    const auto tu = Clock::now();
                    if (src.unpin_complement(uerr))
                        basalt::log::info("basalt serve", "SLEEP: the RAM copy's registration dropped in %.0f ms",
                                          std::chrono::duration<double, std::milli>(Clock::now() - tu).count());
                    else
                        basalt::log::info("basalt serve", "SLEEP: the RAM copy stays registered (%s)", uerr.c_str());
                }
                if (!down && peer.valid() && peer_ok && !peer.wake_wait(perr)) {
                    if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
                } else if (!down && !peer_ok) {
                    if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
                } else if (!peer_ok) {
                    basalt::log::warn("basalt serve", "SLEEP: the peer tier stays as it is (%s)", perr.c_str());
                } else if (peer.valid()) {
                    basalt::log::info("basalt serve", "%s: the peer tier %s; %.0f ms with the main cache",
                                      line.c_str(), down ? "gave its VRAM back" : "is back",
                                      std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
                }
                if (verr.empty()) std::fputs(reply.c_str(), stdout);
            }
            // asleep, the driver holds the engine's GPU state in RAM until the wake: the RAM guard judges no reading
            // then (no shed, no admission wait from it), and judges a fresh one at the wake
            if (down && verr.empty()) basalt::platform::ram_guard_pause(true);
            if (!down) basalt::platform::ram_guard_pause(false);
            if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("ERR_GENT ", 0) == 0) {
            std::printf("ERR %s\n", line.c_str() + 9);
            std::fflush(stdout);
            continue;
        }
        basalt::platform::ram_guard_pause(false);   // a request without a WAKE first: the guard judges again
        admit_slot = -1;
        solo_slot = -1;
        if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
            const bool simg = line.rfind("SGENI ", 0) == 0;
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
            if (!slot0_is_solo || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
                std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
                std::fflush(stdout);
                continue;
            }
            solo_slot = (int) b;
            line = std::string(simg ? "GENI" : "GEN") + e1;
        }
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            const bool bimg = line.rfind("BGENI ", 0) == 0;
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
            char* e2 = nullptr;
            const long long mn = std::strtoll(e1, &e2, 10);
            if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
                std::fflush(stdout);
                continue;
            }
            admit_slot = (int) b;
            admit_max_new = mn;
            line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
        }
        // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
        if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
            Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
            save_profile("periodic");
        if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
        if (line == "QUIT") break;
        // the watchdog watches a request from here until this iteration ends, whichever way it ends
        struct BusyScope {
            BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
            ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
        } busy_scope;
        stop_req.store(false);   // a STOP that arrived between requests is stale
        ram_stop.store(false);
        err.clear();
        if (mtp_on && !mtp.use_main(err)) { std::printf("ERR %s\n", err.c_str()); return 1; }   // (batch windows swap the draft K/V)
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...>, GENI <max_new> <file> <id,id,...> or GENT <max_new> <nbytes> + text\n");
            continue;
        }
        char* endp = nullptr;
        long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
        // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
        // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
        // only).  Absent keys keep today's behavior: greedy, no penalties.
        float req_temperature = 0.0f, req_top_p = 1.0f;
        int req_top_k = 20;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
        unsigned long long req_seed = 0;
        float req_min_p = 0.0f, req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
        int req_penalty_last_n = 0;
        std::string req_bias_text;   // logit_bias=id:value,...: added to the verifier's logits (see ver.set_logit_bias)
        bool req_bias_present = false;
        int64_t req_gprompt = -1;       // gprompt=N: the first N ids are the prompt, the rest tokens the request has already generated
        std::string req_grammar_text;   // grammar=<compact ASCII JSON, spaces as \u0020>: the request's grammar (--grammar on)
        int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
        float req_relax = o.relaxed_drafting;   // relax=P: this request's --relaxed-drafting (0 = exact)
        // ppl=K [ppl_t=T]: quality gate (basalt/program/ppl_gate.hpp) - chunk K of --ppl-base, the rest of
        // the chunk teacher-forced through verify windows of T tokens, every token committed
        int req_ppl = -1, req_ppl_t = 3;
        int req_clip = 0;   // clip=1 (GENT): max_new is a cap, cut to the room the context leaves after the prompt
        if (endp != nullptr) {   // GENI takes the same keys (image requests were always greedy); its
                                 // embedding file path is the first token without an =
            for (;;) {
                while (*endp == ' ') ++endp;
                const char* start = endp;
                while (*endp != '\0' && *endp != ' ') ++endp;
                if (endp == start) break;
                const std::string tok(start, (size_t) (endp - start));
                const size_t eq = tok.find('=');
                if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                const std::string key = tok.substr(0, eq);
                const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
                if (key == "logit_bias") { req_bias_text = tok.substr(eq + 1); req_bias_present = true; }
                else if (key == "grammar") req_grammar_text = tok.substr(eq + 1);
                else if (key == "gprompt") req_gprompt = std::atoll(tok.c_str() + eq + 1);
                else if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
                else if (key == "temperature") req_temperature = fv;
                else if (key == "top_p") req_top_p = fv;
                else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
                else if (key == "min_p") req_min_p = fv;
                else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
                else if (key == "penalty_repeat") req_penalty_repeat = fv;
                else if (key == "penalty_freq") req_penalty_freq = fv;
                else if (key == "penalty_present") req_penalty_present = fv;
                else if (key == "seed") req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
                else if (key == "relax") req_relax = std::clamp(fv, 0.0f, 1.0f);
                else if (key == "clip") req_clip = std::atoi(tok.c_str() + eq + 1);
                else if (key == "ppl") req_ppl = std::atoi(tok.c_str() + eq + 1);
                else if (key == "ppl_t") req_ppl_t = std::atoi(tok.c_str() + eq + 1);
                // unknown keys are skipped: the ids start at the first token without '='
            }
        }
        std::vector<float> req_bias;
        if (req_bias_present) {
            std::string bias_error;
            if (o.batch > 0) bias_error = "continuous batching is not supported";
            const bool ok = o.batch <= 0 && basalt::core::parse_logit_bias(req_bias_text, (int) n_vocab, req_bias, bias_error);
            if (!ok) { std::printf("ERR logit_bias: %s\n", bias_error.c_str()); continue; }
        }
        std::string emb_path;
        if (geni && endp != nullptr) {
            while (*endp == ' ') ++endp;
            char* gap = std::strchr(endp, ' ');
            if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
        }
        std::vector<int64_t> ids;
        std::string pe;
        bool ids_ok = false;
        bool from_text = false;   // the ids came from a GENT request's text
        if (endp != nullptr) {
            while (*endp == ' ') ++endp;
            if (*endp == '@') {
                std::lock_guard<std::mutex> lk(gent_mu);
                const auto it = gent_ids.find(std::strtoull(endp + 1, nullptr, 10));
                if (it != gent_ids.end()) {
                    ids.assign(it->second.begin(), it->second.end());
                    gent_ids.erase(it);
                    ids_ok = from_text = true;
                } else {
                    pe = "the tokenized prompt is gone";
                }
            } else {
                ids_ok = cli::parse_i64_list(endp, ids, pe);
            }
        }
        if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !ids_ok) {
            std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
            continue;
        }
        const int64_t n = (int64_t) ids.size();
        std::unique_ptr<basalt::program::grammar::Guard> gguard;
        if (!req_grammar_text.empty() && grammar_vocab) {
            std::string gerr;
            if (o.batch > 0 && piped) gerr = "pipelined batch groups are not supported";
            else {
                gguard = std::make_unique<basalt::program::grammar::Guard>(*grammar_vocab);
                const size_t gp = req_gprompt > 0 && (size_t) req_gprompt < ids.size() ? (size_t) req_gprompt : ids.size();
                if (!gguard->begin(req_grammar_text, ids.data(), gp, gerr)) gguard.reset();
                else if (gp < ids.size() && gguard->on()) {
                    const std::vector<int32_t> done(ids.begin() + (std::ptrdiff_t) gp, ids.end());
                    if (gguard->scan(done.data(), (int) done.size()) >= 0) {
                        gerr = "the tokens already generated do not satisfy the grammar";
                        gguard.reset();
                    }
                }
                else if (!gguard->on()) gguard.reset();
            }
            if (!gerr.empty()) { std::printf("ERR grammar: %s\n", gerr.c_str()); continue; }
        } else if (!req_grammar_text.empty()) {
            basalt::log::warn("basalt grammar", "the request has a grammar but this engine runs with --grammar off; ignored");
        }
        std::vector<float> grammar_bias;
        int64_t grammar_resamples = 0;
        req_imgs.clear();
        if (geni && !o.vision) { std::printf("ERR this engine was started without --vision\n"); continue; }
        basalt::program::PplGate* ppl = nullptr;
        if (req_ppl >= 0) {
            std::string pe2;
            basalt::program::PplGate::configure(ppl_base, ppl_out);
            ppl = &basalt::program::PplGate::get();
            if (geni || !ppl->begin(req_ppl, ids.data(), n, n_vocab, pe2)) {
                std::printf("ERR %s\n", geni ? "ppl: text only" : pe2.c_str());
                continue;
            }
            req_temperature = 0.0f;
            req_ppl_t = std::clamp(req_ppl_t, 1, S);
        }
        if (geni || !mrope_identity) {
            // positions for every cell this request can reach; the identity again for a text request
            std::string ve;
            row_ptr.assign((size_t) n, nullptr);
            const int64_t cells = (int64_t) mrope_host.size() / 3;
            auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                mrope_host[(size_t) c * 3] = (int32_t) t;
                mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
            };
            if (!geni) {
                for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
            } else {
                struct Img { int64_t n, nx, ny; size_t off; };
                std::vector<Img> imgs;
                img_rows.clear();
                std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                if (!f) ve = "cannot open " + emb_path;
                while (f && ve.empty()) {
                    int32_t hdr[5];
                    const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                    if (got == 0) break;
                    if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                        (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                        ve = "bad embeddings file (expected basalt-vision records of width " +
                             std::to_string((long long) g.n_embd) + ")";
                        break;
                    }
                    const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                    img_rows.resize(off + cnt);
                    if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                    imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                }
                if (f) std::fclose(f);
                int64_t p = 0, i = 0;
                size_t k = 0;
                while (ve.empty() && i < n) {
                    if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                    if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                    const Img& im = imgs[k++];
                    {   // what the conversation cache compares: a picture is its grid and its embeddings
                        const int64_t grid[3] = {im.n, im.nx, im.ny};
                        uint64_t h = fnv1a(grid, sizeof grid);
                        h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                        req_imgs.push_back({i, h});
                    }
                    for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                        if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                            ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                 " rows but fewer <|image_pad|> tokens";
                    for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                        const int64_t y = j / im.nx, x = j % im.nx;
                        put(i + j, p, p + y, p + x);
                        row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                    }
                    i += im.n;
                    p += std::max(im.nx, im.ny);
                }
                if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
            }
            cudaDeviceSynchronize();
            // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
            auto upload_mrope = [&]() -> bool {
                bool ok = cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice) == cudaSuccess;
                for (auto& st : stages) {
                    const basalt::core::OnDevice on(st->dev);
                    cudaDeviceSynchronize();
                    ok = ok && cudaMemcpy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                          cudaMemcpyHostToDevice) == cudaSuccess;
                }
                return ok;
            };
            if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
            if (!ve.empty()) {
                // leave the table as the identity so the next text request is untouched
                for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                upload_mrope();
                mrope_identity = true;
                std::printf("ERR %s\n", ve.c_str());
                std::fflush(stdout);
                continue;
            }
            mrope_identity = !geni;
        }
        sp.embd_rows = geni ? row_ptr.data() : nullptr;
        if (from_text && req_clip && admit_slot < 0) {
            const long long room = (long long) (o.max_context - 8 - n);
            if (room < 1) {
                std::printf("ERR prompt (%lld tokens) leaves no room to answer in the context (%lld)\n", (long long) n,
                            (long long) o.max_context);
                std::fflush(stdout);
                continue;
            }
            max_new = std::min(max_new, room);
        }
        {
            const int64_t ctx = admit_slot >= 0 ? slot_cap[(size_t) admit_slot] : o.max_context;
            const long long want = admit_slot >= 0 ? admit_max_new : (long long) max_new;   // a slot's request: its own max_new
            if (n + want + 8 > ctx) {
                if (admit_slot >= 0)
                    std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds slot %d's context (%lld)\n", (long long) n,
                                want, admit_slot, (long long) ctx);
                else
                    std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                                want, (long long) ctx);
                std::fflush(stdout);
                admit_slot = -1;
                continue;
            }
        }
        if (from_text) { std::printf("PT %lld %lld\n", (long long) n, (long long) max_new); std::fflush(stdout); }
        bool bad = false;
        for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); continue; }
        std::array<int64_t, 3> remote_before{};
        std::array<int64_t, 3> launches_before{};
        std::array<uint64_t, 3> compact_before{}, full_before{};
        // ms_begin/ms_wait are cumulative since boot; take deltas here like every other column, so the host time
        // reads as belonging to this request.
        std::array<double, 3> begin_before{}, wait_before{};
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        {
            remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
            launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
            compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
            full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
            begin_before[(size_t) r] = remote_experts[(size_t) r].ms_begin();
            wait_before[(size_t) r] = remote_experts[(size_t) r].ms_wait();
        }
        cur = ids;
        // The conversation cache's keys are int32: one narrow view per request, so a parked entry or a checkpoint
        // is matched by one bulk compare over the slice instead of a per-token scan.
        const std::vector<int32_t> ids32(ids.begin(), ids.end());
        const Clock::time_point r0 = Clock::now();
        if (!drive.d.usage.empty()) sp.route_hist_reset();   // counts this request's prompt rows for the usage seed
        // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
        // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
        auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
            const int64_t L = (int64_t) pre.size();
            if (L < 1 || L > n - 1) return false;
            if (!basalt::core::conversation_ids_equal(pre.data(), pre.size(), ids32.data())) return false;
            return imgs_below(req_imgs, L) == pre_imgs;
        };
        const bool want_cvec = basalt::kernels::cvec().loaded() ? req_cvec != 0 : true;
        // the last request's final commit may still be running on the verifier's stream (set_commit_async):
        // everything below reads, restores or zeroes the session from other streams and the host (the end of the
        // last request waited already; this covers a request that ended on an error path)
        if (!ver.wait_commit(err)) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        // A request for a slot other than 0 reads its prompt in that slot: the request path works on a view of the
        // solo session that carries the slot's state, with the slot's own conversation bookkeeping swapped in
        // for the request (the solo one is put back after it)
        const int slot_b = solo_slot >= 0 ? solo_slot : admit_slot;
        const bool in_slot = slot0_is_solo && slot_b > 0;
        const bool solo_req = in_slot && solo_slot > 0;   // decoded here, solo-style, on the slot's own state
        basalt::core::SessionState& rview = rviews[(size_t) std::max(slot_b, 0)];
        struct SoloBook {
            std::vector<int32_t> live;
            std::vector<ImgKey> live_imgs;
            bool live_ok = false, cvec = true;
            std::vector<ConvCheckpoint> checks;
        } solo_book;
        bool solo_moved = false;   // the solo session gave its cells to this request (hold_slot): it holds nothing after
        bool own_partial = false, own_partial_from0 = false;   // the slot holds a read that gave way (BYIELD)
        int64_t own_partial_chunk = 0;
        if (in_slot) {
            solo_book = {std::move(live), std::move(live_imgs), live_ok, cvec_cached, std::move(checks)};
            BSlot& sb = bs[(size_t) slot_b];
            live.clear();
            live_imgs.clear();
            checks.clear();
            live_ok = false;
            cvec_cached = sb.cvec;
            if (sb.cached) {
                live = sb.ids;
                live_imgs = sb.imgs;
                live_ok = o.prompt_cache > 0;
                checks = std::move(sb.checks);
                if (batch_spec && !mtp.copy_store(slot_b, mtp.main_store(), (int64_t) live.size(), err)) {
                    err.clear();
                    live_ok = false;
                }
                own_partial = sb.partial;
                own_partial_from0 = sb.partial_from0;
                own_partial_chunk = sb.partial_chunk;
            }
            sb.cached = false;
            sb.partial = false;
            sb.checks.clear();
            make_view(rview, slot_b);
            rsp = &rview;
            if (solo_req && !ver.bind_session(&rview, err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            sp.set_session(rview);
            if (mtp_on) mtp.set_ple_session(&rview);
        }
        // slot 0 is the solo session: what an idle slot 0 holds is the live conversation, in place
        if (slot0_is_solo && !in_slot && !bs.empty() && bs[0].cached && !bs[0].active) {
            BSlot& s0 = bs[0];
            live = std::move(s0.ids);
            live_imgs = std::move(s0.imgs);
            live_ok = o.prompt_cache > 0;
            cvec_cached = s0.cvec;
            for (ConvCheckpoint& c : s0.checks) checks.push_back(std::move(c));
            s0.checks.clear();
            s0.cached = false;
            if (batch_spec && !mtp.copy_store(0, mtp.main_store(), (int64_t) live.size(), err)) {   // the draft K/V with it
                err.clear();
                live_ok = false;
            }
        }
        int64_t resume = 0;
        bool from_live = false;
        if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
            if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
            for (const ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                    resume = (int64_t) c.ids.size();
                    from_live = false;
                }
        }
        // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a
        // source too - its sessions are copied back below, so only the new part is read
        int slot_source = -1;
        int64_t slot_tokens = 0;
        bool resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
        int64_t resumed_chunk = 0;    // its chunk, if it was read beside decoding slots: its first part's plan
        const ConvCheckpoint* slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
        // the slot this request reads into: its own conversation is `live` already
        const int own_slot = in_slot ? slot_b : slot0_is_solo ? 0 : -1;
        // A decoding slot's checkpoints are sources too: the K/V before a checkpoint never changes, so requests that
        // share a start (a system prompt) read it once, whichever slot holds it.
        if (o.prompt_cache > 0)
            for (int b = 0; b < (int) bs.size(); ++b) {
                const BSlot& sl = bs[(size_t) b];
                if (b == own_slot || (!sl.active && !sl.cached) || sl.cvec != want_cvec) continue;
                if (!in_slot && !sl.active && (int64_t) sl.ids.size() > std::max(resume, slot_tokens) &&
                    starts_with(sl.ids, sl.imgs)) {
                    slot_source = b;
                    slot_tokens = (int64_t) sl.ids.size();
                    slot_ck = nullptr;
                }
                for (const ConvCheckpoint& c : sl.checks)
                    if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                        slot_source = b;
                        slot_tokens = (int64_t) c.ids.size();
                        slot_ck = &c;
                    }
            }
        // a request in another slot: the solo session's chain is slot 0's conversation, a source like the rest
        if (o.prompt_cache > 0 && in_slot && slot0_is_solo && solo_book.cvec == want_cvec)
            for (const ConvCheckpoint& c : solo_book.checks)
                if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                    slot_source = 0;
                    slot_tokens = (int64_t) c.ids.size();
                    slot_ck = &c;
                }
        const auto parked = conversations.best(ids32, req_imgs, want_cvec);
        std::optional<basalt::core::SavedConversation> incoming;
        int64_t in_tokens = parked.tokens;   // where the incoming image continues: the RAM tier's match or the disk's
        bool in_live = parked.live;
        // the disk tier, when it holds a longer start than every other source: read back cut at its match, after the
        // RAM guard's wait below (the read takes RAM)
        const basalt::core::ConversationStore::Match on_disk =
            disk.enabled() ? disk.best(ids32.data(), ids32.size(), req_imgs, want_cvec)
                           : basalt::core::ConversationStore::Match{};
        const bool from_disk = on_disk.tokens > std::max({resume, slot_tokens, parked.tokens});
        if (!from_disk && parked.tokens > std::max(resume, slot_tokens)) incoming.emplace(conversations.take(parked.index));
        // RAM below the guard's floor (another program's peak): shed (after the conversation this request
        // continues is taken out of the cache), then let it come back before the request starts; a STOP ends the
        // wait, and the request then ends as cancelled.  A request that only continues the live conversation by a
        // few tokens (the server's wrap-up after a thinking budget) is part of a reply already streaming: no wait.
        if (!batch_on() && basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok) {
            shed_if_low();
            if (!basalt::platform::ram_continues_live(from_live && !incoming && !from_disk && slot_source < 0,
                                                       n - resume)) {
                uint64_t avail = 0, floor = 0;
                (void) basalt::platform::ram_guard_level(&avail, &floor);
                basalt::log::info("basalt serve", "RAM low (%llu MiB free < %llu MiB guard): the request waits up "
                                  "to %llu s", (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20),
                                  (unsigned long long) (basalt::platform::kRamAdmitWaitMs / 1000));
                basalt::core::progress_at("RAM wait");
                const auto w0 = Clock::now();
                auto waited = [&] {
                    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - w0).count();
                };
                while (!stop_req.load() && !stdin_lost.load() &&
                       basalt::platform::ram_admit_waits(basalt::platform::ram_guard_level(), waited())) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    basalt::core::progress_beat();
                }
                // a STOP during the wait ends the request here, before anything moves for it: the conversation it
                // would continue goes back to the cache, the live one stays as it is (a batch slot's request ends
                // in its prompt read, which hands the slot back)
                if ((stop_req.load() || stdin_lost.load()) && !in_slot && admit_slot < 0) {
                    if (incoming) (void) conversations.put(std::move(*incoming));
                    basalt::log::info("basalt serve", "RAM wait: the request was stopped after %.1f s",
                                      waited() / 1000.0);
                    basalt::log::info("basalt serve", "prompt %lld tokens = 0 reused + 0 of %lld read in 0 ms "
                                      "(0.0 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, %zu "
                                      "checkpoints (cancelled)", (long long) n, (long long) n, checks.size());
                    std::printf("DONE 0 %lld 0.0 0.0 cancel 0 0 0 0 0 0 0 0.0 0 0\n", (long long) n);
                    std::fflush(stdout);
                    continue;
                }
                const basalt::platform::RamLevel level = basalt::platform::ram_guard_level(&avail, &floor);
                if (level == basalt::platform::RamLevel::Ok)
                    basalt::log::info("basalt serve", "RAM back after %.1f s (%llu MiB free): the request starts",
                                      waited() / 1000.0, (unsigned long long) (avail >> 20));
                else if (!stop_req.load() && !stdin_lost.load())
                    basalt::log::warn("basalt serve", "RAM still low after %.1f s (%llu MiB free < %llu MiB guard): "
                                      "the request starts", waited() / 1000.0, (unsigned long long) (avail >> 20),
                                      (unsigned long long) (floor >> 20));
                basalt::core::progress_at("request");
            }
        }
        if (from_disk) {
            const uint64_t need = disk.load_bytes(on_disk);
            const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
            const auto t0 = Clock::now();
            basalt::core::SavedConversation image;
            std::string de;
            if (!capture_admit(need, floor)) {
                basalt::log::info("basalt serve", "conversation cache: %lld tokens on disk not read (physical RAM "
                                  "admission; need %llu MiB)", (long long) on_disk.tokens,
                                  (unsigned long long) (need >> 20));
            } else if (!disk.load(on_disk, image, de)) {
                basalt::log::warn("basalt serve", "conversation cache: reading from disk failed (%s); the file is "
                                  "dropped", de.c_str());
            } else {
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                basalt::log::info("basalt serve", "conversation cache: read %lld tokens from disk in %.1f ms "
                                  "(%.2f GB of state)", (long long) on_disk.tokens, ms, (double) image.bytes() / 1e9);
                incoming.emplace(std::move(image));
                in_tokens = on_disk.tokens;
                in_live = true;
            }
            if (!incoming) {   // not read: the RAM tier's match, when the shed left one
                const auto again = conversations.best(ids32, req_imgs, want_cvec);
                if (again.tokens > std::max(resume, slot_tokens)) {
                    incoming.emplace(conversations.take(again.index));
                    in_tokens = again.tokens;
                    in_live = again.live;
                }
            }
        }
        if (incoming) slot_source = -1;
        // a read that gave way in this slot, the same request again: on with it as if it had not stopped
        if (own_partial && from_live && !incoming && slot_source < 0) {
            resumed_from0 = own_partial_from0;
            resumed_chunk = own_partial_chunk;
        }
        // Reject the entire image before parking/overwriting the outgoing
        // state. Invalid entries can safely fall back to its existing prefix.
        if (incoming && incoming->stage_images.size() != stages.size()) {
            basalt::log::info("basalt serve", "conversation cache: discard snapshot (stage count)");
            incoming.reset();
        }
        if (incoming) for (size_t i = 0; i < stages.size(); ++i) {
            const basalt::core::OnDevice on(stages[i]->dev);
            if (!basalt::core::conversation_snapshot_validate(incoming->stage_images[i], stages[i]->ss, g,
                    nullptr, err)) {
                basalt::log::warn("basalt serve", "conversation cache: discard invalid stage snapshot (%s)",
                             err.c_str());
                incoming.reset();
                err.clear();
                break;
            }
        }
        if (incoming && !basalt::core::conversation_snapshot_validate(*incoming, *rsp, g,
                mtp_kv, err)) {
            basalt::log::warn("basalt serve", "conversation cache: discard invalid snapshot (%s)", err.c_str());
            incoming.reset();
            err.clear();
        }
        // Preserve the outgoing branch before any checkpoint rewind, reset,
        // or incoming restore overwrites the positional state it requires.
        // the elastic KV: every cell this request writes is backed before anything (a restore, a slot's copy, the
        // read) writes it
        // (a restore writes the snapshot's whole live K/V, which can be longer than this prompt)
        const int64_t kv_need = std::max<int64_t>(n, incoming ? (int64_t) incoming->live.ids.size() : 0);
        struct KeepRouted {   // only this admission's fits, while slots decode
            bool& on;
            ~KeepRouted() { on = false; }
        } keep_routed{kv_keep_routed};
        kv_keep_routed = batch_on();
        bool kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
        if (!kv_ok && kv_short) {   // the other sessions' spare cells first
            int64_t given = 0;
            std::string ge;
            if (!kv_give_back(in_slot ? (int64_t) solo_book.live.size() : (int64_t) live.size(), rsp,
                              in_slot ? slot_b : -1, nullptr, given, ge)) {
                err = ge;
                kv_short = false;
            } else if (given > 0) {
                err.clear();
                kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
            }
        }
        // the idle slot this request continues: its conversation waits in RAM and its cells come here, so the two
        // copies never hold VRAM at once
        if (!kv_ok && kv_short && slot_source >= 0 && !bs[(size_t) slot_source].active &&
            (bslot_ss[0][(size_t) slot_source].get() != &ss || in_slot) &&
            bslot_ss[0][(size_t) slot_source]->gdn_state != rsp->gdn_state) {
            basalt::core::SessionState& from = *bslot_ss[0][(size_t) slot_source];
            const int64_t had = basalt::core::session_kv_cells(from);
            std::string he;
            if (hold_slot(slot_source, slot_ck, he) && kv_release(from, 0, he)) {
                bs[(size_t) slot_source].cached = false;   // its cells are this request's now
                solo_moved = &from == &ss;
                basalt::log::info("basalt batch", "slot %d's conversation waits in RAM while its %lld KV cells back "
                                  "this request", slot_source, (long long) (had - basalt::core::session_kv_cells(from)));
                err.clear();
                kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
            } else {
                slot_held = SlotHeld{};
                err = he;
                kv_short = false;
            }
        }
        if (!kv_ok && kv_short) {
            // declined: the engine stays up, the solo session and its bookkeeping as they were
            slot_held = SlotHeld{};
            basalt::log::warn("basalt serve", "prompt of %lld tokens refused: %s", (long long) n, err.c_str());
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            if (in_slot) {
                if (solo_req && !ver.bind_session(&ss, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                rsp = &ss;
                sp.set_session(ss);
                if (mtp_on) mtp.set_ple_session(&ss);
                live = std::move(solo_book.live);
                live_imgs = std::move(solo_book.live_imgs);
                live_ok = solo_book.live_ok;
                cvec_cached = solo_book.cvec;
                checks = std::move(solo_book.checks);
                if (solo_moved) {
                    live.clear();
                    live_imgs.clear();
                    live_ok = false;
                    checks.clear();
                }
            }
            admit_slot = -1;
            continue;
        }
        if (!kv_ok) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        if ((!from_live || incoming || slot_source >= 0) &&
            !park_current(incoming ? incoming->bytes() : 0, incoming || slot_source >= 0 ? 0 : resume)) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        if (slot_source >= 0) {
            const auto t0 = Clock::now();
            if (!copy_from_slot(slot_source, slot_ck, err)) {
                // the main session may be half written: read this prompt from the start
                basalt::log::warn("basalt batch", "restoring slot %d failed (%s); reading the prompt",
                             slot_source, err.c_str());
                err.clear();
                slot_held = SlotHeld{};
                live.clear();
                live_imgs.clear();
                checks.clear();
                resume = 0;
                from_live = false;
            } else {
                checks.clear();   // the main session's checkpoints were of the conversation it held before
                conversations.limit_reuse(0);   // its cells now hold the slot's conversation, not the retained buffers'
                // a read that gave way, the same request again (into its own slot): on with it - its segments and
                // checkpoints as if it had not stopped
                if (slot_ck == nullptr && bs[(size_t) slot_source].partial && admit_slot == slot_source) {
                    resumed_from0 = bs[(size_t) slot_source].partial_from0;
                    resumed_chunk = bs[(size_t) slot_source].partial_chunk;
                    for (const ConvCheckpoint& c : bs[(size_t) slot_source].checks) checks.push_back(c);
                    bs[(size_t) slot_source].partial = false;
                }
                if (slot_ck != nullptr) {
                    live = slot_ck->ids;
                    checks.push_back(*slot_ck);   // the main session's chain has it now
                    checks.back().used = ++check_clock;
                } else {
                    live = bs[(size_t) slot_source].ids;
                }
                live_imgs = slot_ck != nullptr ? slot_ck->imgs : bs[(size_t) slot_source].imgs;
                resume = slot_tokens;
                from_live = true;
                basalt::log::info("basalt batch", "slot %d gave back %lld tokens of this conversation (%s) in "
                             "%.1f ms", slot_source, (long long) slot_tokens,
                             slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        }
        const auto restore_t0 = Clock::now();
        if (incoming) {
            // a snapshot that cannot be restored is dropped and the prompt read from the start
            bool restored = basalt::core::conversation_snapshot_restore(*incoming, *rsp,
                    g, mtp_kv, err) == basalt::core::ConversationRestore::restored;
            int failed_stage = -1;
            for (size_t i = 0; restored && i < stages.size(); ++i) {   // the later stages' parts
                const basalt::core::OnDevice on(stages[i]->dev);
                if (basalt::core::conversation_snapshot_restore(incoming->stage_images[i], stages[i]->ss, g,
                        nullptr, err) !=
                    basalt::core::ConversationRestore::restored) {
                    restored = false;
                    failed_stage = stages[i]->dev;
                    break;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR stage sync\n"); return 1; }
            }
            if (!restored) {
                basalt::log::warn("basalt serve", "conversation cache: restoring a parked conversation failed%s "
                             "(%s); reading the prompt from the start",
                             failed_stage >= 0 ? (" on CUDA" + std::to_string(failed_stage)).c_str() : "",
                             err.c_str());
                (void) cudaGetLastError();
                if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR restore sync\n"); return 1; }
                err.clear();
                incoming.reset();
                live.clear();
                live_imgs.clear();
                checks.clear();
                resume = 0;
                from_live = false;
            }
        }
        if (incoming) {
            live = std::move(incoming->live.ids);
            live_imgs = std::move(incoming->live.imgs);
            if (stages.empty()) {
                checks = std::move(incoming->checkpoints);
            } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
                basalt::core::ConversationCheckpointSplit cs;
                cs.stage0 = std::move(incoming->checkpoints);
                for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
                if (!basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
            }
            cvec_cached = incoming->cvec;
            resume = in_tokens;
            from_live = in_live;
            std::vector<std::vector<basalt::core::ConversationKv>> stage_kv;   // every stage's, for its next park
            for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
            conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv), rsp);
            incoming.reset(); // Running-state/checkpoint copies are no longer needed.
            basalt::log::info("basalt serve", "conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu",
                         (long long) resume, from_live ? "live" : "checkpoint",
                         std::chrono::duration<double, std::milli>(Clock::now() - restore_t0).count(),
                         conversations.size(), conversations.bytes());
        }
        ckd_finish();
        if (want_cvec != cvec_cached) {
            live_ok = false;
            checks.clear();
            cvec_cached = want_cvec;
        }
        if (basalt::kernels::cvec().loaded()) basalt::kernels::cvec_set_enabled(want_cvec);
        // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
        // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
        checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                         return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                     }), checks.end());
        live_ok = false;   // until this request has finished, the session is in between
        if (resume == 0) {
            basalt::core::session_zero(*rsp, g, nullptr, main_cs);
            cudaStreamSynchronize(main_stream);
            for (auto& st : stages) {
                const basalt::core::OnDevice on(st->dev);
                basalt::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                cudaStreamSynchronize(st->stream);
            }
            checks.clear();
        } else if (!from_live) {
            ConvCheckpoint* c = nullptr;
            for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
            if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
            if (c == nullptr || !checkpoint_restore(*c, *rsp, g) || c->stage_parts.size() != stages.size() ||
                       [&] {
                           for (size_t i = 0; i < stages.size(); ++i) {
                               const basalt::core::OnDevice on(stages[i]->dev);
                               if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                           }
                           return false;
                       }()) {
                std::printf("ERR restoring a conversation checkpoint failed\n");
                return 1;
            }
        }
        // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
        // host copies and slots are always current (every writer writes both), so they need nothing
        if (mtp_on) {
            if (resume > 0) mtp.kv_restore(resume);
            mtp.set_prompt_len(n);
            mtp.begin_request(ids.data(), (int64_t) ids.size());
        }
        const int64_t read_from = resume;
        conversations.drop_reuse_unless(rsp);   // another session may have overwritten the cells since
        conversations.limit_reuse(read_from);
        pp_total = n;
        pp_from = read_from;
        pp_reached = read_from;
        pp_t0 = r0;
        pp_next_check = resume + o.prompt_cache_every;
        {
            std::lock_guard<std::mutex> lk(part_mu);
            part_at.clear();
            std::fill(part_next.begin(), part_next.end(), pp_next_check);
        }
        // the cells past this request's room hold nothing it reads (the outgoing conversation is parked by now):
        // a longer one before it gives them back to the expert cache
        if (!kv_release(*rsp, n + kKvGrowCells, err)) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
        basalt::core::progress_at("reading the prompt, from token", read_from);
        std::fflush(stdout);
        // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
        // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path has
        // a high fixed cost per run however few tokens it has (it streams every expert the chunk routes to that is
        // not in VRAM over PCIe) and it borrows slots it must refill after; a window costs little per token, with
        // the misses on the CPU.  Each part below is decided on its own, so a long first message is read batched
        // and its header still goes through the windows.  Picture rows need the batched path.
        auto windows_ok = [&](int64_t a, int64_t b) -> bool {
            if ((in_slot && !solo_req) || b - a > o.short_read) return false;   // (the verifier is bound to this request's state)
            if (sp.embd_rows != nullptr)
                for (int64_t i = a; i < b; ++i)
                    if (sp.embd_rows[i] != nullptr) return false;
            return true;
        };
        // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
        auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
            basalt::core::progress_at("reading the prompt (verify windows), from token", a);   // not "batched"
            // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
            struct NoHeadSampling {
                basalt::core::Verifier& v;
                explicit NoHeadSampling(basalt::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                ~NoHeadSampling() { v.set_head_sampling(true); }
            } no_head_sampling(ver);
            std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
            for (int64_t q = a; q < b;) {
                if (halted()) { e = "cancelled"; return false; }
                const int T = (int) std::min<int64_t>(S, b - q);
                for (int t = 0; t < T; ++t) {
                    win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                    nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                }
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                    if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                    return false;
                }
                if (!ver.commit(T, e) || (mtp_on && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
                q += T;
                pp_reached = q;
            }
            // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
            if (!ver.wait_commit(e)) return false;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
            basalt::core::progress_beat();
            std::fflush(stdout);
            return true;
        };
        // the batched path's slots are lent just before its first run and given back (refilled) before a window
        // reads - so the windows always see the whole expert cache - or once the prompt is read
        // Every participant gives its loan back here: the rows it lent are refilled into the SAME slots from the
        // arena, the residency table is restored, and one upload puts it on every device.  A stage refills through
        // its own cache and its own device - a slot refilled into the wrong cache would leave that stage's cache
        // holding an expert it does not own, which is silent and produces plausible tokens.
        // (split in two halves so a layer split can queue every stage's copies before it waits for any)
        // `copied`: the copies already ran (and were waited for) on the overlapped read's thread, so only the table is left
        bool ov_copied = false;
        auto refill_issue = [&](PfPart& p, std::string& e, bool assign = true) -> bool {
            const basalt::core::OnDevice on(p.dev);
            for (const auto& [i, slot] : p.lent) {   // queued, one wait
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(p.cache->fill_slot_queued(slot, b, e, nb)))
                    return false;
                if (assign) host_res[(size_t) i] = slot;
            }
            return true;
        };
        auto refill_wait = [&](PfPart& p, std::string& e) -> bool {
            const basalt::core::OnDevice on(p.dev);
            if (!p.cache->sync_queued(e)) return false;
            p.lent.clear();
            p.lent_chunk = 0;
            return true;
        };
        auto refill_one = [&](PfPart& p, std::string& e) -> bool {
            if (p.lent.empty()) return true;
            if (ov_copied) {
                for (const auto& [i, slot] : p.lent) host_res[(size_t) i] = slot;
                p.lent.clear();
                p.lent_chunk = 0;
                ov_copied = false;
                return true;
            }
            return refill_issue(p, e) && refill_wait(p, e);
        };
        // with several stages every stage's copies go out first (each card on its own link), then each is waited
        // for - the same copies into the same slots, so the same cache; one stage: refill_one exactly.
        auto refill = [&](std::string& e) -> bool {
            const auto t_rf = Clock::now();
            int64_t n_lent = 0, n_parts = 0;
            for (PfPart& p : pf_parts)
                if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
            if (n_parts > 1) {
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty() && !refill_issue(p, e)) return false;
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty() && !refill_wait(p, e)) return false;
            } else {
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty() && !refill_one(p, e)) return false;
            }
            if (n_parts > 0) res_upload();
            ov_copied = false;
            return true;
        };
        // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
        // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
        // participant's own cache, and marking only that participant's own layers
        // `chunk` > 0: that chunk (a split read's), its first one `head` tokens (0: none)
        auto lend = [&](int64_t tokens, std::string& e, int64_t chunk = 0, int64_t head = 0) -> bool {
            if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
            const auto t_ln = Clock::now();
            // the ring depth this read's layout uses; every size computed below (and the layout) follows it, and
            // the full depth is restored on the way out for the sizing done elsewhere
            const bool beside = batch_on() && (tokens > kReadAlone || resumed_chunk > 0);
            basalt::prefill::Prefill::set_ring_depth(beside ? kRingDecoding : 1 << 20);
            struct RingReset { ~RingReset() { basalt::prefill::Prefill::set_ring_depth(1 << 20); } } ring_reset;
            // what this segment needs, capped by the configured chunk: a request lends only what its own
            // prompt needs, so a large chunk costs a short prompt nothing
            // A bare `auto` reads a prompt in equal chunks of at most this many tokens: past a few thousand tokens
            // a chunk's cost per token stops falling (its experts stream once whatever its size) and then rises
            // with the chunk, so the rule is the fewest equal parts of at most this size, never a small tail.
            // `auto:N` and a fixed N keep the operator's number.  (Chunking changes the GEMM tiles and so the rounding of
            // a prompt's K/V; the answers can differ from another chunking by greedy near-ties.)
            int64_t cap = read_cap();
            // Loan limits while slots decode: the loan takes only slots whose expert has a RAM copy (a lent slot
            // is refilled from RAM, never read from the file, before the slots decode again), and a read beside
            // them stays within kLendCapDecoding slots (they keep their experts in the cache).  A read on an idle
            // engine may refill lent slots from the file, so a larger chunk is worth it.  The chunk shrinks to the
            // loan, but not below the smallest chunk the prompt path runs.
            int64_t loan_lim = std::numeric_limits<int64_t>::max();
            if (batch_on()) {
                if (beside) loan_lim = kLendCapDecoding;
                // the RAM-backed slots are the cache's last ones: those still live after the KV took its share
                if (src.complement_ready())
                    loan_lim = std::min<int64_t>(loan_lim, std::max<int64_t>(0, src.resident_lent_slots() -
                                                                                 (xcache.full_slots() - xcache.slots())));
            }
            if (loan_lim != std::numeric_limits<int64_t>::max() && !pf_parts.empty() && pf_parts[0].first >= 0) {
                const int64_t lc = std::max<int64_t>(256, engine_local::biggest_chunk(cap, [&](int64_t t) {
                    return part_slots(pf_parts[0], t) <= loan_lim;
                }));
                cap = std::min(cap, lc);
            }
            auto [want_full, head_tok] = serve_loans::one_card_chunks(tokens, cap, !beside && stages.empty(),
                                                                   o.prefill_auto_fit);
            // a chunk the read must have: a split read's, or the one the read this continues had (if it fits)
            const int64_t fixed_chunk = chunk > 0 ? chunk : (resumed_chunk > 0 && resumed_chunk <= cap) ? resumed_chunk : 0;
            if (fixed_chunk > 0) {
                want_full = fixed_chunk;
                head_tok = chunk > 0 ? head : 0;
            }
            sp.set_head_chunk(head_tok);
            const int64_t want = want_full;
            if (want <= 0) {
                e = "prefill: cannot lend buffers for an empty request segment";
                return false;
            }
            bool any = false;
            const int ring_now = (int) basalt::prefill::Prefill::ring_slots_for(want);
            for (PfPart& p : pf_parts) {
                if (p.first < 0) continue;
                const int32_t first = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, want)));
                if (!p.lent.empty()) {
                    if (want <= p.lent_chunk && p.sp->ring_depth() == ring_now && !ov_copied &&
                        (fixed_chunk <= 0 || want == p.sp->chunk()))
                        continue;   // its loan covers this (a fixed chunk must be the layout's)
                    // the slots the new loan takes again are not refilled first (their experts would be
                    // overwritten by the new layout): only their table entries go back, and the loop below
                    // lends them again; the new loan's refill brings their experts back after the prompt
                    if (!ov_copied) {
                        std::vector<std::pair<int32_t, int32_t>> back;
                        for (const auto& [i, slot] : p.lent) {
                            if (slot >= first) host_res[(size_t) i] = slot;
                            else back.emplace_back(i, slot);
                        }
                        p.lent = std::move(back);
                    }
                    // ONLY this participant's loan goes back: `refill` would return the other participants'
                    // loans too, and their buffers are still laid out in their caches - marking those slots
                    // resident again would hand the next window a prompt buffer in place of an expert
                    if (!refill_one(p, e)) return false;
                }
                const basalt::core::OnDevice on(p.dev);
                if (beside && &p == &pf_parts[0] && p.lent.empty()) {
                    if (ba_job) { ba_thr.join(); ba_job = false; }   // no adaptive copy lands in a slot it moves
                    if (!xq_drain()) { e = "an adaptive refill failed"; return false; }
                    if (!move_routed_below(first, "prompt loan", e)) return false;
                    any = true;
                }
                if (want != p.sp->chunk() || first != p.first_now || p.sp->ring_depth() != ring_now) {
                    if (!p.sp->relayout(want, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                    p.first_now = first;
                }
                for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        const size_t i = (size_t) (l * g.n_expert + ex);
                        if (host_res[i] >= first) {
                            p.lent.emplace_back((int32_t) i, host_res[i]);
                            host_res[i] = basalt::core::kNotResident;
                            any = true;
                        }
                    }
                p.lent_chunk = want;
            }
            if (any) res_upload();
            return true;
        };
        // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
        // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
        // decode for a share of the time the chunk took (half), so a long prompt does not stop the others.  The
        // slots' windows then see the cache without the slots this prompt borrowed (marked missing: those experts
        // run on the CPU), never a prompt buffer.
        constexpr double decode_share = 0.5;
        bool batch_fatal = false;
        int64_t il_parts = 0;   // the slots' windows run between this prompt's chunks
        double il_ms = 0;
        int64_t yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
        int yielded_slot = -1;
        // `BYIELD <slot>` (the server: a shorter request is waiting): at the next chunk boundary the part read so
        // far is copied into <slot> (this admission's own, or a free one the server reserved for a solo request),
        // the request ends with `YIELDED <slot> <tokens>` + DONE cancel, and the server sends it again later: it
        // continues from the slot with the same chunks.  Cooperative preemption, with a slot as the park.
        // BYIELD: a read gives way at a chunk boundary after `q` tokens, into slot `ys`, when the rest is read the same
        // way afterwards (more than a chunk and a short read left; beside decoding slots, `beside`, more than a short
        // read: the rest is read beside them in the same chunk), text only, and the slot is not decoding
        auto can_yield = [&](int64_t q, int64_t b0, int ys, bool beside = false) -> bool {
            return ys >= 0 && ys < (int) bs.size() && !bs[(size_t) ys].active &&
                   n + (admit_slot >= 0 ? admit_max_new : (long long) max_new) + 8 <= slot_cap[(size_t) ys] &&
                   (admit_slot < 0 || ys == admit_slot) && req_imgs.empty() && o.prompt_cache > 0 &&
                   (!in_slot || ys == slot_b) &&
                   b0 - q > (beside ? o.short_read : std::max<int64_t>(std::max<int64_t>(sp.chunk(), 1), o.short_read));
        };
        // the first `q` tokens of this read go to slot `ys`, which holds them until the request is sent again
        // (`chunk`: the chunk of a read beside decoding slots, 0 for any other)
        auto park_read = [&](int64_t q, int ys, std::string& ye, int64_t chunk = 0) -> bool {
            std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
            if (!copy_to_slot(ys, pre, ye)) {
                bs[(size_t) ys].cached = false;   // half written
                return false;
            }
            if (in_slot) {   // the read ran in this slot's own state: its PLE token window goes with it
                basalt::core::SessionState& t = *bslot_ss[0][(size_t) ys];
                t.ple_prev[0] = rview.ple_prev[0];
                t.ple_prev[1] = rview.ple_prev[1];
                t.ple_token = rview.ple_token;
            }
            BSlot& sl = bs[(size_t) ys];
            sl = BSlot{};
            sl.ids = std::move(pre);
            sl.cached = true;
            sl.partial = true;
            sl.partial_from0 = read_from == 0 || resumed_from0;
            sl.partial_chunk = chunk;
            sl.cvec = cvec_cached;
            for (const ConvCheckpoint& c : checks)   // the root / periodic checkpoints of this read
                if ((int64_t) c.ids.size() <= q &&
                    basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), ids32.data()))
                    sl.checks.push_back(c);
            yielded_at = q;
            yielded_slot = ys;
            return true;
        };
        // ---- the prompt-time layer split: a long part of the prompt is read across the cards when its price there
        // (split_price) beats the main card's alone; the chunk, its head chunk, the stage count and the cuts, or none
        struct SplitPlan { int64_t chunk = 0, head = 0; int stages = 0; std::vector<int> cuts; };
        auto split_plan = [&](int64_t a0, int64_t b0, const char* no_split) -> SplitPlan {
            SplitPlan plan;
            if (!split_ready) return plan;   // (the start line said why)
            // every part longer than a chunk that is not read across says why, with its numbers
            auto decline = [&](const std::string& why) {
                basalt::log::info("basalt serve", "prompt split: %lld tokens read on CUDA0 (%s)", (long long) (b0 - a0),
                                  why.c_str());
            };
            const char* skip = no_split != nullptr ? no_split : batch_on() ? "slots are decoding" : !req_imgs.empty() ? "the prompt has images" : nullptr;
            if (skip != nullptr) {
                if (b0 - a0 > read_cap()) decline(skip);
                return plan;
            }
            const engine_local::SplitPricing pr = split_price(a0, b0, n - 1, split_stages);
            if (!pr.why.empty()) decline(pr.why);
            if (pr.chunk <= 0) return plan;   // (no reason: the part is one chunk, not longer than one)
            if (pr.chunk > split_chunk_max) {
                decline("chunks of " + std::to_string(pr.chunk) + " tokens, above the stages' " +
                        std::to_string(split_chunk_max));
                return plan;
            }
            if (pr.price.stages < 2) {
                decline("no stage set fits the cards' layer windows");
                return plan;
            }
            basalt::log::info("basalt serve", "prompt split: %lld tokens priced %.0f ms on CUDA0, %.0f ms with %s%s; read %s",
                              (long long) (b0 - a0), pr.price.tier_ms, pr.price.split_ms,
                              split_layers_text(pr.price.stages, pr.price.cuts).c_str(),
                              split_stages_text(pr.price).c_str(), pr.price.split ? "across" : "on CUDA0");
            if (!pr.price.split) return plan;
            for (int k = 1; k < pr.price.stages; ++k) {
                PromptStage& st = *pstages[(size_t) (pr.price.stages - 1 - k)];
                const int64_t l1 = k + 1 < pr.price.stages ? pr.price.cuts[(size_t) k] : g.n_layers;
                const basalt::core::OnDevice on(st.dev);
                uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
                if (!split_region(st, b0 + 64, pr.chunk, pr.price.cuts[(size_t) k - 1], l1, w_at, s_at, b_at)) {
                    basalt::log::info("basalt serve", "prompt split: %lld tokens do not fit CUDA%d's slots; read on CUDA0",
                                      (long long) (b0 - a0), st.dev);
                    return plan;
                }
            }
            plan.chunk = pr.chunk;
            plan.head = pr.head;
            plan.stages = pr.price.stages;
            plan.cuts = pr.price.cuts;
            return plan;
        };
        // [a0, b0) across the cards in chunks of `chunk` tokens: the main card reads layers [0, cuts[0]) of each chunk
        // and hands it on, each stage reads its layers of it and hands it on while the cards before read the next; the
        // main card keeps the last chunk and reads its later layers once the stages' state is back.  Main's loan is
        // made by `lend`.
        // `ran` stays false when it stopped before reading anything (the caller reads the part on the main card)
        auto read_split = [&](int64_t a0, int64_t b0, const SplitPlan& plan, bool& ran, std::string& e) -> bool {
            ran = false;
            const auto t0 = Clock::now();
            const int64_t n_tok = b0 - a0, chunk = plan.chunk, head = plan.head;
            const int64_t lb = plan.cuts[0];
            // the last chunk's first cell, from the chunks the main card reads (the one it keeps)
            const std::vector<int64_t> part_chunks = serve_loans::chunk_list(n_tok, chunk, head);
            const int64_t c_last = b0 - (part_chunks.empty() ? 0 : part_chunks.back());
            if (sp.chunk() != chunk) {
                e = "prompt split: the main card's prompt buffers are not laid out for the split's chunk";
                return false;
            }
            // the stages in pipeline order, each with its layers
            std::vector<PromptStage*> run;
            for (int k = 1; k < plan.stages; ++k) {
                PromptStage* st = pstages[(size_t) (plan.stages - 1 - k)].get();
                st->l0 = plan.cuts[(size_t) k - 1];
                st->l1 = k + 1 < plan.stages ? plan.cuts[(size_t) k] : g.n_layers;
                run.push_back(st);
            }
            PromptStage& last = *run.back();
            std::vector<basalt::prefill::PrefillStats> st0;
            for (PromptStage* st : run) st0.push_back(st->sp.stats());
            const basalt::prefill::PrefillStats sm0 = sp.stats();
            // the tiers' swaps land first: no copy may still write a slot a stage takes
            for (PromptStage* st : run) {
                st->tier->apply_pending(true);
                if (st->tier_at == 0) peer_landing = false;
                else xlanding[st->tier_at - 1] = 0;
            }
            auto give_back_now = [&](PromptStage& st) {
                std::string ge;
                const basalt::core::OnDevice on(st.dev);
                (void) cudaDeviceSynchronize();
                (void) cudaGetLastError();
                (void) (st.tier->give_back(st.lent, ge) && st.tier->give_back_wait(st.lent, ge));
            };
            auto give_back_all = [&]() {
                for (PromptStage* st : run) give_back_now(*st);
            };
            for (PromptStage* stp : run) {
                PromptStage& st = *stp;
                const basalt::core::OnDevice on(st.dev);
                const basalt::core::ExpertCache& pc = st.tier->cache();
                uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
                if (st.l0 >= st.l1 || !split_region(st, b0 + 64, chunk, st.l0, st.l1, w_at, s_at, b_at)) {
                    e = "prompt split: too few tier slots for the prompt";
                    give_back_all();
                    return false;
                }
                uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
                st.lent_slots = serve_loans::cache_slots_for(pc, (uint64_t) (base + pc.bytes() - b_at));
                const int32_t first = (int32_t) (pc.slots() - st.lent_slots);
                if (!st.tier->lend(first, st.lent, e)) {
                    give_back_all();
                    return false;
                }
                st.buf = base + (pc.slot_offsets() ? (size_t) pc.slot_offsets()[first]
                                                   : (size_t) first * basalt::kernels::cpu::expert_layout().max_blob);
                st.buf_bytes = (uint64_t) (s_at - st.buf);
                // the stage's experts in the tier: its layers' that kept their slots
                const std::vector<int32_t>& r = st.tier->residency();
                std::fill(st.res.begin(), st.res.end(), basalt::core::kNotResident);
                std::copy(r.begin() + (std::ptrdiff_t) (st.l0 * g.n_expert), r.begin() + (std::ptrdiff_t) (st.l1 * g.n_expert),
                          st.res.begin() + (std::ptrdiff_t) (st.l0 * g.n_expert));
                if (cudaMemcpyAsync(w_at, st.image, (size_t) st.image_bytes, cudaMemcpyHostToDevice, st.stream) !=
                    cudaSuccess) {
                    e = std::string("prompt split: the weights' upload: ") + cudaGetErrorString(cudaGetLastError());
                    give_back_all();
                    return false;
                }
                basalt::core::prompt_session_clear(st.ss, g, (void*) st.stream);
            }
            // a part after the start of the context: each stage's layers' state so far, every cell (after the clear),
            // all cards at once
            if (a0 > 0) {
                bool ok = true;
                for (PromptStage* st : run) {
                    const basalt::core::OnDevice on(st->dev);
                    ok = ok && cudaStreamSynchronize(st->stream) == cudaSuccess &&
                         basalt::core::session_copy_state(st->ss, *rsp, g, st->l0, st->l1, 0, a0,
                                                          [&](void* d, const void* sr, uint64_t nb) {
                                                              return st->state_in.copy(d, sr, nb, e);
                                                          }, e);
                }
                for (PromptStage* st : run) ok = ok && st->state_in.wait(e);
                if (!ok) {
                    if (e.empty()) e = "prompt split: a stage's state from the main session";
                    give_back_all();
                    return false;
                }
            }
            // (relayout waits for the stream: the weights and the session are in place after it); each stage runs alone
            // until the chain is set below
            for (PromptStage* st : run) {
                const basalt::core::OnDevice on(st->dev);
                st->sp.set_stage(st->l0, st == &last ? -1 : st->l1, nullptr);
                if (!st->sp.relayout(chunk, st->buf, st->buf_bytes, e)) {
                    give_back_all();
                    return false;
                }
            }
            const double ms_setup = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ran = true;
            // the main card: layers [0, lb) of every chunk, handing each but the last on; the tier cards compute no
            // rows of the main card's layers meanwhile
            sp.set_peer_share(false);
            sp.set_stage(0, lb, &run[0]->sp);
            sp.set_keep_last(true);
            for (PromptStage* st : run) {
                st->sp.should_stop = sp.should_stop;
                if (!drive.d.usage.empty()) st->sp.route_hist_reset();   // its layers' routed entries, counted apart
            }
            // `BYIELD <slot>` (a shorter request waits): at the end of a chunk's first layers the main card keeps that
            // chunk as the last one and reads its later layers once the stages' state is back; the slot then holds
            // the part read up to that chunk's end.  `c_cut`: the kept chunk's first cell.
            int64_t main_done = a0, yield_q = -1;
            int yield_ys = -1;
            std::atomic<int64_t> c_cut{c_last};
            std::atomic<int64_t> last_end{a0};   // the end of the last chunk the last stage read
            sp.keep_here = [&](int64_t done) -> bool {
                if (done >= c_last) return false;
                int ys = -1;
                {
                    std::lock_guard<std::mutex> lk(in_mu);
                    for (auto it = in_lines.begin(); it != in_lines.end();) {
                        if (it->rfind("BYIELD ", 0) == 0) {
                            ys = std::atoi(it->c_str() + 7);
                            it = in_lines.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
                if (ys < 0) return false;
                if (can_yield(done, b0, ys)) {
                    yield_q = done;
                    yield_ys = ys;
                    c_cut.store(main_done);
                    return true;
                }
                basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys, (long long) done,
                                  (long long) n);
                return false;
            };
            for (size_t k = 0; k + 1 < run.size(); ++k) run[k]->sp.set_stage(run[k]->l0, run[k]->l1, &run[k + 1]->sp);
            // the last stage, after each chunk: the rows the draft layer reads, back into the hand-off buffer the chunk
            // came in (the card before writes it again two chunks later, and not at all for the last); after its last
            // chunk, every stage's state to the main session and the slots' experts back, while the main card reads
            // its own last chunk
            const int64_t first_cell = mtp_on ? std::max<int64_t>(0, mtp.first_needed()) : INT64_MAX;
            bool given = false;
            int64_t given_at = -1;   // the cell the stages' state came back up to
            double ms_copy = 0;
            std::future<bool> refill_issue;   // the tier slots' copies, queued on a thread of their own
            std::string refill_err;
            // the final rows of the last stage's chunks the draft layer reads, in the hand-off buffers they came in
            struct Relay { int64_t cell0, n; const float* rows; };
            std::vector<Relay> relays;
            // those of the last two chunks handed on stay where they came in; an earlier chunk's go past `chunk` rows
            // in the card before's two hand-off buffers, which no chunk of this read writes
            int64_t in_place_from = a0;
            {
                const std::vector<int64_t> cl = serve_loans::chunk_list(n_tok, chunk, head);
                int64_t at = a0;
                for (size_t i = 0; i + 1 < cl.size(); ++i) {
                    if (i + 3 == cl.size()) in_place_from = at;
                    at += cl[i];
                }
            }
            const basalt::prefill::Prefill& before_last = run.size() >= 2 ? run[run.size() - 2]->sp : sp;
            const int64_t aside_rows = std::max<int64_t>(0, before_last.hand_rows() - chunk);
            int aside_buf = 0;
            int64_t aside_at = 0;
            std::mutex relay_mu;
            // mid-prompt checkpoints at the positions one card takes them: the main card saves its state when it
            // hands such a chunk on, each stage puts its layers' state in once it has read that chunk, the last one
            // files it
            const bool ck_on = o.prompt_cache > 0 && o.prompt_cache_every > 0;
            int64_t ck_next = pp_next_check;
            std::map<int64_t, ConvCheckpoint> ck_main;
            sp.on_stage_chunk = [&](int64_t done, std::string& ce) -> bool {
                main_done = done;
                if (!ck_on || done < ck_next) return true;
                ck_next = done + o.prompt_cache_every;
                ConvCheckpoint part;   // (its stream is synchronized)
                part.ids.assign(cur.begin(), cur.begin() + done);
                if (!checkpoint_save(part, *rsp, g)) {
                    ce = "saving a conversation checkpoint failed";
                    return false;
                }
                std::lock_guard<std::mutex> lk(relay_mu);
                ck_main[done] = std::move(part);
                return true;
            };
            for (PromptStage* stp : run) {
                if (stp == &last) continue;
                stp->sp.on_stage_chunk = [&, stp](int64_t done, std::string& ce) -> bool {
                    if (!ck_on) return true;
                    std::lock_guard<std::mutex> lk(relay_mu);
                    const auto it = ck_main.find(done);
                    std::string se;
                    if (it != ck_main.end() &&
                        !basalt::core::conversation_checkpoint_save_layers(it->second, stp->ss, g, stp->l0, stp->l1, se)) {
                        ce = "saving a conversation checkpoint failed: " + se;
                        return false;
                    }
                    return true;
                };
            }
            // every stage done with its chunks before the kept one at `c_end`: their state to the main session, and
            // the slots' experts back on a thread while the main card reads the kept chunk
            auto state_back = [&](int64_t c_end, std::string& ce) -> bool {
                const auto tc = Clock::now();
                bool ok = true;
                for (PromptStage* st : run)
                    ok = ok && basalt::core::session_copy_state(*rsp, st->ss, g, st->l0, st->l1, a0, c_end,
                                                                [&](void* d, const void* sr, uint64_t nb) {
                                                                    return st->state_back.copy(d, sr, nb, ce);
                                                                }, ce);
                for (PromptStage* st : run) ok = ok && st->state_back.wait(ce);
                if (!ok) {
                    if (ce.empty()) ce = "prompt split: a stage's state back to the main session";
                    return false;
                }
                given = true;
                given_at = c_end;
                ms_copy = std::chrono::duration<double, std::milli>(Clock::now() - tc).count();
                refill_issue = std::async(std::launch::async, [&] {
                    bool issued = true;
                    for (PromptStage* st : run) issued = st->tier->give_back(st->lent, refill_err) && issued;
                    return issued;
                });
                return true;
            };
            last.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
                const int64_t HCN = g.hc * g.n_embd;
                const int64_t r0 = std::max<int64_t>(0, first_cell - p0);
                for (int64_t r = r0; r < T && p0 < in_place_from;) {
                    if (aside_at >= aside_rows) {
                        if (++aside_buf > 1) {
                            ce = "prompt split: the draft layer's rows do not fit the hand-off buffers";
                            return false;
                        }
                        aside_at = 0;
                        continue;
                    }
                    float* h = before_last.hand_buffer(aside_buf);
                    const int64_t nr = std::min(aside_rows - aside_at, T - r);
                    float* dst = h == nullptr ? nullptr : h + (size_t) (chunk + aside_at) * HCN;
                    if (dst == nullptr || cudaMemcpy(dst, R_rows + (size_t) r * HCN, (size_t) nr * HCN * sizeof(float),
                                                     cudaMemcpyDeviceToHost) != cudaSuccess) {
                        ce = "prompt split: reading the draft layer's rows back failed";
                        return false;
                    }
                    std::lock_guard<std::mutex> lk(relay_mu);
                    relays.push_back({p0 + r, nr, dst});
                    aside_at += nr;
                    r += nr;
                }
                if (r0 < T && p0 >= in_place_from) {
                    float* h = const_cast<float*>(last.sp.hand_in());
                    if (h == nullptr || cudaMemcpy(h + (size_t) r0 * HCN, R_rows + (size_t) r0 * HCN,
                                                   (size_t) (T - r0) * HCN * sizeof(float),
                                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
                        ce = "prompt split: reading the draft layer's rows back failed";
                        return false;
                    }
                    std::lock_guard<std::mutex> lk(relay_mu);
                    relays.push_back({p0 + r0, T - r0, h + (size_t) r0 * HCN});
                }
                std::vector<ConvCheckpoint> whole(1);
                {
                    std::lock_guard<std::mutex> lk(relay_mu);
                    const auto it = ck_main.find(p0 + T);
                    if (it != ck_main.end()) {
                        whole[0] = std::move(it->second);
                        ck_main.erase(it);
                    }
                }
                std::string se;
                if (!whole[0].ids.empty() &&
                    (!basalt::core::conversation_checkpoint_save_layers(whole[0], last.ss, g, last.l0, last.l1, se) ||
                     !checkpoint_at(p0 + T, &whole))) {
                    ce = "saving a conversation checkpoint failed" + (se.empty() ? ckpt_why : ": " + se);
                    return false;
                }
                // its last chunk before the kept one: every stage is done with its own
                if (p0 + T >= c_cut.load() && !state_back(p0 + T, ce)) return false;
                const int64_t done = p0 + T;
                last_end.store(done);
                pp_reached = done;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
                basalt::core::progress_at("reading the prompt (batched), done up to token", done);
                basalt::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            bool ok = sp.run(ids.data() + a0, n_tok, a0, e);   // (waits for every stage's chunks)
            sp.keep_here = nullptr;
            // the kept chunk [c_end, b_end): the last one, or the one a read that gives way ends with (the stages may
            // have been done before the cut)
            const int64_t c_end = c_cut.load();
            const int64_t b_end = yield_q >= 0 ? yield_q : b0;
            if (ok && !given && yield_q >= 0) ok = state_back(c_end, e);
            // the stages' state must come back exactly at the main card's kept chunk: anything else is an error
            if (ok && (!given || (yield_q < 0 && given_at != c_end))) {
                ok = false;
                e = "prompt split: the stages' state did not come back at the kept chunk (planned at " +
                    std::to_string(c_end) + ", the stages ended at " + std::to_string(last_end.load()) +
                    (given ? ", it came back at " + std::to_string(given_at) : std::string(", it never came back")) +
                    "; " + std::to_string(n_tok) + " tokens in chunks of " + std::to_string(chunk) + ", head " +
                    std::to_string(head) + ")";
            }
            // KV streaming: the stages wrote their layers' host copies, so a block resident before may be stale
            if (given && g.n_qsa_layers() > 0 && rsp->qsa_states[rsp->qsa_primary()].kv_mode == 1) {
                const basalt::core::OnDevice on0(0);
                const int64_t I = std::max<int64_t>(g.qsa_interval, 1);
                for (PromptStage* st : run)
                    for (int64_t q = st->l0 / I; q < st->l1 / I; ++q)
                        basalt::kernels::kv_stream_reset(rsp->qsa_states[q].map, nullptr);
                if (cudaStreamSynchronize(nullptr) != cudaSuccess && ok) {
                    ok = false;
                    e = "prompt split: resetting the streamed K/V's residency";
                }
            }
            sp.set_keep_last(false);
            sp.on_stage_chunk = nullptr;
            for (PromptStage* st : run) {
                st->sp.on_stage_chunk = nullptr;
                st->sp.on_chunk = nullptr;
                st->sp.set_stage(st->l0, -1, nullptr);   // alone again: its routed entries are taken on their own
            }
            pp_next_check = ck_next;   // the last chunk's check, on this card as on one card
            const double ms_stages = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            // the draft layer's K/V of the last stage's rows, through this card's prompt path; then the last chunk's
            // layers [lb, n_layers) here
            const auto td = Clock::now();
            for (const Relay& rl : relays) {
                if (!ok || !mtp_on) break;
                std::vector<int32_t> nxt((size_t) rl.n);
                for (int64_t t = 0; t < rl.n; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (rl.cell0 + t + 1)];
                const bool batched = sp.draft_kv(mtp, rl.rows, nxt.data(), rl.n, rl.cell0, e);
                ok = e.empty() && (batched || mtp.prefill(rl.rows, nxt.data(), rl.n, rl.cell0, e));
            }
            relays.clear();
            const double ms_draft = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
            sp.set_stage(lb, -1, nullptr);
            if (ok) {
                sp.continue_in_place();
                ok = sp.run(ids.data() + c_end, b_end - c_end, c_end, e);
            }
            sp.set_stage(0, -1, nullptr);
            sp.set_peer_share(true);
            {
                const bool issued = refill_issue.valid() && refill_issue.get();
                for (PromptStage* st : run) {
                    const basalt::core::OnDevice on(st->dev);
                    std::string ge;
                    if (!st->lent.empty() && !(ok && issued && st->tier->give_back_wait(st->lent, ge)))
                        give_back_now(*st);   // stopped early: nothing of the stage may still write the slots
                }
            }
            split_hist.clear();
            split_hist_tokens = 0;
            if (!drive.d.usage.empty())
                for (PromptStage* st : run) st->sp.route_hist_take(split_hist, split_hist_tokens);
            if (ok && yield_q >= 0) {   // (a run that ends early leaves the token window of the PLE hash where it began)
                if (yield_q >= 2) {
                    rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                    rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
                }
                std::string ye;
                const auto ty = Clock::now();
                if (!park_read(yield_q, yield_ys, ye)) {
                    e = "BYIELD: parking the prompt read failed: " + ye;
                    return false;
                }
                basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d holds it "
                                     "(copied in %.1f ms)", (long long) yield_q, (long long) n, yield_ys,
                                  std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                e = "yield";
                return false;
            }
            if (!ok) return false;
            const basalt::prefill::PrefillStats& sm1 = sp.stats();
            std::string where, cards;
            for (size_t k = 0; k < run.size(); ++k) {
                const PromptStage& st = *run[k];
                const basalt::prefill::PrefillStats& st1 = st.sp.stats();
                size_t pfree = 0, ptotal = 0;
                {
                    const basalt::core::OnDevice on(st.dev);
                    if (cudaMemGetInfo(&pfree, &ptotal) != cudaSuccess) (void) cudaGetLastError();
                }
                char b[256];
                std::snprintf(b, sizeof b, "%s%lld on CUDA%d, layers %lld-%lld", k ? ", " : "",
                              (long long) (st1.chunks - st0[k].chunks), st.dev, (long long) st.l0, (long long) st.l1 - 1);
                where += b;
                std::snprintf(b, sizeof b, "CUDA%d %lld lent slots, its stage %.0f ms, %lld experts streamed, %zu MiB free "
                              "after; ", st.dev, (long long) st.lent_slots, st1.ms_total - st0[k].ms_total,
                              (long long) (st1.experts_streamed - st0[k].experts_streamed), pfree >> 20);
                cards += b;
            }
            basalt::log::info("basalt serve", "prompt split: %lld tokens in chunks of %lld (%s), %.0f ms (setup %.1f, stages "
                                 "%.0f, state back %.0f, draft rows %.0f); %sCUDA0 %.0f ms, %lld streamed",
                              (long long) n_tok, (long long) chunk, where.c_str(),
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), ms_setup, ms_stages,
                              ms_copy, ms_draft, cards.c_str(), sm1.ms_total - sm0.ms_total,
                              (long long) (sm1.experts_streamed - sm0.experts_streamed));
            return true;
        };
        auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
            // (a layer split reads its stages as a pipeline over one run's chunks: in pieces only beside slots)
            if (o.batch <= 0 || piped || (!stages.empty() && !batch_on()) || (b0 - a0 <= kReadAlone && resumed_chunk <= 0)) {
                // no window runs meanwhile, so the CPU pool is free: it takes its share of the streamed experts
                const bool lend_pool = o.batch > 0 && !multi_gpu && !o.no_pool;
                if (lend_pool) { sp.set_cpu_pool(&pool); sp.set_cpu_share(o.prefill_cpu_share); }
                const bool ok = sp.run(ids.data() + a0, b0 - a0, a0, e);
                if (lend_pool) sp.set_cpu_pool(nullptr);
                return ok;
            }
            if (batch_on() && stages.empty()) {
                // the whole range is read on a side thread, in the prompt path's own chunks, while this thread keeps
                // running the slots' windows; the work that touches the draft layer or the output runs here, between
                // two windows, when the prompt path asks for it
                if (!xq_drain()) { e = "an adaptive refill failed"; return false; }
                apply_pending(true);
                struct Mail {
                    std::mutex mu;
                    std::condition_variable cv;
                    std::function<void()> job;
                    bool job_done = false, fin = false;
                } mail;
                std::atomic<bool> abort_read{false};
                bool ok = false;
                std::string terr;
                int yield_req = -1;                 // a BYIELD <slot> not acted on yet
                int64_t yield_q = -1;               // the chunk boundary the read stopped at, to give way
                int yield_slot = -1;
                auto orig_chunk = sp.on_chunk;
                auto orig_stop = sp.should_stop;
                sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
                    bool r = false;
                    std::unique_lock<std::mutex> lk(mail.mu);
                    mail.job = [&] {
                        r = orig_chunk ? orig_chunk(R_rows, T, p0, ce) : true;
                        if (!r || yield_req < 0) return;
                        // a shorter request waits: the read stops after this chunk (the draft layer has it) when the
                        // slot can take what was read, or goes on
                        const int ys = yield_req;
                        yield_req = -1;
                        if (can_yield(p0 + T, b0, ys, true)) {
                            yield_q = p0 + T;
                            yield_slot = ys;
                            abort_read.store(true);
                        } else {
                            basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys,
                                         (long long) (p0 + T), (long long) n);
                        }
                    };
                    mail.job_done = false;
                    mail.cv.wait(lk, [&] { return mail.job_done; });
                    return r;
                };
                sp.should_stop = [&] { return abort_read.load() || (orig_stop && orig_stop()); };
                ov_active = true;
                std::thread th([&] {
                    const basalt::core::OnDevice on(0);
                    ok = sp.run(ids.data() + a0, b0 - a0, a0, terr);
                    if (ok && ov_post) {   // (the checkpoint's copy to the host beside the windows)
                        ok = ov_post();
                        ov_post_done = ok;
                        if (!ok) terr = "saving a conversation checkpoint failed" + ckpt_why;
                    }
                    std::lock_guard<std::mutex> lk(mail.mu);
                    mail.fin = true;
                    mail.cv.notify_all();
                });
                bool fatal = false;
                int64_t ov_jobs = 0, ov_wins = 0;
                double ov_job_ms = 0, ov_job_max = 0, ov_win_ms = 0, ov_win_max = 0;
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lk(mail.mu);
                        if (mail.job && !mail.job_done) job = mail.job;
                        else if (mail.fin) break;
                    }
                    if (job) {
                        const auto tj = Clock::now();
                        job();
                        const double jm = std::chrono::duration<double, std::milli>(Clock::now() - tj).count();
                        ++ov_jobs; ov_job_ms += jm; ov_job_max = std::max(ov_job_max, jm);
                        std::lock_guard<std::mutex> lk(mail.mu);
                        mail.job = nullptr;
                        mail.job_done = true;
                        mail.cv.notify_all();
                        continue;
                    }
                    {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                        std::lock_guard<std::mutex> lk(in_mu);
                        for (auto it = in_lines.begin(); it != in_lines.end();) {
                            if (it->rfind("BSTOP ", 0) == 0) {
                                const int b = std::atoi(it->c_str() + 6);
                                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                                it = in_lines.erase(it);
                            } else if (it->rfind("BYIELD ", 0) == 0) {
                                yield_req = std::atoi(it->c_str() + 7);
                                it = in_lines.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (!fatal && batch_on()) {
                        const auto tw = Clock::now();
                        if (!batch_step()) { fatal = true; abort_read.store(true); }
                        const double wm = std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                        ++ov_wins; ov_win_ms += wm; ov_win_max = std::max(ov_win_max, wm);
                    } else {
                        std::unique_lock<std::mutex> lk(mail.mu);
                        mail.cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return mail.fin || (mail.job && !mail.job_done); });
                    }
                }
                th.join();
                ov_active = false;
                basalt::log::info("basalt batch", "overlapped prompt read: %lld windows (%.1f ms avg, %.0f ms max) beside "
                                     "%lld chunk jobs run between windows (%.1f ms avg, %.0f ms max)",
                             (long long) ov_wins, ov_wins ? ov_win_ms / (double) ov_wins : 0.0, ov_win_max,
                             (long long) ov_jobs, ov_jobs ? ov_job_ms / (double) ov_jobs : 0.0, ov_job_max);
                sp.on_chunk = std::move(orig_chunk);
                sp.should_stop = std::move(orig_stop);
                if (yield_q >= 0 && !fatal) {   // the read stopped at a chunk boundary to give way (BYIELD)
                    std::string ye;
                    const auto ty = Clock::now();
                    if (yield_q >= 2) {   // a run that ends early leaves the token window of the PLE hash where it began
                        rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                        rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
                    }
                    if (!park_read(yield_q, yield_slot, ye, sp.chunk())) {
                        e = "BYIELD: parking the prompt read failed: " + ye;
                        return false;
                    }
                    basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                         "holds it (copied in %.1f ms)", (long long) yield_q, (long long) n, yield_slot,
                                 std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                    e = "yield";
                    return false;
                }
                if (fatal) { batch_fatal = true; e = "a batch window failed"; return false; }
                if (!ok) e = terr;
                return ok;
            }
            const int64_t C = std::max<int64_t>(sp.chunk(), 1);
            // while slots are decoding the prompt goes in chunks of a fixed small size, so a long prompt holds the
            // slots' windows back for one short chunk at a time
            constexpr int64_t live_chunk = 1024;
            for (int64_t q = a0; q < b0;) {
                const int64_t r = std::min(b0, q + (batch_on() ? std::min(C, live_chunk) : C));
                const auto tq = Clock::now();
                if (!sp.run(ids.data() + q, r - q, q, e)) return false;
                q = r;
                if (q >= b0) continue;
                int ys = -1;
                {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                    std::lock_guard<std::mutex> lk(in_mu);
                    for (auto it = in_lines.begin(); it != in_lines.end();) {
                        if (it->rfind("BSTOP ", 0) == 0) {
                            const int b = std::atoi(it->c_str() + 6);
                            if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                            it = in_lines.erase(it);
                        } else if (it->rfind("BYIELD ", 0) == 0) {
                            ys = std::atoi(it->c_str() + 7);
                            it = in_lines.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
                if (ys >= 0) {
                    std::string ye;
                    const auto ty = Clock::now();
                    if (can_yield(q, b0, ys) && park_read(q, ys, ye)) {
                        basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                             "holds it (copied in %.1f ms)", (long long) q, (long long) n, ys,
                                     std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                        e = "yield";
                        return false;
                    }
                    basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld%s%s", ys, (long long) q,
                                 (long long) n, ye.empty() ? "" : ": ", ye.c_str());
                }
                if (decode_share <= 0.0 || !batch_on()) continue;
                const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                const auto td = Clock::now();
                do {
                    if (!batch_step()) { batch_fatal = true; e = "a batch window failed"; return false; }
                } while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
                ++il_parts;
                il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
            }
            return true;
        };
        apply_pending(true);
        if (!xq_drain()) {
            std::printf("ERR an adaptive refill failed\n");
            return 1;
        }
        // per-request sampling for the verify window's head (greedy when temperature is absent)
        basalt::kernels::SamplerParams req_sp;
        req_sp.greedy = req_temperature <= 0.0f;
        req_sp.temperature = req_temperature;
        req_sp.top_p = req_top_p;
        req_sp.top_k = req_top_k;
        req_sp.seed = req_seed ? req_seed
                               : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
        req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
        req_sp.penalty_repeat = req_penalty_repeat;
        req_sp.penalty_freq = req_penalty_freq;
        req_sp.penalty_present = req_penalty_present;
        req_sp.counter = 0;
        if (!ver.set_logit_bias(req_bias, err)) {
            std::printf("ERR %s\n", err.c_str());
            continue;
        }
        ver.set_sampling(req_sp);
        // relaxed acceptance: sampled text requests without a logit_bias; never the quality gate
        const bool relax_on = req_relax > 0.0f && !req_sp.greedy && !req_bias_present && ppl == nullptr && gguard == nullptr;
        ver.set_draft_probe(relax_on);
        bool in_call = false;   // inside a tool call (its tags seen last in the prompt or the output): exact
        for (size_t i = ids.size(); i-- > 0;)
            if (ids[i] == o.relax_call_open || ids[i] == o.relax_call_close) { in_call = ids[i] == o.relax_call_open; break; }
        int64_t relaxed_kept = 0;
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        // a layer split: CUDA0's share; a later GPU keeps its own (its link)
        for (int st = 0; st < split_drive.n; ++st)
            split_drive.pcie_num[st] = (st == 0 || split_same)
                                           ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
        const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
        ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
        if (pipe_on) {   // the second chain draws the same way
            if (!ver_b.set_logit_bias(req_bias, err)) {
                std::printf("ERR %s\n", err.c_str());
                continue;
            }
            ver_b.set_sampling(req_sp);
            ver_b.set_draft_probe(relax_on);
            ver_b.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            for (int st = 0; st < split_drive_b.n; ++st) split_drive_b.pcie_num[st] = split_drive.pcie_num[st];
        }
        bool cancelled = false;
        // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
        // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
        // chat renders the same history - but not always the same header or the thinking of this reply - so that
        // checkpoint is the one it reuses.
        int64_t turn_at = -1;
        if (o.prompt_cache > 0 && o.turn_token >= 0)
            for (int64_t i = n - 1; i > resume; --i)
                if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
        // (opt-in) a short turn of --tail-role-token's role right before the new assistant turn (the server's
        // trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another effort, or the
        // next turn of the chat - finds the conversation without it
        if (turn_at > 0 && o.tail_role_token >= 0)
            for (int64_t i = turn_at - 1; i > resume; --i)
                if (ids[(size_t) i] == o.turn_token) {
                    if (ids[(size_t) i + 1] == o.tail_role_token) turn_at = i;
                    break;
                }
        // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with the
        // tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root, which
        // the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.  Only
        // for a system prompt of --prompt-cache-root tokens or more: a small one is cheaper to read again than the
        // extra part costs.
        int64_t root_at = -1;
        if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
            for (int64_t i = 1; i < turn_at; ++i)
                if (ids[(size_t) i] == o.turn_token) {
                    if (i >= o.prompt_cache_root) root_at = i;
                    break;
                }
        // A short read into a slot beside decoding slots is not split: each part pays the lend, refill and
        // side-thread costs again, and a checkpoint stalls the windows.  One that runs alone stops at its root,
        // and at its turn boundary only past --prompt-cache-root tokens since the root: a short history is
        // cheaper to read again than the extra part costs.  The rest of a read that gave way keeps its parts.
        if (in_slot && batch_on() && resumed_chunk <= 0 && n - 1 - read_from <= std::max<int64_t>(sp.chunk(), 1)) {
            if (n - 1 - read_from > kReadAlone) {
                turn_at = -1;
                root_at = -1;
            } else if (turn_at - std::max(read_from, root_at) < o.prompt_cache_root) {
                turn_at = -1;
            }
        }
        // the parts after the first read their n-gram rows from memory: once the first part has its own, the rest
        // are read into the row cache beside it
        {
            int64_t first_to = 0;
            for (const int64_t to : {root_at, turn_at, n - 1})
                if (to > read_from) { first_to = to; break; }
            sp.set_ple_warm(ids.data(), read_from, first_to, n - 1);
        }
        int64_t at = read_from;
        // the last window of the last windowed part carries the last prompt token too: the first generated
        // token is sampled from it, and the draft layer takes its rows like any accepted window
        int64_t fold_p0 = n - 1;
        for (const int64_t to : {root_at, turn_at, n - 1}) {
            if (to <= at) continue;
            err.clear();
            const bool win = windows_ok(at, to);
            int64_t rto = to;
            if (win && to == n - 1 && to != turn_at && to != root_at && ppl == nullptr && gguard == nullptr) {
                const int64_t p0 = at + (int64_t) S * ((to - at) / S);
                if (!mtp_on || n - p0 <= S_mtp) { rto = p0; fold_p0 = p0; }
            }
            if (win && !refill(err)) {
                std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            const SplitPlan split = split_plan(at, to, win ? "a windowed read" : resumed_chunk > 0 ? "a resumed chunked read" : nullptr);
            const int64_t split_c = split.chunk;
            if (!win && !lend(to - at, err, split_c, split.head)) {
                std::printf("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                return 1;
            }
            // beside slots: a window between the serial steps of a short read keeps them from stopping
            auto ov_window = [&]() -> bool {
                if (!batch_on()) return true;
                if (!batch_step()) { batch_fatal = true; err = "a batch window failed"; return false; }
                if (mtp_on && !mtp.use_main(err)) return false;
                return true;
            };
            if (win && !ov_window()) { std::printf("ERR %s\n", err.c_str()); return 1; }
            ov_post_done = false;
            ov_post = [&, to]() -> bool {
                if ((to == turn_at || to == root_at) && !checkpoint_at(to)) return false;
                // the lent slots get their experts back here, beside the windows; the table follows on the main thread
                for (PfPart& p : pf_parts) {
                    if (p.lent.empty()) continue;
                    std::string re;
                    const basalt::core::OnDevice on(p.dev);
                    if (!refill_issue(p, re, false) || !p.cache->sync_queued(re)) { ckpt_why = ": " + re; return false; }
                }
                ov_copied = true;
                return true;
            };
            const auto tsp = Clock::now();
            pp_part_ckpt = (to == turn_at || to == root_at) ? to : -1;
            bool split_ran = true;
            bool sp_ok = win ? (rto <= at || read_windows(at, rto, err))
                       : split_c > 0 ? read_split(at, to, split, split_ran, err)
                                     : read_part(at, to, err);
            if (!sp_ok && !split_ran) {   // the split could not start: the main card reads the part
                basalt::log::warn("basalt serve", "%s; read on CUDA0", err.c_str());
                err.clear();
                sp_ok = read_part(at, to, err);
            }
            pp_part_ckpt = -1;
            if (sp_ok && win && !ov_window()) { std::printf("ERR %s\n", err.c_str()); return 1; }
            if (!sp_ok) {
                if (yielded_at < 0 && (batch_fatal || !(stop_req.load() || stdin_lost.load() || ram_stop.load()))) {
                    basalt::log::error("basalt serve", "%s", err.c_str());
                    std::printf("ERR %s\n", err.c_str());
                    // a CUDA fault (an illegal address) poisons the context for the whole process, and unwinding
                    // the destructors on it could hang until the watchdog: leave at once
                    if (cudaPeekAtLastError() != cudaSuccess) {
                        std::fflush(stdout);
                        std::_Exit(1);
                    }
                    return 1;
                }
                cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                break;
            }
            at = to;
            resumed_chunk = 0;   // the later parts are planned as the read they continue planned them
            ov_post = nullptr;
            if ((to == turn_at || to == root_at) && !ov_post_done && !checkpoint_at(to, nullptr, !in_slot && !batch_on())) {
                std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                return 1;
            }
        }
        if (!refill(err)) {
            std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
            return 1;
        }
        if (il_parts > 0)
            basalt::log::info("basalt batch", "the prompt was read in %lld parts, the slots decoding %.0f ms between "
                                 "them", (long long) il_parts + 1, il_ms);
        const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
        std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
        std::fflush(stdout);
        // the verify windows: the first holds the last prompt token, with the rest of the last window
        if (mtp_on && !mtp.use_main(err)) { std::printf("ERR %s\n", err.c_str()); return 1; }   // the windows between prompt chunks swapped it
        int64_t p = fold_p0;
        int32_t x = (int32_t) ids[(size_t) fold_p0];
        {   // a request decoded here starts its adaptation from the experts its own prompt routed to
            std::vector<uint32_t> hist;
            int64_t hist_tokens = 0;
            const bool seeded = !drive.d.usage.empty() && sp.route_hist_take(hist, hist_tokens) && (!in_slot || solo_req);
            // a split read counted its stages' layers there
            if (seeded && split_hist.size() == hist.size())
                for (size_t i = 0; i < hist.size(); ++i) hist[i] += split_hist[i];
            split_hist.clear();
            if (seeded) seed_usage_from_prompt(drive.d.usage, hist, hist_tokens, host_res, g.n_expert, peer);
        }
        std::vector<int32_t> drafts((size_t) std::max(S, basalt::core::kMtpChainMax), 0), window((size_t) S), outv((size_t) S);
        std::vector<float> dprob((size_t) std::max(S, basalt::core::kMtpChainMax), 0.0f);
        basalt::core::DraftCumAdapt cum_adapt;   // this request's draft threshold
        std::vector<int32_t> sbuf((size_t) S, 0);
        if (o.suffix_draft > 0) {
            sfx.reset();
            for (int64_t t : ids) sfx.append((int32_t) t);
        }
        bool first_window = true;
        int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        int64_t draft_offered = 0, draft_accepted = 0;
        // what the session holds once this request is done: the prompt read so far, then every committed token
        std::vector<int32_t> consumed;
        consumed.reserve((size_t) (n + max_new + S));
        for (int64_t i = 0; i < fold_p0; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
        const char* finish = "length";
        std::string kv_stop;   // the elastic KV could not back the next window: the request ends on this error
        // pipelined windows: `cv` runs the next window, `ov` is the other chain; a guess that held leaves the next
        // window's verdict in `held`
        basalt::core::Verifier* cv = &ver;
        basalt::core::Verifier* ov = &ver_b;
        struct HeldWindow { bool on = false; int T = 0; std::vector<int32_t> win, out; } held;
        held.win.assign((size_t) S, 0);
        held.out.assign((size_t) S, 0);
        std::vector<int32_t> win_b((size_t) S, 0);
        int n_chain = 0;                       // drafts the last chain produced
        double full_ema = 0.5;                 // how often windows are accepted whole
        bool pair_hint = false;                // windows are accepted whole often enough to run the chain on for a guess
        const bool pair_req = pipe_on && ppl == nullptr && !relax_on && gguard == nullptr && hist_n == 0 && !in_slot &&
                              kv_floor == 0;
        if (pipe_on) mtp.set_max_drafts((pair_req ? basalt::core::kMtpChainMax : S_mtp) - 1);
        int64_t pair_runs = 0, pair_holds = 0;
        const Clock::time_point d0 = Clock::now();
        // --decode-timing: where a request's decode time goes (one line per request)
        const bool dec_timing = basalt::core::diagnostics().decode_timing;
        struct DecSnap {
            double wait, pool, host, plan, actq, jobs, run, peer_wait;
            int64_t misses, entries, hits, pcie, peer;
        };
        auto dec_snap = [&]() {
            return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                           drive.d.ms_run, drive.d.peer != nullptr ? drive.d.peer->ms_wait : 0.0,
                           drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits, drive.d.pcie_experts,
                           drive.d.peer_entries};
        };
        const DecSnap ds0 = dec_snap();
        double dt_run = 0, dt_commit = 0, dt_draft = 0;
        int64_t dec_windows = 0, dec_T = 0;
        const int64_t decode_hits0 = drive.d.cache_hits;
        // CS-T: the RAM and file tiers of this request (the mmap source; 0 with the arena)
        const int64_t ram0 = src.ram_reads(), files0 = src.file_reads();
        const uint64_t file_bytes0 = src.file_read_bytes();
        const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
        const int64_t offload0 = drive.d.offload_entries;
        static const bool chain = basalt::core::Verifier::chain_on();
        int adapt_age = 2;   // loop tops since the last adaptive round
        bool applied_early = false;
        JobRunner apply_thr, adapt_thr;
        bool copy_live = false, copy_done = false;   // exchanges applied early whose RAM copies are under way
        if (cancelled) finish = "cancel";
        // --draft-log FILE: one line per window (T, accepted, draft probabilities, ms) for draft-policy study
        static FILE* const draft_log = basalt::core::diagnostics().draft_log.empty() ? nullptr
            : std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
        if (draft_log != nullptr) {
            std::fprintf(draft_log, "R temp=%g S=%d\n", (double) req_temperature, S_mtp);
            std::fflush(draft_log);
        }
        // spec_log: per window, the target's sampling distribution at each verified row beside the draft head's top-4
        // of the draft that row checks, for offline tree-draft studies (written next to --draft-log as FILE.tk)
        static FILE* const tk_log = basalt::core::exp_flag("spec_log", 0) != 0 && !basalt::core::diagnostics().draft_log.empty()
            ? std::fopen((basalt::core::diagnostics().draft_log + ".tk").c_str(), "a") : nullptr;
        struct TkStep { int n = 0; int32_t id[4] = {}; float p[4] = {}; };
        TkStep tk_prev[16];
        bool tk_valid = false;
        if (tk_log != nullptr) { std::fprintf(tk_log, "R temp=%g S=%d\n", (double) req_temperature, S_mtp); std::fflush(tk_log); }
        while (!cancelled && produced_n < max_new && (!in_slot || solo_req)) {   // (a batch slot's first token comes from the batch window)
            int T = 1;
            if (held.on) T = held.T;
            else
            while (T < S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
            if (first_window) T = (int) (n - fold_p0);
            const int e0 = first_window ? T - 1 : 0;   // rows before it are prompt tokens: forced, not emitted
            if (ppl != nullptr) {
                if (p >= ppl->end()) break;
                T = (int) std::min<int64_t>(req_ppl_t, ppl->end() - p);
            }
            // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
            // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window && ppl == nullptr && !held.on) {
                const int k = sfx.propose(S - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && sbuf[0] == drafts[0]) {
                    const basalt::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            const Clock::time_point round0 = Clock::now();
            if (p + T > o.max_context) break;
            window[0] = x;
            for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
            if (held.on)
                for (int i = 0; i < T; ++i) window[(size_t) i] = held.win[(size_t) i];
            if (e0 > 0)
                for (int i = 1; i < T; ++i) window[(size_t) i] = (int32_t) ids[(size_t) (p + i)];
            if (ppl != nullptr)
                for (int i = 0; i < T; ++i) window[(size_t) i] = ppl->token(p + i);
            // the guess: this window is accepted whole and the target's next token is the chain's draft T - 1; the next
            // window then holds that draft and the ones the chain made after it
            int TB = 0;
            const bool try_pair = pair_req && pair_hint && !held.on && !first_window && !from_sfx && e0 == 0 &&
                                  p + T + S_mtp + 64 < o.max_context;
            const std::function<int(int32_t*)> guess = [&](int32_t* wb) -> int {
                n_chain = mtp.draft_more(n_chain, drafts.data(), dprob.data());   // made beside this window's stage 0
                if (n_chain <= T || !basalt::core::draft_keep(dprob.data(), T - 1, kPairTheta)) return 0;
                int tb = 1;
                while (tb < S_mtp && T - 1 + tb < n_chain && basalt::core::draft_keep(dprob.data() + T, tb - 1, cum_adapt.th()))
                    ++tb;
                if (tb < 2) return 0;
                wb[0] = drafts[(size_t) T - 1];
                for (int i = 1; i < tb; ++i) wb[i] = drafts[(size_t) T - 1 + (size_t) i];
                return tb;
            };
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert
            // ran on the GPU or the CPU (they round differently) would depend on the copy's timing
            basalt::core::decode_gaps().mark(17);
            // an adaptive round's swaps land before the second window after it (a fixed window, so reproducible),
            // applied beside the draft before it (apply_thr) when their copies are done; what that moved to the
            // peer lands one window later
            if (apply_thr.joinable()) apply_thr.join();
            if (copy_live && copy_done) {   // the RAM copies are done: their bookkeeping, between windows
                basalt::core::decode_events().tag(1u << 3);
                src.commit_exchanges();
                copy_live = false;
            }
            ++adapt_age;
            if (!chain || (adapt_age >= 2 && !(applied_early && adapt_age == 2))) {
                if (!pending.empty()) basalt::core::decode_events().tag(1u << 1);
                apply_pending(true);
            }
            // the elastic KV: the next window's cells backed before it runs (between windows, rare)
            if (kv_floor > 0 && p + T + 64 > basalt::core::session_kv_cells(*rsp)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                if (copy_live) { src.commit_exchanges(); copy_live = false; }   // as at the request's end
                if (!kv_fit(*rsp, p + kKvGrowCells, err, true, p + kKvWindowCells)) {
                    if (kv_short) {
                        kv_stop = err;
                        break;
                    }
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
            }
            basalt::core::decode_gaps().mark(18);
            if (hist_n > 0) {
                // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the fed-back
                // head `x` (it joins `consumed` only after this window commits), then the drafts before that row -
                // what plain decode would have counted there
                basalt::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                              hist_n, hist_stage.data());
                const basalt::core::OnDevice on_h(hist_dev);
                cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
            }
            basalt::core::decode_gaps().mark(19);
            basalt::core::decode_events().stamp(3);
            const Clock::time_point tw0 = Clock::now();
            void* const user_c = cv == &ver ? win_pool_user : (void*) &split_drive_b;
            void* const user_o = cv == &ver ? (void*) &split_drive_b : win_pool_user;
            const bool ran = held.on ? true
                : try_pair ? cv->run_pair(*ov, T, window.data(), p, guess, win_b.data(), TB, win_pool_fn, user_c, user_o,
                                          outv.data(), err)
                           : cv->run(T, window.data(), p, win_pool_fn, user_c, outv.data(), err);
            const bool paired = TB > 0;
            const bool held_run = held.on;
            if (held.on) {
                for (int i = 0; i < T; ++i) outv[(size_t) i] = held.out[(size_t) i];
                held.on = false;
            }
            if (!ran || drive.d.failed) {
                std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                return 1;
            }
            int a = 0;
            if (e0 > 0) {
                for (int i = 0; i < e0; ++i) outv[(size_t) i] = window[(size_t) i + 1];
                a = T - 1;
                if (relax_on) {
                    if (outv[(size_t) a] == o.relax_call_open) in_call = true;
                    else if (outv[(size_t) a] == o.relax_call_close) in_call = false;
                }
            } else if (relax_on) {
                // a draft the target's draw disagrees with is kept when the target gives it probability >= P,
                // outside tool calls and never for an added token; the row then emits the draft
                bool call = in_call;
                while (a < T - 1) {
                    const int32_t d = window[(size_t) a + 1];
                    if (d != outv[(size_t) a]) {
                        if (call || d >= o.relax_special_from || cv->draft_prob(a) < req_relax) break;
                        outv[(size_t) a] = d;
                        ++relaxed_kept;
                    }
                    if (d == o.relax_call_open) call = true;
                    else if (d == o.relax_call_close) call = false;
                    ++a;
                }
                for (int i = 0; i <= a; ++i)
                    if (outv[(size_t) i] == o.relax_call_open) in_call = true;
                    else if (outv[(size_t) i] == o.relax_call_close) in_call = false;
            } else
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (gguard != nullptr && gguard->on()) {
                // the first sampled token the grammar refuses is drawn again under the grammar's mask, and the window ends there
                const int miss = gguard->scan(outv.data(), a + 1);
                if (miss >= 0) {
                    gguard->mask_bias(req_bias, grammar_bias);
                    int32_t gtok = 0;
                    if (!cv->resample_row(miss, p, grammar_bias, gtok, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    outv[(size_t) miss] = gtok;
                    a = miss;
                    if (!gguard->take(gtok))
                        basalt::log::info("basalt grammar", "the masked draw at row %d of %d was token %d", miss, T, (int) gtok);
                    ++grammar_resamples;
                }
            }
            if (ppl != nullptr) {   // every row scored and committed
                a = T - 1;
                for (int t = 0; t < T; ++t) {
                    std::vector<float> lg((size_t) n_vocab);
                    if (!cv->copy_logits(t, lg.data()) || !ppl->score(p + t, std::move(lg), err)) {
                        std::printf("ERR ppl: %s\n", err.empty() ? "the head logits copy failed" : err.c_str());
                        return 1;
                    }
                    outv[(size_t) t] = ppl->token(p + t + 1);   // as if every draft were accepted: x, the MTP
                }
            }
            // the guess held when the window was accepted whole, its last pick is the next window's first token, and
            // the request goes on past it
            if (!from_sfx && e0 == 0 && !held_run) full_ema = 0.9 * full_ema + (a == T - 1 ? 0.1 : 0.0);
            bool hold = false;
            if (paired) {
                ++pair_runs;
                int n_emit_h = 0;
                bool eos_h = false;
                for (int i = e0; i <= a && produced_n + n_emit_h < max_new && !eos_h; ++i) {
                    ++n_emit_h;
                    eos_h = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                hold = a == T - 1 && outv[(size_t) a] == win_b[0] && !eos_h && produced_n + n_emit_h < max_new && !halted();
                if (!hold && !cv->pair_miss(*ov, win_pool_fn, user_o, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
            }
            if (logit_hash)   // the accepted rows' logits, per position (equality of the batch rows with these)
                for (int t = 0; t <= a; ++t) basalt::log::raw("LH -1 %lld %016llx\n", (long long) (p + t), (unsigned long long) row_hash(t));
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            if (tk_log != nullptr && tk_valid && !from_sfx && ppl == nullptr && req_temperature > 0.0) {
                std::vector<float> lg((size_t) n_vocab);
                std::fprintf(tk_log, "W %d %d\n", T, a);
                for (int j = 0; j < T; ++j) {
                    if (j > a) {   // past the first miss: only the draft side
                        std::fprintf(tk_log, "R %d - - T D", j);
                        for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                        std::fputc('\n', tk_log);
                        continue;
                    }
                    if (!cv->copy_logits(j, lg.data())) break;
                    // top_k, top_p, then temperature over the survivors: the sampler chain without penalties
                    std::vector<int> ord((size_t) n_vocab);
                    for (int i = 0; i < n_vocab; ++i) ord[(size_t) i] = i;
                    const int K = std::max(1, std::min<int>(req_top_k > 0 ? req_top_k : 20, (int) n_vocab));
                    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) {
                        return lg[(size_t) x] > lg[(size_t) y] || (lg[(size_t) x] == lg[(size_t) y] && x < y);
                    });
                    const double mx = lg[(size_t) ord[0]];
                    std::vector<double> e((size_t) K);
                    double s1 = 0.0;
                    for (int i = 0; i < K; ++i) s1 += (e[(size_t) i] = std::exp((double) lg[(size_t) ord[(size_t) i]] - mx));
                    int keep = K;
                    double cum = 0.0;
                    for (int i = 0; i < K; ++i) { cum += e[(size_t) i] / s1; if (cum >= (double) req_top_p) { keep = i + 1; break; } }
                    double s2 = 0.0;
                    for (int i = 0; i < keep; ++i) s2 += (e[(size_t) i] = std::exp(((double) lg[(size_t) ord[(size_t) i]] - mx) / (double) req_temperature));
                    std::fprintf(tk_log, "R %d %d %d", j, j + 1 < T ? (int) window[(size_t) j + 1] : -1, (int) outv[(size_t) j]);
                    std::fprintf(tk_log, " T");
                    for (int i = 0; i < std::min(keep, 8); ++i) std::fprintf(tk_log, " %d:%.5f", ord[(size_t) i], e[(size_t) i] / s2);
                    std::fprintf(tk_log, " D");
                    for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                    std::fputc('\n', tk_log);
                }
                std::fflush(tk_log);
            }
            tk_valid = false;
            const Clock::time_point tw1 = Clock::now();
            bool adapt_ok = true;
            const bool do_adapt = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0 &&
                                  (adapt_fast ? xq_stage == 0
                                              : (!chain || (pending.empty() && !copy_live && !src.exchanges_staged())));
            auto spawn_adapt = [&] {
                if (adapt_fast) {   // a round, or the next step of the one in flight, beside the draft
                    if (!do_adapt && xq_stage == 0) return;
                    if (do_adapt) adapt_age = 0;
                    basalt::core::decode_events().tag(1u << 0);
                    basalt::core::decode_events().tag(1u << (6 + xq_stage));
                    adapt_thr.run([&] {
                        const auto ta = Clock::now();
                        basalt::core::decode_events().stamp(8);
                        adapt_ok = xq_stage != 0 ? xq_advance(false) : xq_round();
                        basalt::core::decode_events().stamp(6);
                        std::string perr;
                        if (adapt_ok && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                        basalt::core::decode_events().stamp(7);
                        basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                    });
                    return;
                }
                if (!do_adapt) return;
                adapt_age = 0;
                adapt_thr.run([&] {
                    const auto ta = Clock::now();
                    adapt_ok = adapt();
                    if (chain && adapt_ok) {   // the evictions' residency, uploaded beside the draft
                        std::string perr;
                        if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    }
                    basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                });
            };
            if (chain) {
                // the draft launches first; the commit, the adaptive round and the output go in its shadow
                int n_emit = 0;
                bool eos_pre = false;
                for (int i = e0; i <= a && produced_n + n_emit < max_new && !eos_pre; ++i) {
                    ++n_emit;
                    eos_pre = ppl == nullptr &&
                              std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                const bool will_draft = !eos_pre && produced_n + n_emit < max_new;
                float wprob[16];
                const int wprob_n = std::min(S_mtp, 16);
                if (draft_log != nullptr)
                    for (int i = 0; i < wprob_n; ++i) wprob[i] = first_window || from_sfx ? -1.0f : dprob[(size_t) i];
                // the adaptive round and the swaps' residency, beside the commit and the draft
                auto adapt_side = [&] {
                    spawn_adapt();
                    // the swaps' residency at the next window, then their RAM copies a share per draft
                    applied_early = adapt_age == 1 && !do_adapt && !pending.empty();
                    if (applied_early || copy_live) {
                        const bool boundary = applied_early, copies = applied_early || copy_live;
                        if (copies) {
                            copy_live = true;
                            copy_done = false;
                        }
                        basalt::core::decode_events().tag(1u << 2);
                        apply_thr.run([&, boundary, copies] {
                            if (boundary) {
                                apply_pending(true, false);
                                std::string perr;
                                if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                            }
                            if (copies) copy_done = !src.exchanges_staged() || src.copy_exchanges(8ull << 20);
                        });
                    }
                };
                basalt::core::decode_events().stamp(9);
                if (!hold && !cv->commit(a + 1, err, 2)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                draft_offered += T - 1 - e0;
                draft_accepted += a - e0;
                first_window = false;
                bool commit_ok = true, after_ran = false, eos = false;
                std::string cerr;
                auto after_launch = [&] {
                    after_ran = true;
                    commit_ok = hold ? true : cv->commit(a + 1, cerr, 1);
                    if (!hold) adapt_side();
                    for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                        std::printf("T %d\n", (int) outv[(size_t) i]);
                        basalt::core::progress_beat();
                        ++produced_n;
                        if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                        eos = ppl == nullptr &&
                              std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                    }
                    std::fflush(stdout);
                };
                ++rounds;
                const Clock::time_point tw2 = Clock::now();
                basalt::core::decode_gaps().mark(5);
                if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
                mtp.set_cum_th(cum_adapt.th());
                bool drafted = true;
                if (hold) {
                    // the tokens go out; the next window's last stage runs behind this window's commit while its first
                    // stage finishes, and the drafter (on the first card) takes this window's rows once that card is
                    // free
                    ++pair_holds;
                    if (!cv->pair_hold_commit(T, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    after_launch();
                    const float* staged = nullptr;
                    drafted = mtp.stage_rows(cv->final_R_all(), T, &staged, err);   // before the last card is busy again
                    if (!drafted || !cv->pair_hold_begin(*ov, T, p, TB, win_b.data(), err) ||
                        !cv->pair_hold_finish_first(*ov, win_pool_fn, user_o, err) ||
                        !(drafted = mtp.prefill(staged, outv.data(), T, p, err))) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    if (!cv->pair_hold_finish(*ov, win_pool_fn, user_o, held.out.data(), err) || drive.d.failed) {
                        std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                        return 1;
                    }
                    held.on = true;
                    held.T = TB;
                    for (int i = 0; i < TB; ++i) held.win[(size_t) i] = win_b[(size_t) i];
                    std::swap(cv, ov);
                } else if (will_draft && mtp_on) {
                    mtp.set_launch_hook(after_launch);
                    pair_hint = pair_req && full_ema >= kPairFull;
                    mtp.set_cum_th(pair_hint ? cum_adapt.th() * kPairChainScale : cum_adapt.th());
                    mtp.set_host_th(cum_adapt.th());
                    drafted = mtp.draft_from(cv->final_R_all(), T, outv.data(), p, a, drafts.data(), err, dprob.data(), &n_chain);
                    mtp.set_launch_hook(nullptr);
                    if (drafted && pipe_on) {   // the next window's first stage starts after the chain's kernels
                        const cudaEvent_t ev = mtp.done_event();
                        ver.set_wait_event(ev);
                        ver_b.set_wait_event(ev);
                    }
                    if (tk_log != nullptr && drafted) {
                        for (int j = 0; j < 16 && j < S_mtp; ++j) {
                            tk_prev[j].n = mtp.diag_top(j, 4, tk_prev[j].id, tk_prev[j].p);
                        }
                        tk_valid = true;
                    }
                }
                if (!after_ran) after_launch();
                basalt::core::decode_events().stamp(0);
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += T;
                    if (draft_log != nullptr) {
                        std::fprintf(draft_log, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0),
                                     msd(tw0, tw1), msd(tw1, tw3));
                        for (int i = 0; i < wprob_n; ++i) std::fprintf(draft_log, " %.4f", wprob[i]);
                        std::fputc('\n', draft_log);
                    }
                }
                if (adapt_thr.joinable()) adapt_thr.join();
                basalt::core::decode_events().stamp(1);
                basalt::core::decode_gaps().mark(16);
                if (!commit_ok) {
                    std::printf("ERR %s\n", cerr.c_str());
                    return 1;
                }
                if (!adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (!drafted) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (timed_round && !eos)
                    policy.observe(from_sfx, T, a, sfx_match,
                                   std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                basalt::core::decode_gaps().mark(17);
                if (eos) { finish = "stop"; break; }
                shed_if_low();
                if (halted()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
                basalt::core::decode_events().stamp(2);
                continue;
            }
            spawn_adapt();
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
            for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
            draft_offered += T - 1 - e0;
            draft_accepted += a - e0;
            first_window = false;
            bool eos = false;
            for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                std::printf("T %d\n", (int) outv[(size_t) i]);
                basalt::core::progress_beat();
                ++produced_n;
                if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                eos = ppl == nullptr &&
                      std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            std::fflush(stdout);
            ++rounds;
            const Clock::time_point tw2 = Clock::now();
            basalt::core::decode_gaps().mark(5);
            if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
            mtp.set_cum_th(cum_adapt.th());
            const bool drafted = eos || produced_n >= max_new || !mtp_on ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data());
            {
                const Clock::time_point tw3 = Clock::now();
                auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                ++dec_windows; dec_T += T;
            }
            if (adapt_thr.joinable()) adapt_thr.join();
            basalt::core::decode_gaps().mark(16);
            if (!adapt_ok) {
                std::printf("ERR an adaptive refill failed\n");
                return 1;
            }
            if (!drafted) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (timed_round && !eos)
                policy.observe(from_sfx, T, a, sfx_match,
                               std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
            basalt::core::decode_gaps().mark(17);
            if (eos) { finish = "stop"; break; }
            shed_if_low();
            if (halted()) { finish = "cancel"; break; }
            x = outv[(size_t) a];
            p += a + 1;
        }
        if (gguard != nullptr)
            basalt::log::info("basalt grammar", "%lld windows checked, %lld masked resamples, %.3f ms in grammar code (%.2f us/window)",
                         (long long) gguard->windows(), (long long) grammar_resamples, gguard->ms(),
                         gguard->windows() > 0 ? gguard->ms() * 1000.0 / (double) gguard->windows() : 0.0);
        if (apply_thr.joinable()) apply_thr.join();
        if (copy_live) src.commit_exchanges();   // swaps applied early: the rest of their copies
        copy_live = false;
        if (ppl != nullptr && !ppl->finish(err)) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
        // the last commit (set_commit_async): the session is complete before anything reads or copies it
        if (!ver.wait_commit(err) || (pipe_on && !ver_b.wait_commit(err))) {
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        if (pipe_on)
            basalt::log::info("basalt serve", "pipelined windows: %lld launched, %lld held", (long long) pair_runs,
                              (long long) pair_holds);
        if (dec_timing && dec_windows > 0) {
            const DecSnap d1 = dec_snap();
            const double w = (double) dec_windows, L = (double) g.n_layers;
            auto fmt2 = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return std::string(b); };
            basalt::log::info("basalt decode timing", "%lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                 "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                 "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                 "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f%s",
                         (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                         (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                         (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                         (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                         (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L),
                         drive.d.peer == nullptr ? "" : (", peer GPU " + fmt2((d1.peer - ds0.peer) / (w * L)) +
                                                         " (the pool waits " + fmt2((d1.peer_wait - ds0.peer_wait) / w) +
                                                         " ms/window for it)").c_str());
            const std::string evr = basalt::core::decode_events().report();
            if (!evr.empty()) basalt::log::raw("%s\n", evr.c_str());
            const std::string pr = ver.profile_report();
            if (!pr.empty()) basalt::log::raw("basalt decode GPU stages (ms/window):%s\n", pr.c_str());
        }
        if (!cancelled) {
            // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
            // (the checkpoints taken while reading it are still good)
            live.swap(consumed);
            live_imgs = imgs_below(req_imgs, (int64_t) live.size());
            live_ok = o.prompt_cache > 0;
        }
        ckd_finish();
        const int64_t req_hits = drive.d.cache_hits - decode_hits0;
        const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
        const int64_t req_offload = drive.d.offload_entries - offload0;
        // the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a cancel
        // stopped it part-way
        const int64_t fresh = n - resume;
        const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
        //      [RAM blobs] [file blobs] [file MB]   (file tiers; appended, so an older server reads the rest)
        //      [prompt tokens read]   (fewer than <prompt> - <reused> when a cancel stopped the read)
        //      [offloaded]   (the decode's routed experts the GPU read over PCIe or another GPU computed;
        //                    not in [lookups])
        const bool ram_out = ram_stop.load() || !kv_stop.empty();
        if (!kv_stop.empty()) {
            basalt::log::warn("basalt serve", "request stopped after %lld tokens: %s", (long long) produced_n,
                              kv_stop.c_str());
            std::printf("ERR %s\n", kv_stop.c_str());
        } else if (ram_out) {
            // the client gets the reason as the request's error; the engine stays up for the next request
            uint64_t avail = 0, floor = 0;
            basalt::platform::MemoryStalls stalls;
            (void) basalt::platform::ram_guard_level(&avail, &floor, &stalls);
            // what the engine grew by since it was ready, and what the rest of the drop in free RAM went to
            const uint64_t rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
            const uint64_t own = rss > ready_rss ? rss - ready_rss : 0;
            const uint64_t drop = ready_avail > avail ? ready_avail - avail : 0;
            const std::string msg = basalt::platform::ram_stop_message(avail, floor, basalt::platform::kRamHardFloor,
                                                                       stalls, drop > own ? drop - own : 0, own);
            basalt::log::error("basalt", "%s", msg.c_str());
            std::printf("ERR %s\n", msg.c_str());
        }
        if (!ram_out && yielded_at >= 0) std::printf("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
        if (!ram_out) std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                    (long long) produced_n,
                    (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                    (long long) resume, (long long) req_hits, (long long) req_look,
                    (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                    (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                    (long long) req_offload);
        std::fflush(stdout);
        // no swap stays in flight between requests: the last round's copies land once the reply is out
        if (!xq_drain()) {
            std::printf("ERR an adaptive refill failed\n");
            return 1;
        }
        xq_saturated = false;   // the next request starts at the base cap
        if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
            bool cont = ram_out ? false : in_slot ? !cancelled && (int64_t) live.size() == p
                                : !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                                  (int64_t) live.size() == p;
            const auto tc0 = Clock::now();
            // the draft layer's cell for the prompt's last token (it pairs with the first generated one), then its K/V
            if (cont && !in_slot && batch_spec && !mtp.prefill(ver.final_R_all(), &x, 1, p - 1, err)) {
                basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                cont = false;
            }
            if (cont && in_slot) {   // the slot holds the prompt already; only its draft K/V is staged
                basalt::core::SessionState& t = *bslot_ss[0][(size_t) admit_slot];
                t.ple_prev[0] = rview.ple_prev[0];
                t.ple_prev[1] = rview.ple_prev[1];
                t.ple_token = rview.ple_token;
                if (batch_spec && !mtp.copy_store(mtp.main_store(), admit_slot, (int64_t) live.size(), err)) {
                    basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                    cont = false;
                }
            } else if (cont && !copy_to_slot(admit_slot, live, err)) {
                basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                bs[(size_t) admit_slot].cached = false;   // its sessions may be half written
                cont = false;
            }
            if (cont) {
                if (slot0_is_solo && admit_slot == 0) live_ok = false;   // the batch windows move the state on
                ver.set_slot_sampling(admit_slot, req_sp);   // the request's own sampling, row by row
                BSlot& sl = bs[(size_t) admit_slot];
                sl = BSlot{};
                sl.active = true;
                sl.script = std::move(bscript[(size_t) admit_slot].first);
                sl.rng.seed(bscript[(size_t) admit_slot].second);
                bscript[(size_t) admit_slot].first.clear();
                if (batch_spec && o.suffix_draft > 0) {
                    sl.sfx = std::make_unique<basalt::spec::SuffixDrafter>(std::max(1, o.suffix_draft), 64,
                                                                           (size_t) o.max_context + 4096);
                    for (int32_t t : live) sl.sfx->append(t);
                    sl.sfx->append(x);
                }
                sl.x = x;
                sl.p = p;
                sl.produced = in_slot ? 0 : 1;
                sl.max_new = admit_max_new;
                sl.t0 = Clock::now();
                sl.ids = live;
                sl.cvec = cvec_cached;
                sl.imgs = live_imgs;
                                sl.guard = std::move(gguard);   // the request's grammar, already past the tokens the request decoded here
                // the slot keeps its conversation's longest checkpoint (its next turn) and its shortest (the
                // system prompt, which other requests may share)
                const ConvCheckpoint *best = nullptr, *root = nullptr;
                for (const ConvCheckpoint& c : checks)
                    if (c.ids.size() < live.size() &&
                        basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), live.data())) {
                        if (best == nullptr || c.ids.size() > best->ids.size()) best = &c;
                        if (root == nullptr || c.ids.size() < root->ids.size()) root = &c;
                    }
                if (root != nullptr && root != best) sl.checks.push_back(*root);
                if (best != nullptr) sl.checks.push_back(*best);
                basalt::log::info("basalt batch", "slot %d takes %lld tokens (copied in %.1f ms)", admit_slot,
                             (long long) live.size(), std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
            }
            if (!ram_out)   // (after ERR the server reads no BADM)
                std::printf("BADM %d %d\n", admit_slot, cont ? (in_slot ? 2 : 1) : 0);   // 2: no token yet
            std::fflush(stdout);
            admit_slot = -1;
        }
        if (solo_req) {   // the conversation stays in the slot it ran in
            basalt::core::SessionState& t = *bslot_ss[0][(size_t) slot_b];
            t.ple_prev[0] = rview.ple_prev[0];
            t.ple_prev[1] = rview.ple_prev[1];
            t.ple_token = rview.ple_token;
            if (yielded_at < 0) {   // (a read that gave way was parked in the slot already)
                BSlot& sb = bs[(size_t) slot_b];
                sb = BSlot{};
                sb.ids = live;
                sb.cvec = cvec_cached;
                sb.imgs = live_imgs;
                sb.cached = live_ok && o.prompt_cache > 0;
                if (sb.cached) sb.checks = std::move(checks);
                if (sb.cached && batch_spec && !mtp.copy_store(mtp.main_store(), slot_b, (int64_t) live.size(), err)) {
                    err.clear();
                    sb.cached = false;
                }
            }
        }
        if (in_slot) {   // the solo session and its bookkeeping again
            if (solo_req && !ver.bind_session(&ss, err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            rsp = &ss;
            sp.set_session(ss);
            if (mtp_on) mtp.set_ple_session(&ss);
            live = std::move(solo_book.live);
            live_imgs = std::move(solo_book.live_imgs);
            live_ok = solo_book.live_ok;
            cvec_cached = solo_book.cvec;
            checks = std::move(solo_book.checks);
            if (solo_moved) {   // its cells went to this request (hold_slot)
                live.clear();
                live_imgs.clear();
                live_ok = false;
                checks.clear();
            }
        }
        if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
        // "12288 of 98179" when cancelled mid-read, the rate from what was read
        char read_txt[64];
        if (cancelled)
            std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
        else
            std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
        basalt::log::info("basalt serve", "prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                             "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s",
                     (long long) n, (long long) resume, read_txt, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                     decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                     (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
        if (relax_on)
            basalt::log::info("basalt serve", "relaxed drafting %.2f: %lld of the %lld accepted drafts were kept by it",
                         (double) req_relax, (long long) relaxed_kept, (long long) draft_accepted);
        // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU to
        // read are in neither count, so that share is said beside it
        if (req_look > 0) {
            char off[160] = "";
            if (req_offload > 0)
                std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                              "all %lld routed)", (long long) req_offload,
                              100.0 * (double) req_offload / (double) (req_look + req_offload),
                              (long long) (req_look + req_offload));
            basalt::log::info("basalt serve", "decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s",
                         100.0 * (double) req_hits / (double) req_look,
                         (long long) req_hits, (long long) req_look, off);
        }
        // the resident RAM mode, cumulative: experts read from the experts section since the copy was made (what the plain
        // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
        if (src.complement_ready())
            basalt::log::info("basalt serve", "resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                 "VRAM tier, %lld blob reads from the file",
                         (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                         (long long) src.file_reads());
        // CS-T: the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
        if (srcp == &src)
            basalt::log::info("basalt serve", "expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                                 "%.1f MB read", (long long) req_hits, (long long) src.ram_reads(),
                         (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6);
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
            // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
            // (this device's owned ordinals; a split's other stages hold theirs)
            uint64_t miss = 0, look = 0;
            bool over = false;
            for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                const basalt::kernels::KvStreamCounters c =
                    basalt::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
                miss += c.misses; look += c.lookups; over = over || c.overflow;
            }
            basalt::log::info("basalt serve", "KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                 "from RAM%s", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                         (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                         over ? " - OVERFLOW (too few resident cells)" : "");
        }
        if (sfx_windows > 0)
            basalt::log::info("basalt serve", "suffix drafts: %lld windows, %lld of %lld drafts accepted",
                         (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            basalt::log::info("basalt serve", "CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                 "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                 "waiting for it in this request", r + 1,
                         (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                         (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                         (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                         (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                         remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                         remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);
    }
    // the conversations this process holds go to the disk tier before it ends
    if (disk.enabled()) {
        const auto t0 = Clock::now();
        const size_t written = spill_to_disk(true, SIZE_MAX);
        const auto st = disk.stats();
        basalt::log::info("basalt serve", "conversation cache: %zu conversations written to disk at exit in %.0f ms; "
                          "on disk %zu (%.2f GiB); since the start written %zu (%.2f GiB, %.2f GiB with the shared "
                          "files they reuse), read back %zu (%.2f GiB), evicted %zu, dropped %zu", written,
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), st.files,
                          (double) st.bytes / 1073741824.0, st.written, (double) st.written_bytes / 1073741824.0,
                          (double) st.whole_bytes / 1073741824.0, st.restored, (double) st.read_bytes / 1073741824.0,
                          st.evicted, st.dropped);
    }
    save_profile("exit");   // QUIT, or the server closed stdin
    return 0;
}

}  // namespace basalt::program::serve
