# DFlash2

The DFlash2 drafter uses the target's existing token embedding and native output head, over the rows a pack's
`proposal_ids` shortlist names when it has one (see [round cost](#round-cost-and-window-width)); a shortlisted token's
logits are the full head's logits, because the rows are the head's own bytes. Its BF16 checkpoint is stored
unchanged in `dflash2/model`; norms and convolution base kernels are widened exactly to FP32 at load. All projection
weights and selector codebooks stay BF16. Projection inputs, accumulation, convolution, normalization and attention
are FP32. The head uses its existing Q8_1 activation contract. No NVFP4 target weight is converted.

For a Q8_0 output head the drafter uses `q80_head_mmvq`, the same specialized head kernel as target verification.
Its logits are bitwise equal to `native_mmvq` across window widths and column positions. Other head formats use
their native MMVQ kernels. This dispatch preserves the head's output numerics.

The Q4_K variant quantizes projection matrices directly from BF16 with Basalt's ggml reference quantizer. Each
256-value block stores 4-bit codes, group scales and minima, and FP16 block scales. Norms, convolution kernels and
selector codebooks retain the BF16 source values. Projection inputs are rounded to Q8_1; the existing dense
tensor-core path accumulates the integer dots and FP32 scale terms. The attention and selector arithmetic stays FP32.
The block and context branches have separate activation scratch, so their captured graphs cannot overwrite one
another's inputs. This changes drafter numerics and can change acceptance and verify-window widths.

Features are the completed outputs of target layers 5, 15, 25, 35 and 45, averaged across the four residual streams.
The prompt path captures the most recent rows of each chunk; verification captures each window's rows. Only the
accepted rows enter the drafter's context ring. Draft block keys are separate from that ring, so a rejected draft
does not leave speculative keys in the context.

The checkpoint configures eight positions: one anchor and seven masks. DFlash2 `spec` selects 2–16 positions at
inference, including the anchor. Five draft layers process the selected block together. The native head proposes
sixteen candidates per position, and the rank-256 selector chooses a path. The selector's value is `1/z` over those
sixteen candidates, not the token's own probability, and it sizes the next window through the same cumulative rule the
MTP chain uses (`draft_cum.hpp`); neither it nor the window replaces the target's sampling or acceptance decisions.
Projections and the native head process at most eight columns per launch; larger blocks use consecutive launches.
Attention tiles eight query positions while attending to the full block. The target verifier keeps its eight-row limit.
For larger blocks, a fully accepted eight-row window can reuse the remaining proposal chain when its bonus token
matches the next proposal. Committed target features enter the context ring before the next window. Rejection, a
shortened window or a mismatched bonus token discards the tail and drafts a new block.
Changing the block size changes attention, draft probabilities and acceptance. Sizes above eight extrapolate beyond
the checkpoint's configured block size and require separate quality validation.

## Round cost and window width

One process per arm, the prose fixture, 512 new tokens, temperature 0, int8 KV, 475 W; the engine's own per-round
lines at exit:

| Drafter | Rounds | Tokens/window | Window sizes | Drafting ms/window | Verify pool ms/window |
| --- | ---: | ---: | --- | ---: | ---: |
| MTP, `--spec 8` | 243 | 2.11 | T1-T7 (T2 79, T3 64, T4 28) | 0.557 | 5.34 |
| DFlash2, `--spec 7` | 300 | 1.71 | T2 in 296 of 300 | 1.062 (0.999 on the GPU) | 3.90 |
| DFlash2, 32,768-row proposal shortlist | 272 | 1.88 | T2 113, T3 69, T1 58 | 0.653 (0.622 on the GPU) | 4.60 |

The block is computed whole whatever the window is — five layers over every position plus one pass of the
248,320-row head — so a round that verifies one draft pays what a round that verifies six pays. The MTP chain stops
on the device at its rule, so its drafting follows its depth instead. Where the window settles at one or two drafts,
that fixed millisecond is the whole difference between the two drafters; both are dominated by the verify window's
4-5 ms, not by the drafter (CPU-served experts, 1.9-2.6 distinct per layer per window here).

The head pass is the block's largest term: 675 MB of Q8_0 rows per block, against about 225 MB for the five Q4_K
layers. `--dflash2-proposal` narrows it to a ranked shortlist of head rows, gathered once at load, so a 32,768-row
list takes the block from 1.062 to 0.653 ms per round. Acceptance pays for that: the prose fixture keeps 65.2% of the
offered drafts against 70.5% with the full head, and structured falls from 84.1% to 76.3% — its JSON keys and
punctuation are rare in a prose ranking, and a 54,638-row list (every token a 23 MB prose-and-code corpus produced)
recovers only part of it (82.3%). In this placement the 0.4 ms is 2-4% of a round while a few points of acceptance are
worth more, so the shortlist pays only where a ranking covers the deployment's own text.

Structured output is the other case: on that fixture the same arm accepts 436 of 448 drafts (97%) in windows of seven
at 6.6 tokens per window, where the block's 1.03 ms is 2% of the round. A block above eight positions costs a second
pass of the head (11 drafts: 1.69 ms) and its positions past the eighth extrapolate beyond the checkpoint's
configured block, so acceptance drops there (89% at `--spec 12` against 97% at `--spec 7` on that fixture). Chain
reuse does work — 33 of 62 blocks came from a fully accepted window's remaining proposals — but it does not pay for
the second pass.

## Speed

RTX 5090, enforced power limit 475 W, server, context 262144, int8 KV, expert cache auto, prefill auto:32768, 15 pool
workers, PLE in RAM, auto conversation cache. Four requests per workload (1024 tokens, temperature 1, seed 1,
thinking off, prompts of 47-52 tokens), two trials per arm, one model loaded at a time. Decode tok/s:

| Drafter | Conversation | Structured | Prose |
| --- | ---: | ---: | ---: |
| MTP, `--spec 8` | 239.3 | 284.0 | 231.2 |
| DFlash2, `--spec 7` (first measurement) | 233.5 | 284.7 | 219.1 |
| DFlash2, `--spec 7` (this version) | 231.0 | 288.9 | 215.6 |
| DFlash2, `--spec 7`, block started before the window's commit | 224.4 | 293.8 | 216.1 |
| DFlash2, `--spec 7`, width learned from measured tokens per ms | 229.3 | 291.5 | 210.1 |
| DFlash2, `--spec 7`, 32,768-row proposal shortlist | 228.1 | 275.9 | 219.3 |
| DFlash2, `--spec 7`, 54,638-row proposal shortlist | 221.4 | 273.9 | 218.3 |
| DFlash2, `--spec 12` (block 12, windows of at most 8) | 219.8 | 276.7 | 203.9 |

The two width rules, the earlier launch and the shortlists are in the table to be ruled out, not as options: each
lands inside the spread between trials of one arm (up to 5% on conversation), and the learned width is 4% behind the
first measurement on prose. What the drafter's own arithmetic decides is how many drafts survive to depth, not how
fast a round is.
DFlash2 accepts more of what it offers than MTP on structured (84% against 80.5%, at 4.9-5.2 rows against MTP's
6.2-6.5) and slightly less in prose, where both settle at one or two rows and MTP's per-round draft cost — about half
of DFlash2's — wins. Prompt work is not in these numbers: DFlash2 rebuilds its context per request (prompt reuse is
off), so total-request rates are 3-8% below its decode rates.

The target still verifies every emitted token. Different draft windows can change rounding in target kernels that
are not invariant across window widths. Greedy identity and the decode-path Q8_0 PPL/KLD gate therefore apply even
though the target tensors are unchanged. Kernel parity covers random inputs at the real geometry against double
CPU references, with the existing BF16 projection kernel also checked across single-row and multi-row calls.

The implementation supports one GPU, one session and draft blocks of at most sixteen positions. Unsupported
configurations are refused. Prompt reuse is disabled: each request rebuilds the target and drafter context from its
full prompt.
Measure each drafter format and the verifier together with fixed expert placement. A faster projection alone does
not establish an end-to-end speedup.

```sh
build/dflash2_parity
build/q80_head_parity
python3 tools/convert/test_dflash2.py
```

The parity executable prints absolute and relative errors and an attention microbenchmark. Performance numbers
must state the enforced GPU power limit. The full quality gate requires a separate Q8_0 `.kld` reference file.
