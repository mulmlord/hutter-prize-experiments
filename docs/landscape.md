# Landscape

## Lineage
```
cmix (Byron Knoll)
 └─ fx-cmix (Kaido Orav, 2024-02)            112,578,322
     └─ fx2-cmix (Orav & Knoll, 2024-09)     110,793,128
         ├─ fxcm_v26 -> cmix-lex (Marcouch & Orav, 2026-06)   109,671,639  awarded 2026-09-30
         │   └─ cmix-obias (Freelan, 2026-07)                 108,521,870  awarded 2026-09-30; BITLSTM correction model
         └─ fx2-cmix-transformer (Ivanov)    100,424,672  <- official record (awarded 2026-09-30)
             │                                            LSTM -> 6M-param transformer, offline-trained, 4-bit
             ├─ zmix 1.0 (Byrne)              99,312,424  Zig port, weights retrained with a weight-size term, smaller binary
             │   └─ makimono 1.0 (Sklyarenko, not eligible)   99,788,850  5/7-bit weights, 28 more models;
             │                                                archive9 95,793,872 but comp 3,984,978
             ├─ fx4-cmix-transformer -> rata-cmix 1.0.0   100,219,121  more experts, online LSTM, order-25 PPM
             ├─ + cmix-lex (Marcouch) -> cmix-lex-transformer lexth11c (Patel)   99,313,750
             │    retrained larger int4 transformer (12 layers, codebook, QAT); archive9 95,836,613 but comp 3,477,137
             └─ + cmix-lex + cmix-obias + zmix -> cmix-neif 1.0 (Wölk, 2026-10-05)   95,914,051  <- plan target
                  8M transformer trained on its own stream, prompt injection, page form; see below
                  └─ ex-girlfriend 1.0 (Landay, not eligible)   95,773,359  online aux predictor on MLP features;
                       archive9 91,778,237 (−135 KB), comp 3,995,122; 241,636 s, 10,204 MB (LTCB note 124)
```

## cmix-neif 1.0 (Wölk, submitted 2026-10-05; passed all committee tests 2026-10-18)
Source: https://github.com/marwoelk/cmix-neif-1.0 (GPL-3.0, one commit; README.md sections 3–4 itemise every change with
its measured gain). S = comp9 4,000,689 + archive9 91,913,343 + comp9.args 19 = 95,914,051, 3.42% below zmix 1.0.
Mahoney's announced 97,261,675 entry (2026-09-28) was its pre3; an author's older submissions count as withdrawn
when they beat them within 30 days, so 1.0 is the one to beat.
- **Fork base cmix-lex (C++), not zmix.** Coder stream 552,750,198 B over 208 symbols (zmix: 587,138,814, 205).
- **Transformer:** Ivanov's engine and layer pattern, 16 layers, d_model 192, ~7.8 M params, window 2048, trained
  from scratch on its own stream (1,000 epochs bf16, then QAT). Container 3,716,667 B (zmix 2,815,630), counted twice.
- **Free context for the transformer ("prompt injection"):** rows the decoder can derive (page id, contributor, date,
  redirect synonyms, incoming link text, section heading) are fed but never coded.
- **Stream form:** XML markup dropped, metadata in one sorted block at the head (gated away from the transformer),
  census numbers recomputed, comment index, article order as rule + residual (−100 KB vs a list).
- **From cmix-obias:** BITLSTM, a 16-cell logit correction after the mixer (−32 KB of S), PPMd prior on a small
  80-cell online byte LSTM kept beside the transformer.
- **From zmix:** adaptive depth, indel recovery, end-to-end mixer gradient, word-label model, SSE field widths.
- **Packaging:** its own CM stub for x86-64 code, −101 KB of S vs its earlier CM+BCJ stub (cf. our E308 zpack).
- **Resources:** compression 48.72 h at GB5 1306 (W 63,628; EPYC 7B13), 39.35 h at 1707 (5950X); peak 9.75 GiB.
  Mahoney (i7-1165G7, LTCB note 106): decompression 82.87 h (W 124,222), 10,185 MB: over 10^10 bytes whether MB
  means 10^6 B or 1000 KiB, under 10 GiB either way. So the GB vs GiB question (docs/rules.md) decides it too.
- **Verified** (Mahoney, Hutter-Prize group, 2026-10-18): passed all tests, open for public comment. Bowery's run
  (LTCB note 122, Google Cloud EPYC 7B13, GB5 1288): compression 48.99 h, decompression 49.16 h (W 63,318),
  9,749 / 9,750 MB. So the committee accepted the 9,750 MB peak and timed it on AMD, not on the i7.

### Reproduction (T106, in progress)
- **Build: byte-identical.** `scripts/containers.sh judge noble-neif cmix-neif-1.0` (offline, ~5 min on 4 cores)
  gives comp9 4,000,689 B, sha256 `8b07d85f…` as submitted. snapshot.ubuntu.com answered 503 on 2026-10-21, so
  `install.sh` fell back to the current noble archive; 32 of its 33 pinned packages still matched (only
  `linux-libc-dev`, 6.8.0-147 vs -142). The PGO profile ships in `pgo_data/`, so the build runs no training pass.
- **Portability (docs/rules.md) on the unpacked image** (`cmix.unpacked`, 120,644 instructions; positive control
  1,167 `vfmadd`, 12,485 ymm): no RCPPS/RSQRTPS/RCP14, no `%zmm`, no `{%k}` masks. Built with `-march=x86-64-v3
  -mrecip=none`. Unlike zmix it imports libm (`sqrt`, exact, and `std::sin`/`std::cos` in the transformer's
  rotary fallback for positions ≥ 131,072, which clang merges into one `sincosf`). Confirmed unreachable in a run
  that exits 0 (T107, `scripts/neif_gate.sh` header): the position never exceeds the document's row count, and
  `~Tf1Engine` exits 1 on a document over 131,072 rows. `scripts/neif_gate.sh` now checks all of this per build.
- **Memory.** The PPMd page pool grows to 10 GiB − 256 MiB (`src/models/ppmd_page.h`, polled from
  `/proc/self/statm`), so run it under bench.py with `--mem-cap 10Gi`: `10G` (10^10 B) kills it mid-run.
- **Disk.** Its `tmpfile()`s (~1.45 GB) are unlinked files in `/tmp`, invisible to bench.py's `comp_disk_mb`.
- **Speed work carries over in part.** Same Ivanov `opt/` engine as our fork. E301's `qk_scan` fix (two prefetches on
  one line) applies unchanged: it ships `AttnKind::KVI8` with the same code. The `run_packed` fix doesn't: it ships
  unpacked int8 weights (`qmat.h`), whose kernels already prefetch every line. That stream is twice the bytes of
  zmix's packed int4 for a ~7.8 M model. Ivanov's `qmat.h` measured unpacked faster than packed, but with the packed
  kernel's prefetch gap still in; packed int4 with E301's fix is the first speed candidate to measure on this base.
- **To do:** compress-only run on the laptop (README step 3), then the rebase decision below.

## Judge timings
Bowery (LTCB note 122) times on Google Cloud, AMD EPYC 7B13 (Zen 3, GB5 1288): W = 70,000 is 54.3 h there.
`hp-runner` (cloud/) is the same CPU, so a lone job on a c2d VM reproduces his clock, apart from L3 shared with
the VM's other jobs. cmix-neif 1.0 took 48.99 h / 49.16 h on it.

Mahoney's i7-1165G7 under WSL2 (LTCB note 106, GB5 1499). PPMd's mmap is slow under WSL2, so these are far above
the AMD times and over W = 70,000:
- fx2-cmix-transformer, 21 Aug 2026 revision: decompression 695,558 s (193 h). Sizes: comp 3,426,642 + archive9 96,994,188.
- zmix 1.0: decompression 318,947 s (88.6 h, W = 132,806).
- cmix-neif 1.0: decompression 82.87 h (W = 124,222).

Mahoney (2026-10-18): zmix's decompression was over the time limit on his machine, and it "would still qualify
if it runs fast enough on another machine". cmix-neif 1.0 passed on Bowery's EPYC. So a failed i7 time isn't the
end, but it delays the verdict; an entry that is faster under WSL2 mmap is safer.

## What the current best stack does
1. **Preprocessing:** an enwik9-specific article reordering and a word dictionary (in zmix the
   dictionary is derived during compression instead of shipped). This yields a 205-symbol byte alphabet.
2. **Models:** the fx2-cmix / cmix context-model bank plus PPMd (order ~25, memory-mapped to disk
   to stay under 10 GB) plus a 6M-parameter transformer with a sliding attention window. The
   transformer takes the PPMd distribution as an extra input. It is trained offline on 8× RTX 5090
   for 26 h, then quantized to 4-bit weights and 8-bit activations, and runs on AVX2.
3. **Mixing:** mixer network, SSE/APM chain, then a binary arithmetic coder.

## Where the bytes are (zmix 1.0 = B0, measured in T105)
Sliced from our T103 build and archive using the 16-byte trailers (`{dict, order, payload, tfweights}` as
little-endian u32). The program image and weights in archive9.exe are `cmp`-equal to those in comp9.
comp9 = prefix ++ recipe ++ order ++ weights ++ trailer; archive9.exe = prefix ++ dict ++ payload ++ weights ++ trailer.

| Component | Bytes | Charged | Lives in | Share of S |
|---|---:|---|---|---:|
| **coded enwik9 payload** | **93,020,509** | 1× | archive | 93.66% |
| transformer weights (`6m-q4-fp32-t1lambda1.tfwc3`) | 2,815,630 | 2× | both | 5.67% |
| packed program image (`armc_prefix`, UPX 5.2.0 `--ultra-brute`) | 160,884 | 2× | both | 0.32% |
| article order | 173,640 | 1× | comp9 | 0.17% |
| derived dictionary | 99,222 | 1× | archive | 0.10% |
| dictionary-derivation recipe | 65,993 | 1× | comp9 | 0.07% |
| trailer | 16 | 2× | both | 0.00% |
| comp9.args | 0 | 1× | — | |

The fixed cost (everything except the payload) is 6,291,915 bytes, 6.3% of S.
Every byte of weights or program image is worth two bytes of payload.

The program image unpacks (`upx -d`) to a 408,864-byte stripped ELF, sha256 `b8b4d741…` as in the build log:

| Segment | Unpacked | xz estimate |
|---|---:|---:|
| code (R E) | 335,584 | ~143,900 |
| read-only (ELF and dynamic headers, rodata) | 72,328 | ~14,400 |
| initialised data (RW; plus 176,544 B bss) | 896 | |
| alignment padding | 56 | |

Raw LZMA with the x86 BCJ filter (`xz --format=raw --x86 --lzma1=preset=9e,pb=0`) packs the whole image to
157,691 bytes, so the UPX stub and headers cost about 3.2 KB (6.4 KB of S). The image is 0.3% of S.
LZMA is far from what context mixing gets on this code (paq8px: 100,923 B); E308's packer takes 36 KB per copy
off it (see "Program packing (E308)"). The weights are still 18× larger (E303).

## Where the time goes (zmix 1.0 = B0, measured in T201)
Cycle counters (`src/src/profile.zig`) compiled into a separate profile build of our fork; comp9 itself is
unchanged (still sha256 `0fbc9cd2…`). `scripts/profile.py run` compresses enwik9 pinned to one core and
stops after a prefix. Compression is 5.2 min of setup, then the coder over 587,138,814 bytes
(the article-reordered, WRT-coded stream), so the coder is 99.8% of the run.

Coder loop, first 10 MiB (`runs/20261006-185607_comp9_profile_T201`, core 2, PPMd on disk as shipped):

| Component | Share | ns/byte | What it covers |
|---|---:|---:|---|
| transformer | 51.7% | 121,564 | one 6M-parameter step per byte, plus the per-bit readout of its distribution |
| fxcm_v26 context-model bank | 29.2% | 68,689 | slot computation and update (`fxcm.perceive`) |
| mixer | 10.2% | 23,968 | input staging, adaptive-depth head, 23 layer-0 mixers, mixer1, end-to-end gradient |
| PPMd | 7.5% | 17,525 | order-20, 14 GB arena mmapped to disk; bit and byte predict/update |
| SSE/APM | 0.5% | 1,158 | |
| word-identity expert | 0.4% | 877 | |
| match models | 0.3% | 806 | |
| context manager | 0.2% | 385 | |
| arithmetic coder, loop | 0.1% | 250 | |
| **total** | | **235,222** | 0.934 bits per coder byte |

At this window's rate the run takes 0.09 h + 587,138,814 × 235,222 ns = **38.45 h**, which is T103's measured
38.452 h. The counters cost no measurable time, and a 10 MiB prefix predicts the full compression time.
The counters read the TSC, which ticks at a constant rate, so the shares hold whatever the clock does (median 4,393 MHz here).

Setup before coding (wall): read enwik9 0.7 s, decode recipe 18 s, derive dictionary 5 s, code the dictionary
for the archive 84 s, decode the article order 143 s, reorder + WRT 37 s, vocabulary pass + model init 24 s.

**PPMd in RAM vs on disk.** The same build with `-Dppmd-mmap=false`
(`runs/20261006-185609_comp9_profile-ppmd-mmap_false_T201`, core 6, same 10 MiB) spends 4,617 ns/byte in PPMd
instead of 17,525: the disk mapping costs ~12.9 µs/byte, 2.1 h over a full run, 5.5% of the time. The other
components agree within 5% between the two runs. The in-RAM run was stopped after 10 MiB; it was
already at 9.1 GB RSS and growing, and keeping the arena in RAM is what the mapping avoids under the 10 GB limit.

**Time headroom.** W(B0) = 67,407 compress (T103), so 70,000 − W = 2,593, i.e. **1.57 h** on the 5900X at its
official GB5 1648 (1.5 h at the 1,724 zmix measured). zmix's own 5900X time (36.49 h, W 62,905) leaves 7,095, i.e. 4.1 h.
Locally 39.9 − 38.45 = 1.5 h. Anything added has to be paid for: at 235 µs/byte, 1 h buys 6.1 ns/byte (2.6%).

What this means for the plan:
- **E301 (faster transformer) is the lever.** Halving the transformer frees 60 µs/byte, about 10 h, enough for
  E302/E304/E305. fxcm is the second target (69 µs/byte).
  First result: −14% on the transformer in-engine on the dev VM (prefetch coverage; see "Transformer step speed").
- **E305 (port rata-cmix models) must be paid for.** Each extra model costs time from a 1.5 h budget.
- **PPMd on disk costs 2.1 h.** A smaller in-RAM arena or a cheaper mapping is worth a look once memory is freed elsewhere.

**Transformer cache for mixer experiments (open question of T201).** The transformer's input is the coded byte
plus PPMd's byte distribution (`mixer/byte_mixer.zig` byteUpdate). PPMd only sees the coded bits, and nothing
downstream feeds back into either. Their only consumers are mixer inputs [588] (PPMd's bit
prediction) and [589] (the transformer's bit prediction, also the override when it is exactly 0 or 1); the hero
head is off in the recipe. So both streams are pure functions of the text and can be recorded once and replayed:
one f32 per coded bit each, 4 × 8 × 587,138,814 = **18.8 GB per stream, 37.6 GB for both**. A replay run skips
59% of the work (transformer + PPMd), so E306-style mixer/SSE experiments run at ~2.4× speed.
The replay has to be bit-exact, so it is a development tool only, never part of an entry.

## Transformer details (fx2-cmix-transformer writeup; zmix keeps the architecture)
- 6M parameters, byte-level (1 byte = 1 token, 205-symbol alphabet after preprocessing), with the PPMd distribution as an extra input.
- 75% of layers are Kimi Linear, 25% attention with a sliding window of 1024. Tested 512 (worse loss) and 2048 (too slow).
- Modded-NanoGPT style: Muon, RoPE, QK-norm, ReLU², logit softcapping.
- Training: 320 epochs bf16, then 256 epochs QAT, then 16 epochs fp32 QAT at low LR (8× RTX 5090, 26 h).
- Quantization: modified ParetoQ, W4A8, one scale per *input* channel so int8 GEMMs work.
  Weights use the range −7..7 and are stored at log2(15) ≈ 3.81 bits each; embeddings are quantized too.
- Loss 0.911 nats/byte, vs 1.23 for fx2-cmix's online LSTM. The LSTM had used about half of fx2-cmix's runtime.
- Inference: AVX2 int8 kernels (`VPMADDUBSW`). Articles are split and modelled separately in training.
- zmix retrained the weights with an extra term for the weights' compressed size ("t1lambda1"). Those training changes are **not** published.

## Measuring transformer weights (T203)
What the transformer sees in our fork (`fx2_shim.cpp` fx2tf_byte_update): one step per coded byte of
the main region, input the canonical-205 token and PPMd's distribution over the next byte as f16.
It runs in pieces: a fresh context (state reset, RoPE from 0) after every article separator
("  <page>\n    <title>", 243,426 in enwik9) and every 2^17 tokens. A piece's first token is
predicted by the PPMd prior. The r1 side footer (679,505 bytes at the end) bypasses it.

- `src/src/tfdump.zig` (profile build only; comp9, zmix_form1 and zmix_ship stay byte-identical)
  writes that stream: tokens, priors, article bounds in fx2-cmix-transformer's training-data layout
  (410 bytes per token, 241 GB for enwik9), plus the engine's own output rows for a prefix.
- `scripts/eval_transformer.py --engine cpp` runs the fork's transformer C++ itself
  (`scripts/tfeval/build.sh`: Zig 0.15.1's clang, the recipe's per-file flags, the binary's own
  musl `expf`). It reproduced the engine's output **bit for bit** on all 1,200,000 rows of a scratch
  stream (351 pieces, including a 2^17 cut and the 1024-token window transition; shipped weights).
  3,600 tokens/s per core on the cloud box's 2.8 GHz Xeon, so enwik9 is ~45 core-hours.
- `--engine torch` (GPU) has the engine's quantization (exact integer matmuls, same scales, clamps,
  rounding to even) but torch's float rounding. The quantized net turns 1-ulp differences into
  different ints within tens to hundreds of tokens (upstream saw the same between its naive and
  optimized C++), so rows drift apart; the CE does not. 0.2 M tokens, 68 pieces:
  cpp 1.008057, torch fp32 1.008310 (+2.5e-4), torch fp64 1.008127 (+7e-5) nats/token. Per piece, torch − cpp
  averages −6.8e-5 ± 1.9e-4 nats (positive in 56%), and fp64 is as far from cpp as from fp32: noise,
  shrinking with the token count. GPU speed not measured yet (no GPU here).
- Weights: an upstream `.tch` checkpoint exports (upstream's quantizer, CPU) to tensors identical to
  the engine's decode of the `.tfwc2` upstream wrote from it, all 434. The `.tfwc3` differs only in
  the 88 raw fp32 tensors it carries as bf16 (max change 0.0097).

On the scratch stream (English text from /usr/share/doc, not enwik9), exact engine, 1.2 M tokens:

| weights | nats/token | vs upstream |
|---|---:|---:|
| upstream `6m-q4-fp32.tfwc2` (= `.tch`) | 0.5215297 | |
| upstream in zmix's v3 container (`6m-q4-fp32.tfwc3`) | 0.5214737 | −0.011% |
| zmix `6m-q4-fp32-t1lambda1.tfwc3` (shipped) | 0.5209470 | −0.112% |

Torch against cpp on the first 0.4 M tokens of that stream (147 pieces): upstream `.tfwc2` 0.8974740 vs
0.8974988, `.tfwc3` 0.8972791 vs 0.8974667, zmix 0.8959340 vs 0.8957842 nats/token. Both rank zmix's
weights first (torch −0.172%, cpp −0.191% against `.tfwc2`). The 0.004% between the two containers is
below torch's noise at this length.

enwik9 numbers and the factor "transformer CE Δ → payload Δ" (step 4) need the laptop: B0's full
profile run with `--tfdump`, then a prefix run of `scripts/build_variant.sh v-tfwc2 --weights
src/assets/6m-q4-fp32.tfwc2` compared with `curve.py`, against the exact CE Δ over the same tokens.

## Transformer step speed (E301)
Detail: `experiments/E301-inference-kernels/README.md`. Measured on the cloud dev VM (Sapphire Rapids, 2 MB L2),
which runs the engine about 2.4× slower than the laptop; the laptop numbers are still to do.

- The step is bound by its own memory traffic: per token it streams ~2.3 MB of packed int4 weights, the 1.15 MB
  int8 KV cache and the 432 KB of KDA state (read and written), from L3. In-engine it costs only 4% more than alone,
  and the component shares on the VM (transformer 56%, fxcm 30%, mixer 9%) are close to T201's laptop shares.
- zmix's packed GEMV (`-Dtf-packed`) prefetched one line per 4-row group (1 of ~6.5 lines), and the attention QK scan
  every other line of its K stream. With every line prefetched, the step is 19% faster alone and the transformer 14%
  faster in-engine (611K → 525K ticks/byte on a 1.175 MB WRT stream, all components −7.1%), bit for bit the same
  outputs and payload. The packed prefix is 488 B smaller (160,396 B; S −976 B).
- `scripts/tfeval/build.sh` also builds `tfbench`: one transformer step per token over a tfdump stream on one
  core, ns/step, a hash over every step's outputs (two builds must agree), and with `--prof` the per-site table.

## Program packing (E308)
Detail: `experiments/E308-program-size/README.md`. The submitted prefix is packed with zpack (now `tools/zpack/`,
`ZMIX_PACKER=zpack` in `SHIP_RECIPE.env`) instead of UPX: a context-mixing model of x86-64 code shared by the packer
and a freestanding stub, which decodes the image and maps it in-process like UPX's stub (no `execve`;
`/proc/self/exe` is still the packed file).

| packer, same 410,816-B image | packed prefix | S (×2) |
|---|---:|---:|
| UPX 5.2.0 `--ultra-brute` | 160,396 | 320,792 |
| xz raw + x86 BCJ, no stub (reference) | 157,231 | |
| zpack (stub 8,160 + headers 184 + stream 116,303) | 124,647 | 249,294 (**−71,498**) |
| paq8px v217 `-8`, no stub (reference) | 100,923 | |

The biggest single gains: higher mixer learning rates for this small input, x86 field and instruction-history
contexts, and turning call/jmp/Jcc rel32 and RIP-relative disp32 into absolute addresses before modelling (−4.9%).
The stub costs ~10 s per run on the dev VM (~4 s on the laptop) and 105 MB, all freed before the program starts.
The judging harness must be told `COMPRESSOR_FORMAT=executable` / `ARCHIVE_FORMAT=executable` instead of
`upx-overlay`; whether a non-UPX packer is acceptable is for the committee (T903).

## rata-cmix 1.0.0 additions over fx2-cmix-transformer (T104)
From its README and docs/ARCHITECTURE.md. 575 mixed models. The only per-feature gain upstream states is for the weights;
the rest ship as a bundle whose total is archive9 96,849,689 (-144,499 vs fx2-cmix-transformer's 21 Aug 96,994,188).

| Addition | Source | What it does | Claimed gain |
|---|---|---|---|
| GrammarMatch + DeepMix contexts | fx-deepmix (Yttredal) | predicts wiki structure (piped-link labels, title recurrences); 4 extra FXCM context maps, ContextMap3 overflow fix | not stated |
| online 2×200 LSTM expert | fx4 | runs alongside the frozen transformer (BPTT 128): own probability, expected-byte hint, disagreement channel, gate on transformer uncertainty | not stated |
| v22++ stationary-map bank | fx4 | exports signals fxcm_v22 computes and discarded, as stationary maps (after fxcm_v26/fx-deepmix) | not stated |
| order-25 disk PPMd | fx4 | 14,000 MiB `ppm.temp`, MAP_SHARED and pointer-stable, MADV_RANDOM/MADV_DONTNEED | not stated (lets order 25 fit in 10 GB) |
| weight recompression v5 (`.tfwc5`) | fx4 | entropy-codes Q4 symbols with per-tensor histograms and causal column-local counts, no side table | 2,930,652 -> 2,902,452 (**-28,200**, counted 2×) |
| Scr2Match | fx4 | derived pattern table as a probability expert | not stated |
| MorphologyMatch | fx4 | zero-side-data morphology specialist for literal words | not stated |
| CausalDonor | fx4 | zero-side-data match expert over already-decoded history | not stated |
| MATCHTRUST | rata | 8-bit shift register of recent match outcomes -> StateMap | not stated |
| SPECIALIST | rata | small context-gated corrector after SSE | not stated |

Cost: 52:57 h compression on an n4d-highmem-2 (Zen 5), already over the 5900X budget.
Licence GPL-3.0 (reusable in our entry). Its `cpp_infer/` still uses `_mm256_rcp_ps` (vendor-dependent, see below).
Port the models, not its transformer kernels.

## Known determinism bugs (fixed in fx2-cmix-transformer, present in fx2-cmix)
1. `alloc1` in `src/models/fxcmv1.cpp` under-allocates aligned buffers, so RunContextMap reads past its allocation and compressor and decompressor diverge.
2. PPMd re-mmap can move its state and invalidate pointers; crashes on some machines only.
3. Reciprocal-estimate instructions (RCPPS/RSQRTPS) differ between Intel and AMD; fixed with `-mrecip=none`.
   rata-cmix 1.0.0 (and fx4) forked before this fix: 38 such instructions in its unpacked core (T104).

## Awards and pending entries
- Awarded 2026-09-30 (prize page; Bowery's press release in the Hutter-Prize group, 2026-10-02): cmix-lex
  109,671,639 (1.01%), cmix-obias 108,521,870 (1.05%), fx2-cmix-transformer 100,424,672 (7.46%, EUR 37,300).
  The official threshold is now S < 99,420,425.
- Mahoney's status, Hutter-Prize group 2026-10-18 (all open for public comment):
  - zmix 1.0 (09-14) 99,312,424: decompression over time on his machine; eligible if it passes on another.
  - cmix-neif-pre2 (09-16) 98,249,835 and pre3 (09-20) 97,261,675, pre3f (09-28) 97,262,944: decompression over time.
  - cmix-lex-transformer lexth11c (09-27) 99,313,750: decompression passed, compression not tested. Eligible only
    if compression passes and zmix, pre2 and pre3 all fail.
  - **cmix-neif 1.0 (10-05) 95,914,051: passed all tests.** It is the likely next record, so the plan target
    S ≤ 94,954,910 holds. He tested compression on none of the others.
  - He calls the current leader "cmix-lex-transformer"; the awarded record is fx2-cmix-transformer.
- Also in the committee's backlog (Mahoney, 2026-08-31): forge-cmix (Nesner), fx-deepmix (Yttredal), both 108–110 M.
- Not eligible, but informative: fx2-cmix-t-ft8 (Mejuto González, 2026-10-10) 100,367,271, fx2-cmix-transformer
  fine-tuned on 8 GPUs.

## Links
- zmix: https://zmix.frostbyrne.io/v1.0/ (README.md, writeup.md, zmix-src.tar.gz)
- fx2-cmix-transformer: https://github.com/astOwOlfo/fx2-cmix-transformer-v1 (writeup.md, training_recipes/)
- rata-cmix: https://github.com/Dharmesh2015/RATA-CMIX
- fx4-cmix-transformer: https://github.com/naveenbijalwan/fx4-cmix-transformer
- cmix-lex-transformer lexth11c: https://github.com/Neel49/cmix-lex-transformer (release lexth11c; docs/HUTTER_SUBMISSION.md)
- cmix-neif 1.0: https://github.com/marwoelk/cmix-neif-1.0, earlier versions at https://marwoelk.de/cmix-neif-pre3/ etc.
- ex-girlfriend 1.0: https://github.com/rlanday/ex-girlfriend; makimono 1.0: https://github.com/krapuchino83/makimono
- fx2-cmix: https://github.com/kaitz/fx2-cmix, discussion at https://encode.su/threads/4308-fx2-cmix-(HP)
- Benchmark notes for all of the above: https://mattmahoney.net/dc/text.html
- Committee announcements: https://groups.google.com/g/Hutter-Prize; backlog thread https://encode.su/threads/4539-Hutter-prize-backlog
  (encode.su returns 403 to curl and WebFetch; open it in a browser)

## Measured baselines (our machine)
Filled in by T101–T105. Raw rows are in `results/results.csv`.

| Variant | comp | archive | S | comp h | decomp h | peak RSS | round trip |
|---|---|---|---|---|---|---|---|
| zmix-1.0 (T103, B0) | 3,216,163 | 96,096,261 | 99,312,424 | 38.45 (W 67,407) | 38.17 (W 66,917) | 9,641 / 9,242 MiB | OK |

Peak RSS is compress / decompress. GB5 = 1753 (T003). Compression peak RSS is 10.11 × 10^9 bytes, over the decimal
reading of the 10 GB limit (docs/rules.md). T102 and T104 get compress-only rows, so their round trip stays empty.

## Rebase decision (T106 step 4, accepted 2026-10-22): cmix-neif 1.0
**Decision: rebase our entry onto cmix-neif 1.0** (#50). The fork is `neif/` (T107). The zmix fork (`src/`) was retired in T107 step 5; it is at the tag `zmix-fork`.

| | our zmix fork (B0 + E308) | cmix-neif 1.0 |
|---|---|---|
| S | 99,239,950 (projected, `src/ARTIFACT_PIN.env`) | 95,914,051 |
| to the target S ≤ 94,954,910 | −4,285,040 (−4.32%) | −959,141 (−1.00%) |
| committee (2026-10-18) | zmix: decompression over time on Mahoney's i7, pending | passed all tests |
| time | W 67,407 / 66,917 on the laptop (96% of 70,000); not timed on EPYC 7B13 | W 63,099 / 63,318 on EPYC 7B13 (Bowery), 5.2 h under its 54.3 h limit |
| peak RSS | 10.11 × 10^9 B (compress) | 9.75 GiB = 10.47 × 10^9 B, accepted by the committee |
| licence | GPL-3.0 | GPL-3.0 |

Why:
- **The gap.** Our open experiments' kill bars add up to about 0.6% (E302 0.2%, E303 100 KB, E304 and E309
  0.1%, E306 and E307 0.05%, E308 30 KB). Even several times those gains would not close 4.3% on zmix. From
  cmix-neif, a 1% gain is the whole job.
- **It already has zmix's best parts.** Adaptive depth, indel recovery, the end-to-end mixer gradient, the
  word-label model, the SSE field widths and a hashing option, all ported from zmix's Zig source. Going the other
  way, the parts zmix lacks make up most of cmix-neif's 3.4%: page form, prompt injection, the 8M transformer
  trained on its own stream, and BITLSTM. That port would be the bigger job.
- **Verified, and timed on our CPU.** It has 10% time headroom on the committee's EPYC 7B13, which is
  `hp-runner`'s CPU. zmix has 4% on the laptop and failed on the i7.
- **The memory question is settled in practice.** The committee accepted a 10.47 × 10^9 B peak (Mahoney measured
  10,185 MB decompressing). See docs/rules.md.

What carries over:
- **E301.** The `qk_scan` prefetch fix applies unchanged. On Zen 3 it takes 18% off QK and 2.4% off the
  transformer, for S +20 B; merged into `neif/` (2026-10-31). Packed int4 with the
  `run_packed` fix was the first speed candidate, because cmix-neif ships unpacked int8. It is 9% faster on the laptop
  but 11.6% slower in the matmuls on the EPYC 7B13, so it is out (E301 README, "On cmix-neif").
- **E302, E303, E309.** cmix-neif ships its training pipeline (`training/tf/`, `training/bitl/`) and its weight
  container. T203's CE evaluation needs a tfdump of its 208-symbol stream.
- **E304, E306, E307.** The ideas apply as they are. Of E307, its stream form and article-order rule already took
  the obvious steps.
- **T202.** `curve.py` reads only `curve,` lines, and `profile.py run` only needs a binary that prints them.
  Port `src/src/curve.zig` (84 lines before its tests) into its coder loop, then validate the proxy there.

What doesn't:
- **E308 zpack.** cmix-neif has its own CM stub for x86-64 (−101 KB of S against its CM+BCJ stub). Compare
  zpack with it on cmix-neif's image and keep the smaller.
- **T201's counters** (Zig): redo them as C++ rdtsc counters when profiling is needed.
- **The Zig build, `ZMIX_SRC_DIGEST`, and `recip_gate.sh` / `libm_gate.sh`.** T106 ran the gates by hand on the
  unpacked image; script that check.

Risks:
- **archive9 isn't reproduced yet.** T106's compress-only run on `hp-runner` (comp9 sha256 matches) ends around
  2026-10-23. If archive9 ≠ 91,913,343, the decision waits.
- **Others build on it too.** ex-girlfriend is −135 KB of archive9 but too slow. A cmix-neif-based entry accepted
  before ours would move the target; that is equally true on zmix.
- **Headroom is 10%, not more.** Anything that adds model must be paid for with speed, as on zmix, so E301
  stays first.

Next steps:
1. [x] T107 (#51): `neif/`, the curve (comp9 byte-identical with it off), `scripts/neif_gate.sh`.
2. [x] T202 step 3 on that fork, queued as `0013`–`0016` on the cloud runner; the zmix jobs 0010–0012 are cancelled.
3. [x] The critical path in docs/plan.md and the base in README.md. Still on zmix: README's workflow steps and the
   experiment template (T107 step 5).

## Base for our entry: zmix 1.0 (T105)
Our fork lived in `src/` (`variants/ours.env`, built with `scripts/containers.sh judge jammy-zmix src ours`) until
T107 retired it. It is at the tag `zmix-fork`, and the paths in this section are relative to that tag.
Why zmix:
- **Smallest S we can rebuild.** 99,312,424, reproduced byte for byte with a full round trip (T103).
  fx2-cmix-transformer is 1.1 M behind, rata-cmix 0.9 M. cmix-lex-transformer lexth11c ties on S (+1,326)
  and the unreleased 97,261,675 entry had no source. (Chosen 2026-10-06; cmix-neif 1.0, published 2026-10-05,
  is 3.4% smaller. We rebased on it, see "Rebase decision" above.)
- **Within the time budget.** W = 67,407 compress, 66,917 decompress on our machine (limit 70,000).
  rata-cmix takes ~190k s to compress and can't be submitted as is.
- **Vendor-independent by construction.** Its build fails on any reciprocal-estimate instruction or libm import
  (`tools/recip_gate.sh`, `tools/libm_gate.sh`). rata-cmix ships 38 reciprocal estimates.
- **Lean fixed cost.** The program image packs to 160,884 bytes, and the weights were retrained with a size term.
  The dictionary is derived at encode time instead of shipped.
- **GPL-3.0** (`LICENSE`, `LICENSES/GPL-3.0.txt`); `NOTICE.md`, `THIRD-PARTY-NOTICES.md` and
  `THIRD_PARTY_LICENSES.md` stay in the fork.

What we inherit and must change: compression peak RSS is 10.11 × 10^9 bytes, over the decimal reading of the
limit. The code is Zig 0.15.1 plus the C++ transformer kernels in `third_party/fx2_transformer/`, and the
toolchain is pinned in `hp/jammy-zmix`.

Working in the fork:
- `src/SHIP_RECIPE.env` pins `ZMIX_SRC_DIGEST`, the hash of every file under `src/src/` and `src/third_party/`
  plus `build.zig` and `build.zig.zon`. A mismatch aborts the build (rc 3, strict mode).
  Update the digest in the same commit as any engine source change (`zmix_src_digest` in `construct_ship.sh`).
- Our additions so far: `src/src/profile.zig`, `src/src/form1_profile.zig`, the `form1-profile` step in
  `build.zig` and `src/tools/build_profile.sh` (T201). None of them changes comp9.
- `src/ARTIFACT_PIN.env` holds the expected sizes and the comp9 sha256. A mismatch only warns, so a changed
  build still completes. Update the pin when an experiment is kept.
- `src/` started as a verbatim copy of the pinned `upstream/zmix-1.0`, so `diff -r upstream/zmix-1.0 src`
  shows exactly our changes.
