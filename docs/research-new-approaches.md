# New approaches from research (survey, 2026-11-11)

Scope: ideas from 2022–2026 research that are not yet in the tickets, judged against docs/rules.md and the
cmix-neif base. Research and reading only; nothing here was run beyond two small counts on the laptop. Published
numbers are marked as such; every ΔS range is my own estimate unless a source is named.

## Summary
No single published idea has an evidence-backed path to −959 KB that fits the current time, byte and GPU budgets.
The base sits near a local optimum: weight bytes, bit widths and model size are each at about break-even
(E303, Ivanov's writeup), and time buys little (ex-girlfriend's online fast weights: −46 KB for +12.9% time).
What can still move S is (a) information that costs no bytes and little time, (b) better use of the same weights,
and (c) an architecture that does the same work in less time. The best candidates, in order of expected gain per
euro: train the transformer for its role in the mixture instead of for its own cross-entropy; a frozen-feature
harness that settles three head-side ideas for free; decoder-side retrieval as more prompt rows; and, as the one
large bet, a word-rate hierarchical transformer (SpaceByte/BLT style), which is the only family with a published
effect of the needed size but needs a from-scratch training that is beyond the current GPU budget.
JEPA and the other latent or multi-token objectives are rejected for this task: the one paper that reports
next-token perplexity shows them slightly worse than plain next-token training. Stacking the fundable items at
their midpoints gives roughly −300 to −600 KB on top of t205 and E303, so the 1% bar is possible but not likely
without the large bet.

## Facts the ranking rests on
- **Time.** cmix-neif takes 48.99 h on the EPYC 7B13 against a 54.3 h limit (docs/landscape.md). The headroom is
  5.3 h, i.e. 34.5 µs per coder symbol, against ~125 µs for one transformer step on the laptop (E301).
- **Bytes.** The 3.7 MB container costs 7.4 MB of S. 1% of every100 CE ≈ 700 KB of payload (E303). Payload is
  87.9 MB of archive9 (ex-girlfriend's README: 87,896,203 B).
- **Dilution.** A CE gain of the transformer reaches the payload at a factor of 0.55–0.81 (train/GCP.md section 5).
  The mixer and the other models already cover part of whatever the transformer learns.
- **Training cost.** neif's README gives 6 min per bf16 epoch and 22 min per qat epoch on 4 RTX 5090. Our A100
  takes 1.88 h per qat epoch, so about 0.5 h per bf16 epoch (my scaling). The shipped recipe (1,000 bf16 + 32 qat +
  6 fp32 epochs) is then ~590 A100-hours, ~1,180 € at 1.9994 €/h: far beyond the current GPU budget. **Anything that
  changes the architecture needs outside compute to be trained to shipping quality.** Fine-tunes of the shipped
  weights (8 qat + 2 fp32 ≈ 23 h ≈ 46 €) and short A/B runs can.
- **The stream is half tokenized already** (counted on 24 MB of `archiv.cmix.temp` at offset 200 MB): 44.3% of the
  coder symbols are dictionary code bytes, a code word is 1 / 2 / 3 symbols in 42.6 / 40.9 / 16.5% of cases
  (mean 1.74), 24.3% are spaces, and a code word starts at 25.5% of positions.
- **The linear layers are already of the newest family.** 12 of 16 layers are Kimi Delta Attention, an extension of
  Gated DeltaNet (arXiv 2510.26692), which beat Mamba and Mamba2 in its own paper (below).

## Ranked table

| # | Approach | Mechanism | ΔS estimate (mine) | Confidence | CPU / time fit | First test and cost |
|---|---|---|---|---|---|---|
| 1 | Mixture-aware training | fine-tune the transformer on the coder's mixed code length, not its own CE; optionally more CM outputs as side channels | −100 to −400 KB | low–medium | no inference cost | per-bit dump on a 16 MiB run + offline fit (free, 1–2 days); then one 8+2 fine-tune, ~46 € |
| 2 | Frozen-feature harness | hidden states dumped once; test a higher-rank output head, hashed-state contexts for the CM bank, and cheap online output layers offline | −30 to −200 KB per idea that survives | low | each ≤ 5 µs per symbol | free, 2–3 days on the laptop |
| 3 | Decoder-side retrieval rows | more derivable context: retrieved earlier passages, neif's own section-6 link items | −100 to −400 KB | low–medium | +2 to +6% W for the rows | row statistics (free); package rebuild + fine-tune, ~46 € |
| 4 | Word-rate hierarchical transformer | most layers run once per word, a few per symbol (SpaceByte, BLT, MegaByte) | −0.3 to −2 MB, or −20% W at equal CE | low (≈ 20% for ≥ 500 KB) | frees time; new kernels | 2-arm A/B at 40 bf16 epochs, ~80 €; shipping it ~1,200 € |
| 5 | Recurrent depth (tied layers) | fewer unique layers looped: bytes down at equal time | −0 to −1 MB | low | none | same A/B rig as 4, +40 € per arm |
| 6 | Lattice quantization in QAT | E8 rounding instead of scalar: ≤ 0.11 bit per weight at equal distortion | −50 to −210 KB (bounded) | low–medium | none (decode at start) | post-hoc requant + tfeval (free); fine-tune ~46 € |
| 7 | Float teacher for QAT | measure the quantization gap; distil the float model into the int4 one | −0 to −300 KB | low | none | 10 bf16 epochs ≈ 5 h ≈ 10 € for the gap |
| 8 | State carried across articles | KDA state not reset between neighbouring articles | −0 to −300 KB | low | none | trainer change + fine-tune, ~46 € |
| 9 | Checkpoint averaging | average the last epochs' latent weights before export | −0 to −40 KB | low | none | free (laptop tfeval) |

Not new, already ticketed, and therefore not ranked: larger model (E302), fewer levels and a byte term (E303),
dynamic evaluation (E304), rata's models (E305), mixer contexts from transformer state (E306), preprocessing
(E307), training schedule (E309).

## 1. Mixture-aware training
**What.** The transformer is trained on its own cross-entropy, but it is coded through the mixer, next to PPMd, the
context-model bank, the match models and BITLSTM. Train it on what counts: the code length of the mixture.
Two arms, both fine-tunes of the shipped weights:
- *Loss.* Record, per coded bit, the probability the rest of the system gives without the transformer. In
  training, turn the transformer's 208-way distribution into the bit probabilities along the true path, mix them
  with the recorded ones in the logit domain with fixed fitted weights, and take the log loss of the mix. A simpler
  form weights each token's CE by how much the mix depends on the transformer there.
- *Inputs.* The model already gets PPMd's distribution and the match model's predicted byte as side channels. Add
  the text model's most probable byte and a confidence class the same way.

**Why it could pay.** The dilution factor of 0.55–0.81 says 19–45% of a CE gain is lost in the mix, so part of the
model's capacity goes to tokens the other models code well anyway. A capacity-bound model should move that
capacity. Published: Li et al., "N-gram Is Back" (arXiv 2210.14431), train a neural LM on the residual of an
n-gram model and report better perplexity than the plain model on WikiText-103 and other sets; the PPMd input
here is the same idea for one partner model only. No published number at byte level or for compression.

**Fit.** No inference cost, no bytes (a side channel adds a few embedding rows). Deterministic as today.

**Estimate.** If a third of the diluted share is recoverable on the tokens that matter: 0.15–0.6% of payload,
−100 to −400 KB. Confidence low–medium; the sign is not guaranteed because BITLSTM was trained on the current
mixer outputs and must be retrained with any new weights anyway (neif README section 5).

**First test.** (1) A curve build that also writes, per coded bit, the final probability and the transformer's bit
probability, on the 16 MiB prefix (~70 min on the laptop, when the core is free). Offline: refit the two-input mix,
and compute how the per-token gain of `t205-8q2f` over the shipped weights transfers by token class. That shows
how uneven the dilution is, which bounds the gain, for free. (2) A full-stream trace needs one dump run on
hp-runner (~49 h; 4.4 GB at one byte per bit). (3) One 8+2 fine-tune with the mixed loss, ~46 €.

**Risks.** The trace must come from a run without the transformer's influence on the other models (true today:
nothing feeds back, docs/landscape.md "Transformer cache"). The mix weights drift along the file; fixed ones are
a proxy. Trainer work in neif's tree.

## 2. Frozen-feature harness (three head-side ideas, one dump)
`tfeval` has no hidden-state output today. One small addition (write the 192-value final state per scored row for
`below150.ev`, int8, ~30 GB, or a 16 MiB slice) lets three ideas be tested offline in numpy at no cloud cost:

- **Higher-rank output head.** d_model is 192 and the alphabet 208, so the logit matrix has rank ≤ 192: the
  softmax bottleneck of Yang et al. (arXiv 1711.03953; mixture of softmaxes, about −2 to −4% perplexity on PTB and
  WikiText-2, word level; from memory). Each extra softmax is 40 K parameters, ~40 KB of S, and ~3 µs per step. Train K = 2–3
  heads on frozen features. Needs ≥ 0.1% CE per head to pay; the cmix mixer already combines several
  distributions, so the gain may be small.
- **Hashed transformer state as a context-model context** (kNN-LM in CM form). Khandelwal et al. (arXiv
  1911.00172) get 18.65 → 16.12 perplexity on WikiText-103 by interpolating with nearest neighbours in hidden-state
  space; Xu et al. (arXiv 2301.02828) trace the gain to the different, non-parametric output function. A real
  datastore is impossible here (552 M states × 192 B = 106 GB; the run peaks at 9.75 GiB already). A 20–24-bit
  sign hash of a fixed random projection of the int8 state, used as one more context of an existing context map,
  costs one table lookup per bit and is exact on every CPU. Offline: an adaptive count model keyed on the hash,
  mixed with the transformer's own probability.
- **Cheaper online output layers** (dynamic evaluation of the unembedding or a bias only). This is E304's family;
  see "Rejected" for ex-girlfriend's measured rate. The harness can say in an hour whether a 40 K-parameter
  variant keeps most of that gain at a fraction of the time.

**Estimate.** Each: −30 to −200 KB if it survives; I expect at most one to. **Risk:** all three overlap with what
BITLSTM (which reads a sketch of the state) and the fast-weight idea already take.

## 3. Decoder-side retrieval as prompt rows
**What.** neif's prompt injection is retrieval by rule: redirect titles, link context, aliases. Generalise it:
for each article, or each section, feed passages of already-decoded text chosen by a rule the decoder can run
(title words, link targets, BM25 on the last sentence).

**Evidence.** Ram et al. (arXiv 2302.00083, Table 1): prepending the top BM25 passage, re-retrieved every 4
tokens, takes GPT-2 small from 37.5 to 29.6 word perplexity on WikiText-103 and GPT-2 XL from 20.0 to 16.6, with
no training. RETRO (arXiv 2112.04426) and TRIME (arXiv 2205.12674: enwik8 1.16 → 1.05 bpc with a long-term
memory for a 512-context model, but only 0.99 → 0.98 on top of a 2048-context SRU++) show the same shape: large
gains for short-context models that have not seen the text, small ones on top of long context. Our model has seen
the text 1,000 times, has a 2048 window, document-long KDA state and a match channel, so only the small end
applies. neif's own list (README section 6) has two unmeasured items of this kind: lower-case links (+9.3%
link-context tokens, 5,025 pages with none today) and exact-title matching.

**Fit.** Zero bytes. Each fed row is a transformer step (~125 µs on the laptop): 5% more rows is ~+2.5% W.
BM25 needs an index of the decoded text in RAM, which is not there; rule-based retrieval through the existing
temporary file is.

**Estimate.** −100 to −400 KB (0.1–0.45% of payload), low–medium confidence. neif's bug experiment (0.8–0.9 MB
lost when the page rows were dropped) shows the model leans on such rows, not what more of them gain.

**First test.** Count, with the compressor's own dump tools, how many rows each rule would add and where (free).
Then the package rows have to be rebuilt (28 GB RAM, hp-runner; package D upload took 10.8 h) and the weights
fine-tuned 8+2 (~46 €). New row kinds start from zero embeddings, so a gain may need more epochs than that.

**Risks.** A new stream form also changes BITLSTM's trace. The author measured several row kinds and kept only
those that paid.

## 4. Word-rate hierarchical transformer (the large bet)
**What.** Run a few small layers on every symbol and the bulk of the layers only at word starts, as in MegaByte
(arXiv 2305.07185), SpaceByte (arXiv 2404.14408), BLT (arXiv 2412.09871) and H-Net (arXiv 2507.07955).

**Evidence.** SpaceByte Table 1, equal training compute: PG-19 1.009 bits per byte against 1.089 for a byte-level
window transformer and 1.083 for MegaByte, with 251 M against 227 M parameters and 196 M against 529 M FLOPs per
byte (Table 3). So −7% bpb at about the same parameter count and 2.7 times less inference compute. BLT reports
parity with token models at up to 50% fewer inference FLOPs. None of these papers quantizes, runs on a CPU, or
starts from a dictionary-coded stream.

**Here.** Word boundaries are known to both sides for free (the dictionary code tells where a word starts).
A code word starts at 25.5% of positions, so with 12 of 16 layers at word rate the transformer does
(4 + 12 × 0.26) / 16 ≈ 45% of today's work: about −25% of the whole run. Our patches are short (1.74 symbols per
word plus separators), so the published −7% will not transfer in full: SpaceByte's baseline sees raw bytes.

**Estimate.** At equal CE it gives time, not bytes. Converted through a wider window, more prompt rows and
recurrent depth (item 5), or through better CE per parameter as in the paper: −0.3 to −2 MB. Confidence low;
about one in five for ≥ 500 KB. It is still the only candidate whose published effect is of the needed size and
that attacks the binding constraint.

**Cost.** New engine kernels and state handling (weeks), and a from-scratch training: ~1,200 € on spot A100s by
neif's recipe, beyond the current budget. A two-arm comparison at 40 bf16 epochs without QAT (baseline and hierarchical, equal
parameters) is ~80 €; neif compared windows the same way with fewer epochs. A smoke version on the laptop GPU
(document subset, small micro-batch) can only rule out a gross failure.

**Risks.** Early-epoch rankings may not hold at 1,000 epochs. The per-word state adds memory traffic patterns the
int8 kernels were not tuned for. A retrained model needs a new BITLSTM trace.

## 5. Recurrent depth
Tie layers so that k unique blocks run 16 times. Time stays, bytes fall: tying in pairs would halve the 3.7 MB
container (−3.7 MB of S), which pays if CE rises by less than ~5%. Evidence: Takase and Kiyono (arXiv 2104.06022)
keep most of the quality of a deep model with a third of the unique layers on translation; Relaxed Recursive
Transformers (arXiv 2410.20672) report tied models with small per-layer adapters beating untied models of equal
size. No byte-level or 8 M-scale result. In a model trained to capacity the loss could be well above 5%, so the
range is wide: −0 to −1 MB. Needs from-scratch training; test it as a third arm of item 4's A/B (+40 €).

## 6. Lattice quantization in QAT
E303 showed the int4 values have no lossless headroom and that the container is an entropy-coded scalar
quantizer. Theory bounds what any vector quantizer can add at equal distortion: 0.254 bit per weight in the
limit, 0.109 bit for the E8 lattice (granular gain 0.65 dB), which QuIP# (arXiv 2402.04396) and NestQuant
(arXiv 2502.09720) use for LLM weights. On 7,785,008 values that is at most 106 KB of container, **−212 KB of S**,
if CE depends on squared error alone. E8 points are integers or half-integers, so the int8 kernels stay (weights
doubled, activations unchanged). Free first test: requantize `t205-8q2f`'s latent weights to E8 at the same
entropy as the scalar grid and compare every100 CE (laptop tfeval). Then the trainer's quantizer and the
container need the lattice, and one fine-tune (~46 €). This extends E303 and should be filed there.

## 7. Float teacher for QAT
Ivanov reports the quantized loss 2.6% above the bf16 model's; for neif the gap is unpublished and its float
checkpoints are not in the package. Ten bf16 epochs without quantization from `fp32-final.tch` (~5 h, ~10 €)
give a float model and the gap. If it is ≥ 1.5% (1 MB of payload), distil that model into the int4 one during
QAT, as LLM-QAT (arXiv 2305.17888) and BitDistiller (arXiv 2402.10631) do; their gains are large at 2–3 bits and
small at 4. Estimate −0 to −300 KB. The measurement alone is worth the 10 €: it says how much any quantization
work can still win.

## 8. State carried across articles
The KDA state is reset at every document; the article order puts related pages next to each other. Carrying the
state costs nothing at inference. TRIME's long-memory rows (item 3) bound the gain at about 1% of CE for a
long-context model; cross-document only, I expect less: −0 to −300 KB. The trainer shuffles articles per epoch, so
it needs ordered, stateful batches; test as a fine-tune.

## 9. Checkpoint averaging
Average the latent weights of the last qat or fp32 epochs before export (LAWA, arXiv 2209.14981). Free on the
laptop with t205's per-epoch checkpoints from GCS. Gains at the end of a low-rate cosine are usually near zero.

## JEPA, assessed
A joint-embedding predictive loss trains a model to predict the embedding of one view from another. It gives no
distribution over the next symbol, so it cannot code anything by itself. Three ways it could enter:
- **Auxiliary loss on the transformer** (LLM-JEPA, arXiv 2509.14252; NextLat, arXiv 2511.05963; Semformer,
  arXiv 2409.11143; future summaries, arXiv 2510.14751; multi-token prediction, arXiv 2404.19737). LLM-JEPA needs
  paired views of the same content (text and code) and reports task accuracy, not perplexity. NextLat's Table 2
  (1.3 B parameters, 100 B tokens) is the one direct measurement: plain next-token training has the **best**
  in-distribution perplexity, 10.52, against 10.83–10.88 for NextLat, 10.90–11.00 for multi-token and 11.08–11.18
  for joint prediction. The multi-token paper reports gains that grow with model size from 300 M parameters up.
  Our model is 8 M, trained to capacity, and scored only on next-symbol log loss on the training text. An extra
  objective takes capacity from that. Expected ΔS ≥ 0.
- **A JEPA encoder as a feature source for the mixer.** Its weights are paid twice like any others, and per byte
  of weights a model trained on likelihood predicts better than one that is not.
- **A latent chosen by the encoder and sent in the archive.** A deterministic side code costs its full length and
  an autoregressive model gains nothing from it in principle; no compression result shows a net gain at byte level.

Verdict: not a route to bytes here. If someone wants the measurement, a next-latent head on an 8+2 fine-tune costs
~46 € and I expect it to lose 0.1–0.5% CE.

## Rejected, briefly
- **Dynamic evaluation / test-time training** (Krause et al., arXiv 1904.08378: enwik8 0.99 → 0.94 bpc;
  Rannen-Triki et al., arXiv 2403.01518: the gain shrinks as the model gets in-distribution; In-Place TTT,
  arXiv 2604.06169; TTT-NTP, arXiv 2606.21803). Measured on our base by ex-girlfriend: fast weights on MLP
  features give −9,047 B on 160 MB and −46,079 B on the full file for +12.87% wall time, and four feature blocks
  add 5,434 B on 160 MB. That is 0.05% for 13% of W, below E304's own kill bar (0.1% per 10% of W). A cheaper
  form is in item 2; as planned, E304 can be closed on this evidence.
- **Mamba, RWKV-7, xLSTM in place of KDA** (arXiv 2312.00752, 2503.14456, 2405.04517). KDA extends Gated DeltaNet,
  which scores 16.42 WikiText perplexity against Mamba2's 16.56 and Mamba's 17.92 under identical training
  (arXiv 2412.06464, Table 3). Ivanov measured 100% linear layers worse than 75%. A swap needs from-scratch
  training for an expected gain near zero.
- **Mixture of experts.** More parameters at equal time, but each costs a byte of S, and E302's rule says the
  model size is near break-even. Expert tables also leave L3 (the step is memory-bound, E301).
- **Ternary or 2-bit weights with more parameters** (ParetoQ, arXiv 2502.02631; BitNet b1.58, arXiv 2402.17764).
  The published size–accuracy frontier favours 2–3 bits, but with more parameters, and parameters cost time here.
  It becomes relevant only if item 4 frees time.
- **Post-hoc vector or additive quantization** (AQLM, arXiv 2401.06118). E303 idea 3: the values carry no structure.
- **A real kNN datastore or unbounded cache** (arXiv 1711.02604): 106 GB of states, no RAM left.
- **Pretrained LLMs as compressors** (Delétang et al., arXiv 2309.10668; ts_zip) and **online-trained large
  networks** (NNCP): weights are paid twice, or the training does not fit one core. L3TC (arXiv 2412.16642) is a
  fast small RWKV coder that compares itself to gzip, far from this class.
- **Learned mixers / gated linear networks.** E306's results and the mixer's 10% time share leave little.
- **Differential attention, nGPT and similar block changes** (arXiv 2410.05258): results at 100 M+ parameters,
  from-scratch training, no CPU-int8 story.

## What I would run first
1. Item 2's dump and item 1's per-bit trace on one 16 MiB run, then the offline analyses (free; a few days).
2. Item 7's float-gap measurement (~10 €) and item 6's post-hoc E8 comparison (free).
3. One fine-tune for whichever of items 1, 3 or 6 the free tests favour (~46 €).
Item 4 needs a budget decision first: ~80 € for the two-arm A/B, ~1,200 € to ship.

## Sources
Papers (arXiv): 2509.14252 LLM-JEPA · 2511.05963 NextLat · 2409.11143 Semformer · 2510.14751 future summaries ·
2404.19737 multi-token prediction · 2502.08524 CoCoMix · 1904.08378 and 2403.01518 dynamic evaluation ·
2604.06169 In-Place TTT · 2606.21803 TTT-NTP · 1911.00172 kNN-LM · 2301.02828 why kNN-LMs work · 1711.02604
unbounded cache · 2205.12674 TRIME · 2302.00083 in-context RALM · 2112.04426 RETRO · 2210.14431 n-gram residual ·
1711.03953 softmax bottleneck · 2305.07185 MegaByte · 2404.14408 SpaceByte · 2412.09871 BLT · 2507.07955 H-Net ·
2401.13660 MambaByte · 2312.00752 Mamba · 2405.04517 xLSTM · 2503.14456 RWKV-7 · 2412.06464 Gated DeltaNet ·
2510.26692 Kimi Linear · 2104.06022 layer sharing · 2410.20672 relaxed recursive transformers · 2502.02631 ParetoQ ·
2402.17764 BitNet b1.58 · 2401.06118 AQLM · 2402.04396 QuIP# · 2502.09720 NestQuant · 2305.17888 LLM-QAT ·
2402.10631 BitDistiller · 2209.14981 LAWA · 2505.24832 memorization capacity (~3.6 bits per parameter) ·
2106.04426 hash layers · 2309.10668 language modeling is compression · 2412.16642 L3TC · 2410.05258 Differential
Transformer. Each is at `https://arxiv.org/abs/<id>`.

Read in full text for the numbers quoted: 2404.14408 (Tables 1, 3), 2302.00083 (Table 1), 2205.12674 (Table 4),
2511.05963 (Tables 2, 4), 2412.06464 (Table 3), 2403.01518 (conclusions), 2404.19737 (sections 3.1, 3.3). The
others were checked by abstract and title only; statements about them beyond the abstract are from memory.

Entries and code: <https://github.com/rlanday/ex-girlfriend> (README: fast weights, per-change gains, times),
<https://github.com/astOwOlfo/fx2-cmix-transformer-v1> (writeup.md: layer ratio, window, bit widths, 2.6% QAT gap),
<https://zmix.frostbyrne.io/v1.0/writeup.md>, `neif/README.md` sections 3–6, `neif/training/README.md`.

Lattice gains (0.65 dB for E8, 1.53 dB limit, 0.254 bit for entropy-coded scalar quantization) are textbook values
(Conway and Sloane; Gersho and Gray), quoted from memory.

## Review notes (2026-11-11, checked against the repo after the survey)
- **Training cost.** `train/GCP.md` section 3 measured bf16 at 0.47 h per epoch on the A100 (`t205-probe3`), not
  an estimate. neif's recipe is then 1,000 × 0.47 + 32 × 1.89 + 6 × 3.5 ≈ 550 A100-hours, ~1,100 €. The
  conclusion stands.
- **BITLSTM confirmed** (`neif/README.md` item 2 and section 4.4): trained offline on a replay of the first 20% of
  the compression run, reading fixed sketches of the transformer and byte-LSTM states, online from 20% on. Every
  new container (t205, E303) therefore runs with BITLSTM weights fitted to the shipped transformer. t205's curve
  (−66 KB at 150 MiB) shows the mismatch doesn't erase a gain. The curve, not every100, is the real measure.
- **E304's kill bar confirmed** (issue #26: < 0.1% per 10% of added W). ex-girlfriend's fast weights, −46 KB on
  ~91.9 MB for +12.87%, are 0.04% per 10%.
- **Item 6.** The −212 KB bound assumes CE depends on squared error alone. E303's warm start contradicts that: the
  least-squares scales lowered the squared error by 9–13% and raised CE (+3.82% against +3.54%), because clipping
  the largest weights costs more than rounding. A lattice test must compare CE, not distortion.
