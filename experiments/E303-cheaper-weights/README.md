# E303: cheaper weights

- Ticket: E303 (private tracker)
- Started: 2026-11-10
- Base: `t205-8q2f`'s container (`train/weights/t205-8q2f.c4`, 3,715,490 B, the best weights so far; T205)
- Status: running (idea 3, stronger lossless coding of the int4 values: killed 2026-11-10; sensitivity scan
  done; first training run `e303-qkd11-4q1f` on GCP)

## Hypothesis
The weights are shipped twice (comp9 and the decompressor in archive9), so every byte of the container is worth
2 bytes of S. The container is 3.7 MB, about 7.4 MB of S. Ideas from the ticket:
1. Mixed precision per tensor (3-bit for insensitive tensors), chosen by measured CE sensitivity.
2. QAT or more cooldown at the target bit width.
3. Stronger entropy coding of the weight planes (context-modelled), decoded once at startup. Lossless, so
   ΔS = 2 × Δcontainer − Δdecoder code, with no curve run needed.
4. Prune low-magnitude rows or heads, then retrain.

## Budget
Kill criterion (ticket): < 100 KB net S saving.

## Results

### Idea 3: how far is v4 from the int4 values' entropy? (2026-11-10, `wentropy.py`)
v4 codes the 151 int4 tensors (7,785,008 values in −7..7) with one adaptive order-0 15-symbol tree (counts
halved at 1023) and a flag for each all-zero row. To get v4's real cost of the values, the container was
re-encoded with every int4 value zeroed: 88,297 B remain (scales, f32 tensors, names, the row flags). The values
cost the other **3,627,193 B, 3.7274 bits/value**; uniform would be log2(15) = 3.9069.

Adaptive code lengths (−Σ log2 p with a counting estimator like v4's, learning included), all-zero rows skipped:

| model | bytes | bits/value | against v4 |
|---|---:|---:|---:|
| v4 (measured) | 3,627,193 | 3.7274 | |
| order 0, one model for all tensors | 3,631,369 | 3.7317 | +4,176 |
| order 0, one model per tensor | 3,631,949 | 3.7322 | +4,756 |
| per tensor + previous value in the row | 3,642,324 | 3.7429 | +15,131 |
| per tensor + magnitude of the value above (same column) | 3,634,611 | 3.7350 | +7,418 |
| per tensor + the column's mean magnitude in earlier rows | 3,634,536 | 3.7349 | +7,343 |
| per tensor + column mean + magnitude of the previous value | 3,644,397 | 3.7450 | +17,204 |
| per tensor + column mean + the row's mean magnitude so far | 3,643,604 | 3.7442 | +16,411 |
| zstd −22 / xz −9e on one byte per value | 3,666,855 / 3,840,160 | | |

Static bounds, which no adaptive model reaches because they ignore its learning cost: order-0 entropy
3.7341 bits/value over all values, 3.7300 summed per tensor (3,629,801 B). **v4 is already below both**: its fast
counts follow the distribution's drift within and between tensors. Every context added costs learning and returns
nothing, so the values carry no structure such a model can use (row, column, neighbour). That fits weights
trained under quantization noise. The distribution is broad, not peaked:

| value | 0 | ±1 | ±2 | ±3 | ±4 | ±5 | ±6 | ±7 |
|---|---|---|---|---|---|---|---|---|
| share (each sign) | 0.118 | 0.110 | 0.096 | 0.078 | 0.058 | 0.040 | 0.026 | 0.033 |

±7 holds more than ±6, which is clipping. fx4's recompression with per-tensor histograms and column-local counts
saved 28,200 B (1%) on fx2-cmix-transformer's v2/v3-era container (docs/landscape.md); on neif's v4 that
headroom is gone.

**Decision for idea 3: killed.** The int4 values have no lossless headroom. The other 88,297 B could give at most
a few KB.

### The lossy side, for ideas 1, 2 and 4
T203's requantized variants (no retraining) price fewer levels against CE. At t205's measured rate (−0.28% CE on
every100 ≈ −196 KB of full-file payload), about 700 KB of payload per 1% of CE:

| variant | container | ×2 in S | every100 CE | payload equivalent |
|---|---:|---:|---:|---:|
| b0u13 (blocks.0.mlp.up, 13 levels) | −5,963 B | −11.9 KB | +0.263% | ≈ +184 KB |
| b0u7 | −21,963 B | −43.9 KB | +1.022% | ≈ +715 KB |
| f13 (all 16 mlp.up, 13 levels) | −95,534 B | −191 KB | +2.852% | ≈ +2.0 MB |

Without retraining, the trade is about 10 times worse than break-even. A size saving can pay only with training
that keeps CE (ideas 2 and 4: a weight-size term in QAT, as zmix did; T204 step 4). That is rented GPU work.
Break-even for such a run: each 0.1% of CE given up must buy more than ~35 KB of container.

### The lever, priced (2026-11-10)
Every int4 value saved in the container is saved twice in S. One bit less per value on all 7,785,008 values is
973 KB of container, **1.95 MB of S**, at the cost rate above worth about **2.8% of every100 CE**. The engine
keeps the weights as unpacked int8 (`qmat.h`), so fewer levels need no kernel change and cost no time. That makes
fewer levels with retraining, per tensor group, the one lever in sight that is large enough for the record
(about 760 KB still missing after t205).

Groups of the int4 values: `mlp.up` 30.1% and `mlp.down` 30.1%; attention query, key, value and output 7.6%
each; the forget and output gate projections 1.9% each (up and down); the embeddings and tables 1.8% together.

Plan:
1. Sensitivity scan (`scan.sh`, laptop, free). Requantize one group at a time to 11 and 7 levels without
   retraining, then measure container bytes and every100 CE against `t205-8q2f`. `mlp.up` is known already
   (T203: +2.85% CE at 13 levels), so it is left out. This is the worst case per group, and its ratio ranks them.
2. QAT at the chosen levels per group from `t205-8q2f` on the GCP pipeline (`train/gcp.sh`).
   The trainer's quantizer needs a per-group level count. Then tfeval and the net
   ΔS = 2·Δcontainer + ~700 KB × ΔCE%. A 150 MiB curve only for a container that wins on that estimate.

### Sensitivity scan (2026-11-10, `scan.sh`)
Each group of `t205-8q2f` was requantized to 11 and to 7 levels with no retraining and measured on every100 against
the unchanged container (0.905424 nats/token; SE of each Δ ≤ 0.0009 nats). Sorted by container bytes per 0.1% of CE.
Break-even without retraining is 35 KB per 0.1%. The last column is the share of the CE loss that retraining
would have to win back to break even (ΔS = 2·Δcontainer + ~700 KB × ΔCE%).

| group (11 levels) | container | ×2 in S | every100 CE | KB per 0.1% CE | recovery needed |
|---|---:|---:|---:|---:|---:|
| attention.query_projection | −36,091 B | −72.2 KB | +0.550% | 6.6 | 81% |
| attention.key_projection | −36,362 B | −72.7 KB | +0.583% | 6.2 | 82% |
| mlp.down | −169,021 B | −338.0 KB | +3.190% | 5.3 | 85% |
| mlp.up (T203 f13, 13 levels) | −95,534 B | −191.1 KB | +2.852% | 3.3 | 90% |
| attention.output_projection | −34,900 B | −69.8 KB | +1.704% | 2.0 | 94% |
| forget_gate_projection | −17,227 B | −34.5 KB | +0.860% | 2.0 | 94% |
| output_gate_projection | −16,969 B | −33.9 KB | +1.211% | 1.4 | 96% |
| attention.value_projection | −35,897 B | −71.8 KB | +3.493% | 1.0 | 97% |
| embedding | −10,262 B | −20.5 KB | +4.093% | 0.3 | 99% |

| group (7 levels) | container | ×2 in S | every100 CE | KB per 0.1% CE | recovery needed |
|---|---:|---:|---:|---:|---:|
| attention.query_projection | −78,372 B | −156.7 KB | +0.998% | 7.9 | 78% |
| attention.key_projection | −78,579 B | −157.2 KB | +1.153% | 6.8 | 81% |
| mlp.down | −355,316 B | −710.6 KB | +6.683% | 5.3 | 85% |
| forget_gate_projection | −36,749 B | −73.5 KB | +1.296% | 2.8 | 92% |
| attention.output_projection | −76,710 B | −153.4 KB | +3.530% | 2.2 | 94% |
| output_gate_projection | −36,640 B | −73.3 KB | +2.088% | 1.8 | 95% |
| attention.value_projection | −78,193 B | −156.4 KB | +7.581% | 1.0 | 97% |
| embedding | −21,540 B | −43.1 KB | +10.664% | 0.2 | 99% |

Query and key are the least sensitive, then mlp.down, which carries by far the most bytes. Value and the
embeddings are the most sensitive and should keep 15 levels. The order is the same at both widths. From 11 to
7 levels, query and key give twice the bytes for less than twice the CE (7.9 and 6.8 KB per 0.1% against 6.6
and 6.2); mlp.down doubles both. Every group is 5 to 100 times short of break-even without retraining. Whether
QAT at fewer levels wins back more than 80% of the loss is the open question for the training run.

### The warm start for training (2026-11-10, `train/requant_levels.py`, `export.sh`, `evalwarm.sh`)
The trainer with `train/weight_levels.patch` quantizes a group to the uniform grid −h..h (h = (L−1)/2), not to
the scan's subset of the ±7 grid. So the warm start was built and measured as the run will see it: query, key
and mlp.down at 11 levels, from `t205-8q2f`'s `fp32-final.tch`, exported through the patched tree (with no
levels set, that tree gives back `t205-8q2f.c4` bit for bit). Two ways to set each row's scale s:

| scale fit | squared error against 15 levels (down / query / key) | container | every100 CE | ΔCE for net ΔS = 0 |
|---|---|---:|---:|---:|
| range: s · 7/5, same clipping range | ×1.91 / ×2.55 / ×2.86 | 3,509,199 B (−206,291) | +3.541% (± 0.00033 nats) | +0.589% |
| mse: per row, the factor with the least squared error | ×1.73 / ×2.25 / ×2.48 | 3,537,658 B (−177,832) | +3.822% (± 0.00036) | +0.508% |

The least-squares scales lose on both counts. They shrink the grid, clip the largest weights and leave more
values away from zero. Clipping costs more CE than rounding does. **`range` is the warm start**
(`e303-t205-qkd11.tch`; 6.2–6.7% of each group's values sit at ±5). Against the scan's sum for the three groups
(−241,474 B, +4.32%), the uniform grid saves 15% fewer bytes and costs 18% less CE: 5.8 KB per 0.1%, about
the same ratio.

### First training run: `e303-qkd11-4q1f` (2026-11-10)
```bash
train/gcp.sh warm runs/e303/warm/L11-range/e303-t205-qkd11.tch
WARM=e303-t205-qkd11.tch QAT=4 FP32=1 MAX_HOURS=15 \
  TFT_ENV="TFT_WEIGHT_LEVELS=mlp.down=11,attention.query_projection=11,attention.key_projection=11" \
  train/gcp.sh run e303-qkd11-4q1f
```
- Recipe: 4 qat + 1 fp32 epochs, the qat schedule unchanged (a cosine from 25× down to 1× of 2.5e-6). That
  reheat is the recipe's tool to fit weights to a quantization grid. At 1× alone, ~2,350 steps move a weight by
  at most ~3e-3, 0.15 of the new grid step (median scale 1.37e-2 × 7/5). The reheat allows about 4 steps.
  E309 found the reheat harmful for weights that already fit their grid; here the grid changes.
- Cost: ~12 h on one spot A100 at 1.9994 EUR/h, ~24 EUR.
- Bar: every100 ΔCE against `t205-8q2f` below +0.589% for any net saving; below **+0.447%** for the ticket's
  100 KB (ΔS = −412.6 KB + 700 KB × ΔCE%). That is 87% of the warm start's loss won back.
- Check before fp32: qat's end exported on the laptop against `t205-8q2f`'s own qat end
  (`runs/e303/ref/t205-qat-final`), both before their fp32 epochs. That reference scores +0.191%
  (± 0.00006 nats) against `t205-8q2f`, 3,715,474 B, so t205's fp32 epochs were worth −0.19% of CE. Stop
  before fp32 only if the run's qat end is more than ~1.0% behind t205's.

Ran from 2026-11-10 20:28 UTC on one spot A100 in us-central1-a. The VM log confirms `weight levels: 48
quantizers changed` (query, key, mlp.down × 16 blocks, at 11) and `TFT_KEEP_SCALES=1`. qat ran 4 epochs,
2,366 steps, in 7.65 h (1.91 h/epoch). Its end, exported on the laptop:

| state | container | every100 CE against `t205-8q2f` | against t205's qat end |
|---|---:|---:|---:|
| warm start, no training | 3,509,199 B (−206,291) | +3.541% | |
| qat end (`qat-final.tch`) | 3,540,120 B (−175,370) | +0.671% (± 0.00012 nats) | **+0.479%** |

qat won back 86% of the warm start's CE loss and 15% of its bytes. All the bytes came back in the three
11-level groups. Order-0 entropy of the int values from warm start to qat end: mlp.down 3.227 → 3.302
bits/value (+22.1 KB), query 3.250 → 3.307 (+4.2 KB), key 3.231 → 3.292 (+4.6 KB). Their share at ±5 rose
from 6.2–6.7% to 8.1–8.7%. The 15-level groups stayed within 0.002 bits. The trainer spreads the values
over the grid to win back CE, because its loss has no term for bytes. At qat end, the trade is 175.4 KB of
container for 0.479% of CE, 36.6 KB per 0.1%: the break-even rate (35 KB). If fp32 gives the same −0.19% as
for t205, the net is ΔS ≈ −351 KB + 335 KB ≈ −15 KB.

**Result (2026-11-11).** The run was preempted twice in fp32 (05:13 and 06:59 UTC). One resume died in setup on a
`gcp_vm.sh` change (`train/GCP.md`, section 3). The third VM, in us-central1-c, finished fp32 and the export at
11:27 UTC. Four VMs, 14.66 VM-hours, ~29.3 EUR at 1.9994 EUR/h.

| state | container | every100 CE against `t205-8q2f` |
|---|---:|---:|
| warm start | 3,509,199 B (−206,291) | +3.541% |
| qat end | 3,540,120 B (−175,370) | +0.671% |
| **final** (`runs/e303/run1/final`, sha256 `6e7c5b49…`) | **3,540,154 B (−175,336)** | **+0.402%** (± 0.00010 nats) |

fp32 gave −0.269%, more than its −0.191% for t205. **Net ΔS ≈ −350.7 KB + 0.402 × 700 KB ≈ −69 KB**: a real
saving, but under the ticket's 100 KB. Training won back 89% of the warm start's CE loss and kept 85% of its
bytes. The 700 KB per 1% is t205's rate. For the requantized variants of T203, every100 reached the payload at
a factor of 0.55–0.62, not t205's 0.812. If that applies here, the CE cost is lower and the net nearer −130 KB.
`below150.ev` on the laptop gives this container's transformer code length on the first 150 MiB.
Result: +0.004098 ± 0.00002 nats/token (+0.392%) against `t205-8q2f`, +112,423 B of transformer code over
39,612 documents. That agrees with every100 (+0.402%). The payload is measured by a 150 MiB curve run of
`builds/neif-e303r1-curve` (comp9 3,824,545 B, −175,335 B against `neif-t205-curve`; laptop; a first start on 2026-11-11 was cut
by a shutdown, restarted 2026-11-12 15:38, `runs/20261112-153847_comp9_e303r1-curve150`), compared with t205's curve `0037`. At t205's factor
(0.812) the transformer code would be +91 KB of payload at 150 MiB; at the requantized variants' 0.59 it would
be +66 KB.

### A byte term in the loss (2026-11-11, `train/rate_term.py`, `train/rate_term.patch`)
The qat result says the trainer gives bytes away because its loss has no term for them. `TFT_RATE_LAMBDA=X`
adds X × R, where R is the order-0 code length of every quantized tensor's current ints under its own histogram,
in bits. The gradient is straight-through: the slope of the tensor's code-length table, with LSQ's factor on the
scale's share (`train/GCP.md` "Byte term"). At X = 3.23e-9 nats/token per bit (`EXCHANGE_LAMBDA`, the
exchange rate above), the trainer's optimum is the smallest S. Rows with a near-zero scale are left out. t205
has all-zero forget-gate rows with a scale of 9.3e-10, and their gradient (slope / s) was 2·10⁴ times CE's in
the first smoke. Through Muon's per-matrix normalisation and momentum, that would have frozen those matrices.

Laptop smoke (`train/smoke_phase.py`, qat from `e303-t205-qkd11.tch`, documents d % 200 == 10 up to 8,192
rows, 65,536-token steps, 1 epoch = 33 steps, the same batches in every run):

| X (× exchange rate) | R, int bytes, step 1 → 33 | loss, last 10 steps | against X = 0 | gradient norm, weights: rate / CE |
|---|---:|---:|---:|---:|
| 0 | 3,423,120 → 3,422,950 | 0.85564 | | 0 |
| 1 | → 3,420,547 (−2,573) | 0.85581 | +0.02% | 6e-4 |
| 10 | → 3,404,038 (−19,082) | 0.86052 | +0.57% | 6e-3 |
| 100 | → 3,361,752 (−61,368) | 0.88652 | +3.6% | 6e-2 |

The term works in the real trainer: R moves in proportion to X and CE's gradient is unchanged. With the term
off (`TFT_RATE_LAMBDA` unset) the same smoke ran 2.44 s per step and ended at 0.85535. That is 0.03% from the
X = 0 run, which is the noise between two runs, so X = 1's +0.02% is noise. The term cost 0.11 s per step
here, with a log line every step. On the VM's 11.6 s steps, logging every 50, that should be about 1% or less
(estimate). The trade can't be judged from 33 steps inside a 25× reheat. Even at 10×, the trainer's own
objective CE + X·R only broke even there, so this is a transient, not an optimum. On the VM a step has 16 times
the tokens, which raises the term against CE's (partly averaged-out) gradient by 1–4×.

### Second training run: `e303-qkd11-r1-4q1f` (2026-11-11)
The first run plus the byte term at the exchange rate. Same warm start, levels, recipe and data order, so the
difference between the two runs is the term's:
```bash
ZONE=us-central1-c WARM=e303-t205-qkd11.tch QAT=4 FP32=1 MAX_HOURS=15 \
  TFT_ENV="TFT_WEIGHT_LEVELS=mlp.down=11,attention.query_projection=11,attention.key_projection=11 TFT_RATE_LAMBDA=3.2335e-9" \
  train/gcp.sh run e303-qkd11-r1-4q1f
```
X = 1× because along the trainer's path at the exchange rate, S can only fall (in a quadratic picture). A larger
X reaches further in 2,366 steps but can overshoot: at 2× the quadratic optimum's gain is gone. Bar: net S
below the first run's −69 KB (same every100 rate), i.e. ΔS = 2·Δcontainer + 700 KB × ΔCE% < −69 KB against
`t205-8q2f`. The VM's qat log has R every 50 steps.

## Decision
Idea 3 killed (no headroom: v4 sits at the order-0 bound, contexts add nothing). Ideas 1, 2 and 4 stay open as
training experiments; the measured exchange rate above is their bar.
