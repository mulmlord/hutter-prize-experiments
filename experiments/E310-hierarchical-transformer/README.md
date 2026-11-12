# E310: a word-rate hierarchical transformer

- Ticket: E310 (private tracker)
- Started: 2026-11-12
- Base: neif's transformer, geometry `poc8m_w2048` (16 layers, [KDA, KDA, KDA, windowed attention] × 4,
  d_model 192, ~7.8 M parameters), and its trainer (`neif/training/tf`)
- Source: `docs/research-new-approaches.md` item 4 (the one family with a published effect of the size a
  record needs)
- Status: prototype runs in the trainer (laptop smoke); the pilot needs rented or LRZ GPUs

## Hypothesis
The time limit, not the weight bytes, is the binding constraint: cmix-neif uses ~90% of it, and the
transformer runs once per symbol. Word boundaries are free in the coder's stream (dictionary code words are
1–3 bytes ≥ 0x80, separated by space-like bytes), so most layers could run once per word, as in SpaceByte
(arXiv 2404.14408: −7% bits per byte at equal parameters and 2.7× fewer FLOPs per byte) and BLT. Two ways it
can pay: at equal CE it frees ~half the transformer's time (for a wider window, more prompt rows, a larger
model); or, at equal time, it buys a stronger model.

## Change
`train/hier.py` (the trainer tree's `tft/hier.py`) and `train/hier.patch` (one call in `tft/runner.py`, after
every model change). `TFT_HIER="LO-HI[,rule=sb|doc][,pmax=P][,window=W][,bucket=B]"`:
- Layers LO..HI run on a compact sequence of the boundary positions only. Every position then gets the update
  those layers made at its latest boundary (causal: a boundary at b only uses tokens ≤ b). The outer layers
  run per symbol as before. `2-13` keeps layers 0, 1, 14, 15 per symbol; LO + HI = 15 keeps each U-net skip
  pair (layer i with 15 − i) on one level.
- Boundaries come from the input tokens alone, so the decoder can compute them. `rule=sb`: SpaceByte's rule on
  neif's WRT bytes (token → byte through the package's `vocab-bitmap.u8`, byte → WRT byte through
  `src/run_mode.h`'s permutation). A position is a boundary if its byte is space-like (below 0x80 and not a
  letter or digit) and the previous one is not. Every document start is a boundary, and every 16th position
  of a run without one (`pmax`).
- The compact sequence has its own documents (plus a padding document up to a multiple of 1,024, for
  torch.compile), rotary positions in word units, and an attention window of 2,048 words.
- The parameters are exactly the plain model's: any checkpoint of the geometry loads, t205's included.

Tests: `train/tests/test_hier.py` (parsing, the permutation, the space-like table, the boundary rule, the
compact layout's causality and documents, and the whole patch chain as `train/gcp_vm.sh` builds the tree).

## Smoke (2026-11-12, `smoke.sh`)
Laptop RTX 4070 through `train/smoke_phase.py`: bf16 from scratch, documents d % 50 == 10 up to 32,768 rows,
262,144-token steps, 2 epochs = 93 steps, the same initialisation and batches in both arms.

| steps | plain loss | hierarchical (`2-13`) | difference |
|---|---:|---:|---:|
| 0–10 | 3.1357 | 4.0763 | +30.0% |
| 10–30 | 1.8184 | 2.3531 | +29.4% |
| 30–50 | 1.6920 | 1.8366 | +8.6% |
| 50–70 | 1.6720 | 1.7667 | +5.7% |
| 70–93 | 1.6265 | 1.7030 | +4.7% |
| median step time | 3.47 s | 1.82 s | −48% |

- Boundaries: 23.5% of positions over the run (28.2% in the first batch), close to the 25.5% of code-word
  starts counted in the research survey.
- The step time halves. The predicted share of transformer work per symbol is (4 + 12 × 0.235) / 16 = 43%;
  the rest is gather and scatter overhead and the outer layers' fixed costs.
- The loss gap shrinks steadily. At equal training time (the hierarchical arm's 93 steps take as long as
  ~48 plain steps), the two are within 1% (1.703 against 1.692 at steps 30–50).
- This is a pipeline check, not a result: 24 M tokens, with both learning-rate schedules compressed into 93
  steps.

## What is not there yet
- The C++ engine has no hierarchical path, so `tfeval` and the container export cannot use such a model. The
  pilot compares the trainer's own loss on the same batches; since overfitting enwik9 is the goal, training
  loss is the quantity that matters. The engine work (per-word state, compact KV and KDA states) comes only
  if the pilot wins.
- `TFT_HIER` is not in `gcp_driver.py`'s `PASS_ENV` yet, and `gcp_vm.sh` doesn't apply `hier.patch`.

## Next: the pilot
Two arms, plain and `2-13`, 40 bf16 epochs each on the full package D at micro-batch 131,072, on one
A100/RTX 5090-class card: ~0.4–0.5 GPU-hours per plain epoch, about half that for the hierarchical arm, so
~30–40 GPU-hours in all. Compare the training loss on paired batches over the last epochs, at equal steps
and at equal compute. A cheaper variant: both arms warm-started from `t205-8q2f`'s weights, which load
unchanged.
