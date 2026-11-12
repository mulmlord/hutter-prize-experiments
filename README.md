# Training the transformer inside a Hutter Prize compressor

The [Hutter Prize](http://prize.hutter1.net/) asks for the smallest self-extracting archive of enwik9, 1 GB of
Wikipedia. Compressor and archive count together, so a model's weights are paid for twice. Both programs must
run on one CPU core with less than 10 GB of RAM, within a time limit of roughly 40–55 hours depending on the
machine (`docs/rules.md`). The best verified entry, [cmix-neif 1.0](https://github.com/marwoelk/cmix-neif-1.0)
(95,914,051 bytes), combines classic context models with an 8M-parameter int4 transformer that predicts the
text. A new record needs 1% less, about 959 KB.

This repository is a curated excerpt of my working repository: the experiments on that transformer, their
measured results, and the training code I added. The compressor itself is cmix-neif (GPL-3.0); it is linked
here, not copied.

## Results so far

All training runs on one spot A100. Cross-entropy (CE) is measured on an evaluation set made of every 100th
article. As a rule of thumb from these runs, 1% of CE is worth about 700 KB of archive.

| Experiment | What | Result |
|---|---|---|
| T205 | 8 qat + 2 fp32 more epochs on the shipped weights (23 h) | **−0.280% CE**; −66,007 B on the first 150 MiB of enwik9 (measured with the full compressor) |
| E309 | 16 + 4 epochs instead of 8 + 2 | +0.021% against T205: no gain. The learning-rate reheat sets the end point, not the run length |
| E303 | stronger lossless coding of the int4 weights | no headroom: the weight coder already sits at the order-0 bound |
| E303 | 11 instead of 15 quantization levels for query, key and mlp.down, retrained 4 + 1 epochs | **weight file −175,336 B** (counted twice) at +0.402% CE, net ≈ −69 KB (estimate) |
| E303 | a byte term in the loss: the code length of the int weights, with a straight-through gradient | implemented, tested, smoke-run on a laptop GPU; first GPU run in progress |
| E310 | a word-rate hierarchical transformer (most layers once per word) | prototype runs in the trainer; laptop smoke: **training steps −48%**, loss gap +4.7% after 93 steps and shrinking |

Details: `experiments/E303-cheaper-weights/README.md`, `experiments/E309-longer-training/README.md`,
`experiments/E310-hierarchical-transformer/README.md`.

## Next: a word-rate hierarchical transformer (pilot)

The time limit, not the weights, is the binding constraint: cmix-neif already uses about 90% of it. The
transformer runs once per symbol. Word boundaries are free in the coder's dictionary-coded stream (a word starts
at 25.5% of positions), so most layers could run once per word instead, as in SpaceByte, BLT and MegaByte. With
12 of 16 layers at word rate, the transformer would do about 45% of today's work per symbol. That time could buy
a stronger model. SpaceByte reports −7% bits per byte at equal parameters and 2.7× fewer FLOPs per byte. Nobody
has tested this, to my knowledge, with int4 weights on a CPU and a dictionary-coded stream.

**The prototype exists** (`train/hier.py`, `TFT_HIER="2-13"`). Layers 2–13 run on the boundary positions
only. Boundaries follow SpaceByte's rule applied to the compressor's own byte classes, so the decoder can
compute them. Every position gets the update made at its latest boundary, and the parameters are those of the
plain model. A laptop smoke (RTX 4070, 93 paired bf16 steps from scratch):

| | plain | hierarchical |
|---|---:|---:|
| boundaries | | 23.5% of positions |
| median training step | 3.47 s | 1.82 s |
| loss, steps 0–30 | | +30% |
| loss, last 23 steps | | **+4.7%**, still shrinking |

At equal training time the two arms are within 1%. That only checks the pipeline, so the plan continues:

1. **Pilot:** both arms, 40 bf16 epochs each on the full training data: about 30–40 GPU-hours on an A100- or
   RTX 5090-class card. Compare the training loss on paired batches at equal steps and at equal compute.
2. **If the hierarchical arm wins:** train it from scratch with the full recipe, about 500 GPU-hours, and
   add the hierarchical path to the compressor's C++ inference engine.

The alternatives I surveyed, ranked with sources (JEPA, state-space models, test-time training, retrieval,
lattice quantization and more): `docs/research-new-approaches.md`.

## Contents

| Path | |
|---|---|
| `docs/research-new-approaches.md` | survey of new approaches from 2022–2026 research, judged against the prize's constraints |
| `docs/landscape.md` | the recent entries and what each one changed |
| `docs/rules.md` | prize rules, scoring and the time budget |
| `experiments/E303-cheaper-weights/` | fewer levels per weight group, the byte term: write-up and scripts |
| `experiments/E309-longer-training/` | longer training at the same schedule: write-up |
| `experiments/E310-hierarchical-transformer/` | the word-rate hierarchical transformer: write-up and smoke script |
| `train/hier.py`, `hier.patch` | the hierarchical layers for the trainer (`TFT_HIER`) |
| `train/rate_term.py`, `rate_term.patch` | the byte term for the trainer's loss (`TFT_RATE_LAMBDA`) |
| `train/weight_levels.patch`, `requant_levels.py` | quantization levels per weight group (`TFT_WEIGHT_LEVELS`) and the matching warm start |
| `train/keep_scales.patch` | qat keeps the quantization scales it loads (for warm starts and resumes) |
| `train/smoke_phase.py` | one trainer phase on a data subset at a micro-batch an 8 GB card holds |
| `train/tests/` | tests for the above (`python -m pytest train/tests -q`) |

The patches apply to the trainer tree built from cmix-neif's `training/` directory (upstream engine plus its
overlay). Some write-ups refer to files of the working repository that are not part of this excerpt.

## License

GPL-3.0, as cmix-neif (`LICENSE`).
