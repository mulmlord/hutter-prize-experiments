# E309: longer training at fixed model size

- Ticket: E309 (private tracker)
- Started: 2026-11-08
- Base: the shipped weights (B0, curve `0022-t202-neif-b0`) and `t205-8q2f` (T205: 8 qat + 2 fp32 epochs on
  from the shipped weights; `train/GCP.md` section 5)
- Status: running (arm 1, longer at the same schedule: killed 2026-11-10)

## Hypothesis
The shipped weights are 32 qat + 6 fp32 epochs of neif's chain, and they are not done learning enwik9:
8 qat + 2 fp32 more (`t205-8q2f`) lowered CE on `every100.ev` by 0.280% and the payload at 150 MiB by 0.223%
(−66,007 B). Overfitting enwik9 is the goal, so more epochs at the same size should keep lowering CE. Twice the
extra epochs, 16 + 4 from the shipped weights, measures how the gain grows with training. If the gain is still
growing, longer runs are worth the GPU-hours. If it has flattened, the next lever is the schedule: the warm
restart's learning-rate multiplier of 25 raised the qat loss for 3 epochs (`train/GCP.md` section 4).

## Budget
The architecture and weight bytes stay the same. The export prunes and zeroes by the same rules, so the
container may move by about 1 KB. The gain is archive bytes only.
- Kill criterion (ticket): < 0.1% CE improvement per training run. Here, 16+4 must beat `t205-8q2f` by
  ≥ 0.1% on `every100.ev` to justify running longer.
- Cost: ~45 h on one spot A100 (16 × 1.88 h + 4 × 3.51 h, plus compile, setup and export), ~$59 at $1.30/h.

## Change
No code change. The weights come from the T205 pipeline with other epoch counts:
```bash
QAT=16 FP32=4 MAX_HOURS=52 REV=a74306a train/gcp.sh run e309-16q4f
systemd-run --user --unit e309-16q4f-watch ... train/gcp.sh watch e309-16q4f   # resume after a preemption
```
Then the T205 route (`train/GCP.md` section 5): fetch, `tfeval.py run` + `compare` on `every100.ev` against
the shipped weights and `t205-8q2f`, `build_neif.sh --curve --weights`, the 150 MiB curve on hp-runner.

## Checks
Only the weight container changes. The coder and its decoder read the same container and its
`WeightsFingerprint()` is stamped by `build_neif.sh`. With T205's container the 150 MiB curve matched B0 to the
byte until the transformer starts (4–8 MiB).

## Results
| Stage | Metric | shipped | t205-8q2f (8+2) | e309-16q4f (16+4) |
|---|---|---:|---:|---:|
| tfeval, `every100.ev` | nats/token | 0.907967 | 0.905424 (−0.280%) | 0.905615 (−0.259%) |
| tfeval, `every100.ev`, against t205-8q2f | Δ nats/token | | | **+0.000192 ± 0.000084 (+0.021%)**, +188 B |
| tfeval, `below150.ev` | nats/token | 1.049610 | 1.046647 (−0.282%) | not run |
| prefix, 150 MiB (against B0) | payload | 29,552,818 B | −66,007 B (−0.223%) | not run |
| container | bytes | 3,716,667 | 3,715,490 | 3,721,714 (+6,224 against t205) |
| training | VM-hours, $ | | 22.95 h, ~$29.83 | 45.07 h, ~$58.59 |

`e309-16q4f` ran 2026-11-08/10 on one spot A100 and was never preempted: setup 27 min, qat 16 × ~1.9 h, fp32
4 × ~3.5 h, export 8.6 min. The export had the same structure as T205's (70 MLP neurons pruned, 86 attention rows
zeroed in 29 tensors). Its container is `d695eaa1…`, 3,721,714 B; the extra bytes come from the entropy-coded int4
values, not the structure.

Mean training loss per epoch (paired data order; `train/GCP.md` section 4):

| run | qat | qat end, last 60 steps | first 30 fp32 batches (shipped: 0.90063) | fp32 end, last 60 steps |
|---|---|---:|---:|---:|
| t205-8q2f | 0.8966 → 0.8988 (epoch 3) → 0.8947 | 0.89363 | 0.89838 | 0.88164 |
| e309-16q4f | 0.8966 → **0.9025** (epoch 7) → 0.8970 | 0.90370 | 0.90034 | 0.89260 |

qat restarts from the shipped weights with the learning-rate multiplier at 25× and lets it fall by a cosine to
1× over the whole phase (`WARMUP_FRACTION` qat = 1.0, `tft/config.py`; the "warmup" in Ivanov's schedule runs
downward). 16 epochs spend twice as long near the peak, so the model drifted further from the shipped end
(loss up for 7 epochs) and reached fp32 hardly better than the shipped weights.

## Decision
**Longer at the same schedule: killed** (2026-11-10). 16 + 4 is +0.021% against 8 + 2 on every100, the wrong
side of the 0.1% bar, and its container is 6.2 KB larger. `t205-8q2f` stays the best container. The gain of
8 + 2 came despite the warm restart, not from its length.

Next arm: a gentler schedule. `TFT_WARMUP_FRACTION=0` makes qat a plain 1× → 0 cosine from the shipped weights,
with no reheat at 25×. That needs no trainer change; fp32's fraction is 0 already. `gcp.sh` (config.env) and
`gcp_driver.py` have to pass it through, because the driver drops every `TFT_*` variable from its own environment. First a short qat-only probe of its first epochs' loss against T205's paired
batches, then 8 + 2 if it holds. That waits for GPU quota.
