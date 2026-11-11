#!/usr/bin/env python3
"""One phase of neif's trainer on a subset of package D, at a micro-batch an 8 GB card holds (T204 smoke).

neif's recipe needs a micro-batch of 131,072 rows: the dump loader puts a whole document in one micro-batch,
and package D's longest has 131,037. At that size neither bf16 nor qat fits the laptop's RTX 4070 (8 GB;
train/README.md). For a pipeline check this wrapper trains on documents d with d % EVERY == OFFSET that fit
TFT_MICRO_BATCH_TOKENS, and nothing else changes: same model, loader, rows, schedule code and checkpoints.

    cd builds/tftrain && set -a && . ../../train/env/local.env && set +a
    HP_SMOKE_EVERY=20 HP_SMOKE_OFFSET=10 TFT_MICRO_BATCH_TOKENS=32768 TFT_EPOCHS_BF16=2 \
        .venv/bin/torchrun --nproc_per_node 1 ../../train/smoke_phase.py bf16

The default OFFSET keeps the subset disjoint from tfeval's every100.ev (documents d % 100 == 0).
Steps per epoch follow from the subset: the base loader counts batches by running the same iterator.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())  # run from the training tree, as `torchrun -m tft.train_poc` is

import pysrc.data as data  # noqa: E402
from zctx.zpdump import PaketDumpDataLoader

EVERY = int(os.environ.get("HP_SMOKE_EVERY", "20"))
OFFSET = int(os.environ.get("HP_SMOKE_OFFSET", "10"))

# The coder cut the documents at 2^17 already; tft/runner.py passes max_article_tokens = micro-batch for pkd,
# which at a smaller micro-batch would cut them again (and zpdump refuses that). Keep them whole.
data.split_article_lengths = lambda lengths, max_tokens: (lengths, 0)

_iter = PaketDumpDataLoader._iter_micro_batch_articles


def _subset(self, articles, drop_last):
    rows = self._doc_rows
    keep = [a for a in articles if a % EVERY == OFFSET and rows[a + 1] - rows[a] <= self.micro_batch_tokens]
    return _iter(self, keep, drop_last)


PaketDumpDataLoader._iter_micro_batch_articles = _subset

# HP_SMOKE_CKPT=1: gradient checkpointing in bf16 too (neif's config has it only for qat and fp32). Same
# arithmetic, activations recomputed in the backward pass; without it bf16 doesn't fit 8 GB even at 32,768 rows.
if os.environ.get("HP_SMOKE_CKPT", "1") != "0":
    import dataclasses

    from tft import config as tftcfg

    _make_config = tftcfg.make_config
    tftcfg.make_config = lambda *a, **k: dataclasses.replace(_make_config(*a, **k), gradient_checkpointing=True)

from tft.runner import run_phase  # noqa: E402

if __name__ == "__main__":
    phase = sys.argv[1]
    assert phase in ("bf16", "qat", "fp32"), phase
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"smoke: documents d % {EVERY} == {OFFSET} with at most "
              f"{os.environ.get('TFT_MICRO_BATCH_TOKENS')} rows", flush=True)
    run_phase(phase)
