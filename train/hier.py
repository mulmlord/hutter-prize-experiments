"""A word-rate hierarchical transformer for neif's trainer (TFT_HIER; research survey item 4).

On the VM and the laptop this file is the trainer tree's tft/hier.py (copied like rate_term.py), and
train/hier.patch calls install_hier() in tft/runner.py once the model and its embedding are built. With
TFT_HIER unset or empty nothing changes.

    TFT_HIER="LO-HI[,rule=sb|doc][,pmax=P][,window=W][,bucket=B]"      e.g. TFT_HIER="2-13"

Layers LO..HI run once per *word* instead of once per symbol, as in SpaceByte (arXiv 2404.14408): they see a
compact sequence made of the boundary positions only, and every position then receives the update that the
middle layers made at its latest boundary. The layers outside LO..HI run on every position as before. The
parameters are exactly those of the plain model, so any checkpoint of the same geometry loads.

Boundaries are computed from the input tokens alone, so the decoder can compute them too:
  rule=sb (default)  SpaceByte's rule on neif's WRT bytes: a position is a boundary when its byte is
                     space-like (below 0x80 and not a letter or digit: space, punctuation, line feed, the
                     capital/uppercase/escape marks) and the previous byte is not. Dictionary code bytes
                     (>= 0x80) and letters are word content. Inserted sentinel rows count as space-like.
  rule=doc           only document starts (with pmax this is a fixed stride, as in MegaByte)
Every document start is a boundary, and a boundary is forced wherever pmax (16) positions passed without one.
The token is bits 0..7 of the packed input (zctx/zpaket.py); its byte comes from the package's
vocab-bitmap.u8, and the WRT byte from neif's fixed stream permutation (src/run_mode.h BuildPerm).

The compact sequence is its own set of documents (one per real document, plus one padding document up to a
multiple of `bucket`, 1024, so that torch.compile sees few shapes), with rotary positions in word units and
an attention window of `window` words (default: the model's window). LO + HI must equal n_layers - 1: the
U-net skip connections pair layer i with layer n_layers - 1 - i, and a symmetric range keeps every pair on
one level.
"""

from __future__ import annotations

import math
import os
import types

import numpy as np
import torch

ENV = "TFT_HIER"
TOKEN_MASK = 0xFF


def parse(text: str | None, n_layers: int) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    first, *rest = [t.strip() for t in text.split(",") if t.strip()]
    lo_s, sep, hi_s = first.partition("-")
    if not sep or not lo_s.isdigit() or not hi_s.isdigit():
        raise SystemExit("%s=%r: LO-HI first, e.g. 2-13" % (ENV, text))
    lo, hi = int(lo_s), int(hi_s)
    if not (1 <= lo <= hi <= n_layers - 2):
        raise SystemExit("%s: need 1 <= LO <= HI <= %d (one symbol-rate layer at each end)" % (ENV, n_layers - 2))
    if lo + hi != n_layers - 1:
        raise SystemExit("%s: LO + HI must be %d (symmetric, for the skip connections)" % (ENV, n_layers - 1))
    spec = {"lo": lo, "hi": hi, "rule": "sb", "pmax": 16, "window": None, "bucket": 1024}
    for item in rest:
        k, sep, v = item.partition("=")
        if not sep or k not in spec or k in ("lo", "hi"):
            raise SystemExit("%s item %r: one of rule=, pmax=, window=, bucket=" % (ENV, item))
        if k == "rule":
            if v not in ("sb", "doc"):
                raise SystemExit("%s: rule is sb or doc" % ENV)
            spec[k] = v
        else:
            if not v.isdigit() or int(v) < 1:
                raise SystemExit("%s: %s must be a positive integer" % (ENV, k))
            spec[k] = int(v)
    return spec


def wrt_perm() -> np.ndarray:
    """stream byte -> WRT byte, neif's src/run_mode.h BuildPerm()"""
    perm = np.zeros(256, dtype=np.int64)
    for i in range(256):
        c = i
        if ord("{") <= c < 127:
            c += ord("P") - ord("{")
        elif ord("P") <= c < ord("T"):
            c -= ord("P") - ord("{")
        elif ord(":") <= c <= ord("?") or ord("J") <= c <= ord("O"):
            c ^= 0x70
        if c == ord("X") or c == ord("`"):
            c ^= ord("X") ^ ord("`")
        perm[i] = c
    return perm


def spacelike_wrt(r: int) -> bool:
    return r < 0x80 and not (ord("a") <= r <= ord("z") or ord("A") <= r <= ord("Z") or ord("0") <= r <= ord("9"))


def spacelike_table(bitmap: bytes | np.ndarray) -> np.ndarray:
    """bool[256] over token ids: tokens 0..207 through the vocab bitmap and the permutation; above: True"""
    bits = np.frombuffer(bytes(bitmap), dtype=np.uint8)
    assert bits.size == 32, "vocab bitmap: 32 bytes"
    values = [b * 8 + k for b in range(32) for k in range(8) if bits[b] >> k & 1]
    perm = wrt_perm()
    table = np.ones(256, dtype=bool)
    for token, stream_byte in enumerate(values):
        table[token] = spacelike_wrt(int(perm[stream_byte]))
    return table


def boundaries(tokens: torch.Tensor, starts: torch.Tensor, table: torch.Tensor, rule: str,
               pmax: int) -> torch.Tensor:
    """bool[L]: tokens int64[L] (bits 0..7 used), starts bool[L] (document starts), table bool[256]"""
    sl = table[tokens & TOKEN_MASK]
    b = starts.clone()
    if rule == "sb":
        prev = torch.roll(sl, 1)
        b |= sl & ~prev
    pos = torch.arange(tokens.numel(), device=tokens.device)
    last = torch.cummax(torch.where(b, pos, torch.zeros_like(pos)), dim=0).values
    gap = pos - last
    return b | ((gap > 0) & (gap % pmax == 0))


def compact_layout(b: torch.Tensor, cu: torch.Tensor, bucket: int):
    """(idx int64[Lp], j int64[L], cu_c like cu, Lc, Lp) for boundary mask b and cu_seqlens cu ([1, D+1])

    idx: the compact sequence's source positions (padding repeats position 0); j[t]: the compact index of
    the latest boundary <= t; cu_c: the compact documents, plus a padding document up to Lp."""
    L = b.numel()
    idx = b.nonzero().squeeze(1)
    lc = idx.numel()
    lp = max(bucket, math.ceil(lc / bucket) * bucket)
    cm = torch.cumsum(b.to(torch.int64), dim=0)
    j = cm - 1
    cm_ex = torch.cat([cm.new_zeros(1), cm])
    cu_c = cm_ex[cu.clamp(max=L).to(torch.int64)].to(cu.dtype)
    tail = []
    if int(cu_c[0, -1]) < lc:
        tail.append(lc)
    if lp > lc:
        tail.append(lp)
    if tail:
        cu_c = torch.cat([cu_c, torch.tensor([tail], dtype=cu.dtype, device=cu.device)], dim=1)
        idx = torch.cat([idx, idx.new_zeros(lp - lc)])
    return idx, j, cu_c, lc, lp


def install_hier(raw_model, cfg, spec: dict | None = None, bitmap_path: str | None = None,
                 log=print) -> dict | None:
    """switch raw_model's compute_last_activation to the hierarchical one; returns the spec or None"""
    spec = spec if spec is not None else parse(os.environ.get(ENV), cfg.n_layers)
    if spec is None:
        return None
    assert cfg.value_embedding_pattern is None, "TFT_HIER: value embeddings are not supported"
    assert not cfg.prior_embedding_connections, "TFT_HIER: prior embedding connections are not supported"
    if bitmap_path is None:
        data = os.environ.get("PK_DUMP_DIR") or os.environ.get("TFT_DATA_DIR") or "."
        bitmap_path = os.path.join(data, "vocab-bitmap.u8")
    with open(bitmap_path, "rb") as f:
        table = spacelike_table(f.read())
    raw_model._hier = {**spec, "table": torch.from_numpy(table), "calls": 0, "frac": 0.0}
    raw_model.compute_last_activation = types.MethodType(compute_last_activation_hier, raw_model)
    log("hier: layers %d-%d at word rate (rule %s, pmax %d, window %s, bucket %d); %d of 256 token ids "
        "space-like" % (spec["lo"], spec["hi"], spec["rule"], spec["pmax"], spec["window"] or "model",
                        spec["bucket"], int(table.sum())))
    return spec


def compute_last_activation_hier(self, input_tokens, priors, cumulative_sequence_lengths,
                                 override_window_size=None, initialize_scales=None):
    """pysrc/model.py Transformer.compute_last_activation with layers LO..HI on the compact sequence"""
    from pysrc.attention_function import make_flex_attention_block_mask
    from pysrc.model import InitializeScales, make_rope_args, rms_norm
    from torch.nn.functional import dropout

    if initialize_scales is None:
        initialize_scales = InitializeScales(None, None)
    cfg, h = self.cfg, self._hier
    assert input_tokens.ndim == 2 and input_tokens.size(0) == 1, "TFT_HIER needs batch size 1 (packed)"
    _, sequence_length = input_tokens.shape
    device = input_tokens.device
    window_size = override_window_size if override_window_size is not None else cfg.window_size
    window_c = h["window"] or window_size
    flex = cfg.attention_implementation == "flex_attention"

    cu = cumulative_sequence_lengths
    if cu is None:
        cu = torch.tensor([[0, sequence_length]], dtype=torch.int32, device=device)
    starts = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    first = cu[0, :-1].to(torch.int64)
    starts[first[first < sequence_length]] = True
    if h["table"].device != device:
        h["table"] = h["table"].to(device)
    b = boundaries(input_tokens[0], starts, h["table"], h["rule"], h["pmax"])
    idx, j, cu_c, lc, lp = compact_layout(b, cu, h["bucket"])
    h["calls"] += 1
    h["frac"] += (lc / sequence_length - h["frac"]) / h["calls"]
    if h["calls"] == 1 or h["calls"] % 500 == 0:
        print("hier: call %d, boundaries %.2f%% of positions (this batch %d of %d, compact %d)"
              % (h["calls"], 100 * h["frac"], lc, sequence_length, lp), flush=True)

    if priors is not None:
        priors = priors.to(cfg.activation_dtype)
    token_embeddings = self.embedding(input_tokens, initialize_scales=initialize_scales)
    if cfg.embedding_norm:
        token_embeddings = rms_norm(token_embeddings)
    if cfg.dropout != 0:
        token_embeddings = dropout(token_embeddings, cfg.dropout, training=self.training)
    x = token_embeddings
    if cfg.prior_embedding:
        prior_embeddings = self.prior_embedding(self.prior_embedding_input(priors),
                                                initialize_scales=initialize_scales)
        if cfg.prior_embedding_norm:
            prior_embeddings = rms_norm(prior_embeddings)
        if cfg.dropout != 0:
            prior_embeddings = dropout(prior_embeddings, cfg.dropout, training=self.training)
        x = x + prior_embeddings

    levels = {
        "symbol": dict(cu=cumulative_sequence_lengths, rope=make_rope_args(sequence_length, cfg, device),
                       window=window_size, tok=token_embeddings,
                       mask=make_flex_attention_block_mask(cumulative_sequence_lengths, sequence_length,
                                                           window_size, device) if flex else None),
        "word": dict(cu=cu_c, rope=make_rope_args(lp, cfg, device), window=window_c,
                     tok=token_embeddings[:, idx],
                     mask=make_flex_attention_block_mask(cu_c, lp, window_c, device) if flex else None),
    }
    n = cfg.n_layers
    sources, destinations = list(range(n // 2)), list(range((n + 1) // 2, n))
    skips = []
    weights = self.skip_connection_weights() if cfg.skip_connections else None
    x_word_in = None
    for layer, block in enumerate(self.blocks):
        if layer == h["lo"]:
            x_word_in = x[:, idx]
            x_symbol, x = x, x_word_in
        level = levels["word" if h["lo"] <= layer <= h["hi"] else "symbol"]
        if cfg.skip_connections and layer in destinations:
            x = x + weights[destinations.index(layer)] * skips.pop()
        x = self.maybe_checkpoint(
            block, x,
            cumulative_sequence_lengths=level["cu"], rope_args=level["rope"],
            token_embeddings=level["tok"], prior_embeddings=None, value_embedding=None,
            window_size=level["window"], flex_attention_block_mask=level["mask"],
            initialize_scales=initialize_scales,
        )
        if cfg.skip_connections and layer in sources:
            skips.append(x)
        if layer == h["hi"]:
            x = x_symbol + (x - x_word_in)[:, j]
    return rms_norm(x)
