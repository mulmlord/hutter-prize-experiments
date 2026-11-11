#!/usr/bin/env python3
"""A trainer checkpoint's weight scales for fewer levels per group (E303), as a warm start for qat.

    builds/tftrain/.venv/bin/python train/requant_levels.py IN.tch OUT.tch \
        --levels mlp.down=11,attention.query_projection=11 [--fit range|mse]

The trainer with train/weight_levels.patch quantizes the weights that TFT_WEIGHT_LEVELS selects to L levels
(ints -(L-1)/2 .. (L-1)/2) instead of 15. qat then runs with TFT_KEEP_SCALES=1 (the loaded state has scales),
so the scales it starts from must already fit the new range. This sets them once, in the warm start; a resume
and the fp32 phase load the trained state and need nothing. --levels takes the same SUBSTR=L items as
TFT_WEIGHT_LEVELS, and the run must use the same string.

The latent fp32 weights stay as they are; only each selected row's scale s changes:
  range  s * 7 / h with h = (L-1)/2: the clipping range stays at 7 s, and the grid steps widen by 7/h
  mse    per row, the factor f * 7 / h (f in 0.70..1.05) with the least squared quantization error
The new scale is rounded to bfloat16, which is what the quantizer computes with. The report gives, per group,
the squared error against the 15-level quantization and the share of values at the range limit.
"""

from __future__ import annotations

import argparse
import sys

import torch

SCALE = ".quantize_weight.scale"


def parse_levels(text: str) -> dict[str, int]:
    """the TFT_WEIGHT_LEVELS syntax of tft/config.py (train/weight_levels.patch)"""
    out = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        sub, _, val = item.partition("=")
        if not sub or not val.isdigit() or int(val) % 2 == 0 or not 5 <= int(val) <= 15:
            raise SystemExit("--levels item %r: SUBSTR=L with L odd, 5..15" % item)
        out[sub] = int(val)
    return out


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def quantize(w: torch.Tensor, s: torch.Tensor, h: int) -> torch.Tensor:
    """QuantizeFunction.forward for one scale per row: s * clamp(round(w / s), -h, h), fp32, bf16 scale"""
    sb = bf16(s).unsqueeze(-1)
    return sb * (w / sb).round().clamp(-h, h)


def fit_mse(w: torch.Tensor, s: torch.Tensor, h: int) -> torch.Tensor:
    best, best_err = None, None
    for f in torch.linspace(0.70, 1.05, 36).tolist():
        cand = bf16(s * (f * 7 / h))
        err = ((quantize(w, cand, h) - w) ** 2).sum(-1)
        if best is None:
            best, best_err = cand.clone(), err
        else:
            better = err < best_err
            best = torch.where(better, cand, best)
            best_err = torch.where(better, err, best_err)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--levels", required=True, help="SUBSTR=L,... as TFT_WEIGHT_LEVELS")
    ap.add_argument("--fit", choices=["range", "mse"], default="range")
    a = ap.parse_args(argv)
    levels = parse_levels(a.levels)
    sd = torch.load(a.inp, map_location="cpu")
    groups: dict[str, dict] = {sub: {"tensors": 0, "values": 0, "err15": 0.0, "err": 0.0, "edge": 0}
                               for sub in levels}
    for key in list(sd):
        if not key.endswith(SCALE):
            continue
        wkey = key[: -len(SCALE)] + ".weight"
        if wkey not in sd:
            continue                               # an activation scale
        hits = [sub for sub in levels if sub in wkey]
        if len(hits) > 1:
            raise SystemExit("%s matches several --levels items: %s" % (wkey, hits))
        if not hits:
            continue
        sub = hits[0]
        h = (levels[sub] - 1) // 2
        w = sd[wkey].detach().to(torch.float32)
        s = sd[key].detach().to(torch.float32)
        if w.ndim != 2 or s.shape != (w.shape[0],):
            raise SystemExit("%s: expected one scale per row, got weight %s scale %s"
                             % (wkey, tuple(w.shape), tuple(s.shape)))
        new = bf16(s * 7 / h) if a.fit == "range" else fit_mse(w, s, h)
        g = groups[sub]
        g["tensors"] += 1
        g["values"] += w.numel()
        g["err15"] += float(((quantize(w, s, 7) - w) ** 2).sum())
        g["err"] += float(((quantize(w, new, h) - w) ** 2).sum())
        g["edge"] += int(((w / bf16(new).unsqueeze(-1)).round().abs() >= h).sum())
        sd[key] = new.to(sd[key].dtype)
    unused = [sub for sub, g in groups.items() if not g["tensors"]]
    if unused:
        raise SystemExit("--levels items match no quantized weight: %s" % unused)
    torch.save(sd, a.out)
    for sub, g in groups.items():
        print("%-28s L=%-2d %3d tensors %9d values  squared error x%.2f of 15 levels, %.2f%% at +-%d"
              % (sub, levels[sub], g["tensors"], g["values"], g["err"] / max(g["err15"], 1e-30),
                 100 * g["edge"] / g["values"], (levels[sub] - 1) // 2))
    print("wrote %s (--fit %s); train with TFT_WEIGHT_LEVELS=%s" % (a.out, a.fit, a.levels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
