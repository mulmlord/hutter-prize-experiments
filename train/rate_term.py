"""E303: a byte term in the trainer's loss, so that qat prices the weight container (TFT_RATE_LAMBDA).

On the VM this file is the trainer tree's tft/rate.py (train/gcp_vm.sh copies it), and train/rate_term.patch
calls it in pysrc/train.py once per optimizer step, after the micro-batches' backward and before the step.
With TFT_RATE_LAMBDA unset or empty, RateTerm.from_env() is None and nothing changes.

The trainer minimizes the mean cross-entropy per token; this adds LAMBDA nats/token per bit of int weights:
    loss = CE + LAMBDA * R,   R = sum over quantized weights of -log2 p_t(q)
q = clamp(round(w / s), -h, h) is the int the export ships; p_t is that tensor's own current histogram of q
(order 0, counts + 0.5). R's value is the code length of the current ints: v4 codes them adaptively at
about that (E303 README: per-tensor order 0 is +4.8 KB on 3.63 MB). Its gradient is straight-through, as
the quantizer's is: d(-log2 p(q))/dx is the slope of the tensor's code-length table at q (centred, one-sided
at +-h), x = w / s unclamped inside the grid, 0 outside. The histogram is held fixed for the gradient.
The scale's share of that gradient is multiplied by 1 / sqrt(numel * h), the factor QuantizeFunction.backward
puts on the cross-entropy's scale gradient (LSQ), so both reach the optimizer in the same proportion.
Rows whose |scale| is at most DEAD_ROW times the tensor's median are left out (see rate_bits).

Units: pysrc/train.py backpropagates sum(token losses) / gradient_accumulation_steps over the micro-batches,
that is (unmasked tokens / accumulation steps) * mean CE, so add_grad takes that token count and scales R
by it. The exchange rate measured for t205-8q2f (E303 README) gives the LAMBDA at which the trainer's optimum
is the smallest S:
    dS = 2 * d(container bytes) + 700,000 B * dCE%,  1% of CE = 0.009054 nats/token
    => LAMBDA = (2 / 8) / (700,000 * 100 / 0.905424) = 3.23e-9 nats/token per bit
"""

from __future__ import annotations

import math
import os

import torch

EXCHANGE_LAMBDA = 0.25 / (700_000 * 100 / 0.905424)   # nats/token per bit, ~3.23e-9
DEAD_ROW = 1e-3   # a row whose |scale| is at most this times the tensor's median |scale| is left out


def weight_quantizers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, torch.nn.Module]]:
    """(name, weight, Quantize) for every weight on an int grid (buckets >= 5: round and clamp to qmin..qmax)"""
    out = []
    for name, mod in model.named_modules():
        q = getattr(mod, "quantize_weight", None)
        if q is None or getattr(q, "quant", None) is None or q.quant.buckets < 5:
            continue
        assert q.quant.qmin == -q.quant.qmax, (name, q.quant.qmin, q.quant.qmax)
        out.append((name, mod.weight, q))
    return out


def rate_bits(quantizers) -> tuple[torch.Tensor, int, int]:
    """(R in bits with the straight-through gradient, int values, tensors)"""
    total = None
    values = 0
    for _, w, q in quantizers:
        h = q.quant.qmax
        k_levels = 2 * h + 1
        blocks = q.reshape_input(w).to(torch.float32)
        s = q.scale.view(-1)
        g = 1 / math.sqrt(blocks.numel() * h)
        s = s * g + (s - s * g).detach()                     # value s, gradient times g
        s = s.to(q.quant.scale_dtype).to(torch.float32).view(1, -1, 1)
        # dead rows: a scale near zero (all-zero rows carry ~1e-9, LSQ can also drive one through zero). Their
        # values are numerically zero and the export zeroes such rows; d/dw = slope / s would be ~1e9 there
        # and, through Muon's per-matrix normalisation, swamp the whole matrix's update. Left out entirely.
        with torch.no_grad():
            live = s.abs() > DEAD_ROW * s.abs().median()
        x = torch.where(live, blocks / torch.where(live, s, torch.ones_like(s)), torch.zeros_like(blocks))
        inside = (x >= -h) & (x <= h) & live
        with torch.no_grad():
            k = x.round().clamp(-h, h).to(torch.int64) + h
            n = torch.bincount(k[live.expand_as(k)], minlength=k_levels).to(torch.float32)
            cost = -torch.log2((n + 0.5) / (n.sum() + 0.5 * k_levels))
            slope = torch.empty_like(cost)
            slope[1:-1] = (cost[2:] - cost[:-2]) / 2
            slope[0] = cost[1] - cost[0]
            slope[-1] = cost[-1] - cost[-2]
            bits = (n * cost).sum()
        # value: the code length of the current ints; gradient: slope at q times dx, only inside the grid
        dx = torch.where(inside, x - x.detach(), torch.zeros_like(x))
        r = bits + (slope[k] * dx).sum()
        total = r if total is None else total + r
        values += int(live.sum()) * blocks.size(-1)
    if total is None:
        total = torch.zeros(())
    return total, values, len(quantizers)


class RateTerm:
    def __init__(self, lam: float, every: int = 50) -> None:
        assert lam > 0, lam
        self.lam = lam
        self.every = every
        self.steps = 0

    @classmethod
    def from_env(cls) -> "RateTerm | None":
        v = os.environ.get("TFT_RATE_LAMBDA", "").strip()
        if not v:
            return None
        print("rate: TFT_RATE_LAMBDA=%g nats/token per bit, %.3g x the E303 exchange rate"
              % (float(v), float(v) / EXCHANGE_LAMBDA), flush=True)
        return cls(float(v), every=int(os.environ.get("TFT_RATE_EVERY", "50")))   # log line every N steps

    def add_grad(self, model: torch.nn.Module, tokens: float) -> float:
        """adds tokens * LAMBDA * dR/dparams to the .grad of model's weights and scales; returns R in bits

        tokens: unmasked tokens of this optimizer step / gradient accumulation steps (see the module doc)
        """
        quantizers = weight_quantizers(model)
        self.steps += 1
        if not quantizers:
            if self.steps == 1:
                print("rate: no quantized weights in this phase, the term does nothing", flush=True)
            return 0.0
        log = self.steps == 1 or self.steps % self.every == 0
        params = [p for _, w, q in quantizers for p in (w, q.scale)]
        before = [None if p.grad is None else p.grad.detach().clone() for p in params] if log else None
        r, values, tensors = rate_bits(quantizers)
        (r * (self.lam * tokens)).backward()
        bits = float(r.detach())
        if log:
            ce_w = ce_s = rt_w = rt_s = 0.0
            for i, p in enumerate(params):
                old = before[i] if before[i] is not None else torch.zeros_like(p)
                new = p.grad if p.grad is not None else torch.zeros_like(p)
                ce, rt = float(old.float().pow(2).sum()), float((new - old).float().pow(2).sum())
                if i % 2 == 0:
                    ce_w, rt_w = ce_w + ce, rt_w + rt
                else:
                    ce_s, rt_s = ce_s + ce, rt_s + rt
            print("rate: step %d, %d B order 0 in %d tensors (%.4f bits/value), lambda*R %.5f nats/token; "
                  "grad norm weights CE %.3e rate %.3e, scales CE %.3e rate %.3e"
                  % (self.steps, round(bits / 8), tensors, bits / values, self.lam * bits,
                     math.sqrt(ce_w), math.sqrt(rt_w), math.sqrt(ce_s), math.sqrt(rt_s)), flush=True)
        return bits
