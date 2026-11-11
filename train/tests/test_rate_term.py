"""train/rate_term.py (E303's byte term) on small stand-ins for the trainer's quantized layers."""
import importlib.util
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rate_term", ROOT / "train" / "rate_term.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


class Quant:   # the fields of pysrc.quantization.Quantization that rate_term reads
    def __init__(self, buckets):
        self.buckets, self.qmax, self.qmin = buckets, (buckets - 1) // 2, -((buckets - 1) // 2)
        self.scale_dtype = torch.bfloat16


class FakeQuantize(torch.nn.Module):   # one scale per row, as the trainer's weight quantizers
    def __init__(self, rows, cols, buckets):
        super().__init__()
        self.quant, self.size = Quant(buckets), (rows, cols)
        self.scale = torch.nn.Parameter(torch.zeros(rows))

    def reshape_input(self, w):
        assert w.shape == self.size
        return w.view(1, *self.size)


class Layer(torch.nn.Module):
    def __init__(self, rows, cols, buckets, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(rows, cols, generator=g) * 0.03)
        self.quantize_weight = FakeQuantize(rows, cols, buckets)
        h = (buckets - 1) // 2
        with torch.no_grad():
            self.quantize_weight.scale.copy_(self.weight.abs().amax(-1) / h * 0.8)   # some values clipped


def model():
    m = torch.nn.Module()
    m.a, m.b = Layer(16, 48, 15, 1), Layer(24, 32, 11, 2)
    m.c = torch.nn.Linear(4, 4)   # not quantized
    return m


def ints(layer):
    q = layer.quantize_weight
    s = q.scale.detach().to(torch.bfloat16).float().unsqueeze(-1)
    return (layer.weight.detach() / s).round().clamp(-q.quant.qmax, q.quant.qmax).long(), s


def test_from_env(monkeypatch):
    monkeypatch.delenv("TFT_RATE_LAMBDA", raising=False)
    assert rt.RateTerm.from_env() is None
    monkeypatch.setenv("TFT_RATE_LAMBDA", "")
    assert rt.RateTerm.from_env() is None
    monkeypatch.setenv("TFT_RATE_LAMBDA", "3.2e-9")
    assert rt.RateTerm.from_env().lam == pytest.approx(3.2e-9)
    assert rt.EXCHANGE_LAMBDA == pytest.approx(3.2335e-9, rel=1e-3)


def test_value_is_order0_code_length_of_the_ints():
    m = model()
    qs = rt.weight_quantizers(m)
    assert [n for n, _, _ in qs] == ["a", "b"]
    r, values, tensors = rt.rate_bits(qs)
    want = 0.0
    for layer in (m.a, m.b):
        k, _ = ints(layer)
        h = layer.quantize_weight.quant.qmax
        n = torch.bincount((k + h).view(-1), minlength=2 * h + 1).double()
        want += float(-(n * torch.log2((n + 0.5) / (n.sum() + 0.5 * (2 * h + 1)))).sum())
    assert float(r) == pytest.approx(want, rel=1e-5)
    assert (values, tensors) == (16 * 48 + 24 * 32, 2)


def test_gradient_is_the_code_length_slope():
    m = model()
    r, _, _ = rt.rate_bits(rt.weight_quantizers(m))
    r.backward()
    layer = m.b
    q = layer.quantize_weight
    h = q.quant.qmax
    k, s = ints(layer)
    x = layer.weight.detach() / s
    n = torch.bincount((k + h).view(-1), minlength=2 * h + 1).float()
    cost = -torch.log2((n + 0.5) / (n.sum() + 0.5 * (2 * h + 1)))
    slope = torch.cat([(cost[1:2] - cost[0:1]), (cost[2:] - cost[:-2]) / 2, cost[-1:] - cost[-2:-1]])
    inside = (x >= -h) & (x <= h)
    want_w = torch.where(inside, slope[k + h] / s, torch.zeros_like(x))
    assert torch.allclose(layer.weight.grad, want_w, rtol=1e-4, atol=1e-6)
    assert (~inside).any() and (layer.weight.grad[~inside] == 0).all()
    # scale: d/ds of slope * w / s, times LSQ's 1 / sqrt(numel * h)
    g = 1 / math.sqrt(x.numel() * h)
    want_s = (torch.where(inside, slope[k + h] * -x / s, torch.zeros_like(x)).sum(-1)) * g
    # the gradient comes back through the bf16 cast of the scale (as in the forward), so it is bf16-rounded
    assert torch.allclose(q.scale.grad, want_s, rtol=1e-2, atol=1e-6)
    # most weight sits near zero and the costs grow outward, so the term pulls the scales up
    assert float(q.scale.grad.sum()) < 0


def test_add_grad_scales_by_lambda_and_tokens_and_keeps_ce_grads():
    m1, m2 = model(), model()
    for m in (m1, m2):
        m.a.weight.grad = torch.full_like(m.a.weight, 0.25)   # stands for the cross-entropy's gradient
    rate = rt.RateTerm(2e-3)
    bits = rate.add_grad(m1, tokens=1000.0)
    r, _, _ = rt.rate_bits(rt.weight_quantizers(m2))
    r.backward()
    assert bits == pytest.approx(float(r), rel=1e-6)
    assert torch.allclose(m1.a.weight.grad, 0.25 + 2.0 * (m2.a.weight.grad - 0.25), rtol=1e-4, atol=1e-6)
    assert torch.allclose(m1.b.quantize_weight.scale.grad, 2.0 * m2.b.quantize_weight.scale.grad, rtol=1e-4)


def test_descent_on_the_term_alone_lowers_the_bits():
    m = model()
    params = [p for _, w, q in rt.weight_quantizers(m) for p in (w, q.scale)]
    opt = torch.optim.Adam(params, lr=2e-3)
    first = None
    for _ in range(60):
        opt.zero_grad()
        r, _, _ = rt.rate_bits(rt.weight_quantizers(m))
        first = float(r) if first is None else first
        r.backward()
        opt.step()
    assert float(rt.rate_bits(rt.weight_quantizers(m))[0]) < 0.9 * first


def test_dead_rows_are_left_out():
    """an all-zero row with a ~1e-9 scale (as t205's forget gates have) and a row whose scale crossed zero
    (|s| 1e-6): no bits, no gradient; the other rows are counted as before"""
    m, ref = model(), model()
    with torch.no_grad():
        m.b.weight[3].zero_()
        m.b.quantize_weight.scale[3] = 9.313e-10
        m.b.weight[5] *= 1e-4
        m.b.quantize_weight.scale[5] = -1e-6
    r, values, _ = rt.rate_bits(rt.weight_quantizers(m))
    r.backward()
    assert values == 16 * 48 + 22 * 32
    assert (m.b.weight.grad[[3, 5]] == 0).all() and (m.b.quantize_weight.scale.grad[[3, 5]] == 0).all()
    assert torch.isfinite(m.b.weight.grad).all() and float(m.b.weight.grad.abs().max()) < 1e4
    # the same rows removed from the reference layer give the same bits
    keep = [i for i in range(24) if i not in (3, 5)]
    ref.b.weight = torch.nn.Parameter(ref.b.weight.detach()[keep].clone())
    ref.b.quantize_weight = FakeQuantize(22, 32, 11)
    with torch.no_grad():
        ref.b.quantize_weight.scale.copy_(m.b.quantize_weight.scale.detach()[keep])
    assert float(r) == pytest.approx(float(rt.rate_bits(rt.weight_quantizers(ref))[0]), rel=1e-6)


def test_no_quantized_weights_is_a_no_op():
    m = torch.nn.Linear(3, 3)
    assert rt.RateTerm(1e-9).add_grad(m, tokens=10.0) == 0.0
    assert m.weight.grad is None


TRAIN_PY = Path(os.environ.get("NEIF_TRAIN_PY", ROOT / "neif/training/tf/tree_upstream/pysrc/train.py"))


@pytest.mark.skipif(shutil.which("patch") is None or not TRAIN_PY.exists(),
                    reason="needs patch(1) and cmix-neif's training/tf/tree_upstream/pysrc/train.py (NEIF_TRAIN_PY)")
def test_patch_applies_to_the_upstream_train_loop(tmp_path):
    (tmp_path / "pysrc").mkdir()
    shutil.copy(TRAIN_PY, tmp_path / "pysrc/train.py")
    p = subprocess.run(["patch", "-p1", "-s", "-i", str(ROOT / "train/rate_term.patch")], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    src = (tmp_path / "pysrc/train.py").read_text()
    assert src.index("rate.add_grad(") < src.index("optimizers.step()")
