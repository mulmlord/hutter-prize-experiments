"""train/requant_levels.py on a synthetic checkpoint (E303)."""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "requant_levels.py"


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def make(tmp_path):
    g = torch.Generator().manual_seed(1)
    sd = {}
    for name in ("blocks.0.mlp.down", "blocks.0.mlp.up", "blocks.0.attention.key_projection"):
        w = torch.randn(32, 64, generator=g)
        sd[name + ".weight"] = w
        sd[name + ".quantize_weight.scale"] = bf16(w.abs().amax(-1) / 7)
        sd[name + ".quantize_activation.scale"] = torch.tensor([0.5])
    p = tmp_path / "in.tch"
    torch.save(sd, p)
    return sd, p


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def err(w, s, h):
    sb = bf16(s).unsqueeze(-1)
    return float(((sb * (w / sb).round().clamp(-h, h) - w) ** 2).sum())


def test_range_and_mse(tmp_path):
    sd, inp = make(tmp_path)
    for fit in ("range", "mse"):
        r = run(inp, tmp_path / (fit + ".tch"), "--levels", "mlp.down=11,key_projection=7", "--fit", fit)
        assert r.returncode == 0, r.stderr
        out = torch.load(tmp_path / (fit + ".tch"))
        for k in sd:                                      # weights, activation scales, unselected: unchanged
            if not (k.endswith("quantize_weight.scale") and ("mlp.down" in k or "key_projection" in k)):
                assert torch.equal(out[k], sd[k]), k
        for name, h in (("blocks.0.mlp.down", 5), ("blocks.0.attention.key_projection", 3)):
            s0, s1 = sd[name + ".quantize_weight.scale"], out[name + ".quantize_weight.scale"]
            assert torch.equal(s1, bf16(s1))              # already what the quantizer computes with
            if fit == "range":
                assert torch.equal(s1, bf16(s0 * 7 / h))
        if fit == "range":
            rng = out
    for name, h in (("blocks.0.mlp.down", 5), ("blocks.0.attention.key_projection", 3)):
        w = sd[name + ".weight"]
        assert err(w, out[name + ".quantize_weight.scale"], h) <= err(w, rng[name + ".quantize_weight.scale"], h)


def test_bad_levels(tmp_path):
    _, inp = make(tmp_path)
    assert "match no quantized weight" in run(inp, tmp_path / "o.tch", "--levels", "nothing=11").stderr
    assert "several" in run(inp, tmp_path / "o.tch", "--levels", "mlp=11,mlp.down=9").stderr
    assert "odd, 5..15" in run(inp, tmp_path / "o.tch", "--levels", "mlp.down=8").stderr
    assert not (tmp_path / "o.tch").exists()
