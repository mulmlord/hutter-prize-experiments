"""train/hier.py (word-rate hierarchical layers): the parts the decoder must reproduce, on CPU."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hier", ROOT / "train" / "hier.py")
hier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hier)


def test_parse():
    assert hier.parse("", 16) is None and hier.parse(None, 16) is None
    s = hier.parse("2-13", 16)
    assert (s["lo"], s["hi"], s["rule"], s["pmax"], s["window"], s["bucket"]) == (2, 13, "sb", 16, None, 1024)
    s = hier.parse("4-11,rule=doc,pmax=4,window=512,bucket=256", 16)
    assert (s["lo"], s["hi"], s["rule"], s["pmax"], s["window"], s["bucket"]) == (4, 11, "doc", 4, 512, 256)
    for bad in ("2-12", "0-15", "13-2", "2", "2-13,rule=x", "2-13,pmax=0", "2-13,foo=1"):
        with pytest.raises(SystemExit):
            hier.parse(bad, 16)


def test_wrt_perm_is_a_permutation_and_matches_run_mode():
    perm = hier.wrt_perm()
    assert sorted(perm.tolist()) == list(range(256))
    # spot values of src/run_mode.h BuildPerm, worked by hand
    assert perm[ord("{")] == ord("P") and perm[ord("P")] == ord("{")
    assert perm[ord(":")] == ord(":") ^ 0x70
    assert perm[ord("X")] == ord("`") and perm[ord("`")] == ord("X")
    assert perm[ord("a")] == ord("a") and perm[0x80] == 0x80


def bitmap_for(stream_bytes):
    bits = bytearray(32)
    for v in stream_bytes:
        bits[v // 8] |= 1 << (v % 8)
    return bytes(bits)


def test_spacelike_table_through_bitmap_and_permutation():
    inv = {int(r): s for s, r in enumerate(hier.wrt_perm())}
    wrt = [ord(" "), ord("a"), ord("5"), 0x40, 0x80, 0xD3, ord("."), ord("\n"), ord("Q")]
    stream = sorted(inv[r] for r in wrt)
    table = hier.spacelike_table(bitmap_for(stream))
    perm = hier.wrt_perm()
    for token, s in enumerate(stream):
        r = int(perm[s])
        want = r in (ord(" "), 0x40, ord("."), ord("\n"))
        assert table[token] == want, (token, hex(r))
    assert table[len(stream):].all()          # sentinel rows and unused ids count as space-like


def test_boundaries_spacebyte_rule_doc_starts_and_pmax():
    table = torch.zeros(256, dtype=torch.bool)
    table[1] = True                           # token 1 is the space; 0 a letter, 2 a code byte
    tok = torch.tensor([0, 0, 1, 2, 2, 1, 1, 0, 0, 1], dtype=torch.int64)
    starts = torch.zeros(10, dtype=torch.bool)
    starts[0] = starts[7] = True              # two documents: 0..6 and 7..9
    b = hier.boundaries(tok, starts, table, "sb", pmax=16)
    assert b.nonzero().squeeze(1).tolist() == [0, 2, 5, 7, 9]   # 6: a space after a space
    # the packed carrier: only bits 0..7 are the token
    b2 = hier.boundaries(tok | (5 << 8) | (77 << 24), starts, table, "sb", pmax=16)
    assert torch.equal(b, b2)
    run = torch.zeros(10, dtype=torch.int64)
    st = torch.zeros(10, dtype=torch.bool)
    st[0] = True
    assert hier.boundaries(run, st, table, "sb", pmax=3).nonzero().squeeze(1).tolist() == [0, 3, 6, 9]
    assert hier.boundaries(tok, starts, table, "doc", pmax=4).nonzero().squeeze(1).tolist() == [0, 4, 7]


def test_compact_layout_is_causal_and_per_document():
    b = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    cu = torch.tensor([[0, 7, 10]], dtype=torch.int32)
    idx, j, cu_c, lc, lp = hier.compact_layout(b, cu, bucket=8)
    assert lc == 5 and lp == 8
    assert idx[:lc].tolist() == [0, 2, 5, 7, 9] and idx[lc:].tolist() == [0, 0, 0]
    assert cu_c.tolist() == [[0, 3, 5, 8]] and cu_c.dtype == cu.dtype     # two documents + padding
    for t in range(10):
        src = int(idx[j[t]])
        assert src <= t and b[src] and not b[src + 1:t + 1].any()         # the latest boundary <= t
    doc = lambda p: 0 if p < 7 else 1
    assert all(doc(int(idx[j[t]])) == doc(t) for t in range(10))
    # no padding when Lc fills the bucket exactly
    idx2, _, cu2, lc2, lp2 = hier.compact_layout(b, cu, bucket=5)
    assert (lc2, lp2) == (5, 5) and cu2.tolist() == [[0, 3, 5]] and idx2.numel() == 5


@pytest.mark.skipif(shutil.which("patch") is None or not (ROOT / "neif/training/tf/tree_upstream").is_dir(),
                    reason="needs patch(1) and neif's training tree")
def test_vm_tree_patches_apply_in_order(tmp_path):
    """the trainer tree as train/gcp_vm.sh builds it: upstream + overlay + paket11, then the patches"""
    t = tmp_path / "tree"
    shutil.copytree(ROOT / "neif/training/tf/tree_upstream", t)
    shutil.copytree(ROOT / "neif/training/tf/tree_overlay", t, dirs_exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "neif/training/tf/2_package/patch_paket11.py"), str(t)],
                   check=True, capture_output=True)
    for p in ("keep_scales", "weight_levels", "rate_term", "hier"):
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(ROOT / "train" / (p + ".patch"))], cwd=t,
                           capture_output=True, text=True)
        assert r.returncode == 0, p + ": " + r.stdout + r.stderr
    src = (t / "tft/runner.py").read_text()
    assert src.index("apply_weight_levels(raw_model)") < src.index("install_hier(raw_model") < src.index("# Warm start")
