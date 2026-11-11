"""How much smaller could the int4 weights of a v4 container be coded? (E303 feasibility)

  python experiments/E303-cheaper-weights/wentropy.py CONTAINER.c4   (after source scripts/tfeval/env.sh)

1. v4's real cost of the int4 values: the container's size minus the size of the same container
   re-encoded with every int4 value set to 0 (all-zero rows are one flag bit each).
2. Adaptive code lengths (what an arithmetic coder pays, learning included) under several contexts,
   with a counting estimator like v4's (counts halved at a limit): -sum log2 p.
3. xz and zstd on the raw values, for reference.
"""
import lzma
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tfeval"))
import perturb  # noqa: E402

path = sys.argv[1]
w = perturb.codec()
tensors = perturb.load(path)
size = os.path.getsize(path)

int4 = [(n, arr.astype(np.int64)) for n, (c, arr) in tensors.items() if perturb.is_int4(n, c)]
scales = {n[:-len(".q")]: tensors[n[:-len(".q")] + ".scale"][1] for n, _ in int4 if n[:-len(".q")] + ".scale" in tensors}
nvals = sum(a.size for _, a in int4)
print(f"{path}: {size:,} B, {len(tensors)} tensors, {len(int4)} int4 tensors with {nvals:,} values")

# 1. v4's cost of the int4 values
entries_zero = [(n, c, (np.zeros_like(arr) if perturb.is_int4(n, c) else arr)) for n, (c, arr) in tensors.items()]
with tempfile.TemporaryDirectory() as d:
    p0 = os.path.join(d, "zero.c4")
    w.write_tensor_file_v4(p0, entries_zero)
    size0 = os.path.getsize(p0)
    p1 = os.path.join(d, "same.c4")
    w.write_tensor_file_v4(p1, [(n, c, arr) for n, (c, arr) in tensors.items()])
    size1 = os.path.getsize(p1)
v4_int4 = size1 - size0
print(f"re-encoded as is: {size1:,} B (file {size:,}); int4 values zeroed: {size0:,} B")
print(f"v4 spends {v4_int4:,} B on the int4 values = {8 * v4_int4 / nvals:.4f} bits/value; "
      f"everything else (scales, f32, names) {size0:,} B")
zero_rows = sum(int((a.reshape(a.shape[0], -1) == 0).all(axis=1).sum()) for _, a in int4 if a.ndim >= 2)


class Counter:
    """adaptive frequency model over 15 symbols (-7..7), counts halved when the total passes limit"""
    __slots__ = ("c", "t", "limit")

    def __init__(self, limit=1023, init=0.5):
        self.c = [init] * 15
        self.t = init * 15
        self.limit = limit

    def cost(self, s):
        b = -math.log2(self.c[s] / self.t)
        self.c[s] += 1
        self.t += 1
        if self.t > self.limit:
            self.c = [max(x / 2, 0.25) for x in self.c]
            self.t = sum(self.c)
        return b


def adaptive(ctx_fn, limit=1023, label=""):
    """total bits with one Counter per context returned by ctx_fn(tensor index, name, arr2d, row, col)"""
    models = {}
    bits = 0.0
    for ti, (name, a) in enumerate(int4):
        a2 = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)
        state = {}
        for r in range(a2.shape[0]):
            row = a2[r]
            if not row.any():
                continue                      # v4 flags all-zero rows; same here
            for col in range(a2.shape[1]):
                k = ctx_fn(ti, name, a2, r, col, state)
                m = models.get(k)
                if m is None:
                    m = models[k] = Counter(limit)
                s = int(row[col]) + 7
                bits += m.cost(s)
                state.setdefault("update", lambda *_: None)(r, col, s)
    return bits


def mag(v):
    v = abs(v)
    return 0 if v == 0 else 1 if v <= 1 else 2 if v <= 3 else 3


results = []


def report(label, bits):
    B = bits / 8
    results.append((label, B))
    print(f"  {label:<58} {B:>12,.0f} B  {bits / nvals:.4f} bits/value  {B - v4_int4:>+10,.0f} B vs v4")


print("adaptive code lengths of the int4 values (all-zero rows skipped, as in v4):")
report("order 0, one model for all tensors (v4's model)", adaptive(lambda ti, n, a, r, c, st: 0))
report("order 0, one model per tensor", adaptive(lambda ti, n, a, r, c, st: ti))


# per tensor + previous value in the row
def prev_in_row(ti, n, a, r, c, st):
    return (ti, int(a[r, c - 1]) if c else 99)


report("per tensor + previous value in the row", adaptive(prev_in_row))


# per tensor + magnitude of the same column in the previous row
def col_above(ti, n, a, r, c, st):
    return (ti, mag(a[r - 1, c]) if r else 9)


report("per tensor + |value| above (same column, previous row)", adaptive(col_above))


# per tensor + the column's running mean |value| over earlier rows (bucketed)
colsum = {}


def col_mean(ti, n, a, r, c, st):
    key = (ti, c)
    s, k = colsum.get(key, (0.0, 0))
    if r == 0:
        bucket = 9
    else:
        m = s / max(k, 1)
        bucket = min(int(m * 2), 12)
    # update after use (the decoder knows rows < r)
    colsum[key] = (s + abs(int(a[r, c])), k + 1)
    return (ti, bucket)


report("per tensor + column's mean |value| in earlier rows", adaptive(col_mean))
colsum.clear()


# per tensor + column mean + previous value in the row
def col_mean_prev(ti, n, a, r, c, st):
    key = (ti, c)
    s, k = colsum.get(key, (0.0, 0))
    bucket = 9 if r == 0 else min(int(s / max(k, 1) * 2), 12)
    colsum[key] = (s + abs(int(a[r, c])), k + 1)
    return (ti, bucket, mag(a[r, c - 1]) if c else 9)


report("per tensor + column mean + |previous in row|", adaptive(col_mean_prev))
colsum.clear()


# per tensor + row's running mean |value| so far + column mean
def row_col(ti, n, a, r, c, st):
    key = (ti, c)
    s, k = colsum.get(key, (0.0, 0))
    cb = 9 if r == 0 else min(int(s / max(k, 1) * 2), 12)
    colsum[key] = (s + abs(int(a[r, c])), k + 1)
    rb = 9 if c < 8 else min(int(np.abs(a[r, :c]).mean() * 2), 12)
    return (ti, cb, rb)


report("per tensor + column mean + row's mean |value| so far", adaptive(row_col))
colsum.clear()

# 3. generic compressors on one byte per value
raw = b"".join(((a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)) + 7).astype(np.uint8).tobytes()
               for _, a in int4)
xz = len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
try:
    zs = len(subprocess.run(["zstd", "-19", "--ultra", "-22", "-c"], input=raw, capture_output=True).stdout)
except FileNotFoundError:
    zs = -1
print(f"generic compressors on {len(raw):,} raw values: xz -9e {xz:,} B, zstd -22 {zs:,} B")
print(f"all-zero int4 rows: {zero_rows}")
best = min(results[1:], key=lambda x: x[1])
print(f"best context model here: {best[0]}: {best[1]:,.0f} B, {best[1] - v4_int4:+,.0f} B against v4; "
      f"x2 in S: {2 * (best[1] - v4_int4):+,.0f} B")
