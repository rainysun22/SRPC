#!/usr/bin/env python3
"""hebb 臂浅迭代：hebb_free 下 iters ∈ {8,16,24,32}，err 臂固定 32。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN


def make_seq(cfg, rng, n, delay, lo, hi):
    d = cfg.d_feat
    steps = n + delay
    feat = rng.uniform(0.0, 1.0, (steps, d))
    b0 = rng.random(steps) < 0.5
    b1 = rng.random(steps) < 0.5
    feat[:, 0] = np.where(b0, hi, lo)
    feat[:, 1] = np.where(b1, hi, lo)
    y = (b0[:-delay] != b1[:-delay]) if delay > 0 else (b0 != b1)
    X = np.empty((n, (delay + 1) * d))
    for t in range(n):
        X[t] = feat[t:t + delay + 1].ravel()
    return X, y.astype(float)


def run(cfg, seed, mode, iters, eta, hebb_free):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = make_seq(cfg, rng, 11500, 4, 0.0, 1.0)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), mode)
    m.orth = False
    m.eta_inf = eta
    m.iters = iters
    m.hebb_free = hebb_free
    for t in range(10000):
        yoh = np.array([1.0, 0.0]) if ytr[t] == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=hebb_free)
        m._learn(Xtr[t], yoh)
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
               h1=32, h2=16)
eta = 0.09
errs = [run(base, s, "error", 32, eta, False) for s in range(3)]
print(f"err(iters=32): " + " ".join(f"{a:.3f}" for a in errs))
for ih in (8, 16, 24, 32):
    he = [run(base, s, "hebb", ih, eta, True) for s in range(3)]
    gaps = [e - h for e, h in zip(errs, he)]
    flag = " <== ALLgap>=0.30" if min(gaps) >= 0.30 else ""
    print(f"hebb(iters={ih:2d}): " + " ".join(f"{a:.3f}" for a in he)
          + f" | gap=" + " ".join(f"{g:.3f}" for g in gaps)
          + f" | mean_gap={np.mean(gaps):.3f}{flag}")
