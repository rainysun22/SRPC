#!/usr/bin/env python3
"""训练时长 × h1 宽度对 seed1 的影响（none 变体，h2=16，iters=24）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, seed, train_n, eval_n, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_n + eval_n, 4)
    Xtr, ytr = X[:train_n], y[:train_n]
    Xev, yev = X[train_n:], y[train_n:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(train_n):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== train_n × h1（eval_n=500, iters=24）===")
for h1 in (32, 48):
    for tn in (10000, 15000, 20000):
        accs = [run(replace(base, h1=h1), s, tn, 500, 24) for s in range(3)]
        print(f"h1={h1:2d} train={tn} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}")
