#!/usr/bin/env python3
"""α>>β（文档规则1 设计意图：顶层类拉动远大于底层重建）对全判据的影响。"""
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


def run(cfg, seed, mode, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full")
for a, b in ((1.0, 1.0), (1.5, 0.5), (2.0, 1.0), (2.0, 0.5), (3.0, 1.0)):
    cfg = replace(base, alpha=a, beta=b)
    for iters in (16, 24):
        accs, gaps = [], []
        for s in range(3):
            ae = run(cfg, s, "error", iters)
            ah = run(cfg, s, "hebb", iters)
            accs.append(ae); gaps.append(ae - ah)
        print(f"a={a:.1f} b={b:.1f} iters={iters:2d} | "
              + " ".join(f"{g:.3f}" for g in gaps)
              + f" | err_mean={np.mean(accs):.3f} gap_mean={np.mean(gaps):.3f}")
