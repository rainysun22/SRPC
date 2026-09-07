#!/usr/bin/env python3
"""稳定性审计：eval 分段×500 测真实水平；energy_mode full vs class。"""
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


def run(cfg, seed, iters, em):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 11500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4, energy_mode=em),
                  np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    seg = [float(np.mean(preds[i:i + 500] == yev[i:i + 500])) for i in (0, 500, 1000)]
    return seg, float(np.mean(preds == yev))


cfg = replace(CreditConfig(alpha=1.5, beta=1.0), h1=32, h2=16)
print("a=1.5 b=1.0 iters=24：3 段×500 评估")
for em in ("full", "class"):
    for s in range(3):
        seg, tot = run(cfg, s, 24, em)
        print(f"  {em:5s} seed{s}: seg={seg} total={tot:.3f}")
