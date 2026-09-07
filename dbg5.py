#!/usr/bin/env python3
"""侧抑制(orth) × 学习率 × iters 对种子方差的组合（none 变体，h2=16）。"""
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


def run(cfg, seed, iters, orth, eta_w=None):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = orth
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    if eta_w:
        m.cfg = replace(m.cfg, eta_w=eta_w)
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== orth × iters（h1=32 h2=16, eta_w=0.05）===")
for iters in (16, 24, 32):
    for orth in (False, True):
        accs = [run(base, s, iters, orth) for s in range(3)]
        print(f"iters={iters:2d} orth={str(orth):>5s} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}")
print("=== orth=True × eta_w（iters=24）===")
for ew in (0.02, 0.05, 0.10, 0.20):
    accs = [run(base, s, 24, True, eta_w=ew) for s in range(3)]
    print(f"eta_w={ew:.2f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
