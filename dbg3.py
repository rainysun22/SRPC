#!/usr/bin/env python3
"""谱半径 / eta_inf 波动诊断：自适应 vs 固定步长对种子方差的影响。"""
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


def run(cfg, seed, iters, eta_fixed=None):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    eta = eta_fixed if eta_fixed else min(0.3, 0.6 * 1.2 / lam)
    m.orth = False
    m.eta_inf = eta
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    return lam, eta, acc


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
print("seed |  lam  | eta_adapt | acc@adaptive | acc@eta=0.08 | acc@eta=0.05")
for s in range(3):
    la, ea, aa = run(cfg, s, 24)
    _, _, a8 = run(cfg, s, 24, eta_fixed=0.08)
    _, _, a5 = run(cfg, s, 24, eta_fixed=0.05)
    print(f"  {s}  | {la:5.2f} |   {ea:.3f}   |    {aa:.3f}     |    {a8:.3f}     |    {a5:.3f}")
