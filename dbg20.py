#!/usr/bin/env python3
"""T≤5 约束下的规模搜索：网络越小谱半径越低 => 步长越大 => 迭代内收敛。"""
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


def run(cfg, seed, iters, energy_mode):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4, energy_mode=energy_mode),
                  np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev)), lam


base = replace(CreditConfig(alpha=1.0, beta=1.0),
               fan_in_frac=0.75, kwta_frac=0.5)
for h1, h2 in ((32, 16), (24, 12), (16, 8), (12, 6)):
    for iters in (3, 5):
        for em in ("class", "full"):
            accs, lams = [], []
            for s in range(3):
                a, lam = run(replace(base, h1=h1, h2=h2), s, iters, em)
                accs.append(a)
                lams.append(lam)
            print(f"h={h1}/{h2} iters={iters} {em:5s} | "
                  + " ".join(f"{a:.3f}" for a in accs)
                  + f" | mean={np.mean(accs):.3f} lam~{np.mean(lams):.1f}")
