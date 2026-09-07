#!/usr/bin/env python3
"""energy=class × eta_w × iters 全判据（压低 hebb + 推高 err）。"""
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


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="class",
               alpha=1.0, beta=1.0)
for iters in (24, 32):
    for ew in (0.03, 0.05):
        cfg = replace(base, eta_w=ew)
        rows = []
        for s in range(3):
            ae = run(cfg, s, "error", iters)
            ah = run(cfg, s, "hebb", iters)
            rows.append((ae, ah, ae - ah))
        m = np.mean(np.array(rows), axis=0)
        print(f"iters={iters:2d} eta_w={ew:.2f} | "
              + " ".join(f"{r[2]:+.3f}" for r in rows)
              + f" | err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:+.3f}")
