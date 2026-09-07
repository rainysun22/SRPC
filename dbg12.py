#!/usr/bin/env python3
"""energy 口径(class/full) × delay(4/6) 全判据（iters=24）。"""
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


def run(cfg, seed, delay, mode, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10000 + 500, delay)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


base = replace(CreditConfig(), h1=32, h2=16, alpha=1.0, beta=1.0)
for delay in (4, 6):
    for em in ("full", "class"):
        cfg = replace(base, delay=delay, energy_mode=em)
        rows = []
        for s in range(3):
            ae, de = run(cfg, s, delay, "error", 24)
            ah, _ = run(cfg, s, delay, "hebb", 24)
            rows.append((ae, ah, ae - ah, de))
        m = np.mean(np.array(rows), axis=0)
        print(f"Δ={delay} {em:<5s} | "
              + " ".join(f"{r[2]:+.3f}" for r in rows)
              + f" | err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:+.3f} distal={m[3]:.3f}")
