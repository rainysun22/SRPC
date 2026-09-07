#!/usr/bin/env python3
"""最优配置全判据矩阵：per-seed error/hebb/gap/distal @ iters=16/24。"""
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


def run(cfg, seed, delay, train_n, eval_n, mode, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_n + eval_n, delay)
    Xtr, ytr = X[:train_n], y[:train_n]
    Xev, yev = X[train_n:], y[train_n:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(train_n):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
for iters in (16, 24):
    print(f"=== iters={iters}：Δ4 per-seed 全判据（train=10000, eval=500）===")
    rows = []
    for s in range(3):
        ae, de = run(base, s, 4, 10000, 500, "error", iters)
        ah, _ = run(base, s, 4, 10000, 500, "hebb", iters)
        gap = ae - ah
        rows.append((ae, ah, gap, de))
        print(f"seed{s}: err={ae:.3f} hebb={ah:.3f} gap={gap:+.3f} distal={de:.3f}"
              + ("  <==" if not (ae >= 0.80 and ah <= 0.68 and gap >= 0.30 and de >= 0.25) else "  PASS"))
    m = np.mean(np.array(rows), axis=0)
    print(f"MEAN: err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:+.3f} distal={m[3]:.3f}")
