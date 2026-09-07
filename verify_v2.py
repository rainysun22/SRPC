#!/usr/bin/env python3
"""验证 alpha=1.5 收敛配置的全部判据（error vs hebb，Δ=1 与 Δ=4，eval=1000）。"""
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


def run(cfg, delay, train, eval_n, seed, mode, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


cfg = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
iters = 24
print("per-seed（Δ=4: train 10000 eval 1000；Δ=1: train 5000 eval 1000）")
rows = []
for s in range(3):
    a4e, d4e = run(cfg, 4, 10000, 1000, s, "error", iters)
    a4h, _ = run(cfg, 4, 10000, 1000, s, "hebb", iters)
    a1e, d1e = run(cfg, 1, 5000, 1000, s, "error", iters)
    a1h, _ = run(cfg, 1, 5000, 1000, s, "hebb", iters)
    rows.append((a4e, a4h, a4e - a4h, d4e, a1e, a1h, d1e))
    print(f"seed{s}: Δ4 err={a4e:.3f} hebb={a4h:.3f} gap={a4e-a4h:.3f} "
          f"distal={d4e:.3f} | Δ1 err={a1e:.3f} hebb={a1h:.3f} distal={d1e:.3f}")
m = np.mean(np.array(rows), axis=0)
print(f"\nMEAN: Δ4 err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:.3f} distal={m[3]:.3f} "
      f"| Δ1 err={m[4]:.3f} hebb={m[5]:.3f} distal={m[6]:.3f}")
print(f"PASS(mean): err>=0.80: {m[0]>=0.80}  hebb<=0.68: {m[1]<=0.68}  "
      f"gap>=0.30: {m[2]>=0.30}  distal>=0.25: {m[3]>=0.25}  Δ1 err>=0.75: {m[4]>=0.75}")
print(f"PASS(per-seed err>=0.80): {all(r[0]>=0.80 for r in rows)}")
