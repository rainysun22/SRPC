#!/usr/bin/env python3
"""3 seeds 验证 h1=32/h2=16 收敛配置的全部判据（error vs hebb，Δ=1 与 Δ=4）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def run(cfg, delay, train, eval_n, seed, mode, eta_inf, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    m.orth = False
    m.eta_inf = eta_inf
    m.iters = iters
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]))
    return acc, distal / total


cfg = replace(CreditConfig(alpha=1.0, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
eta_inf, iters = 0.095, 8
print("per-seed (Δ=4: train 10000; Δ=1: train 5000)")
rows = []
for s in range(3):
    a4e, d4e = run(cfg, 4, 10000, 300, s, "error", eta_inf, iters)
    a4h, _ = run(cfg, 4, 10000, 300, s, "hebb", eta_inf, iters)
    a1e, d1e = run(cfg, 1, 5000, 300, s, "error", eta_inf, iters)
    a1h, _ = run(cfg, 1, 5000, 300, s, "hebb", eta_inf, iters)
    rows.append((a4e, a4h, a4e - a4h, d4e, a1e, a1h, d1e))
    print(f"seed{s}: Δ4 err={a4e:.3f} hebb={a4h:.3f} gap={a4e-a4h:.3f} "
          f"distal={d4e:.3f} | Δ1 err={a1e:.3f} hebb={a1h:.3f} distal={d1e:.3f}")
m = np.mean(np.array(rows), axis=0)
print(f"\nMEAN: Δ4 err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:.3f} distal={m[3]:.3f} "
      f"| Δ1 err={m[4]:.3f} hebb={m[5]:.3f} distal={m[6]:.3f}")
print(f"PASS: Δ4 err>=0.80: {m[0]>=0.80}  hebb<=0.68: {m[1]<=0.68}  "
      f"gap>=0.30: {m[2]>=0.30}  distal>=0.25: {m[3]>=0.25}  "
      f"Δ1 err>=0.75: {m[4]>=0.75}")
