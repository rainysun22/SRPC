#!/usr/bin/env python3
"""Δ4 方差修复扫描 v5：瞄准 seed1 —— α强度 × θ门限 × 训练时长（h1=32, iters=16）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(cfg, seed, delay=4):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, delay, train, eval_n, seed, mode, iters, orth=False):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(cfg, seed, delay)
    m.orth = orth
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
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


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
eval_n = 500

print("=== alpha sweep @ iters=16 ===")
for alpha in (1.0, 1.5, 2.0, 2.5):
    accs = []
    for s in range(3):
        a, _ = run(replace(base, alpha=alpha), 4, 10000, eval_n, s, "error", 16)
        accs.append(a)
    print(f"a={alpha:.1f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")

print("=== theta sweep @ a=1.0, iters=16 ===")
for th in (0.01, 0.005, 0.003, 0.001):
    accs = []
    for s in range(3):
        a, _ = run(replace(base, theta_event=th), 4, 10000, eval_n, s, "error", 16)
        accs.append(a)
    print(f"theta={th:.3f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")

print("=== train-length sweep @ a=1.0, iters=16 ===")
for tr in (10000, 15000, 20000):
    accs = []
    for s in range(3):
        a, _ = run(base, 4, tr, eval_n, s, "error", 16)
        accs.append(a)
    print(f"train={tr} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
