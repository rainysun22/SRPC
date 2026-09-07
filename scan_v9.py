#!/usr/bin/env python3
"""Δ4：orth(侧抑制) × 深度 × h2 宽度组合 + hebb 基线测量。"""
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


def run(cfg, delay, train, eval_n, seed, mode, iters, orth):
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
train, eval_n = 10000, 500

print("=== orth on/off @ iters=16/24 ===")
for iters in (16, 24):
    for orth in (False, True):
        accs = []
        for s in range(3):
            a, _ = run(base, 4, train, eval_n, s, "error", iters, orth)
            accs.append(a)
        print(f"iters={iters:2d} orth={str(orth):>5s} | "
              + " ".join(f"{a:.3f}" for a in accs) + f" | mean={np.mean(accs):.3f}")

print("=== h2=24 @ iters=16/24, orth=False ===")
b24 = replace(base, h2=24)
for iters in (16, 24):
    accs = []
    for s in range(3):
        a, _ = run(b24, 4, train, eval_n, s, "error", iters, False)
        accs.append(a)
    print(f"iters={iters:2d} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")

print("=== hebb 基线 @ iters=16 (需 ≤0.68) ===")
accs = []
for s in range(3):
    a, _ = run(base, 4, train, eval_n, s, "hebb", 16, False)
    accs.append(a)
print("hebb  | " + " ".join(f"{a:.3f}" for a in accs) + f" | mean={np.mean(accs):.3f}")
