#!/usr/bin/env python3
"""Δ4 方差修复扫描 v2：规模 × energy_mode × kwta_frac × theta_event（逐 seed）。"""
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
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


base = replace(CreditConfig(alpha=1.0, beta=1.0), h1=32, h2=16)
eta_inf, iters, eval_n = 0.095, 8, 200
grid = []
for h1, h2 in ((32, 16), (48, 24)):
    for em in ("full", "class"):
        for kw in (0.5, 0.7):
            for th in (0.01, 0.001):
                grid.append((h1, h2, em, kw, th))

print(f"{'h1/h2':>6s} {'em':>5s} {'kwta':>4s} {'theta':>5s} | "
      + " ".join(f"seed{s}" for s in range(3)) + " | mean")
for h1, h2, em, kw, th in grid:
    accs, dists = [], []
    for s in range(3):
        cfg = replace(base, h1=h1, h2=h2, energy_mode=em, kwta_frac=kw, theta_event=th)
        a, dd = run(cfg, 4, 10000, eval_n, s, "error", eta_inf, iters)
        accs.append(a); dists.append(dd)
    print(f"{h1:>2d}/{h2:<2d} {em:>5s} {kw:>4.1f} {th:>5.3f} | "
          + " ".join(f"{a:.3f}" for a in accs) + f" | {np.mean(accs):.3f} "
          + f"distal={np.mean(dists):.3f}")
