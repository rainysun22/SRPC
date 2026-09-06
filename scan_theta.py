#!/usr/bin/env python3
"""theta_event × iters × eta_w 对 3 seeds Δ=4 的分布（随机块分配，η=0.095）。"""
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
    return float(np.mean(preds == yev))


cfg = replace(CreditConfig(alpha=1.0, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
eta_inf = 0.095
print(f"{'theta':>7s} {'iters':>6s} {'eta_w':>6s} | " + " ".join(f"seed{s}" for s in range(3)) + " | mean")
for theta in (0.01, 0.001):
    for iters in (8, 12):
        for eta_w in (0.05, 0.03):
            accs = [run(replace(cfg, theta_event=theta, eta_w=eta_w), 4, 10000, 300,
                        s, "error", eta_inf, iters) for s in range(3)]
            print(f"{theta:7.3f} {iters:6d} {eta_w:6.2f} | " +
                  " ".join(f"{a:.3f}" for a in accs) + f" | {np.mean(accs):.3f}")
