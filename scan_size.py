#!/usr/bin/env python3
"""网络规模 × 阻尼 × 迭代数：找满足 T≤5 且双延迟过 0.80 的最小配置。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev(cfg, delay, train, eval_n=300, seed=0, eta_inf=0.15, iters=8):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    m.orth = False
    m.eta_inf = eta_inf
    m.iters = iters
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev)), \
        float(np.mean(preds[yev == 0] == 0)), float(np.mean(preds[yev == 1] == 1))


def spectral_of(cfg, delay=1, seed=11):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return np.linalg.eigvalsh(H).max()


base = CreditConfig(alpha=1.0, beta=1.0, energy_mode="full")
for h1, h2 in ((16, 8), (24, 12), (32, 16), (48, 24)):
    cfg = replace(base, h1=h1, h2=h2)
    lm = spectral_of(cfg)
    eta = min(0.3, 1.2 / lm)
    print(f"\nh1={h1} h2={h2}: λmax(H)={lm:.1f} eta={eta:.3f}")
    for iters in (5, 8, 12):
        accs = []
        for delay, train in ((1, 10000), (4, 10000)):
            acc, a0, a1 = ev(cfg, delay, train, eta_inf=eta, iters=iters)
            accs.append(acc)
            print(f"  iters={iters:2d} Δ={delay}: acc={acc:.3f} a0={a0:.3f} a1={a1:.3f}")
        print(f"  -> both>=0.80: {all(a >= 0.80 for a in accs)}")
