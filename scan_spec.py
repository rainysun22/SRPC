#!/usr/bin/env python3
"""谱半径分析 + 阻尼步长扫描：验证收敛推断下分类是否成立。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral(cfg, seed=11):
    m = CreditPCN(replace(cfg, delay=1), np.random.default_rng(seed), "error")
    W1, W2, W3 = m.W1, m.W2, m.W3
    # H = [[W1.TW1+I, -W2],[-W2.T, W2.TW2+I]]  (unit α/β)
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    lm = np.linalg.eigvalsh(H).max()
    lw2 = np.linalg.eigvalsh(W2.T @ W2).max()
    return lm, lw2


def ev_damped(cfg, delay, train, eval_n=300, seed=0, eta_inf=0.5, iters=None):
    iters = iters or cfg.settle_iters
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
    return float(np.mean(preds == yev))


for name, kw in [("base(α.6β.1)", dict(alpha=0.6, beta=0.1)),
                 ("unit(1,1)", dict(alpha=1.0, beta=1.0)),
                 ("half(0.5,0.5)", dict(alpha=0.5, beta=0.5))]:
    cfg = replace(CreditConfig(), **kw)
    lm, lw2 = spectral(cfg)
    print(f"{name}: λmax(H)={lm:.1f} λmax(W2.TW2)={lw2:.1f} -> 稳定需 η<=2/{lm:.0f}={2/lm:.3f}")
