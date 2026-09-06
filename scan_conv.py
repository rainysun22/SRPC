#!/usr/bin/env python3
"""收敛推断（小步长+多迭代）下 XOR 是否可学：α/β × energy_mode 扫描。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev(cfg, delay, train, eval_n=300, seed=0, eta_inf=0.04, iters=40):
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


base = CreditConfig()
print(f"{'αβ/energy':24s} {'Δ':>2s} {'acc':>6s} {'a0':>6s} {'a1':>6s}")
for ab, e_mode in [((1.0, 1.0), "full"), ((1.0, 1.0), "class"),
                   ((0.2, 1.0), "full"), ((0.2, 1.0), "class"),
                   ((0.1, 0.6), "full"), ((0.1, 0.6), "class"),
                   ((1.0, 0.2), "full"), ((1.0, 0.2), "class"),
                   ((0.5, 0.5), "class")]:
    cfg = replace(base, alpha=ab[0], beta=ab[1], energy_mode=e_mode)
    tag = f"α{ab[0]}/β{ab[1]}/{e_mode}"
    for delay, train in ((1, 5000), (4, 10000)):
        acc, a0, a1 = ev(cfg, delay, train)
        print(f"{tag:24s} {delay:>2d} {acc:6.3f} {a0:6.3f} {a1:6.3f}")
