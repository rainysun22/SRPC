#!/usr/bin/env python3
"""收敛推断 + 参数调优：free/compare × eta_w × train 步数。"""
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


base = CreditConfig(alpha=1.0, beta=1.0)
print(f"{'cfg':34s} {'Δ':>2s} {'train':>6s} {'acc':>6s} {'a0':>6s} {'a1':>6s}")
for mode, e_mode in [("compare", "full"), ("free", "full")]:
    for eta_w in (0.02, 0.05):
        for delay, train in ((1, 10000), (4, 10000)):
            cfg = replace(base, predict_mode=mode, energy_mode=e_mode, eta_w=eta_w)
            acc, a0, a1 = ev(cfg, delay, train)
            tag = f"{mode}/e{eta_w}"
            print(f"{tag:34s} {delay:>2d} {train:>6d} {acc:6.3f} {a0:6.3f} {a1:6.3f}")
