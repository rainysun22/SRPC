#!/usr/bin/env python3
"""Δ=1：扫 α/β（推断稳定性）—— 标准 PCN 单位增益 vs 顶层拉动偏置。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev(cfg, delay=1, train=5000, eval_n=300, seed=0):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    acc = float(np.mean([m.predict(x) == float(yev[t]) for t, x in enumerate(Xev)]))
    # 逐类
    a0 = float(np.mean([m.predict(Xev[t]) == 0 for t in np.where(yev == 0)[0]]))
    a1 = float(np.mean([m.predict(Xev[t]) == 1 for t in np.where(yev == 1)[0]]))
    return acc, a0, a1


base = CreditConfig()
print(f"{'name':22s} {'acc':>6s} {'a0':>6s} {'a1':>6s}")
for name, kw in [
    ("ab_1.0_1.0", dict(alpha=1.0, beta=1.0)),
    ("ab_0.5_0.5", dict(alpha=0.5, beta=0.5)),
    ("ab_0.65_0.10", dict(alpha=0.65, beta=0.10)),
    ("ab_1.0_0.5", dict(alpha=1.0, beta=0.5)),
    ("ab_0.6_0.4", dict(alpha=0.6, beta=0.4)),
    ("ab_0.7_0.3", dict(alpha=0.7, beta=0.3)),
]:
    cfg = replace(base, **kw)
    acc, a0, a1 = ev(cfg)
    print(f"{name:22s} {acc:6.3f} {a0:6.3f} {a1:6.3f}")
