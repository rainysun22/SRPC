#!/usr/bin/env python3
"""Δ=1 修复粗筛（1 seed, 6000 步）：推断收敛/容量/系数方向。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev1(cfg, train=6000, eval_n=300):
    seed = 0
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, 1)
    Xtr, ytr, Xev, yev = X[:train], y[:train], X[train:], y[train:]
    m = CreditPCN(cfg, np.random.default_rng(seed * 3000 + 11), "error")
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    x2s = {0: [], 1: []}
    for t in range(0, train, 6):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        x2s[c].append(m.x2.copy())
    c0 = np.mean(x2s[0], axis=0); c1 = np.mean(x2s[1], axis=0)
    ccos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    return acc, ccos


base = CreditConfig(alpha=0.65)
variants = [
    ("base", {}),
    ("sit_30", dict(settle_iters=30)),
    ("h2_128", dict(h2=128)),
    ("h1_256h2_128", dict(h1=256, h2=128)),
    ("alpha_0.80", dict(alpha=0.80)),
    ("alpha.5beta.25", dict(alpha=0.50, beta=0.25)),
    ("kwta_0.7", dict(kwta_frac=0.7)),
    ("sit_30+h2_128", dict(settle_iters=30, h2=128)),
]
for name, kw in variants:
    cfg = replace(base, delay=1, **kw)
    acc, ccos = ev1(cfg)
    print(f"{name:16s} d1: acc={acc:.3f} x2cos={ccos:.3f}")
