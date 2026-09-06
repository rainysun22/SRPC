#!/usr/bin/env python3
"""Δ=1 修复配置矩阵：测哪些因素能阻止 W3 坍缩（1 seed，6000 步）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev(cfg, delay, train=6000, eval_n=300):
    seed = 0
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr, Xev, yev = X[:train], y[:train], X[train:], y[train:]
    m = CreditPCN(cfg, np.random.default_rng(seed * 3000 + 11), "error")
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    w0, w1 = m.W3[:, 0], m.W3[:, 1]
    wcos = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1) + 1e-9))
    return acc, wcos


base = CreditConfig(alpha=0.65)
variants = [
    ("base(α.65)", {}),
    ("kwta_off", dict(kwta_on=False)),
    ("kwta_0.8", dict(kwta_frac=0.8)),
    ("beta_0.25", dict(beta=0.25)),
    ("h1_256", dict(h1=256)),
    ("sit_30", dict(settle_iters=30)),
    ("eta_0.02", dict(eta_w=0.02)),
    ("kwta.8+beta.25", dict(kwta_frac=0.8, beta=0.25)),
    ("full_energy", dict(energy_mode="full")),
]
for name, kw in variants:
    cfg = replace(base, delay=1, **kw)
    acc, wcos = ev(cfg, 1)
    print(f"{name:16s} d1: acc={acc:.3f} W3cos={wcos:.3f}")
