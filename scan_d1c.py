#!/usr/bin/env python3
"""Δ=1 训练过程 acc 监控：5 配置 × 每 1000 步冻结评估（1 seed, 8000 步）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence

SEED = 0


def ev_curve(cfg, train=8000, eval_n=300, lr_decay=None):
    seed = SEED
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + 300, 1)
    Xtr, ytr = X[:train], y[:train]
    ri = np.random.default_rng(123).choice(train, size=eval_n, replace=False)
    Xm, ym = Xtr[ri], ytr[ri]
    m = CreditPCN(cfg, np.random.default_rng(seed * 3000 + 11), "error")
    curve = []
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
        if lr_decay and (t + 1) % lr_decay[0] == 0:
            m.cfg = replace(m.cfg, eta_w=m.cfg.eta_w * lr_decay[1])
        if (t + 1) % 1000 == 0:
            m.set_learning(False)
            preds = np.array([m.predict(x) for x in Xm])
            m.set_learning(True)
            curve.append(float(np.mean(preds == ym)))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in X[train:]])
    final = float(np.mean(preds == y[train:]))
    return curve, final


base = CreditConfig(alpha=0.65)
configs = [
    ("base", replace(base), None),
    ("lr_x0.6/2000", replace(base), (2000, 0.6)),
    ("sit20_eta.03", replace(base, settle_iters=20, eta_w=0.03), None),
    ("beta.25", replace(base, beta=0.25), None),
    ("sit25_eta.03", replace(base, settle_iters=25, eta_w=0.03), None),
]
for name, cfg, dec in configs:
    cfg = replace(cfg, delay=1)
    curve, final = ev_curve(cfg, lr_decay=dec)
    print(f"{name:16s} final={final:.3f} curve=" + " ".join(f"{v:.2f}" for v in curve))
