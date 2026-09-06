#!/usr/bin/env python3
"""训练后 x1/x2 是否类可分（free 推断）：质心、线性探针（仅诊断）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def train_eval(cfg, delay, train, seed=0):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + 300, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    m.orth = False
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    # 收集 free 推断的 x1/x2
    x1s, x2s = [], []
    for x in Xev:
        m.x1[:] = 0.0
        m.x2[:] = 0.0
        m._infer(x, np.zeros(2), free_out=True)
        x1s.append(m.x1.copy())
        x2s.append(m.x2.copy())
    X1, X2 = np.array(x1s), np.array(x2s)
    for tag, Z in (("x1", X1), ("x2", X2)):
        c0 = Z[yev == 0].mean(0)
        c1 = Z[yev == 1].mean(0)
        cos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
        print(f"{tag}: centroid cos={cos:.4f} |c0|={np.linalg.norm(c0):.3f} |c1|={np.linalg.norm(c1):.3f}")
    # 线性探针（诊断用，非学习）
    A = np.hstack([X2, np.ones((len(X2), 1))])
    w, *_ = np.linalg.lstsq(A, yev, rcond=None)
    probe = float(np.mean((A @ w > 0.5) == yev))
    print(f"x2 linear probe acc={probe:.3f}  (W3 readout 2-col 是 x2 上的 2 维线性投影)")
    return probe


for name, kw in [("ab_1.0_1.0", dict(alpha=1.0, beta=1.0)),
                 ("ab_0.5_0.5", dict(alpha=0.5, beta=0.5)),
                 ("ab_0.65_0.10", dict(alpha=0.65, beta=0.10))]:
    cfg = replace(CreditConfig(), **kw)
    print(f"--- {name} Δ=1 ---")
    train_eval(cfg, 1, 5000)
