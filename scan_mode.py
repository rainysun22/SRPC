#!/usr/bin/env python3
"""Δ=1/Δ=4 全协议扫参：compare/free × αβ × orth on/off × train 步数。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def ev(cfg, delay, train, seed=0, orth=True):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + 300, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    m.orth = orth
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev)), \
        float(np.mean(preds[yev == 0] == 0)), float(np.mean(preds[yev == 1] == 1))


base = CreditConfig(alpha=1.0, beta=1.0)
print(f"{'name':30s} {'Δ':>2s} {'train':>6s} {'acc':>6s} {'a0':>6s} {'a1':>6s}")
for mode in ("free", "compare"):
    for name, kw in [("ab_1.0_1.0", dict(alpha=1.0, beta=1.0)),
                     ("ab_0.5_0.5", dict(alpha=0.5, beta=0.5))]:
        for orth in (True, False):
            for delay, train in ((1, 5000), (4, 10000)):
                cfg = replace(base, predict_mode=mode, **kw)
                try:
                    acc, a0, a1 = ev(cfg, delay, train, orth=orth)
                except Exception as e:
                    print(f"{mode}/{name}/o{orth}/Δ{delay}: ERR {e}")
                    continue
                tag = f"{mode}/{name}/orth={int(orth)}"
                print(f"{tag:30s} {delay:>2d} {train:>6d} {acc:6.3f} {a0:6.3f} {a1:6.3f}")
