#!/usr/bin/env python3
"""iters=24 下统一 eta_inf 的精细搜索：找全种子 >=0.80 的步长。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def run(cfg, seed, iters, eta):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    m.orth = False
    m.eta_inf = eta
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
print("iters=24: 统一 eta_inf 网格")
best = None
for eta in (0.045, 0.05, 0.055, 0.06, 0.065, 0.07):
    accs = [run(cfg, s, 24, eta) for s in range(3)]
    m = np.mean(accs)
    flag = "ALLPASS" if min(accs) >= 0.80 else ""
    print(f"eta={eta:.3f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={m:.3f} {flag}")
    if min(accs) >= 0.80 and (best is None or m > best[0]):
        best = (m, eta, accs)
if best:
    print(f"\nBEST: eta={best[1]:.3f} mean={best[0]:.3f} seeds={best[2]}")
