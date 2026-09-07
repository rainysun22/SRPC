#!/usr/bin/env python3
"""seed1 gap 修复：iters=32 下 eta × alpha 微调（目标 seed1 err>=0.85）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN


def make_seq(cfg, rng, n, delay, lo, hi):
    d = cfg.d_feat
    steps = n + delay
    feat = rng.uniform(0.0, 1.0, (steps, d))
    b0 = rng.random(steps) < 0.5
    b1 = rng.random(steps) < 0.5
    feat[:, 0] = np.where(b0, hi, lo)
    feat[:, 1] = np.where(b1, hi, lo)
    y = (b0[:-delay] != b1[:-delay]) if delay > 0 else (b0 != b1)
    X = np.empty((n, (delay + 1) * d))
    for t in range(n):
        X[t] = feat[t:t + delay + 1].ravel()
    return X, y.astype(float)


def run(cfg, seed, iters, eta):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = make_seq(cfg, rng, 11500, 4, 0.0, 1.0)
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


base = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
               h1=32, h2=16)
print("iters=32: eta × alpha（err 臂，eval=1500）")
for eta in (0.08, 0.09, 0.10, 0.11):
    for alpha in (1.5, 1.6, 1.7):
        cfg = replace(base, alpha=alpha)
        accs = [run(cfg, s, 32, eta) for s in range(3)]
        flag = " <== ALL>=0.85" if min(accs) >= 0.85 else ""
        print(f"eta={eta:.2f} a={alpha:.1f} | " + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}{flag}")
