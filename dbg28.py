#!/usr/bin/env python3
"""bit 编码变体 × alpha：低输入位型感知修复（eval 分段审计稳定性）。"""
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


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, seed, iters, lo, hi):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = make_seq(cfg, rng, 11500, 4, lo, hi)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full")
print("bit 编码 × alpha（iters=24, eval=1500）")
for lo, hi, tag in ((0.2, 0.8, "0.2/0.8"), (0.1, 0.9, "0.1/0.9"), (0.0, 1.0, "0/1")):
    for alpha in (1.0, 1.5):
        cfg = replace(base, alpha=alpha, beta=1.0)
        accs = [run(cfg, s, 24, lo, hi) for s in range(3)]
        flag = " <== ALL>=0.80" if min(accs) >= 0.80 else ""
        print(f"{tag} a={alpha:.1f} | " + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}{flag}")
