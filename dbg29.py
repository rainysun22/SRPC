#!/usr/bin/env python3
"""0/1 编码 + a=1.5 下：hebb 臂 hebb_free 效果（目标 hebb<=0.551 -> gap>=0.30）。"""
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


def run(cfg, seed, mode, iters, hebb_free):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = make_seq(cfg, rng, 11500, 4, 0.0, 1.0)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    m.hebb_free = hebb_free
    for t in range(10000):
        yoh = np.array([1.0, 0.0]) if ytr[t] == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=hebb_free)
        m._learn(Xtr[t], yoh)
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
               h1=32, h2=16)
print("0/1 编码：hebb_free × mode（iters=24, eval=1500）")
for hf in (False, True):
    accs = [run(base, s, "hebb", 24, hf) for s in range(3)]
    print(f"hebb_free={str(hf):>5s} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
ae = [run(base, s, "error", 24, False) for s in range(3)]
print(f"error(钳制)      | " + " ".join(f"{a:.3f}" for a in ae)
      + f" | mean={np.mean(ae):.3f}")
