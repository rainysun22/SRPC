#!/usr/bin/env python3
"""seed1 逐位型准确率 + 候选能量裕度（预测稳定性）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def diag(cfg, seed):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = 24
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)

    # 逐位型：远端块 (bit0,bit1) 4 种组合的准确率 + 能量裕度
    pat = {(0.2, 0.2): [0, 0], (0.2, 0.8): [0, 0],
           (0.8, 0.2): [0, 0], (0.8, 0.8): [0, 0]}
    margins = []
    for t in range(len(Xev)):
        x0, yt = Xev[t], float(yev[t])
        b0 = x0[0]; b1 = x0[1]
        key = (round(float(b0), 1), round(float(b1), 1))
        if key not in pat:
            continue
        # 手动 compare：记录两个候选能量
        es = {}
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            m.x1[:] = 0.0; m.x2[:] = 0.0
            m._infer(x0, yoh, free_out=False)
            es[c] = m._energy
        pred = 0 if es[0] < es[1] else 1
        pat[key][0] += int(pred == yt)
        pat[key][1] += 1
        margins.append(abs(es[0] - es[1]))
    acc = sum(v[0] for v in pat.values()) / max(1, sum(v[1] for v in pat.values()))
    print(f"seed{seed}: acc={acc:.3f} | "
          + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in pat.items())
          + f" | 裕度 mean={np.mean(margins):.4f} sd={np.std(margins):.4f}")
    return pat, margins


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
for s in range(3):
    diag(cfg, s)
