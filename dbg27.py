#!/usr/bin/env python3
"""seed0 vs seed1 深度诊断（a=1.5 配置）：类级/位型级准确率、x2 几何、能量裕度。"""
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


def diag(cfg, seed, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 11500, 4)
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
    # 位型/类级准确率 + 裕度
    pat = {(0.2, 0.2): [0, 0], (0.2, 0.8): [0, 0],
           (0.8, 0.2): [0, 0], (0.8, 0.8): [0, 0]}
    cl = {0: [0, 0], 1: [0, 0]}
    margins = []
    for t in range(len(Xev)):
        x0, yt = Xev[t], float(yev[t])
        key = (round(float(x0[0]), 1), round(float(x0[1]), 1))
        es = {}
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            m.x1[:] = 0.0; m.x2[:] = 0.0
            m._infer(x0, yoh, free_out=False)
            es[c] = m._energy
        pred = 0 if es[0] < es[1] else 1
        if key in pat:
            pat[key][0] += int(pred == yt)
            pat[key][1] += 1
        cl[int(yt)][0] += int(pred == yt)
        cl[int(yt)][1] += 1
        margins.append(abs(es[0] - es[1]))
    # x2 类质心
    m.x1[:] = 0.0; m.x2[:] = 0.0
    x2c = {0: [], 1: []}
    for t in range(0, 2000, 2):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        x2c[c].append(m.x2.copy())
    c0, c1 = np.mean(x2c[0], axis=0), np.mean(x2c[1], axis=0)
    cos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    w0, w1 = m.W3[:, 0], m.W3[:, 1]
    wcos = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1) + 1e-9))
    return pat, cl, cos, wcos, float(np.mean(margins)), float(np.std(margins))


cfg = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
for s in (0, 1, 2):
    pat, cl, cos, wcos, mm, ms = diag(cfg, s, 24)
    tot = sum(v[1] for v in pat.values())
    pacc = {k: v[0] / v[1] for k, v in pat.items() if v[1]}
    print(f"seed{s}: 位型={pacc} | 类={ {k: v[0]/v[1] for k, v in cl.items()} }")
    print(f"        x2cos={cos:.3f} W3cos={wcos:.3f} margin={mm:.4f}±{ms:.4f}")
