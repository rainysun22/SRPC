#!/usr/bin/env python3
"""诊断 Δ=1 分类偏置：W3 符号结构 + 推断轨迹 + 能量组成。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence

cfg = CreditConfig(alpha=0.65)


def _kwta(x, frac):
    k = max(1, int(round(frac * x.size)))
    if k >= x.size:
        return x
    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


seed = 0
rng = np.random.default_rng(seed * 3000 + 7)
X, y = _make_sequence(cfg, rng, 10000 + 500, 1)
Xtr, ytr = X[:10000], y[:10000]
Xev, yev = X[10000:], y[10000:]

m = CreditPCN(replace(cfg, delay=1), np.random.default_rng(seed * 3000 + 11), "error")
for t in range(10000):
    m.train_step(Xtr[t], float(ytr[t]))
m.set_learning(False)

w0, w1 = m.W3[:, 0], m.W3[:, 1]
print("W3[:,0] pos=", float(w0[w0 > 0].sum()), "neg=", float(w0[w0 < 0].sum()),
      "npos=", int((w0 > 0).sum()))
print("W3[:,1] pos=", float(w1[w1 > 0].sum()), "neg=", float(w1[w1 < 0].sum()),
      "npos=", int((w1 > 0).sum()))

# 推断轨迹：单个真实类 0 与类 1 样本，两个钳制下各误差项随 settle 变化
for xi, yi in ((Xev[0], 0), (Xev[200], 1)):
    print(f"\n--- sample true class {yi} ---")
    for c in (0, 1):
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0
        m.x2[:] = 0.0
        for it in (5, 10, 15):
            # 手动重跑 _infer 到 it 步比较麻烦；改为每步记录
            pass
        # 记录完整轨迹
        m.x1[:] = 0.0; m.x2[:] = 0.0
        a, b = cfg.alpha, cfg.beta
        x3 = yoh.copy()
        hist = []
        for _ in range(15):
            e0 = xi - m.W1 @ m.x1
            e1 = m.x1 - m.W2 @ m.x2
            e2 = m.x2 - m.W3 @ x3
            u1 = b * (m.W1.T @ e0) - a * e1
            u2 = b * (m.W2.T @ e1) - a * e2
            m.x1 = np.clip(m.x1 + u1 * (np.abs(u1) > cfg.theta_event), 0.0, cfg.x_max)
            m.x2 = np.clip(m.x2 + u2 * (np.abs(u2) > cfg.theta_event), 0.0, cfg.x_max)
            if cfg.kwta_on:
                m.x1 = _kwta(m.x1, cfg.kwta_frac)
                m.x2 = _kwta(m.x2, cfg.kwta_frac)
            hist.append((float(np.dot(e0, e0)), float(np.dot(e1, e1)),
                         float(np.dot(e2, e2))))
        print(f"clamp c={c}: e0^2 e1^2 e2^2 at it1={hist[0]} it8={hist[7]} it15={hist[14]}")
