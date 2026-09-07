#!/usr/bin/env python3
"""seed1 失败模式：类级 acc / 掩码远端单元数 / kwta_frac 扫描。"""
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


def run(cfg, seed, iters=24):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    cl = [float(np.mean(preds[yev == c] == c)) for c in (0, 1)]
    # 掩码远端单元数（mask1 前 d_feat 行有连接的单元）
    distal_units = int(np.sum(m.mask1[:cfg.d_feat].any(axis=0)))
    # x2 类质心余弦（训练窗口采样）
    x2c = {0: [], 1: []}
    for t in range(0, 1200, 3):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        x2c[c].append(m.x2.copy())
    c0, c1 = np.mean(x2c[0], axis=0), np.mean(x2c[1], axis=0)
    cos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    return acc, cl, distal_units, cos


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
print("=== 基础诊断（kwta=0.5）===")
for s in range(3):
    acc, cl, du, cos = run(cfg, s)
    print(f"seed{s}: acc={acc:.3f} c0={cl[0]:.3f} c1={cl[1]:.3f} "
          f"distal_units={du} x2cos={cos:.3f}")

print("=== kwta_frac 扫描 ===")
for kf in (0.35, 0.5, 0.75):
    accs = [run(replace(cfg, kwta_frac=kf), s)[0] for s in range(3)]
    print(f"kwta={kf:.2f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
