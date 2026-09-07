#!/usr/bin/env python3
"""seed1 根因定位：类级准确率 / W3-x2 几何 / 网络-序列交叉测试。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(cfg, seed, delay=4):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return float(np.linalg.eigvalsh(H).max())


def make(cfg, seed):
    return CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")


def train(m, cfg, Xtr, ytr, iters, lam):
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(len(Xtr)):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)


def diag(cfg, net_seed, seq_seed, train_n, eval_n, iters):
    rng = np.random.default_rng(seq_seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_n + eval_n, 4)
    Xtr, ytr = X[:train_n], y[:train_n]
    Xev, yev = X[train_n:], y[train_n:]
    m = make(cfg, net_seed)
    lam = spectral_of(cfg, net_seed)
    train(m, cfg, Xtr, ytr, iters, lam)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    cl = [float(np.mean(preds[yev == c] == c)) for c in (0, 1)]
    # 几何：x2 类质心 vs W3 列
    m.x1[:] = 0.0; m.x2[:] = 0.0
    x2c = {0: [], 1: []}
    for t in range(0, 1200, 3):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        x2c[c].append(m.x2.copy())
    c0, c1 = np.mean(x2c[0], axis=0), np.mean(x2c[1], axis=0)
    cos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    w0, w1 = m.W3[:, 0], m.W3[:, 1]
    wcos = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1) + 1e-9))
    return acc, cl, cos, wcos


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
train_n, eval_n, iters = 10000, 500, 16
print("net_seed/seq_seed | acc | class0 class1 | x2cos | W3cos")
for ns in (0, 1, 2):
    acc, cl, cos, wcos = diag(cfg, ns, ns, train_n, eval_n, iters)
    print(f"  net{ns}/seq{ns}  | {acc:.3f} | {cl[0]:.3f}  {cl[1]:.3f}  | {cos:.3f} | {wcos:.3f}")
print("--- 交叉：seed1 网络 vs seed0 序列；seed0 网络 vs seed1 序列 ---")
for ns, ss in ((1, 0), (0, 1), (1, 2)):
    acc, cl, cos, wcos = diag(cfg, ns, ss, train_n, eval_n, iters)
    print(f"  net{ns}/seq{ss}  | {acc:.3f} | {cl[0]:.3f}  {cl[1]:.3f}  | {cos:.3f} | {wcos:.3f}")
