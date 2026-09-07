#!/usr/bin/env python3
"""出生极性初始化：每块单元轮转正方向检测器 (1,0)/(0,1)/(1,1)/√2，
消除远端块 bit 极性的种子运气（整流模型下负权单元无效，正方向保证可学习）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _colnorm, _make_sequence

POLAR = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) / np.array(
    [[1.0], [1.0], [np.sqrt(2.0)]])


def apply_polar(m):
    cfg = m.cfg
    d, n_blk = cfg.d_feat, cfg.delay + 1
    W1 = np.zeros_like(m.W1)
    for j in range(m.W1.shape[1]):
        b = j % n_blk
        W1[b * d:(b + 1) * d, j] = POLAR[j % len(POLAR)]
    m.W1 = _colnorm(W1 * m.mask1)
    m.W1_init = m.W1.copy()


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, seed, delay, train_n, eval_n, mode, iters, polar):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_n + eval_n, delay)
    Xtr, ytr = X[:train_n], y[:train_n]
    Xev, yev = X[train_n:], y[train_n:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    if polar:
        apply_polar(m)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(train_n):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== polar 极性初始化 @ iters=16/24 ===")
for iters in (16, 24, 32):
    for polar in (False, True):
        accs = [run(base, s, 4, 10000, 500, "error", iters, polar)[0]
                for s in range(3)]
        print(f"iters={iters:2d} polar={str(polar):>5s} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}")
