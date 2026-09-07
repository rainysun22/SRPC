#!/usr/bin/env python3
"""修复：出生近正交列（代码注释声明的设计意图）。
- orth_w2:  W2 列出生正交化（隐表征多样性）
- spread_w1: W1 按时间块子空间均匀铺开方向（保证块内 2 维方向覆盖）
- both: 两者组合
"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence, _colnorm, _orth_cols


def spread_w1(m, cfg, rng):
    """层1按块均匀铺开方向：块内第 j 个单元角度 = 2π*(j/n_units + 均匀偏移)。"""
    d, n_blk = cfg.d_feat, cfg.delay + 1
    W1 = m.W1.copy()
    for b in range(n_blk):
        rows = slice(b * d, (b + 1) * d)
        cols = np.where(m.mask1[rows].any(axis=0))[0]
        n = len(cols)
        if n == 0:
            continue
        off = rng.uniform(0, 1)
        for k, j in enumerate(cols):
            th = 2 * np.pi * (k / n + off)
            W1[rows, j] = [np.cos(th), np.sin(th)] if d >= 2 else W1[rows, j]
    m.W1 = _colnorm(W1 * m.mask1)


def apply_variant(m, cfg, variant, rng):
    if variant in ("orth_w2", "both"):
        m.W2 = _orth_cols(m.W2 * m.mask2) * m.mask2
        m.W2 = _colnorm(m.W2)
    if variant in ("spread_w1", "both"):
        spread_w1(m, cfg, rng)
    m.W1_init = m.W1.copy()


def spectral_of(cfg, seed, delay=4):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, delay, train, eval_n, seed, mode, iters, variant, orth=False):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    apply_variant(m, cfg, variant, np.random.default_rng(seed * 3000 + 55))
    lam = spectral_of(cfg, seed, delay)
    m.orth = orth
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(train):
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
train, eval_n, iters = 10000, 500, 16
print(f"{'variant':>11s} | " + " ".join(f"seed{s}" for s in range(3)) + " | mean  distal")
for v in ("none", "orth_w2", "spread_w1", "both"):
    accs, dists = [], []
    for s in range(3):
        a, dd = run(base, 4, train, eval_n, s, "error", iters, v)
        accs.append(a); dists.append(dd)
    print(f"{v:>11s} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | {np.mean(accs):.3f}  {np.mean(dists):.3f}")
