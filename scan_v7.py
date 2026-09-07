#!/usr/bin/env python3
"""Δ4 方差修复扫描 v7：iters=16 下 energy×eta_w×掩码组合。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence, _colnorm


def spectral_of(cfg, seed, delay=4):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, delay, train, eval_n, seed, mode, iters, m2_guar=False, orth=False):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    if m2_guar:
        d = cfg.d_feat
        vrng = np.random.default_rng(seed * 3000 + 99)
        distal_rows = np.where(m.mask1[:d].any(axis=0))[0]
        k2 = max(1, int(round(cfg.fan_in_frac * cfg.h1)))
        m2 = np.zeros((cfg.h1, cfg.h2), dtype=bool)
        for j in range(cfg.h2):
            pool = np.setdiff1d(np.arange(cfg.h1), distal_rows)
            need = k2 - len(distal_rows)
            extra = (vrng.choice(pool, size=need, replace=False)
                     if 0 < need <= len(pool) else np.array([], dtype=int))
            m2[np.concatenate([distal_rows, extra]), j] = True
        m.mask2 = m2
        m.W2 = _colnorm(m.W2 * m2)
    lam = spectral_of(cfg, seed, delay)
    m.orth = orth
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    return acc


base = replace(CreditConfig(), h1=32, h2=16, alpha=1.0, beta=1.0)
train, eval_n, iters = 10000, 300, 16
print(f"{'em':>5s} {'eta_w':>5s} {'mask':>5s} | " + " ".join(f"seed{s}" for s in range(3)) + " | mean")
for em in ("full", "class"):
    for ew in (0.05, 0.10):
        for mg in (False, True):
            accs = []
            for s in range(3):
                cfg = replace(base, energy_mode=em, eta_w=ew)
                a = run(cfg, 4, train, eval_n, s, "error", iters, mg)
                accs.append(a)
            print(f"{em:>5s} {ew:>5.2f} {'g' if mg else 'b':>5s} | "
                  + " ".join(f"{a:.3f}" for a in accs) + f" | {np.mean(accs):.3f}")
