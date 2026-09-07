#!/usr/bin/env python3
"""弱极性先验（noise 大小）× alpha_eval 组合，专攻 seed1 err。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _colnorm, _make_sequence

POLAR = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) / np.array(
    [[1.0], [1.0], [np.sqrt(2.0)]])


def apply_softpolar(m, rng, noise):
    cfg = m.cfg
    d, n_blk = cfg.d_feat, cfg.delay + 1
    W1 = np.zeros_like(m.W1)
    for j in range(m.W1.shape[1]):
        b = j % n_blk
        base_d = POLAR[j % len(POLAR)]
        W1[b * d:(b + 1) * d, j] = base_d + noise * rng.normal(size=d)
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


def run(cfg, seed, iters, polar_noise, alpha_eval):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    if polar_noise is not None:
        apply_softpolar(m, np.random.default_rng(seed * 3000 + 11), polar_noise)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    m.cfg = replace(m.cfg, alpha=alpha_eval)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== 弱极性 noise × alpha_eval（err 臂，iters=24）===")
for noise in (None, 2.0, 1.5, 1.0):
    for ae in (1.0, 1.5):
        accs = [run(base, s, 24, noise, ae) for s in range(3)]
        tag = "none " if noise is None else f"n={noise:.1f}"
        print(f"{tag} ae={ae:.1f} | " + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}")
