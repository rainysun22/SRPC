#!/usr/bin/env python3
"""软极性初始化（保证覆盖 + 随机扰动破退化）× theta_event 扫描。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _colnorm, _make_sequence

POLAR = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) / np.array(
    [[1.0], [1.0], [np.sqrt(2.0)]])


def apply_softpolar(m, rng, noise=0.5):
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


def run(cfg, seed, mode, iters, variant, theta_event):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    cfg2 = replace(cfg, theta_event=theta_event)
    m = CreditPCN(replace(cfg2, delay=4), np.random.default_rng(seed * 3000 + 11), mode)
    if variant == "softpolar":
        apply_softpolar(m, np.random.default_rng(seed * 3000 + 11), noise=0.5)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== 变体 × theta_event（err 臂，iters=24）===")
for var in ("none", "softpolar"):
    for te in (0.01, 0.05):
        accs = [run(base, s, "error", 24, var, te) for s in range(3)]
        print(f"{var:<10s} theta={te:.2f} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f}")
