#!/usr/bin/env python3
"""能量估计口径：末态 vs 轨迹平均 vs 末 k 平均（降低 compare 裕度噪声）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _kwta, _make_sequence


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def settle_energy(m, x0, yoh, avg, last_k=3):
    cfg = m.cfg
    a, b = cfg.alpha, cfg.beta
    x1 = np.zeros_like(m.x1); x2 = np.zeros_like(m.x2)
    x3 = yoh.copy()
    es = []
    for _ in range(m.iters):
        e0 = x0 - m.W1 @ x1
        e1 = x1 - m.W2 @ x2
        e2 = x2 - m.W3 @ x3
        u1 = b * (m.W1.T @ e0) - a * e1
        u2 = b * (m.W2.T @ e1) - a * e2
        x1 = np.clip(x1 + m.eta_inf * u1 * (np.abs(u1) > cfg.theta_event), 0.0, cfg.x_max)
        x2 = np.clip(x2 + m.eta_inf * u2 * (np.abs(u2) > cfg.theta_event), 0.0, cfg.x_max)
        if cfg.kwta_on:
            x1 = _kwta(x1, cfg.kwta_frac)
            x2 = _kwta(x2, cfg.kwta_frac)
        e0 = x0 - m.W1 @ x1
        e1 = x1 - m.W2 @ x2
        e2 = x2 - m.W3 @ x3
        if cfg.energy_mode == "class":
            es.append(0.5 * (float(np.dot(e1, e1)) + float(np.dot(e2, e2))))
        else:
            es.append(0.5 * (float(np.dot(e0, e0)) + float(np.dot(e1, e1))
                             + float(np.dot(e2, e2))))
    if avg == "final":
        return es[-1]
    if avg == "traj":
        return float(np.mean(es))
    return float(np.mean(es[-last_k:]))


def run(cfg, seed, avg, iters_eval=None, last_k=3):
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
    if iters_eval:
        m.iters = iters_eval
    ok = 0
    for t in range(len(Xev)):
        es = {}
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            es[c] = settle_energy(m, Xev[t], yoh, avg, last_k)
        ok += int((0 if es[0] < es[1] else 1) == float(yev[t]))
    return ok / len(Xev)


cfg = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
              alpha=1.0, beta=1.0)
print("avg | seed accs | mean")
for avg in ("final", "traj", "last3"):
    accs = [run(cfg, s, avg) for s in range(3)]
    print(f"{avg:<6s} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | {np.mean(accs):.3f}")
print("traj + iters_eval=40:")
accs = [run(cfg, s, "traj", iters_eval=40) for s in range(3)]
print("       | " + " ".join(f"{a:.3f}" for a in accs) + f" | {np.mean(accs):.3f}")
