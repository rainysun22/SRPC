#!/usr/bin/env python3
"""bal 平衡掩码 × alpha/beta × iters 联合搜索：救 seed1 同时不破坏 seed0/2。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _colnorm, _make_sequence


def apply_bal(m: CreditPCN) -> None:
    cfg = m.cfg
    d, n_blk = cfg.d_feat, cfg.delay + 1
    m1 = np.zeros((m.d0, m.W1.shape[1]), dtype=bool)
    for j in range(m.W1.shape[1]):
        b = j % n_blk
        m1[b * d:(b + 1) * d, j] = True
    m.mask1 = m1
    m.W1 = _colnorm(m.W1 * m1)
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


def run(cfg, seed, iters, bal):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    if bal:
        apply_bal(m)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.72 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full")
print("bal × alpha × beta（iters=24）")
for bal in (False, True):
    for alpha in (1.0, 1.5, 2.0):
        for beta in (1.0, 0.5):
            cfg = replace(base, alpha=alpha, beta=beta)
            accs = [run(cfg, s, 24, bal) for s in range(3)]
            flag = " <== ALL>=0.80" if min(accs) >= 0.80 else ""
            print(f"bal={int(bal)} a={alpha:.1f} b={beta:.1f} | "
                  + " ".join(f"{a:.3f}" for a in accs)
                  + f" | mean={np.mean(accs):.3f}{flag}")
