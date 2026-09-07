#!/usr/bin/env python3
"""掩码变体实验 v2：只改命名部分，其余继承模型自身掩码（公平对照）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence, _colnorm, _orth_cols


def run(cfg, delay, train, eval_n, seed, mode, eta_inf, iters, variant, orth=False, eta_w=None):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    rcfg = replace(cfg, delay=delay)
    m = CreditPCN(rcfg, np.random.default_rng(seed * 3000 + 11), mode)

    d0, n_blk, d = (cfg.delay + 1) * cfg.d_feat, cfg.delay + 1, cfg.d_feat
    vrng = np.random.default_rng(seed * 3000 + 99)
    m1, m2, m3 = m.mask1.copy(), m.mask2.copy(), m.mask3.copy()

    if variant in ("w1_full", "both"):
        m1 = np.zeros((d0, cfg.h1), dtype=bool)
        for j in range(cfg.h1):
            m1[:d, j] = True
            b = int(vrng.integers(1, n_blk))
            m1[b * d:(b + 1) * d, j] = True
    if variant in ("m2_guar", "both"):
        distal_rows = np.where(m1[:d].any(axis=0))[0]
        k2 = max(1, int(round(cfg.fan_in_frac * cfg.h1)))
        m2 = np.zeros((cfg.h1, cfg.h2), dtype=bool)
        for j in range(cfg.h2):
            pool = np.setdiff1d(np.arange(cfg.h1), distal_rows)
            need = k2 - len(distal_rows)
            extra = (vrng.choice(pool, size=need, replace=False)
                     if 0 < need <= len(pool) else np.array([], dtype=int))
            idx = np.concatenate([distal_rows, extra])
            m2[idx, j] = True

    m.mask1, m.mask2, m.mask3 = m1, m2, m3
    m.W1 = _colnorm(m.W1 * m1)
    m.W2 = _colnorm(m.W2 * m2)
    m.W3 = _orth_cols(m.W3 * m3)
    m.W1_init = m.W1.copy()
    m.orth = orth
    m.eta_inf = eta_inf
    m.iters = iters
    if eta_w is not None:
        m.eta_w = eta_w
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


cfg = replace(CreditConfig(alpha=1.0, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
eta_inf, iters = 0.095, 8
variants = ("base", "m2_guar", "w1_full", "both")
print(f"{'variant':>10s} | " + " ".join(f"seed{s}" for s in range(3)) + " | mean  distal")
for v in variants:
    accs, dists = [], []
    for s in range(3):
        a, dd = run(cfg, 4, 10000, 300, s, "error", eta_inf, iters, v)
        accs.append(a); dists.append(dd)
    print(f"{v:>10s} | " + " ".join(f"{a:.3f}" for a in accs) +
          f" | {np.mean(accs):.3f}  {np.mean(dists):.3f}")
