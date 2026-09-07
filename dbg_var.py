#!/usr/bin/env python3
"""诊断 Δ4 种子间方差：远端块掩码覆盖 / 谱半径 / 权重学习动态。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(cfg, seed=11):
    """推断 Hessian 最大特征值（决定稳定步长上界）。"""
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    m.eta_inf = 0.095
    m.iters = 8
    m.orth = False
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    return float(np.linalg.eigvalsh(H).max())


def run_detail(cfg, delay, train, eval_n, seed, mode, eta_inf, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    m.orth = False
    m.eta_inf = eta_inf
    m.iters = iters

    # 掩码覆盖：远端块（首 d_feat 行）的单元数
    d = cfg.d_feat
    n_blk = delay + 1
    distal_units = int(m.mask1[:d].any(axis=0).sum())
    per_blk = [int(m.mask1[b * d:(b + 1) * d].any(axis=0).sum()) for b in range(n_blk)]
    accs = []
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
        if (t + 1) % 2500 == 0:
            m.set_learning(False)
            preds = np.array([m.predict(x) for x in Xev[:200]])
            accs.append(float(np.mean(preds == yev[:200])))
            m.set_learning(True)
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:d]))
    # 每块权重变化占比
    blk_share = [float(np.linalg.norm(dW[b * d:(b + 1) * d]) / total) for b in range(n_blk)]
    return dict(acc=acc, distal=distal / total, distal_units=distal_units,
                per_blk=per_blk, blk_share=blk_share, accs=accs)


cfg = replace(CreditConfig(alpha=1.0, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
eta_inf, iters = 0.095, 8
for s in range(3):
    lm = spectral_of(cfg, s)
    r = run_detail(cfg, 4, 10000, 300, s, "error", eta_inf, iters)
    print(f"seed{s}: lam_max={lm:.1f} eta_stable={1.2/lm:.4f} "
          f"acc={r['acc']:.3f} distal_units={r['distal_units']}/32 "
          f"per_blk={r['per_blk']} blk_share={['%.3f' % x for x in r['blk_share']]}")
    print(f"        acc@2.5k/5k/7.5k/10k={['%.3f' % a for a in r['accs']]}")
