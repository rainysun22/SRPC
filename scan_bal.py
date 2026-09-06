#!/usr/bin/env python3
"""平衡块分配 + 每 seed 最优 η：验证 3 seeds 稳定性。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence, _kwta


def run(cfg, delay, train, eval_n, seed, mode, eta_inf, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    m.orth = False
    m.eta_inf = eta_inf
    m.iters = iters
    for t in range(train):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]))
    return acc, distal / total


def lm_lim(cfg, delay, seed):
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), "error")
    W1, W2 = m.W1, m.W2
    H = np.zeros((cfg.h1 + cfg.h2, cfg.h1 + cfg.h2))
    H[:cfg.h1, :cfg.h1] = W1.T @ W1 + np.eye(cfg.h1)
    H[:cfg.h1, cfg.h1:] = -W2
    H[cfg.h1:, :cfg.h1] = -W2.T
    H[cfg.h1:, cfg.h1:] = W2.T @ W2 + np.eye(cfg.h2)
    e = np.linalg.eigvalsh(H)
    return e.max(), e.min()


# 平衡块分配：直接改 mask 生成（block 编号均匀轮转）
import srpc.credit as C

_orig_init = C.CreditPCN.__init__


def _balanced_init(self, cfg, rng, learn_mode="error"):
    _orig_init(self, cfg, rng, learn_mode)
    # 重建层1掩码：每块均分单元（锚点轮转），保证远端块覆盖
    n_blk = cfg.delay + 1
    m1 = np.zeros((self.d0, cfg.h1), dtype=bool)
    for j in range(cfg.h1):
        b = j % n_blk
        m1[b * cfg.d_feat:(b + 1) * cfg.d_feat, j] = True
    self.W1 = C._colnorm(self.W1 * m1)
    self.mask1 = m1
    self.W1_init = self.W1.copy()


C.CreditPCN.__init__ = _balanced_init

cfg = replace(CreditConfig(alpha=1.0, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
print(f"{'seed':>4s} {'Δ':>2s} {'λmax':>6s} {'η':>5s} {'it8':>6s} {'it10':>6s} {'it12':>6s}")
for s in range(3):
    for delay in (1, 4):
        lm, lmin = lm_lim(cfg, delay, s)
        eta = 1.8 / lm
        outs = []
        for iters in (8, 10, 12):
            a, _ = run(cfg, delay, 10000, 300, s, "error", eta, iters)
            outs.append(a)
        print(f"{s:>4d} {delay:>2d} {lm:6.1f} {eta:5.3f} " + " ".join(f"{v:6.3f}" for v in outs))
