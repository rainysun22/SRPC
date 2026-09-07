#!/usr/bin/env python3
"""Δ4 种子间方差修复：结构化出生掩码 × 初始化正交化。

方案（对齐文档"出生即定型掩码"不变量 3，消除网络结构的随机运气）：
- bal：层1掩码改为循环平衡分块 —— 每时间块等量单元（单元 j 锚定块 j%(Δ+1)），
  远端块覆盖由构造保证（种子间不再有二项式方差）；
- orthw2：W2 列初始化正交化 —— x2 特征去相关，类表征不坍缩（dbg2 诊断 seed1
  x2cos=0.886 坍缩）；
- h2=24：更多组合容量。
"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _colnorm, _orth_cols, _make_sequence


def apply_variant(m: CreditPCN, variant: str) -> None:
    cfg = m.cfg
    if "bal" in variant:
        d, n_blk = cfg.d_feat, cfg.delay + 1
        m1 = np.zeros((m.d0, m.W1.shape[1]), dtype=bool)
        for j in range(m.W1.shape[1]):
            b = j % n_blk                      # 循环锚定：每块等量单元
            m1[b * d:(b + 1) * d, j] = True
        m.mask1 = m1
        m.W1 = _colnorm(m.W1 * m1)
    if "orthw2" in variant:
        m.W2 = _colnorm(_orth_cols(m.W2 * m.mask2) * m.mask2)
    m.W1_init = m.W1.copy()                    # distal 度量基准 = 出生权重


def spectral_of(m: CreditPCN) -> float:
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, delay, train, eval_n, seed, mode, variant, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + eval_n, delay)
    Xtr, ytr = X[:train], y[:train]
    Xev, yev = X[train:], y[train:]
    m = CreditPCN(replace(cfg, delay=delay),
                  np.random.default_rng(seed * 3000 + 11), mode)
    apply_variant(m, variant)
    lam = spectral_of(m)
    m.orth = False
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


base = replace(CreditConfig(), h1=32, energy_mode="full",
               alpha=1.0, beta=1.0)
train, eval_n, iters = 10000, 500, 24
variants = ["none", "bal", "orthw2", "bal+orthw2"]

print("=== 阶段1：变体 × h2 @ iters=24 ===")
best = None
for h2 in (16, 24):
    for var in variants:
        accs, dists = [], []
        for s in range(3):
            a, d = run(replace(base, h2=h2), 4, train, eval_n, s,
                       "error", var, iters)
            accs.append(a); dists.append(d)
        mn = np.mean(accs)
        flag = all(a >= 0.80 for a in accs)
        tag = "  <== ALL>=0.80" if flag else ""
        print(f"h2={h2:2d} {var:<10s} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={mn:.3f} | distal={np.mean(dists):.3f}{tag}")
        if flag and (best is None or mn > best[0]):
            best = (mn, h2, var)

if best is None:
    print("\n无配置达到逐种子>=0.80，终止。")
    sys.exit(1)

_, h2, var = best
print(f"\n=== 阶段2：最优配置 h2={h2} var={var} 全判据验证 ===")
cfg = replace(base, h2=h2)
for delay, tr in ((4, 10000), (1, 5000)):
    for mode in ("error", "hebb"):
        accs, dists = [], []
        for s in range(3):
            a, d = run(cfg, delay, tr, 300, s, mode, var, iters)
            accs.append(a); dists.append(d)
        print(f"Δ={delay} {mode:<5s} | "
              + " ".join(f"{a:.3f}" for a in accs)
              + f" | mean={np.mean(accs):.3f} | distal={np.mean(dists):.3f}")
