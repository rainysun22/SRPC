#!/usr/bin/env python3
"""评估期 α 扫描（eval-only）：x2 锁定类原型 -> 能量 = 输入-类一致性，裕度增大。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def run(cfg, seed, mode, iters, alpha_eval):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), mode)
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    # 评估期 α 覆盖（只影响 compare 推断动力学）
    ev = replace(m.cfg, alpha=alpha_eval)
    m.cfg = ev
    preds = np.array([m.predict(x) for x in Xev])
    return float(np.mean(preds == yev))


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== 评估期 alpha_eval 扫描（err 臂，iters=24）===")
for ae in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    accs = [run(base, s, "error", 24, ae) for s in range(3)]
    print(f"alpha_eval={ae:.1f} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
