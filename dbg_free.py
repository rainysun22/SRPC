#!/usr/bin/env python3
"""逐迭代跟踪 free 推断，找出 x1/x2 死亡原因。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence

cfg = replace(CreditConfig(), alpha=1.0, beta=1.0)
rng = np.random.default_rng(7)
X, y = _make_sequence(cfg, rng, 2000, 1)
m = CreditPCN(replace(cfg, delay=1), np.random.default_rng(11), "error")
m.orth = False
for t in range(500):
    m.train_step(X[t], float(y[t]))
m.set_learning(False)

x = X[600]
m.x1[:] = 0.0
m.x2[:] = 0.0
x3 = np.zeros(2)
print(f"input x0 norm={np.linalg.norm(x):.3f}")
print(f"W1 col norms: min={np.linalg.norm(m.W1,axis=0).min():.3f} "
      f"max={np.linalg.norm(m.W1,axis=0).max():.3f} nz_per_col={int((m.mask1).sum(0).max())}")
for it in range(15):
    e0 = x - m.W1 @ m.x1
    e1 = m.x1 - m.W2 @ m.x2
    e2 = m.x2 - m.W3 @ x3
    u1 = 1.0 * (m.W1.T @ e0) - 1.0 * e1
    u2 = 1.0 * (m.W2.T @ e1) - 1.0 * e2
    g1 = np.abs(u1) > cfg.theta_event
    g2 = np.abs(u2) > cfg.theta_event
    n1 = float(np.linalg.norm(m.x1))
    n2 = float(np.linalg.norm(m.x2))
    m.x1 = np.clip(m.x1 + u1 * g1, 0.0, cfg.x_max)
    m.x2 = np.clip(m.x2 + u2 * g2, 0.0, cfg.x_max)
    from srpc.credit import _kwta
    m.x1 = _kwta(m.x1, cfg.kwta_frac)
    m.x2 = _kwta(m.x2, cfg.kwta_frac)
    e2 = m.x2 - m.W3 @ x3
    x3 = np.clip(x3 + cfg.eta_out * (m.W3.T @ e2), 0.0, 1.0)
    print(f"it{it:2d} |x1|={n1:6.3f}->{np.linalg.norm(m.x1):6.3f} "
          f"|x2|={n2:6.3f}->{np.linalg.norm(m.x2):6.3f} "
          f"|e0|={np.linalg.norm(e0):6.3f} |e1|={np.linalg.norm(e1):6.3f} "
          f"|e2|={np.linalg.norm(e2):6.3f} |u1|={np.linalg.norm(u1):6.3f} "
          f"|u2|={np.linalg.norm(u2):6.3f} x3={x3}")
