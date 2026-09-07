#!/usr/bin/env python3
"""最终配置全判据验证：0/1编码 a=1.5 b=1.0 h1=32 h2=16 eta=0.09
err臂 iters=32（钳制）；hebb臂 iters=8 + 自由推断。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN


def make_seq(cfg, rng, n, delay, lo, hi):
    d = cfg.d_feat
    steps = n + delay
    feat = rng.uniform(0.0, 1.0, (steps, d))
    b0 = rng.random(steps) < 0.5
    b1 = rng.random(steps) < 0.5
    feat[:, 0] = np.where(b0, hi, lo)
    feat[:, 1] = np.where(b1, hi, lo)
    y = (b0[:-delay] != b1[:-delay]) if delay > 0 else (b0 != b1)
    X = np.empty((n, (delay + 1) * d))
    for t in range(n):
        X[t] = feat[t:t + delay + 1].ravel()
    return X, y.astype(float)


def run(cfg, delay, train_n, eval_n, seed, mode, iters, eta, hebb_free, lo, hi):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = make_seq(cfg, rng, train_n + eval_n, delay, lo, hi)
    Xtr, ytr = X[:train_n], y[:train_n]
    Xev, yev = X[train_n:], y[train_n:]
    m = CreditPCN(replace(cfg, delay=delay), np.random.default_rng(seed * 3000 + 11), mode)
    m.orth = False
    m.eta_inf = eta
    m.iters = iters
    m.hebb_free = hebb_free
    for t in range(train_n):
        yoh = np.array([1.0, 0.0]) if ytr[t] == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=hebb_free)
        m._learn(Xtr[t], yoh)
    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]) / total)
    return acc, distal


cfg = replace(CreditConfig(alpha=1.5, beta=1.0, energy_mode="full"),
              h1=32, h2=16)
eta, lo, hi = 0.09, 0.0, 1.0
ITERS_E, ITERS_H = 32, 8
print("per-seed（Δ=4: train 10000 eval 1500；Δ=1: train 5000 eval 1500）")
rows = []
for s in range(3):
    a4e, d4e = run(cfg, 4, 10000, 1500, s, "error", ITERS_E, eta, False, lo, hi)
    a4h, _ = run(cfg, 4, 10000, 1500, s, "hebb", ITERS_H, eta, True, lo, hi)
    a1e, d1e = run(cfg, 1, 5000, 1500, s, "error", ITERS_E, eta, False, lo, hi)
    a1h, _ = run(cfg, 1, 5000, 1500, s, "hebb", ITERS_H, eta, True, lo, hi)
    rows.append((a4e, a4h, a4e - a4h, d4e, a1e, a1h, d1e))
    print(f"seed{s}: Δ4 err={a4e:.3f} hebb={a4h:.3f} gap={a4e-a4h:.3f} "
          f"distal={d4e:.3f} | Δ1 err={a1e:.3f} hebb={a1h:.3f} distal={d1e:.3f}")
m = np.mean(np.array(rows), axis=0)
print(f"\nMEAN: Δ4 err={m[0]:.3f} hebb={m[1]:.3f} gap={m[2]:.3f} distal={m[3]:.3f} "
      f"| Δ1 err={m[4]:.3f} hebb={m[5]:.3f} distal={m[6]:.3f}")
ok = dict(
    err=all(r[0] >= 0.80 for r in rows),
    hebb=all(r[1] <= 0.68 for r in rows),
    gap=all(r[2] >= 0.30 for r in rows),
    distal=all(r[3] >= 0.25 for r in rows),
    d1err=m[4] >= 0.75,
)
print("PASS per-seed:", ok, "| ALL:", all(ok.values()))
