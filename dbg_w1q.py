#!/usr/bin/env python3
"""LS 上限：读出源 = 核心重建 ŝ = W1@x1（符号空间，变换为精确线性映射）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

# 重建保真度
rng = np.random.default_rng(9)
recon_err = []
for _ in range(200):
    g_in = arc.sample_input()
    s = arc._onehot(g_in).astype(float)
    model.apply_transform(s, arc._cond(0))
    shat = model.Ws[1] @ model.xs[1]
    recon_err.append(float(np.mean((s - shat) ** 2)))
print(f"recon mse={np.mean(recon_err):.5f}")

for nm in arc.train_names:
    cond = arc._cond(arc.train_names.index(nm))
    Xs, Ys = [], []
    for _ in range(1200):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), cond)
        Xs.append(model.Ws[1] @ model.xs[1])       # ŝ 特征
        Ys.append(arc._onehot(TRANSFORMS[nm](g_in)).astype(float))
    X = np.stack(Xs); Y = np.stack(Ys)
    W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y).T
    ca = []
    for _ in range(120):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), cond)
        g_pred = arc.decode_grid(W @ (model.Ws[1] @ model.xs[1]))
        ca.append(float(np.mean(g_pred == TRANSFORMS[nm](g_in).ravel())))
    print(f"  {nm:10s} ls@recon cell={np.mean(ca):.4f}")
