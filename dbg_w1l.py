#!/usr/bin/env python3
"""x1 表征保真上限：冻结 x1 特征后用最小二乘解完美读出，看 x1 能否支撑 0.99。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))


def ls_ceiling(beta_cond, tag):
    dcfg = replace(DeepConfig(), trace_energy=False, beta_cond=beta_cond)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    model = res["model"]
    model.set_learning(False)
    rng = np.random.default_rng(42)
    print(f"--- {tag} (beta_cond={beta_cond}) ---")
    for nm in arc.train_names:
        cond = arc._cond(arc.train_names.index(nm))
        Xs, Ys = [], []
        for _ in range(1200):
            g_in = arc.sample_input()
            model.apply_transform(arc._onehot(g_in).astype(float), cond)
            Xs.append(model.xs[model.ro_src].copy())
            Ys.append(arc._onehot(TRANSFORMS[nm](g_in)).astype(float))
        X = np.stack(Xs); Y = np.stack(Ys)
        # 岭回归最小二乘（正则防病态）
        W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y).T
        ca = []
        for _ in range(120):
            g_in = arc.sample_input()
            model.apply_transform(arc._onehot(g_in).astype(float), cond)
            g_pred = arc.decode_grid(W @ model.xs[model.ro_src])
            ca.append(float(np.mean(g_pred == TRANSFORMS[nm](g_in).ravel())))
        print(f"  {nm:10s} ls-cell={np.mean(ca):.4f}  (x1nz={np.mean(X != 0, axis=1).mean():.3f})")


ls_ceiling(0.8, "default")
ls_ceiling(0.0, "nocond")
