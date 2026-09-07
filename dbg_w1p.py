#!/usr/bin/env python3
"""LS 上限 vs 读出头掩码：fan_in_ro_frac 是否绑定精度上限？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))


def ls_with_mask(fan_ro, tag):
    dcfg = replace(DeepConfig(), trace_energy=False, fan_in_ro_frac=fan_ro,
                   ro_norm="clip", ro_norm_cap=6.0)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    model = res["model"]
    model.set_learning(False)
    print(f"--- {tag} fan_ro={fan_ro} ---")
    for nm in arc.train_names:
        cond = arc._cond(arc.train_names.index(nm))
        Xs, Ys = [], []
        for _ in range(1200):
            g_in = arc.sample_input()
            model.apply_transform(arc._onehot(g_in).astype(float), cond)
            Xs.append(model.xs[model.ro_src].copy())
            Ys.append(arc._onehot(TRANSFORMS[nm](g_in)).astype(float))
        X = np.stack(Xs); Y = np.stack(Ys)
        W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y).T
        idx = arc.train_names.index(nm)
        if model.ro_masks[idx] is not None:
            W = W * model.ro_masks[idx]          # 施加头掩码
        n = np.linalg.norm(W, axis=0, keepdims=True)
        over = n[0] > 6.0
        if over.any():
            W[:, over] *= 6.0 / np.maximum(n[:, over], 1e-8)
        ca = []
        for _ in range(120):
            g_in = arc.sample_input()
            model.apply_transform(arc._onehot(g_in).astype(float), cond)
            g_pred = arc.decode_grid(W @ model.xs[model.ro_src])
            ca.append(float(np.mean(g_pred == TRANSFORMS[nm](g_in).ravel())))
        print(f"  {nm:10s} ls+mask cell={np.mean(ca):.4f}")


ls_with_mask(0.40, "masked")
ls_with_mask(0.00, "dense")
