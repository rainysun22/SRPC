#!/usr/bin/env python3
"""诊断：x1 量级 / 读出头列范数 vs LS 解列范数 —— clip 上限是否绑死？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False, ro_norm="clip", eta_wout=0.05)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

nm = "flip_h"
cond = arc._cond(0)
# x1 量级
vals = []
for _ in range(300):
    g_in = arc.sample_input()
    model.apply_transform(arc._onehot(g_in).astype(float), cond)
    x = model.xs[model.ro_src]
    vals.append(x[x > 0])
v = np.concatenate(vals)
print(f"x1 nonzero: mean={v.mean():.4f} med={np.median(v):.4f} max={v.max():.4f} "
      f"frac_nonzero={len(v) / (300 * x.size):.3f}")

# 读出头列范数
W = model.W_outs[0]
cn = np.linalg.norm(W, axis=0)
print(f"W_out colnorm: mean={cn.mean():.3f} max={cn.max():.3f} frac_sat={np.mean(cn >= 0.999):.3f}")

# LS 解列范数（同特征）
Xs, Ys = [], []
for _ in range(1500):
    g_in = arc.sample_input()
    model.apply_transform(arc._onehot(g_in).astype(float), cond)
    Xs.append(model.xs[model.ro_src].copy())
    Ys.append(arc._onehot(TRANSFORMS[nm](g_in)).astype(float))
X = np.stack(Xs); Y = np.stack(Ys)
Wls = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y).T
cnls = np.linalg.norm(Wls, axis=0)
print(f"LS colnorm: mean={cnls.mean():.3f} max={cnls.max():.3f} frac_gt1={np.mean(cnls > 1):.3f}")

# 用 clip 约束后的 LS 解评估（模拟 clip 绑定的效果）
Wcl = Wls.copy()
n = np.linalg.norm(Wcl, axis=0, keepdims=True)
over = n[0] > 1.0
Wcl[:, over] /= n[:, over]
ca = []
for _ in range(150):
    g_in = arc.sample_input()
    model.apply_transform(arc._onehot(g_in).astype(float), cond)
    g_pred = arc.decode_grid(Wcl @ model.xs[model.ro_src])
    ca.append(float(np.mean(g_pred == TRANSFORMS[nm](g_in).ravel())))
print(f"LS+clip cell acc: {np.mean(ca):.4f}  (LS free: {0.9908:.4f} ref)")
