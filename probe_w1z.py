#!/usr/bin/env python3
"""W1 证据采集 v2（修正 cell 口径：逐格 reshape(-1,4).argmax）：
1) LS 冻结解上限（在线 RLS 差距对照）；
2) 谱证据：eig_max(W1^TW1) 出生 vs 训练后（注意：与 config 旧注释相反，
   实测出生 ~48（重叠正感受野列相关）、训练后 ~3（Hebbian+列归一化去相关））。
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS
from srpc.deepmodel import DeepSRPC

seed = 0
dcfg = DeepConfig()
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

eig_after = float(np.linalg.eigvalsh(model.Ws[1].T @ model.Ws[1])[-1])
m0 = DeepSRPC(dcfg, 0, np.random.default_rng(seed * 3000 + 1), self_loop=True,
              memory=None)
eig_birth = float(np.linalg.eigvalsh(m0.Ws[1].T @ m0.Ws[1])[-1])
print(f"eig_max(W1^TW1): birth={eig_birth:.2f} trained={eig_after:.2f}")

cond = arc._cond(0)
X, Y = [], []
for _ in range(400):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    X.append(v.copy())
    Y.append(arc._onehot(TRANSFORMS["flip_h"](g_in)).astype(float).ravel())
X, Y = np.asarray(X), np.asarray(Y)
W_ls, *_ = np.linalg.lstsq(X, Y, rcond=None)

def cellgrid(v, g_true):
    p = (W_ls.T @ v).reshape(-1, 4).argmax(axis=1).reshape(arc.grid, arc.grid)
    return float(np.mean(p == g_true)), float(np.all(p == g_true))

cs, gs = [], []
for _ in range(200):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    c, g = cellgrid(v, TRANSFORMS["flip_h"](g_in))
    cs.append(c); gs.append(g)
print(f"LS 上限（flip_h, ŝ 源）: cell={np.mean(cs):.4f} grid={np.mean(gs):.4f}")

# 在线 RLS 同口径对照（当前模型读出头）
cs, gs = [], []
for _ in range(200):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    c, g = cellgrid_online = (lambda vv, gt: (
        float(np.mean(vv.reshape(-1, 4).argmax(axis=1).reshape(arc.grid, arc.grid) == gt)),
        float(np.all(vv.reshape(-1, 4).argmax(axis=1).reshape(arc.grid, arc.grid) == gt))))(
        v, TRANSFORMS["flip_h"](g_in))
    cs.append(c); gs.append(g)
print(f"在线 RLS（flip_h, 同样本）: cell={np.mean(cs):.4f} grid={np.mean(gs):.4f}")
