#!/usr/bin/env python3
"""debug ro_on_recon: 重建保真 + 读出头训练轨迹 + 各阶段 eval 精度。"""
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

# 1) 冻结后重建保真（argmax 符号级）
rng = np.random.default_rng(9)
recon_cells = []
for _ in range(300):
    g_in = arc.sample_input()
    s = arc._onehot(g_in).astype(float)
    model.apply_transform(s, arc._cond(0))
    shat = model.Ws[1] @ model.xs[1]
    g_rec = arc.decode_grid(shat).ravel()
    recon_cells.append(float(np.mean(g_rec == g_in.ravel())))
print(f"frozen recon argmax cell acc = {np.mean(recon_cells):.4f}")

# 2) 读出头训练误差轨迹（重建任务：flip_h，重训一个头）
model.set_learning(True)
cond = arc._cond(0)
W0 = model.W_outs[0].copy()
e_hist = []
for t in range(800):
    g_in = arc.sample_input()
    s_in = arc._onehot(g_in).astype(float)
    s_out = arc._onehot(TRANSFORMS["flip_h"](g_in)).astype(float)
    info = model.step_mapping(s_in, s_out, cond)
    e_hist.append(info["e_readout"])
model.set_learning(False)
print(f"readout retrain e_norm: first={e_hist[0]:.3f} mid={e_hist[400]:.3f} last={e_hist[-1]:.3f}")

# 3) 重训后 flip_h 符号精度
ca = []
for _ in range(120):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    g_pred = arc.decode_grid(v).ravel()
    ca.append(float(np.mean(g_pred == TRANSFORMS["flip_h"](g_in).ravel())))
print(f"after retrain flip_h cell = {np.mean(ca):.4f}")

# 4) W_out 列范数分布
n = np.linalg.norm(model.W_outs[0], axis=0)
print(f"W_out colnorm: mean={n.mean():.3f} max={n.max():.3f}")
