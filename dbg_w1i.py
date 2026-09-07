#!/usr/bin/env python3
"""读出学习诊断：跟踪训练中 e_readout 曲线 + W_out 更新幅度。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), inner_iters=6, beta_cond=0.0, gamma_mem=0.0)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(True)

# 手动重训任务0（flip_h），跟踪误差
name = arc.train_names[0]
cond = arc._cond(0)
model.set_condition(cond)
model.W_outs[0] = None  # 重置头
model.d_out = 0
curve = []
for t in range(800):
    s_in, s_out, c = arc.sample(name)
    info = model.step_mapping(s_in, s_out, cond)
    if t % 50 == 0 or t > 750:
        curve.append((t, info["e_readout"]))
        if t % 200 == 0:
            print(f"t={t:4d} e_readout={info['e_readout']:.4f} "
                  f"boost={info['boost']:.2f} evs={[f'{e:.2f}' for e in info['evs']]}")
# 评估符号精度
model.set_learning(False)
ca = []
for _ in range(60):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    g_pred = arc.decode_grid(v)
    ca.append(float(np.mean(g_pred == TRANSFORMS[name](g_in).ravel())))
print(f"flip_h cell acc after 800 steps (fresh head): {np.mean(ca):.4f}")
