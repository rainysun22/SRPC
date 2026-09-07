#!/usr/bin/env python3
"""检查：inner_iters=6 训练下，读出头是否学到了输入->输出映射？"""
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
model.set_learning(False)
print("MSE diag:", [f"{x:.4f}" for x in res["diag"]])
print("combo:", {k: f"{v:.4f}" for k, v in res["combo"].items() if k in arc.novel_names})

# 单样本细看：flip_h
rng = np.random.default_rng(3)
name = "flip_h"
cond = arc._cond(arc.train_names.index(name))
for m in range(3):
    g_in = arc.sample_input()
    oh = arc._onehot(g_in).astype(float)
    v = model.apply_transform(oh, cond)
    g_pred = arc.decode_grid(v)
    g_true = TRANSFORMS[name](g_in)
    cell = float(np.mean(g_pred == g_true.ravel()))
    # 输出值域
    vmat = v.reshape(-1, 4)
    print(f"s{m}: cell={cell:.3f} |pred nz={np.count_nonzero(g_pred)} "
          f"true nz={np.count_nonzero(g_true)} "
          f"| max_v={vmat.max():.3f} ch0_mean={vmat[:,0].mean():.3f} "
          f"ch1..3_mean={vmat[:,1:].mean():.3f}")
    if m == 0:
        print("  pred:\n", g_pred.reshape(8, 8).astype(int))
        print("  true:\n", g_true)
        print("  in :\n", g_in)
