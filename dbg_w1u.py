#!/usr/bin/env python3
"""重建误差定位 + 内迭代数对重建保真的影响（冻结模型）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

for it in (1, 3, 6, 10):
    model.cfg = replace(model.cfg, inner_iters=it)
    empty_err, block_err, n_empty, n_block, cell = [], [], 0, 0, []
    for _ in range(300):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), arc._cond(0))
        shat = model.Ws[1] @ model.xs[1]
        g_rec = arc.decode_grid(shat).ravel()
        gt = g_in.ravel()
        cell.append(float(np.mean(g_rec == gt)))
        e = g_rec != gt
        empty = gt == 0
        n_empty += int(empty.sum()); n_block += int((~empty).sum())
        empty_err.append(float(np.mean(e[empty])))
        block_err.append(float(np.mean(e[~empty])))
    print(f"iters={it}: cell={np.mean(cell):.4f} empty_err={np.mean(empty_err)*100:.2f}% "
          f"({n_empty}格) block_err={np.mean(block_err)*100:.2f}% ({n_block}格)")
model.cfg = replace(model.cfg, inner_iters=3)
