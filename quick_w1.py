#!/usr/bin/env python3
"""快速单 seed 验证：x1 读出源 + 全训练配置下符号精度。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=True)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

for name in arc.train_names + arc.novel_names:
    is_novel = name in arc.novel_names
    ca, ga = [], []
    for _ in range(80):
        g_in = arc.sample_input()
        oh_in = arc._onehot(g_in).astype(float)
        if is_novel:
            i, j = arc.novel_combos[arc.novel_names.index(name)]
            c1 = arc._cond(arc.train_names.index(i))
            c2 = arc._cond(arc.train_names.index(j))
            out1 = model.apply_transform(oh_in, c1)
            v = model.apply_transform(out1, c2)
            g_true = TRANSFORMS[j](TRANSFORMS[i](g_in))
        else:
            cond = arc._cond(arc.train_names.index(name))
            v = model.apply_transform(oh_in, cond)
            g_true = TRANSFORMS[name](g_in)
        g_pred = arc.decode_grid(v).ravel()
        ca.append(float(np.mean(g_pred == np.asarray(g_true).ravel())))
        ga.append(float(np.all(g_pred == np.asarray(g_true).ravel())))
    print(f"{name:14s} cell={np.mean(ca):.4f} grid={np.mean(ga):.4f}")
