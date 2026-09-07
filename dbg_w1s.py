#!/usr/bin/env python3
"""ro_on_recon 读出头收敛性扫描：gate 阈值 / eta / 列范数。"""
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


def eval_flip_h():
    model.set_learning(False)
    ca = []
    for _ in range(120):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), arc._cond(0))
        g_pred = arc.decode_grid(v).ravel()
        ca.append(float(np.mean(g_pred == TRANSFORMS["flip_h"](g_in).ravel())))
    return float(np.mean(ca))


for gate, eta in ((0.3, 0.05), (0.3, 0.02), (0.3, 0.01), (0.5, 0.02), (0.1, 0.02)):
    model.W_outs = [None] * 4          # 重初始化所有头
    model.ro_masks = [None] * 4
    model.ro_fan = [0] * 4
    model.ro_sig = [None] * 4
    model.d_out = 0
    model.set_learning(True)
    for t in range(800):
        g_in = arc.sample_input()
        s_in = arc._onehot(g_in).astype(float)
        s_out = arc._onehot(TRANSFORMS["flip_h"](g_in)).astype(float)
        model.step_mapping(s_in, s_out, arc._cond(0))
    acc = eval_flip_h()
    print(f"gate={gate} eta={eta}: flip_h cell={acc:.4f}")
