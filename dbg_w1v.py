#!/usr/bin/env python3
"""追踪记忆先验在冻结评估中的层间传播（mem vs no_mem，同一 seed 同初始化）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res_m = run_sequential(seed, with_memory=True, dcfg=dcfg,
                       mcfg=mcfg, acfg=ArcConfig(), clcfg=CLConfig())
res_n = run_sequential(seed, with_memory=False, dcfg=dcfg,
                       mcfg=mcfg, acfg=ArcConfig(), clcfg=CLConfig())
m1, m2 = res_m["model"], res_n["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))

# 逐层权重差异
for l in range(1, m1.L + 1):
    print(f"Ws[{l}] max diff: {np.max(np.abs(m1.Ws[l]-m2.Ws[l])):.4f}")
for i in range(len(m1.W_outs)):
    if m1.W_outs[i] is not None and m2.W_outs[i] is not None:
        print(f"W_out[{i}] max diff: {np.max(np.abs(m1.W_outs[i]-m2.W_outs[i])):.4f}")

# 冻结评估任务0，追踪每层差异
m1.set_learning(False); m2.set_learning(False)
cond = arc._cond(0)
s_in, s_out, _ = arc.sample("flip_h")
for model in (m1, m2):
    model.set_condition(cond)
    model.reset_states()
    model.prepare_next(action=None)

print("\n--- 每内迭代后层差异 (mem - no_mem) ---")
for it in range(1, 9):
    for model in (m1, m2):
        model.cfg = replace(model.cfg, inner_iters=1)
        model.observe(s_in)
    diffs = [np.max(np.abs(m1.xs[l] - m2.xs[l])) for l in range(0, m1.L + 1)]
    print(f"iter {it}: " + " ".join(f"x{l}={d:.4f}" for l, d in enumerate(diffs)))
