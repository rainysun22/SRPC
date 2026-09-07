#!/usr/bin/env python3
"""查 beta_cond 为何无效：对比 bc=0.0 vs 0.8 的 pred_self 与 xL。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite

for bc in (0.0, 0.8):
    dcfg = replace(DeepConfig(), trace_energy=False, beta_cond=bc)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(0, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                         acfg=ArcConfig(), clcfg=CLConfig())
    model = res["model"]
    arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
    model.set_learning(False)
    cond = arc._cond(0)
    g_in = arc.sample_input()
    s = arc._onehot(g_in).astype(float)
    model.set_condition(cond)
    model.reset_states()
    model.prepare_next(action=None)
    print(f"bc={bc}: pred_self_norm={np.linalg.norm(model.pred_self):.4f} "
          f"pred_self_nz={np.count_nonzero(model.pred_self)}")
    print(f"  Uc@cond norm={np.linalg.norm(model.Uc @ cond):.4f} "
          f"Wdyn@z norm={np.linalg.norm(model.Wdyn @ model.last_z):.4f}")
    model.observe(s)
    print(f"  after observe: xL_norm={np.linalg.norm(model.xs[4]):.4f} "
          f"x1_norm={np.linalg.norm(model.xs[1]):.4f}")

# 直接查 learn_readout 的 e 与 readout 输出
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(0, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                     acfg=ArcConfig(), clcfg=CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
model.set_learning(False)
cond = arc._cond(0)
g_in = arc.sample_input()
s = arc._onehot(g_in).astype(float)
out = model.apply_transform(s, cond)
print("\nreadout norm:", np.linalg.norm(out), "src norm:", np.linalg.norm(model._ro_src()))
print("W_out[0] shape:", model.W_outs[0].shape, "W_out[0] norm:", np.linalg.norm(model.W_outs[0]))
