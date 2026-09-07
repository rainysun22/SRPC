#!/usr/bin/env python3
"""插桩：每迭代 up[l] 幅度 与 x2 的 u 幅度。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), beta_cond=0.0, gamma_mem=0.0, inner_iters=6)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)
rng = np.random.default_rng(7)

s_in = arc._onehot(arc.sample_input()).astype(float)
c0 = arc._cond(0)
model.set_condition(c0)
model.reset_states()
model.prepare_next(action=None)
L = model.L
for it in range(6):
    preds = model.forward()
    errs, up = model.backward_error(s_in, preds)
    upn = [f"{np.linalg.norm(u):.3f}" for u in up[1:]]
    x_nz = [int(np.count_nonzero(x)) for x in model.xs[1:]]
    cfg = model.cfg
    u2 = cfg.alpha * (preds[2] - model.xs[2]) + cfg.beta * up[2]
    print(f"it{it}: |up|={upn} x_nz={x_nz} "
          f"u2_max={np.abs(u2).max():.4f} u2_nz={(np.abs(u2) > cfg.theta_event).sum()}")
    evs = model.update_states(preds, up)
