#!/usr/bin/env python3
"""逐迭代追踪：空输入下 x1..x4 从哪来。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite

clcfg = CLConfig()
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                     acfg=ArcConfig(), clcfg=clcfg)
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
model.set_learning(False)
cond0 = arc._cond(0)
empty = np.zeros((arc.grid, arc.grid), dtype=int)
s = arc._onehot(empty).astype(float)

model.set_condition(cond0)
model.reset_states()
model.prepare_next(action=None)
print("pred_self norm:", np.linalg.norm(model.pred_self),
      "nz:", np.count_nonzero(model.pred_self))
for it in range(1, 6):
    preds = model.forward()
    errs, up = model.backward_error(s, preds)
    evs = model.update_states(preds, up)
    print(f"iter{it}: x1={np.linalg.norm(model.xs[1]):.3f} "
          f"x2={np.linalg.norm(model.xs[2]):.3f} "
          f"x3={np.linalg.norm(model.xs[3]):.3f} "
          f"x4={np.linalg.norm(model.xs[4]):.3f} "
          f"|sh|={np.linalg.norm(model._ro_src()):.3f}")
print("gamma_mem:", model.cfg.gamma_mem)
print("alpha:", model.cfg.alpha, "beta:", model.cfg.beta, "beta_cond:", model.cfg.beta_cond)
