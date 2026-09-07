#!/usr/bin/env python3
"""内迭代跟踪：每迭代后各层 |x|、非零数，定位 x2 死亡 + xL 输入无关的机制。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), beta_cond=0.05, kwta_frac=1.0)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

s_in = arc._onehot(arc.sample_input()).astype(float)
c0 = arc._cond(0)
model.set_condition(c0)
model.reset_states()
model.prepare_next(action=None)
L = model.L
print(f"init: xs={[int(np.count_nonzero(x)) for x in model.xs[1:]]} "
      f"pred_self_nz={np.count_nonzero(model.pred_self)} "
      f"|pred_self|={np.linalg.norm(model.pred_self):.4f}")
for it in range(model.cfg.inner_iters):
    preds = model.forward()
    errs, up = model.backward_error(s_in, preds)
    evs = model.update_states(preds, up)
    nz = [int(np.count_nonzero(x)) for x in model.xs[1:]]
    norms = [f"{np.linalg.norm(x):.3f}" for x in model.xs[1:]]
    upn = [f"{np.linalg.norm(u):.3f}" for u in up[1:]]
    print(f"it{it}: nz={nz} |x|={norms} |up|={upn} ev={[f'{e:.3f}' for e in evs]}")
# 末尾状态区分度
v = model.readout()
print(f"|readout|={np.linalg.norm(v):.3f} nz={np.count_nonzero(v)}")
