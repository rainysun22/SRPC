#!/usr/bin/env python3
"""读出头学习消融：colnorm / mask / gate 的影响。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), inner_iters=6, beta_cond=0.0, gamma_mem=0.0,
               fan_in_ro_frac=0.0)  # 关闭读出头 mask
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(True)

name = arc.train_names[0]
cond = arc._cond(0)
model.set_condition(cond)
model.W_outs[0] = None
model.d_out = 0

# 手动训练，去掉 colnorm
model.W_outs[0] = model.rng.uniform(0.0, 0.5, (256, model.d_self))
model.d_out = 256
for t in range(800):
    s_in, s_out, c = arc.sample(name)
    model.set_condition(cond)
    model.reset_states()
    model.prepare_next(action=None)
    model.observe(s_in)
    model.learn(s_in)
    e = s_out - (model.W_outs[0] @ model.xs[model.L])
    g = model.xs[model.L] > 1e-2
    model.W_outs[0] += 0.08 * np.outer(e, model.xs[model.L] * g)
    # 无 colnorm
    if t % 200 == 0:
        print(f"t={t:4d} e={np.linalg.norm(e):.4f}")
model.set_learning(False)
ca = []
for _ in range(60):
    g_in = arc.sample_input()
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    g_pred = arc.decode_grid(v)
    ca.append(float(np.mean(g_pred == TRANSFORMS[name](g_in).ravel())))
print(f"no-colnorm flip_h cell acc: {np.mean(ca):.4f}")
