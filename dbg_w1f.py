#!/usr/bin/env python3
"""阻尼测试：update 加 γ<1 阻尼后 xL 是否收敛为输入的线性编码。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), beta_cond=0.0, gamma_mem=0.0, inner_iters=40)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)
rng = np.random.default_rng(7)

# 手动带阻尼迭代
s_in = arc._onehot(arc.sample_input()).astype(float)
c0 = arc._cond(0)
model.set_condition(c0)
model.reset_states()
model.prepare_next(action=None)
L = model.L
for gamma in (1.0, 0.5, 0.2, 0.1):
    model.reset_states()
    model.prepare_next(action=None)
    for it in range(40):
        preds = model.forward()
        errs, up = model.backward_error(s_in, preds)
        cfg = model.cfg
        for l in range(1, L + 1):
            u = gamma * (cfg.alpha * (preds[l] - model.xs[l]) + cfg.beta * up[l])
            m = np.abs(u) > cfg.theta_event
            model.xs[l] = np.clip(model.xs[l] + u * m, 0.0, cfg.x_max)
    nz = [int(np.count_nonzero(x)) for x in model.xs[1:]]
    norms = [f"{np.linalg.norm(x):.3f}" for x in model.xs[1:]]
    # 输入区分度
    xr = model.xs[L].copy()
    model.reset_states(); model.prepare_next(action=None)
    for it in range(40):
        preds = model.forward()
        errs, up = model.backward_error(np.zeros(256), preds)
        cfg = model.cfg
        for l in range(1, L + 1):
            u = gamma * (cfg.alpha * (preds[l] - model.xs[l]) + cfg.beta * up[l])
            m = np.abs(u) > cfg.theta_event
            model.xs[l] = np.clip(model.xs[l] + u * m, 0.0, cfg.x_max)
    xz = model.xs[L].copy()
    print(f"gamma={gamma}: nz={nz} |x|={norms} |xL(real)-xL(zero)|={np.linalg.norm(xr - xz):.4f}")
