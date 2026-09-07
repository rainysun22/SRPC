#!/usr/bin/env python3
"""nobg 下重建质量诊断：W1@x1 是否逼近输入；x1 是否死亡；读出头行为。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
SEED = 0


def _onehot_nobg(self, g):
    oh = np.zeros((g.size, self.n_colors))
    for c in range(1, self.n_colors):
        oh[:, c] = (g.ravel() == c).astype(float)
    return oh.ravel()


ArcLite._onehot = _onehot_nobg
try:
    dcfg = DeepConfig()
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(SEED, True, dcfg, mcfg, ACFG,
                         CLConfig(steps_per_task=400, eval_samples=12))
finally:
    from srpc import arc as _arcmod
    ArcLite._onehot = _arcmod.ArcLite._onehot

model = res["model"]
arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
model.set_learning(False)

# 重建质量
rng = np.random.default_rng(3)
recon_errs, corr, act_frac = [], [], []
for _ in range(30):
    g_in = arc.sample_input()
    s = _onehot_nobg(arc, g_in)
    cond = arc._cond(0)
    model.set_condition(cond); model.reset_states(); model.prepare_next(None)
    model.observe(s)
    src = model._ro_src()
    recon_errs.append(float(np.mean((src - s) ** 2)))
    corr.append(float(np.dot(src, s) / (np.linalg.norm(src) * np.linalg.norm(s) + 1e-9)))
    act_frac.append(float(np.count_nonzero(model.xs[1]) / model.xs[1].size))
print(f"recon mse={np.mean(recon_errs):.4f} corr={np.mean(corr):.3f} x1_act={np.mean(act_frac):.3f}")

# 单样本看重建
g_in = arc.sample_input()
s = _onehot_nobg(arc, g_in)
cond = arc._cond(0)
model.apply_transform(s, cond)
src = model._ro_src()
oh_src = src.reshape(64, 4)
print("input grid:"); print(g_in)
print("src argmax:"); print(oh_src.argmax(axis=1).reshape(8, 8))
print("src max val:", oh_src.max(axis=1).reshape(8, 8).round(2))
