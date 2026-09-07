#!/usr/bin/env python3
"""对比 bg / nobg：重建质量 + x1 激活 + 读出头精度（定位 S1 瓶颈）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
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


for bg in (True, False):
    if not bg:
        ArcLite._onehot = _onehot_nobg
    try:
        res = run_sequential(SEED, True, DeepConfig(),
                             MemoryConfig(d=DeepConfig().dims[-1]), ACFG,
                             CLConfig(steps_per_task=400, eval_samples=12))
    finally:
        from srpc import arc as _arcmod
        ArcLite._onehot = _arcmod.ArcLite._onehot
    model = res["model"]
    arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
    model.set_learning(False)
    rng = np.random.default_rng(3)
    recon_err, corr, act, cell = [], [], [], []
    cond = arc._cond(0)
    for _ in range(40):
        g_in = arc.sample_input()
        s = arc._onehot(g_in).astype(float)
        model.apply_transform(s, cond)
        src = model._ro_src()
        recon_err.append(float(np.mean((src - s) ** 2)))
        corr.append(float(np.dot(src, s) / (np.linalg.norm(src) * np.linalg.norm(s) + 1e-9)))
        act.append(float(np.count_nonzero(model.xs[1]) / model.xs[1].size))
        v = model.readout()
        gp = arc.decode_grid(v).ravel()
        gt = TRANSFORMS["flip_h"](g_in).ravel()
        cell.append(float(np.mean(gp == gt)))
    print(f"bg={bg}: recon_mse={np.mean(recon_err):.4f} corr={np.mean(corr):.3f} "
          f"x1_act={np.mean(act):.3f} flip_h_cell={np.mean(cell):.3f}")
    # 读出头权重统计
    w0 = model.W_outs[0]
    print(f"   W_out0 shape={w0.shape} colnorm_min={np.linalg.norm(w0,axis=0).min():.4f} "
          f"colnorm_max={np.linalg.norm(w0,axis=0).max():.4f}")
