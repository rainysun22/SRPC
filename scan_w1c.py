#!/usr/bin/env python3
"""扫描 inner_iters + 是否去掉 cond/memory 对 xL 的影响与符号精度。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential


def one(inner_iters, beta_cond, gamma_mem, kwta, seed=0, mem=True):
    dcfg = replace(DeepConfig(), inner_iters=inner_iters, beta_cond=beta_cond,
                   gamma_mem=gamma_mem, kwta_frac=kwta)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, mem, dcfg, mcfg, ArcConfig(), CLConfig())
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model = res["model"]
    model.set_learning(False)
    rng = np.random.default_rng(7)
    c0 = arc._cond(0)
    accs = {}
    for name in arc.train_names:
        ca = []
        for _ in range(60):
            g_in = arc.sample_input()
            v = model.apply_transform(arc._onehot(g_in).astype(float),
                                      arc._cond(arc.train_names.index(name)))
            g_pred = arc.decode_grid(v).ravel()
            ca.append(float(np.mean(g_pred == TRANSFORMS[name](g_in).ravel())))
        accs[name] = float(np.mean(ca))
    def enc(s):
        model.set_condition(c0); model.reset_states()
        model.prepare_next(action=None); model.observe(s)
        return model.xs[model.L].copy()
    xr = enc(arc._onehot(arc.sample_input()).astype(float))
    xz = enc(np.zeros(256))
    d = np.linalg.norm(xr - xz)
    mean = float(np.mean(list(accs.values())))
    print(f"it={inner_iters:2d} bc={beta_cond:.2f} gm={gamma_mem:.2f} kw={kwta:.1f} mem={int(mem)} "
          f"| mean={mean:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in accs.items())
          + f" | dxL={d:.4f} xL_nz={np.count_nonzero(xr)}")
    return mean


if __name__ == "__main__":
    one(3, 0.0, 0.0, 0.5)
    one(5, 0.0, 0.0, 0.5)
    one(6, 0.0, 0.0, 0.5)
    one(8, 0.0, 0.0, 0.5)
    one(12, 0.0, 0.0, 0.5)
    one(6, 0.05, 0.0, 0.5)
    one(6, 0.8, 0.0, 0.5)
    one(6, 0.8, 0.35, 0.5)
