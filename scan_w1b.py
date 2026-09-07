#!/usr/bin/env python3
"""隔离实验：无 cond / 无 memory / 无 k-WTA 时 xL 是否携带输入信息？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential


def probe(beta_cond, gamma_mem, kwta, seed=0, mem=True):
    dcfg = replace(DeepConfig(), beta_cond=beta_cond, gamma_mem=gamma_mem,
                   kwta_frac=kwta)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, mem, dcfg, mcfg, ArcConfig(), CLConfig())
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model = res["model"]
    model.set_learning(False)
    rng = np.random.default_rng(7)
    c0 = arc._cond(0)

    def enc(s):
        model.set_condition(c0); model.reset_states()
        model.prepare_next(action=None); model.observe(s)
        return model.xs[model.L].copy()
    xr = enc(arc._onehot(arc.sample_input()).astype(float))
    xz = enc(np.zeros(256))
    d = np.linalg.norm(xr - xz)
    # 符号级精度（flip_h）
    accs = []
    for _ in range(60):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), c0)
        g_pred = arc.decode_grid(v).ravel()
        accs.append(float(np.mean(g_pred == TRANSFORMS["flip_h"](g_in).ravel())))
    print(f"bc={beta_cond:4.2f} gm={gamma_mem:4.2f} kw={kwta:3.1f} mem={int(mem)} "
          f"| |xL(real)-xL(zero)|={d:.4f} xL_nz={np.count_nonzero(xr)} "
          f"flip_h_cell={np.mean(accs):.4f}")


if __name__ == "__main__":
    probe(0.0, 0.0, 1.0)
    probe(0.0, 0.0, 0.5)
    probe(0.05, 0.0, 1.0)
    probe(0.0, 0.35, 1.0)
    probe(0.0, 0.0, 1.0, mem=False)
