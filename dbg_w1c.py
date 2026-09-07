#!/usr/bin/env python3
"""深层探测：底层表征是否携带输入信息？errs/up 各层幅度？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential


def probe(seed=0, beta_cond=0.05, kwta_frac=1.0):
    dcfg = replace(DeepConfig(), beta_cond=beta_cond, kwta_frac=kwta_frac)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    model = res["model"]
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    rng = np.random.default_rng(7)

    print(f"=== beta_cond={beta_cond} kwta={kwta_frac} ===")
    for tag, s_in in (
        ("real", arc._onehot(arc.sample_input()).astype(float)),
        ("zero", np.zeros(256)),
        ("noise", rng.uniform(0, 1, 256)),
    ):
        c0 = arc._cond(0)
        model.set_condition(c0)
        model.reset_states()
        model.prepare_next(action=None)
        model.observe(s_in)
        L = model.L
        xs = model.xs
        preds = model.forward()
        errs, up = model.backward_error(s_in, preds)
        e0 = float(np.linalg.norm(errs[0]))
        nz = [int(np.count_nonzero(xs[l])) for l in range(1, L + 1)]
        ups = [float(np.linalg.norm(up[l])) for l in range(1, L + 1)]
        print(f"  {tag:6s} e0={e0:7.3f} nz={nz} |up|={[f'{u:6.3f}' for u in ups]}")
    # x[L] 区分度：real vs zero
    c0 = arc._cond(0)
    def enc(s):
        model.set_condition(c0); model.reset_states()
        model.prepare_next(action=None); model.observe(s)
        return model.xs[model.L].copy()
    xr = enc(arc._onehot(arc.sample_input()).astype(float))
    xz = enc(np.zeros(256))
    d = np.linalg.norm(xr - xz)
    print(f"  |xL(real)-xL(zero)|={d:.4f} | xL(real) nz={np.count_nonzero(xr)} "
          f"|xL|={np.linalg.norm(xr):.4f}")


if __name__ == "__main__":
    probe(0, 0.05, 1.0)
    probe(0, 0.05, 0.5)
