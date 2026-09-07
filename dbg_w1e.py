#!/usr/bin/env python3
"""内迭代收敛测试：inner_iters 增大后 xL 是否收敛到输入的线性编码？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), beta_cond=0.0, gamma_mem=0.0, inner_iters=20)
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
for it in range(20):
    preds = model.forward()
    errs, up = model.backward_error(s_in, preds)
    evs = model.update_states(preds, up)
    if it % 5 == 0 or it >= 15:
        nz = [int(np.count_nonzero(x)) for x in model.xs[1:]]
        norms = [f"{np.linalg.norm(x):.3f}" for x in model.xs[1:]]
        print(f"it{it}: nz={nz} |x|={norms}")

# 收敛后 xL 输入区分度
def enc(s):
    model.set_condition(c0); model.reset_states()
    model.prepare_next(action=None); model.observe(s)
    return model.xs[model.L].copy()
xr = enc(arc._onehot(arc.sample_input()).astype(float))
xz = enc(np.zeros(256))
print(f"|xL(real)-xL(zero)|={np.linalg.norm(xr - xz):.4f} xL_nz={np.count_nonzero(xr)}")
# 线性性检查：xL(A+B) vs xL(A)+xL(B) 增量
ga = arc.sample_input(); gb = arc.sample_input()
xa = enc(arc._onehot(ga).astype(float)); xb = enc(arc._onehot(gb).astype(float))
print(f"linearity: |xL(ga+gb) - (xL(ga)+xL(gb))| = "
      f"{np.linalg.norm(enc(arc._onehot(ga).astype(float) + arc._onehot(gb).astype(float)) - (xa + xb)):.4f}")
