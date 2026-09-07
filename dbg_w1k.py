#!/usr/bin/env python3
"""读出头可行性测试：用稳定线性编码 x1 训练读出头，能否达到符号级保真？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), inner_iters=3, beta_cond=0.0, gamma_mem=0.0)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))

name = "flip_h"
cond = arc._cond(0)
model.set_learning(False)


def enc_x1(s):
    model.set_condition(cond); model.reset_states()
    model.prepare_next(action=None); model.observe(s)
    return model.xs[1].copy()


# 用 x1 作为特征训练线性读出（小步长梯度下降，无 colnorm）
rng = np.random.default_rng(5)
W = rng.uniform(0.0, 0.5, (256, 160))
eta = 0.02
Xs, Ys = [], []
for _ in range(3000):
    g_in = arc.sample_input()
    Xs.append(enc_x1(arc._onehot(g_in).astype(float)))
    Ys.append(arc._onehot(TRANSFORMS[name](g_in)).astype(float))
for it in range(60):
    e_all = 0.0
    for x, y in zip(Xs, Ys):
        pred = W @ x
        e = y - pred
        e_all += np.mean(e ** 2)
        W += eta * np.outer(e, x)
    if it % 15 == 0:
        print(f"epoch{it}: mse={e_all / len(Xs):.5f}")
# 符号精度
ca = []
for _ in range(100):
    g_in = arc.sample_input()
    x = enc_x1(arc._onehot(g_in).astype(float))
    g_pred = arc.decode_grid(W @ x)
    ca.append(float(np.mean(g_pred == TRANSFORMS[name](g_in).ravel())))
print(f"x1-feature flip_h cell acc: {np.mean(ca):.4f}")
