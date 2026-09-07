#!/usr/bin/env python3
"""假说检验：读出学习期间 x1 非平稳 → 冻结生成权重后继续训练头能否逼近 LS 上限？"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False, ro_norm="clip", eta_wout=0.05)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))


def cell_acc(nm, n=80):
    model.set_learning(False)
    is_novel = nm in arc.novel_names
    ca = []
    for _ in range(n):
        g_in = arc.sample_input()
        oh_in = arc._onehot(g_in).astype(float)
        if is_novel:
            i, j = arc.novel_combos[arc.novel_names.index(nm)]
            out1 = model.apply_transform(oh_in, arc._cond(arc.train_names.index(i)))
            v = model.apply_transform(out1, arc._cond(arc.train_names.index(j)))
            g_true = TRANSFORMS[j](TRANSFORMS[i](g_in))
        else:
            v = model.apply_transform(oh_in, arc._cond(arc.train_names.index(nm)))
            g_true = TRANSFORMS[nm](g_in)
        g_pred = arc.decode_grid(v).ravel()
        ca.append(float(np.mean(g_pred == np.asarray(g_true).ravel())))
    return float(np.mean(ca))


print("after train:", {nm: round(cell_acc(nm, 40), 3) for nm in arc.train_names})

# 阶段 2：冻结生成权重（只训头），每任务再走 head_only 步
model.set_learning(True)
for name in arc.train_names:
    cond = arc._cond(arc.train_names.index(name))
    for t in range(2000):
        s_in, s_out, _ = arc.sample(name)
        model.set_condition(cond)
        model.reset_states()
        model.prepare_next(action=None)
        model.observe(s_in)                       # 感知（不 learn()，生成权重冻结）
        model.learn_readout(s_out)                # 只训当前任务头
    print(f"after head-only +{2000} on {name}:",
          {nm: round(cell_acc(nm, 40), 3) for nm in arc.train_names})
