#!/usr/bin/env python3
"""实测 flip_h 输出质量：全格/符号格精度、是否真的翻转、S2 探针输出。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, with_memory=True, dcfg=dcfg,
                     mcfg=mcfg, acfg=ArcConfig(), clcfg=CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)
rng = np.random.default_rng(0)

cond = arc._cond(0)  # flip_h
print("=== flip_h 样本输出（前 3 个） ===")
for k in range(3):
    g_in = arc.sample_input()
    g_true = TRANSFORMS["flip_h"](g_in)
    v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
    g_pred = arc.decode_grid(v)
    all_acc = np.mean(g_pred.ravel() == np.asarray(g_true).ravel())
    sym = g_true.ravel() != 0
    sym_acc = np.mean(g_pred.ravel()[sym] == np.asarray(g_true).ravel()[sym])
    nz_pred = np.count_nonzero(g_pred); nz_true = np.count_nonzero(g_true)
    print(f"  in:\n{g_in}\n  true:\n{g_true}\n  pred:\n{g_pred}\n  all={all_acc:.3f} sym={sym_acc:.3f} nz {nz_pred}/{nz_true}")

print("\n=== S2 探针：1x1 块在四角 → ŝ 表征 ===")
def enc(grid_int):
    model.apply_transform(arc._onehot(grid_int).astype(float), cond)
    return model._ro_src().copy()
g = arc.grid
pos = []
for (r, cc) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
    gg = np.zeros((g, g), dtype=int); gg[r, cc] = 1
    pos.append(enc(gg))
for i, a in enumerate(pos):
    for b in pos[i + 1:]:
        print(f"  cos={np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9):.4f} "
              f"|a|={np.linalg.norm(a):.3f} nz_a={np.count_nonzero(a)}")

print("\n=== x1 状态检查（同一输入，iter 展开） ===")
s_in = arc._onehot(arc.sample_input()).astype(float)
model.set_condition(cond)
model.reset_states()
model.prepare_next(action=None)
for it in range(1, 7):
    model.cfg = replace(model.cfg, inner_iters=1)
    model.observe(s_in)
    src = model._ro_src()
    print(f"  it{it}: |ŝ|={np.linalg.norm(src):.3f} nz(ŝ)={np.count_nonzero(src)} "
          f"|x1|={np.linalg.norm(model.xs[1]):.3f} cos(ŝ,s)={np.dot(src,s_in)/(np.linalg.norm(src)*np.linalg.norm(s_in)+1e-9):.3f}")
