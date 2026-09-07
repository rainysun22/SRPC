#!/usr/bin/env python3
"""诊断 S2 坍缩：ŝ=W1@x1 对不同输入是否可分辩（cos 矩阵 + 范数 + 与 s 的重建对齐）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

clcfg = CLConfig()
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                     acfg=ArcConfig(), clcfg=clcfg)
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
model.set_learning(False)
cond0 = arc._cond(0)


def enc(grid_int):
    model.apply_transform(arc._onehot(grid_int).astype(float), cond0)
    return model._ro_src().copy(), model.xs[1].copy()


g = arc.grid
inputs = {}
inputs["empty"] = np.zeros((g, g), dtype=int)
for (r, cc) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
    gg = np.zeros((g, g), dtype=int); gg[r, cc] = 1
    inputs[f"corner({r},{cc})"] = gg
for cv in (1, 2, 3):
    gg = np.zeros((g, g), dtype=int); gg[1:3, 1:3] = cv
    inputs[f"block2x2_c{cv}"] = gg
for i in range(3):
    inputs[f"rand{i}"] = arc.sample_input()

keys = list(inputs)
vecs, x1s = {}, {}
for k in keys:
    v, x1 = enc(inputs[k])
    vecs[k] = v; x1s[k] = x1
    print(f"{k:16s} |s_hat|={np.linalg.norm(v):.3f} nz={np.count_nonzero(v)} "
          f"|x1|={np.linalg.norm(x1):.3f} x1nz={np.count_nonzero(x1)}")

print("\ncos matrix (ŝ):")
for i, a in enumerate(keys):
    row = []
    for b in keys:
        va, vb = vecs[a], vecs[b]
        row.append(f"{np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-9):.3f}")
    print(f"  {a:16s} " + " ".join(row))

# ŝ vs 输入重建对齐：ŝ 是否像 s
s0 = arc._onehot(inputs["corner(0,0)"]).astype(float)
print("\n|s_hat| vs |s|:", np.linalg.norm(vecs["corner(0,0)"]), np.linalg.norm(s0))
print("corner ŝ argmax 网格:")
print(arc.decode_grid(vecs["corner(0,0)"]))
print("corner x1 argmax 对应解码(直接看 ŝ 的大致形状):")
# ŝ 与 s 的余弦（归一化）
sn = s0 / (np.linalg.norm(s0) + 1e-9)
vn = vecs["corner(0,0)"] / (np.linalg.norm(vecs["corner(0,0)"]) + 1e-9)
print("cos(ŝ, s):", np.dot(sn, vn))

# 检查 W1 的结构：是否有大量列根本没学到（接近初始化）
W1 = model.Ws[1]
print("\nW1 shape:", W1.shape, "col norm stats:", np.linalg.norm(W1, axis=0).min(),
      np.linalg.norm(W1, axis=0).max())
