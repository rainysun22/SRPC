#!/usr/bin/env python3
"""诊断：1) 记忆为何零贡献（R_mem==R_no 逐位相同）2) S2 区分度为何坍缩。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite

dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
clcfg = CLConfig()
res_m = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                       acfg=ArcConfig(), clcfg=clcfg)
res_n = run_sequential(0, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                       acfg=ArcConfig(), clcfg=clcfg)
model = res_m["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))

print("R_mem==R_no identical:", np.array_equal(res_m["R"], res_n["R"]))

# 记忆内容
mem = res_m["mem"]
print("fast slots:", np.count_nonzero(mem.n_fast), "/", len(mem.n_fast))
print("slow groups:", {g: int(c.sum()) for g, c in mem.n_slow.items()})

# 评估单样本，逐层差异（带/不带记忆的模型分开训，不能用同一 model；
# 用 evaluate 时的 apply_transform 对比 recall 拉动是否改变 xL）
model.set_learning(False)
cond = arc._cond(0)
g_in = arc.sample_input()
s = arc._onehot(g_in).astype(float)
model.set_condition(cond)
model.reset_states()
model.prepare_next(action=None)
# 手动 observe 一步并打印各层
for it in range(3):
    preds = model.forward()
    errs, up = model.backward_error(s, preds)
    evs = model.update_states(preds, up)
    print(f"it{it}: xL_norm={np.linalg.norm(model.xs[4]):.3f} "
          f"xL_nz={np.count_nonzero(model.xs[4])} "
          f"x1_norm={np.linalg.norm(model.xs[1]):.3f} "
          f"x1_nz={np.count_nonzero(model.xs[1])}")

# 检查记忆原型与 xL 的关系
proto = mem.recall(model.xs[4], 0)
print("proto_norm:", np.linalg.norm(proto), "xL_norm:", np.linalg.norm(model.xs[4]))
print("gamma_mem*(proto-xL) norm:", np.linalg.norm(0.35 * (proto - model.xs[4])))

# S2 区分度：读出头源对不同输入的差异
def enc(grid_int):
    model.apply_transform(arc._onehot(grid_int).astype(float), cond)
    return model._ro_src().copy()

g = arc.grid
pos = []
for (r, c) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
    gg = np.zeros((g, g), dtype=int); gg[r, c] = 1
    pos.append(enc(gg))
for i, a in enumerate(pos):
    for b in pos[i + 1:]:
        print(f"pos cos: {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9):.3f} "
              f"|a|={np.linalg.norm(a):.3f}")
