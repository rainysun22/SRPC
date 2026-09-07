#!/usr/bin/env python3
"""B 收尾对比：默认协议（iters=3）下主验收 + 符号保真（基线偏差口径）。

用工作区当前配置跑（git stash 切换旧/新配置）。输出四里程碑判定 + S1/S2 表。
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.runner_b import (run_sequential, eval_task, eval_combination,
                           evaluate_acceptance_b)
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
CLCFG = CLConfig()
SEEDS = [0, 1, 2]


def s1s2(model, arc, n_sym=60, n_div=20):
    """S1 写方向 + S2 读方向（基线偏差口径）。"""
    model.set_learning(False)
    rng = np.random.default_rng(7)
    cond = arc._cond(0)
    ca, ga, nzc, cc = [], [], [], []
    for _ in range(n_sym):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        gp = arc.decode_grid(v).ravel()
        gt = TRANSFORMS["flip_h"](g_in).ravel()
        ca.append(np.mean(gp == gt)); ga.append(np.all(gp == gt))
        nzc.append(np.count_nonzero(gp) == np.count_nonzero(gt))
        cc.append(set(np.unique(gp)) == set(np.unique(gt)))
    cell, grid = float(np.mean(ca)), float(np.mean(ga))
    nz_cons, color_cons = float(np.mean(nzc)), float(np.mean(cc))

    def enc(g):
        model.apply_transform(arc._onehot(g).astype(float), cond)
        if hasattr(model, "_ro_src"):
            return model._ro_src().copy()
        return model.xs[model.L].copy()   # 旧配置读出源 = 顶层 x_self

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    gg = arc.grid
    base = enc(np.zeros((gg, gg), dtype=int))

    def denc(g):
        return enc(g) - base

    pos = []
    for (r, c) in ((0, 0), (0, gg - 1), (gg - 1, 0), (gg - 1, gg - 1)):
        m = np.zeros((gg, gg), dtype=int); m[r, c] = 1
        pos.append(denc(m))
    pc = max(cos(a, b) for i, a in enumerate(pos) for b in pos[i + 1:])
    col = []
    for cv in (1, 2, 3):
        m = np.zeros((gg, gg), dtype=int); m[1:3, 1:3] = cv
        col.append(denc(m))
    cc_ = max(cos(a, b) for i, a in enumerate(col) for b in col[i + 1:])
    div = [denc(arc.sample_input()) for _ in range(n_div)]
    ok = n_ = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            n_ += 1; ok += (cos(div[i], div[j]) < 0.9)
    return dict(cell=cell, grid=grid, nz_cons=nz_cons, color_cons=color_cons,
                pos_cos=pc, color_cos=cc_, div=ok / n_)


mem_runs = [run_sequential(s, True, DeepConfig(),
                           MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
            for s in SEEDS]
no_runs = [run_sequential(s, False, DeepConfig(),
                          MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
           for s in SEEDS]
combo = {k: float(np.mean([r["combo"][k] for r in mem_runs]))
         for k in mem_runs[0]["combo"]}
acc = evaluate_acceptance_b(mem_runs, no_runs, combo, AcceptanceBConfig())
print("learn:", "P" if acc["learning"]["pass_"] else "F")
print("forget:", "P" if acc["forgetting_mem"]["pass_"] else "F",
      f"(fg={acc['forgetting_mem']['forget_mean']*100:.1f}%)")
print("gain:", "P" if acc["memory_gain"]["pass_"] else "F",
      f"(g={acc['memory_gain']['gain_frac']*100:+.1f}% "
      f"mem={acc['memory_gain']['retain_mem']*100:.1f} "
      f"no={acc['memory_gain']['retain_no']*100:.1f})")
print("combo:", "P" if acc["combination"]["pass_"] else "F",
      f"(+{acc['combination']['gain_frac']*100:.0f}% "
      f"c={acc['combination']['combo_err']:.3f} r={acc['combination']['rand_err']:.3f})")
print("ALL:", "P" if acc["all_pass"] else "F")
for s in SEEDS:
    s1 = s1s2(mem_runs[s]["model"], ArcLite(ACFG, np.random.default_rng(s * 3000 + 2)))
    print(f"seed{s}: cell={s1['cell']:.4f} grid={s1['grid']:.3f} "
          f"nz={s1['nz_cons']:.3f} col_cons={s1['color_cons']:.3f} "
          f"pos={s1['pos_cos']:.3f} col={s1['color_cos']:.3f} div={s1['div']:.3f}")
