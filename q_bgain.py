#!/usr/bin/env python3
"""记忆增益扫描：eval 收敛迭代数 × eval 阻尼步长（训练保持 iters=3/eta=1.0 稳定）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential, eval_task, forget_stats
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
SEED = 0
CLCFG = CLConfig(steps_per_task=400, eval_samples=12)


def quick(seed, with_memory):
    dcfg = DeepConfig()
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    return run_sequential(seed, with_memory, dcfg, mcfg, ACFG, CLCFG)


def eval_at(model, arc, eval_iters, eval_eta):
    model.set_learning(False)
    model.cfg = replace(model.cfg, inner_iters=eval_iters, eta_inf=eval_eta)
    n = arc.n_train
    R = np.zeros((n, n))
    for j in range(n):
        for i in range(j + 1):
            R[i, j] = eval_task(model, arc, arc.train_names[i], CLCFG, SEED)
    return R


def s1_cell(model, arc, n=40):
    model.set_learning(False)
    rng = np.random.default_rng(7)
    cond = arc._cond(0)
    ca = []
    for _ in range(n):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        ca.append(np.mean(arc.decode_grid(v).ravel()
                          == TRANSFORMS["flip_h"](g_in).ravel()))
    return float(np.mean(ca))


res_m = quick(SEED, True)
res_n = quick(SEED, False)
arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
print("train cfg: iters=%d eta=%.2f" % (res_m["model"].cfg.inner_iters,
                                        res_m["model"].cfg.eta_inf))
print(f"{'e_iters':>7} {'e_eta':>5} | {'mem_ret':>8} {'no_ret':>8} "
      f"{'gain':>7} {'mem_fg':>7} | {'cell_mem':>8} {'cell_no':>8}")
for ei in (3, 5, 6, 8, 10, 12):
    for ee in (1.0, 0.5, 0.3, 0.2):
        Rm = eval_at(res_m["model"], arc, ei, ee)
        Rn = eval_at(res_n["model"], arc, ei, ee)
        fm = forget_stats(Rm, np.diag(Rm))
        fn = forget_stats(Rn, np.diag(Rn))
        g = (fn["retain_mean"] - fm["retain_mean"]) / (fn["retain_mean"] + 1e-8)
        cm = s1_cell(res_m["model"], arc) if (ei, ee) in ((3, 1.0), (6, 0.3), (8, 0.3)) else float("nan")
        cn = s1_cell(res_n["model"], arc) if (ei, ee) in ((3, 1.0), (6, 0.3), (8, 0.3)) else float("nan")
        flag = " <<" if g >= 0.10 and fm["forget_mean"] <= 0.25 else ""
        print(f"{ei:7d} {ee:5.1f} | {fm['retain_mean']:8.4f} {fn['retain_mean']:8.4f} "
              f"{g*100:+6.1f}% {fm['forget_mean']*100:6.1f}% | {cm:8.4f} {cn:8.4f}{flag}")
