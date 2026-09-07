#!/usr/bin/env python3
"""B 收尾核心实验：推断步长 eta_inf × inner_iters 对（稳定性 / S1 保真 / 记忆增益）。
背景：x1 更新是全步长梯度下降，W1^TW1 大特征值导致 3+ 迭代发散；
顶层条件/记忆信号每迭代只向下传 1 层，inner_iters=3 < L=4 到不了 x1 -> 记忆增益=0。
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS
from srpc.deepmodel import DeepSRPC

ACFG = ArcConfig()


def quick(seed, with_memory, dcfg):
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    clcfg = CLConfig(steps_per_task=300)
    res = run_sequential(seed, with_memory=with_memory, dcfg=dcfg,
                         mcfg=mcfg, acfg=ACFG, clcfg=clcfg)
    return res


def s1_flip(model, arc, n=60):
    """冻结 flip_h 的 cell/grid 精度（seed 固定）。"""
    model.set_learning(False)
    rng = np.random.default_rng(7)
    cond = arc._cond(0)
    ca, ga = [], []
    for _ in range(n):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        gp = arc.decode_grid(v).ravel()
        gt = TRANSFORMS["flip_h"](g_in).ravel()
        ca.append(np.mean(gp == gt)); ga.append(np.all(gp == gt))
    return float(np.mean(ca)), float(np.mean(ga))


def mem_gain(res_m, res_n):
    from srpc.runner_b import forget_stats
    fm = forget_stats(res_m["R"], res_m["diag"])
    fn = forget_stats(res_n["R"], res_n["diag"])
    g = (fn["retain_mean"] - fm["retain_mean"]) / (fn["retain_mean"] + 1e-8)
    return g, fm["retain_mean"], fn["retain_mean"]


def lam_max(model):
    W1 = model.Ws[1]
    return float(np.linalg.eigvalsh(W1.T @ W1)[-1])


for eta_inf in (1.0, 0.6, 0.4):
    for iters in (3, 6, 10):
        dcfg = replace(DeepConfig(), trace_energy=False,
                       eta_inf=eta_inf, inner_iters=iters)
        res_m = quick(0, True, dcfg)
        res_n = quick(1, False, dcfg)
        model, arc = res_m["model"], ArcLite(ACFG, np.random.default_rng(0 * 3000 + 2))
        cell, grid = s1_flip(model, arc)
        g, rm, rn = mem_gain(res_m, res_n)
        print(f"eta={eta_inf} iters={iters}: cell={cell:.4f} grid={grid:.4f} "
              f"mem_gain={g*100:+.1f}% (mem={rm:.4f} no={rn:.4f}) "
              f"lam_max(W1^TW1)={lam_max(model):.1f}")
