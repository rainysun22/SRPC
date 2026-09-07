#!/usr/bin/env python3
"""3 seeds × 完整协议验证 eval 收敛协议（iters, eta）对阶段 B 四项验收的影响。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.runner_b import (run_sequential, eval_task, eval_combination,
                           evaluate_acceptance_b)
from srpc.arc import ArcLite

ACFG = ArcConfig()
CLCFG = CLConfig()
SEEDS = [0, 1, 2]


def eval_protocol(res, ei, ee, seed):
    """在给定 eval 协议下重算 R / diag / combo（训练协议不变）。"""
    model = res["model"]
    arc = ArcLite(ACFG, np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    old = model.cfg
    model.cfg = replace(old, inner_iters=ei, eta_inf=ee)
    try:
        n = arc.n_train
        R = np.zeros((n, n))
        for j in range(n):
            for i in range(j + 1):
                R[i, j] = eval_task(model, arc, arc.train_names[i], CLCFG, seed)
        combo = eval_combination(model, arc, CLCFG, seed)
    finally:
        model.cfg = old
    out = dict(res)
    out["R"] = R
    out["diag"] = np.diag(R)
    out["combo"] = combo
    return out


def s1s2(model, arc, n_sym=60, n_div=20):
    model.set_learning(False)
    rng = np.random.default_rng(7)
    cond = arc._cond(0)
    ca, ga = [], []
    for _ in range(n_sym):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        gp = arc.decode_grid(v).ravel()
        gt = __import__("srpc.arc", fromlist=["TRANSFORMS"]).TRANSFORMS["flip_h"](g_in).ravel()
        ca.append(np.mean(gp == gt)); ga.append(np.all(gp == gt))
    cell, grid = float(np.mean(ca)), float(np.mean(ga))
    def enc(g):
        model.apply_transform(arc._onehot(g).astype(float), cond)
        return model._ro_src().copy()
    gg = arc.grid
    pos = []
    for (r, c) in ((0, 0), (0, gg - 1), (gg - 1, 0), (gg - 1, gg - 1)):
        m = np.zeros((gg, gg), dtype=int); m[r, c] = 1
        pos.append(enc(m))
    pc = max(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
             for i, a in enumerate(pos) for b in pos[i + 1:])
    col = []
    for cv in (1, 2, 3):
        m = np.zeros((gg, gg), dtype=int); m[1:3, 1:3] = cv
        col.append(enc(m))
    cc = max(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
             for i, a in enumerate(col) for b in col[i + 1:])
    div = [enc(arc.sample_input()) for _ in range(n_div)]
    ok = np_ = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            cs = float(np.dot(div[i], div[j]) /
                       (np.linalg.norm(div[i]) * np.linalg.norm(div[j]) + 1e-9))
            np_ += 1; ok += (cs < 0.9)
    return dict(cell=cell, grid=grid, pos_cos=pc, color_cos=cc, div=ok / np_)


mem_runs = [run_sequential(s, True, DeepConfig(),
                           MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
            for s in SEEDS]
no_runs = [run_sequential(s, False, DeepConfig(),
                          MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
           for s in SEEDS]

for ei, ee in ((3, 1.0), (8, 0.5), (10, 0.5), (12, 0.5)):
    mr = [eval_protocol(r, ei, ee, s) for r, s in zip(mem_runs, SEEDS)]
    nr = [eval_protocol(r, ei, ee, s) for r, s in zip(no_runs, SEEDS)]
    combo = {k: float(np.mean([r["combo"][k] for r in mr])) for k in mr[0]["combo"]}
    acc = evaluate_acceptance_b(mr, nr, combo, AcceptanceBConfig())
    s1 = s1s2(mr[0]["model"], ArcLite(ACFG, np.random.default_rng(0 * 3000 + 2)))
    print(f"eval_iters={ei} eta={ee}: "
          f"learn={'P' if acc['learning']['pass_'] else 'F'} "
          f"forget={'P' if acc['forgetting_mem']['pass_'] else 'F'} "
          f"(fg={acc['forgetting_mem']['forget_mean']*100:.1f}%) "
          f"gain={'P' if acc['memory_gain']['pass_'] else 'F'} "
          f"(g={acc['memory_gain']['gain_frac']*100:+.1f}% "
          f"gains={[f'{x*100:+.0f}' for x in acc['memory_gain']['gains']]}) "
          f"combo={'P' if acc['combination']['pass_'] else 'F'} "
          f"(+{acc['combination']['gain_frac']*100:.0f}%) "
          f"ALL={'P' if acc['all_pass'] else 'F'} | "
          f"S1 cell={s1['cell']:.4f} grid={s1['grid']:.3f} "
          f"S2 pos={s1['pos_cos']:.3f} col={s1['color_cos']:.3f} div={s1['div']:.3f}")
