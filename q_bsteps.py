#!/usr/bin/env python3
"""实验1：更多训练步对 S1（读出头收敛）与主验收（记忆增益）的影响（bg，seed0）。
实验2：S2 偏差度量（enc(x)-enc(empty)）在 bg 下是否恢复区分度。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.runner_b import run_sequential, evaluate_acceptance_b
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
SEED = 0


def s1s2(model, arc, n_sym=50, n_div=20):
    model.set_learning(False)
    rng = np.random.default_rng(7)
    out = {}
    for name in arc.train_names:
        cond = arc._cond(arc.train_names.index(name))
        ca, ga = [], []
        for _ in range(n_sym):
            g_in = arc.sample_input()
            v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
            gp = arc.decode_grid(v).ravel()
            gt = TRANSFORMS[name](g_in).ravel()
            ca.append(np.mean(gp == gt)); ga.append(np.all(gp == gt))
        out[name] = (float(np.mean(ca)), float(np.mean(ga)))
    cond0 = arc._cond(0)
    empty = np.zeros((arc.grid, arc.grid), dtype=int)
    base = None
    def enc(g, baseline=False):
        model.apply_transform(arc._onehot(g).astype(float), cond0)
        r = model._ro_src().copy()
        return r - base if baseline and base is not None else r
    base = enc(empty)
    g = arc.grid
    pos = []
    for (r, c) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
        m = np.zeros((g, g), dtype=int); m[r, c] = 1
        pos.append(enc(m, baseline=True))
    def mx(vs):
        return max(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                   for i, a in enumerate(vs) for b in vs[i + 1:])
    # raw pos cos（不去基线）
    pr = []
    for (r, c) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
        m = np.zeros((g, g), dtype=int); m[r, c] = 1
        pr.append(enc(m, baseline=False))
    pc_raw = mx(pr)
    pc = mx(pos)
    col = []
    for cv in (1, 2, 3):
        m = np.zeros((g, g), dtype=int); m[1:3, 1:3] = cv
        col.append(enc(m, baseline=True))
    cc = mx(col)
    div = [enc(arc.sample_input(), baseline=True) for _ in range(n_div)]
    ok = np_ = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            cs = float(np.dot(div[i], div[j]) /
                       (np.linalg.norm(div[i]) * np.linalg.norm(div[j]) + 1e-9))
            np_ += 1; ok += (cs < 0.9)
    return dict(per_task=out, pos_raw=pc_raw, pos_base=pc, color_base=cc,
                div_base=ok / np_)


for steps in (800, 1600, 2400):
    cl = CLConfig(steps_per_task=steps, eval_samples=24)
    mr = run_sequential(SEED, True, DeepConfig(),
                        MemoryConfig(d=DeepConfig().dims[-1]), ACFG, cl)
    nr = run_sequential(SEED, False, DeepConfig(),
                        MemoryConfig(d=DeepConfig().dims[-1]), ACFG, cl)
    arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
    combo = {k: float(np.mean([mr["combo"][k]])) for k in mr["combo"]}
    acc = evaluate_acceptance_b([mr], [nr], combo, AcceptanceBConfig())
    s = s1s2(mr["model"], arc)
    pt = " ".join(f"{k}:{v[0]:.3f}/{v[1]:.2f}" for k, v in s["per_task"].items())
    print(f"steps={steps}: learn={'P' if acc['learning']['pass_'] else 'F'} "
          f"forget={'P' if acc['forgetting_mem']['pass_'] else 'F'} "
          f"(fg={acc['forgetting_mem']['forget_mean']*100:.1f}%) "
          f"gain={'P' if acc['memory_gain']['pass_'] else 'F'} "
          f"(g={acc['memory_gain']['gain_frac']*100:+.1f}%) "
          f"combo={'P' if acc['combination']['pass_'] else 'F'} "
          f"(+{acc['combination']['gain_frac']*100:.0f}%)")
    print(f"   S1 [{pt}]")
    print(f"   S2 pos_raw={s['pos_raw']:.3f} pos_base={s['pos_base']:.3f} "
          f"color_base={s['color_base']:.3f} div_base={s['div_base']:.3f}")
