#!/usr/bin/env python3
"""full 协议下空背景编码 on/off 对 S1/S2 + 主验收的影响（seed 0 定向）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.runner_b import run_sequential, evaluate_acceptance_b
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
CLCFG = CLConfig()   # full: 800 steps, 24 samples
SEED = 0


def _onehot_nobg(self, g):
    oh = np.zeros((g.size, self.n_colors))
    for c in range(1, self.n_colors):
        oh[:, c] = (g.ravel() == c).astype(float)
    return oh.ravel()


def quick(seed, with_memory, bg):
    if not bg:
        ArcLite._onehot = _onehot_nobg
    try:
        dcfg = DeepConfig()
        mcfg = MemoryConfig(d=dcfg.dims[-1])
        return run_sequential(seed, with_memory, dcfg, mcfg, ACFG, CLCFG)
    finally:
        from srpc import arc as _arcmod
        ArcLite._onehot = _arcmod.ArcLite._onehot


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
    def enc(g):
        model.apply_transform(arc._onehot(g).astype(float), arc._cond(0))
        return model._ro_src().copy()
    g = arc.grid
    pos = []
    for (r, c) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
        m = np.zeros((g, g), dtype=int); m[r, c] = 1
        pos.append(enc(m))
    pc = max(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
             for i, a in enumerate(pos) for b in pos[i + 1:])
    col = []
    for cv in (1, 2, 3):
        m = np.zeros((g, g), dtype=int); m[1:3, 1:3] = cv
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
    return dict(per_task=out, pos_cos=pc, color_cos=cc, div=ok / np_)


for bg in (True, False):
    mr = quick(SEED, True, bg)
    nr = quick(SEED, False, bg)
    arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
    combo = {k: float(np.mean([mr["combo"][k]])) for k in mr["combo"]}
    acc = evaluate_acceptance_b([mr], [nr], combo, AcceptanceBConfig())
    s = s1s2(mr["model"], arc)
    pt = " ".join(f"{k}:{v[0]:.3f}/{v[1]:.2f}" for k, v in s["per_task"].items())
    print(f"bg={bg}: learn={'P' if acc['learning']['pass_'] else 'F'} "
          f"forget={'P' if acc['forgetting_mem']['pass_'] else 'F'} "
          f"(fg={acc['forgetting_mem']['forget_mean']*100:.1f}%) "
          f"gain={'P' if acc['memory_gain']['pass_'] else 'F'} "
          f"(g={acc['memory_gain']['gain_frac']*100:+.1f}%) "
          f"combo={'P' if acc['combination']['pass_'] else 'F'} "
          f"(+{acc['combination']['gain_frac']*100:.0f}%) "
          f"ALL={'P' if acc['all_pass'] else 'F'}")
    print(f"   S1 [{pt}]")
    print(f"   S2 pos={s['pos_cos']:.3f} color={s['color_cos']:.3f} div={s['div']:.3f}")
