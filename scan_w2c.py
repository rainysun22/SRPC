#!/usr/bin/env python3
"""扫描：输入缩放（增强底-上驱动）对符号保真 + 记忆增益的影响。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential, forget_stats, evaluate_acceptance_b
from srpc.arc import ArcLite

def sym_fast(model, arc, seed, scale):
    model.set_learning(False)
    rng = np.random.default_rng(seed * 771 + 3)
    cond0 = arc._cond(0)
    g = arc.grid
    def enc(grid_int):
        model.apply_transform((arc._onehot(grid_int) * scale).astype(float), cond0)
        return model._ro_src().copy()
    pos = []
    for (r, cc) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
        gg = np.zeros((g, g), dtype=int); gg[r, cc] = 1
        pos.append(enc(gg))
    pos_cos = max(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                  for i, a in enumerate(pos) for b in pos[i + 1:])
    col = []
    for cval in range(1, arc.n_colors):
        gg = np.zeros((g, g), dtype=int); gg[1:3, 1:3] = cval
        col.append(enc(gg))
    col_cos = max(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                  for i, a in enumerate(col) for b in col[i + 1:])
    # flip_h 符号格精度
    from srpc.arc import TRANSFORMS
    cond = arc._cond(0)
    ca, al = [], []
    for _ in range(60):
        g_in = arc.sample_input()
        v = model.apply_transform((arc._onehot(g_in) * scale).astype(float), cond)
        g_pred = arc.decode_grid(v).ravel()
        g_true = np.asarray(TRANSFORMS["flip_h"](g_in)).ravel()
        sym = g_true != 0
        ca.append(float(np.mean(g_pred[sym] == g_true[sym])))
        al.append(float(np.mean(g_pred == g_true)))
    return dict(pos_cos=pos_cos, col_cos=col_cos, sym=float(np.mean(ca)),
                allc=float(np.mean(al)))

for scale in (1.0, 2.0, 4.0, 8.0):
    dcfg = replace(DeepConfig(), trace_energy=False)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    mem_runs, no_runs = [], []
    s1s = []
    for seed in range(2):
        res_m = run_sequential(seed, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                               acfg=ArcConfig(), clcfg=CLConfig())
        res_n = run_sequential(seed, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                               acfg=ArcConfig(), clcfg=CLConfig())
        mem_runs.append(res_m); no_runs.append(res_n)
        arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
        s1s.append(sym_fast(res_m["model"], arc, seed, scale))
    combo = {k: float(np.mean([c["combo"][k] for c in mem_runs])) for k in mem_runs[0]["combo"]}
    acc = evaluate_acceptance_b(mem_runs, no_runs, combo)
    mg = acc["memory_gain"]
    s1m = {k: float(np.mean([s[k] for s in s1s])) for k in s1s[0]}
    print(f"scale={scale}: sym={s1m['sym']:.3f} allc={s1m['allc']:.3f} "
          f"pos_cos={s1m['pos_cos']:.3f} col_cos={s1m['col_cos']:.3f} "
          f"| mem_gain={mg['gain_frac']*100:.1f}% "
          f"| forget={acc['forgetting_mem']['forget_mean']*100:.1f}% "
          f"| combo={acc['combination']['gain_frac']*100:.0f}% "
          f"| ALL={acc['all_pass']}")
