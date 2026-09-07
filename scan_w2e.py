#!/usr/bin/env python3
"""扫描 alpha / beta_cond：减小编码漏斗 vs 验收不退化。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential, evaluate_acceptance_b
from srpc.arc import ArcLite, TRANSFORMS

def s1_s2(model, arc, seed):
    model.set_learning(False)
    cond0 = arc._cond(0)
    g = arc.grid
    def enc(grid_int):
        model.apply_transform(arc._onehot(grid_int).astype(float), cond0)
        return model._ro_src().copy()
    pos = [enc(np.asarray([[1 if (r, c) == cc else 0 for c in range(g)]
                           for r in range(g)]))
           for cc in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1))]
    pos_cos = max(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                  for i, a in enumerate(pos) for b in pos[i + 1:])
    col = []
    for cval in (1, 2, 3):
        gg = np.zeros((g, g), dtype=int); gg[1:3, 1:3] = cval
        col.append(enc(gg))
    col_cos = max(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                  for i, a in enumerate(col) for b in col[i + 1:])
    div = [enc(arc.sample_input()) for _ in range(20)]
    n_ok = n_pair = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            cs = np.dot(div[i], div[j]) / (np.linalg.norm(div[i]) * np.linalg.norm(div[j]) + 1e-9)
            n_pair += 1; n_ok += (cs < 0.9)
    # S1: 训练变换
    ca, ga, nzc = [], [], []
    for name in arc.train_names:
        cond = arc._cond(arc.train_names.index(name))
        accs, grs, nzs = [], [], []
        for _ in range(40):
            g_in = arc.sample_input()
            v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
            gp = arc.decode_grid(v).ravel()
            gt = np.asarray(TRANSFORMS[name](g_in)).ravel()
            accs.append(float(np.mean(gp == gt)))
            grs.append(float(np.all(gp == gt)))
            nzs.append(float(np.count_nonzero(gp) == np.count_nonzero(gt)))
        ca.append(np.mean(accs)); ga.append(np.mean(grs)); nzc.append(np.mean(nzs))
    return dict(pos_cos=pos_cos, col_cos=col_cos, div=float(n_ok / n_pair),
                cell=float(np.mean(ca)), grid=float(np.mean(ga)),
                nz=float(np.mean(nzc)))

for alpha, beta_cond in ((0.40, 0.80), (0.25, 0.80), (0.15, 0.80),
                         (0.40, 0.40), (0.40, 0.15), (0.15, 0.15)):
    dcfg = replace(DeepConfig(), trace_energy=False, alpha=alpha, beta_cond=beta_cond)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    mem_runs, no_runs, s1s = [], [], []
    for seed in range(2):
        res_m = run_sequential(seed, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                               acfg=ArcConfig(), clcfg=CLConfig())
        res_n = run_sequential(seed, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                               acfg=ArcConfig(), clcfg=CLConfig())
        mem_runs.append(res_m); no_runs.append(res_n)
        arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
        s1s.append(s1_s2(res_m["model"], arc, seed))
    combo = {k: float(np.mean([c["combo"][k] for c in mem_runs])) for k in mem_runs[0]["combo"]}
    acc = evaluate_acceptance_b(mem_runs, no_runs, combo)
    m = {k: float(np.mean([s[k] for s in s1s])) for k in s1s[0]}
    slopes_ok = acc["learning"]["pass_"]
    print(f"a={alpha} bc={beta_cond}: cell={m['cell']:.4f} grid={m['grid']:.3f} "
          f"nz={m['nz']:.3f} pos_cos={m['pos_cos']:.3f} col_cos={m['col_cos']:.3f} "
          f"div={m['div']:.3f} | mem_gain={acc['memory_gain']['gain_frac']*100:.1f}% "
          f"forget={acc['forgetting_mem']['forget_mean']*100:.1f}% "
          f"combo={acc['combination']['gain_frac']*100:.0f}% "
          f"learn={slopes_ok} ALL={acc['all_pass']}")
