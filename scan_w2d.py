#!/usr/bin/env python3
"""扫描：每任务训练步数对符号保真的影响（RLS 收敛）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

for steps in (800, 2000, 4000):
    clcfg = CLConfig(steps_per_task=steps)
    dcfg = replace(DeepConfig(), trace_energy=False)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                         acfg=ArcConfig(), clcfg=clcfg)
    model = res["model"]
    arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
    model.set_learning(False)
    rng = np.random.default_rng(5)
    ca, ga = [], []
    for name in arc.train_names:
        cond = arc._cond(arc.train_names.index(name))
        accs, grs = [], []
        for _ in range(60):
            g_in = arc.sample_input()
            v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
            g_pred = arc.decode_grid(v).ravel()
            g_true = np.asarray(TRANSFORMS[name](g_in)).ravel()
            accs.append(float(np.mean(g_pred == g_true)))
            grs.append(float(np.all(g_pred == g_true)))
        ca.append(np.mean(accs)); ga.append(np.mean(grs))
    print(f"steps={steps}: allcell={np.mean(ca):.4f} (min {min(ca):.4f}) "
          f"grid={np.mean(ga):.4f} (min {min(ga):.4f})")
