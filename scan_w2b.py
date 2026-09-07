#!/usr/bin/env python3
"""测试 inner_iters 加大后记忆增益是否恢复（能力/免遗忘/组合不退化）。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential, forget_stats, evaluate_acceptance_b, eval_combination
from srpc.arc import ArcLite

for inner in (3, 5, 6, 8):
    dcfg = replace(DeepConfig(), trace_energy=False, inner_iters=inner)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    mem_runs, no_runs = [], []
    combos = []
    for seed in range(2):
        res_m = run_sequential(seed, with_memory=True, dcfg=dcfg,
                               mcfg=mcfg, acfg=ArcConfig(), clcfg=CLConfig())
        res_n = run_sequential(seed, with_memory=False, dcfg=dcfg,
                               mcfg=mcfg, acfg=ArcConfig(), clcfg=CLConfig())
        mem_runs.append(res_m); no_runs.append(res_n)
        arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
        combos.append(res_m["combo"])
    combo = {k: float(np.mean([c[k] for c in combos])) for k in combos[0]}
    acc = evaluate_acceptance_b(mem_runs, no_runs, combo)
    mg = acc["memory_gain"]
    fm = forget_stats(mem_runs[0]["R"], mem_runs[0]["diag"])
    print(f"inner_iters={inner}: mem_gain={mg['gain_frac']*100:.1f}% "
          f"retain_mem={mg['retain_mem']*100:.2f}% retain_no={mg['retain_no']*100:.2f}% "
          f"| forget={fm['forget_mean']*100:.1f}% "
          f"| combo={acc['combination']['gain_frac']*100:.0f}% "
          f"| ALL={acc['all_pass']}")
