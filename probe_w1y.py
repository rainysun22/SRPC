#!/usr/bin/env python3
"""W1 探针：阻尼+深迭代训练/评估下，记忆是否干净下传至 ŝ 并产生记忆增益。

假说：记忆增益=0 的根因是训练时（iters=3, eta=1.0）记忆拉动经 4 层下传
混乱/到不了 x1 -> 读出头在"无记忆影响的 ŝ"上学习；评估时同样到不了 ->
mem/no_mem 的 ŝ 无差异 -> gain=0。若训练+评估都用 阻尼(eta_inf<1)+深迭代，
记忆应干净下传至 x1/ŝ，读出头在记忆锚定表征上学习 -> 评估时记忆锚定 ŝ
与训练一致 -> gain>0。
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.runner_b import (run_sequential, eval_task, eval_combination,
                           evaluate_acceptance_b, forget_stats)
from srpc.arc import ArcLite

ACFG = ArcConfig(); CLCFG = CLConfig(); SEEDS = [0, 1, 2]


def probe(dcfg, tag):
    mr = [run_sequential(s, True, dcfg, MemoryConfig(d=dcfg.dims[-1]), ACFG, CLCFG)
          for s in SEEDS]
    nr = [run_sequential(s, False, dcfg, MemoryConfig(d=dcfg.dims[-1]), ACFG, CLCFG)
          for s in SEEDS]
    combo = {k: float(np.mean([r["combo"][k] for r in mr])) for k in mr[0]["combo"]}
    acc = evaluate_acceptance_b(mr, nr, combo)
    fm = [forget_stats(r["R"], r["diag"]) for r in mr]
    fn = [forget_stats(r["R"], r["diag"]) for r in nr]
    gains = [(a["retain_mean"] - b["retain_mean"]) / (a["retain_mean"] + 1e-8)
             for a, b in zip(fm, fn)]
    g = acc["memory_gain"]
    print(f"{tag}: learn={acc['learning']['pass_']} "
          f"forget={acc['forgetting_mem']['pass_']} (fg={acc['forgetting_mem']['forget_mean']*100:.1f}%) "
          f"gain={'P' if g['pass_'] else 'F'} (g={g['gain_frac']*100:+.1f}% gains={[f'{x*100:+.0f}' for x in gains]}) "
          f"combo={acc['combination']['pass_']} (+{acc['combination']['gain_frac']*100:.0f}%) "
          f"ALL={acc['all_pass']}")
    return mr


for ei, ee in ((3, 1.0), (4, 0.5), (6, 0.5), (8, 0.4)):
    dcfg = replace(DeepConfig(), eta_inf=ee, inner_iters=ei, trace_energy=False)
    probe(dcfg, f"train&eval iters={ei} eta={ee}")
