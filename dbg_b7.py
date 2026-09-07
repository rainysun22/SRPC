#!/usr/bin/env python3
"""定向调试：full 协议(800步)下记忆拉动为何到不了读出。

量化 eval 时 (a) recall 拉动力度 (b) 逐层 x 差异 mem vs no (c) 读出差。
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite

ACFG = ArcConfig()
CLCFG = CLConfig()  # full: 800 steps, 24 samples

res_m = run_sequential(0, True, DeepConfig(),
                       MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
res_n = run_sequential(0, False, DeepConfig(),
                       MemoryConfig(d=DeepConfig().dims[-1]), ACFG, CLCFG)
mm, nm = res_m["model"], res_n["model"]
arc = ArcLite(ACFG, np.random.default_rng(0 * 3000 + 2))

mem = res_m["mem"]
print("fast slots:", np.count_nonzero(mem.n_fast), "/", len(mem.n_fast))
print("slow groups:", {g: int(c.sum()) for g, c in mem.n_slow.items()})

# 训练时 consolidate 的 x_L 模长（看原型量级）
probe = None
for g, ps in mem.slow.items():
    print(f"slow group {g} proto norms:", np.linalg.norm(ps, axis=1))
    probe = ps[0]

def eval_prot(model, arc, ei, ee, name, verbose=False):
    model.set_learning(False)
    old = model.cfg
    model.cfg = replace(old, inner_iters=ei, eta_inf=ee)
    rng = np.random.default_rng(0 * 977 + 13)
    errs = []
    for m in range(CLCFG.eval_samples):
        s_in, s_out, cond = arc.sample(name)
        model.set_condition(cond)
        model.reset_states()
        model.prepare_next(action=None)
        model.observe(s_in)
        if verbose:
            xL = model.xs[model.L]
            proto = model.memory.recall(xL, model.group()) if model.memory else None
            print(f"  xL_norm={np.linalg.norm(xL):.3f} "
                  f"pull={(np.linalg.norm(proto - xL) * model.cfg.gamma_mem if proto is not None else 0):.4f} "
                  f"proto_norm={np.linalg.norm(proto) if proto is not None else 0:.3f}")
        out = model.readout()
        errs.append(float(np.mean((out - s_out) ** 2)))
    model.cfg = old
    return float(np.mean(errs))

for ei, ee in ((3, 1.0), (10, 0.5)):
    em = eval_prot(mm, arc, ei, ee, "flip_h", verbose=(ei == 10))
    en = eval_prot(nm, arc, ei, ee, "flip_h")
    print(f"protocol ({ei},{ee}): mem_err={em:.4f} no_err={en:.4f} gain={100*(en-em)/en:+.1f}%")
