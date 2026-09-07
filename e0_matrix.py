#!/usr/bin/env python3
"""E0 实验矩阵：记忆-读出层稳定下传路径三配置对比（3 seeds）。

架构债务 #1：记忆先验只拉顶层 x_L、读出源在底层 ŝ=W1@x1、中间迭代动力学
不下传（mem/no_mem 读出头权重逐元素相同）→ 记忆增益 0%。

配置矩阵（docs/ROADMAP.md E0，2026-09-07 拍板）：
  base = recon 源 + 仅顶层记忆（当前主线，对照）
  b1   = 读出源切回顶层 x_L（ro_recon_mode="self"，RLS 保留）——耦合零传播延迟
  b2   = 双源读出 concat[ŝ, x_L]（ro_recon_mode="dual"）——保真走 ŝ、耦合走 x_L
  a    = recon 源 + x2 级第二记忆（extra_mems[2]）——先验下移，绕开深层传播
  ab   = b2 + a 组合（双源 + x2 记忆）

判定（验收）：记忆增益 ≥10% 且 S1/S2 在线阈值（v2）不回退。
"""
import json
import sys
import time

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.arc import ArcLite
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.memory import PrototypeMemory
from srpc.runner_b import evaluate_acceptance_b, run_sequential
from verify_w1 import sym_fidelity

SEEDS = (0, 1, 2)
# W1 收尾阈值 v2（在线可达，见 results_phaseB/report_w1.md）
TH = dict(cell=0.92, ambig=0.08, pos=0.70, color=0.97, div=0.90, gain=0.10)


def hook_x1(model):
    """E0-a：x1 级附加记忆（读出源 ŝ=W1@x1 的直接上游，零传播距离）。

    插桩实证（2026-09-07）：x2 是 3 迭代协议的动力学死角（激活时序
    iter1→x1、iter2→x2、iter3→x3/x4 混入输入；x2 寿命 2 迭代且评估时
    近零）→ 挂 x2 的记忆拉动精确为零（a≡base 逐位相同）。改挂 x1：
    迭代 1 即激活，记忆先验经 W1 直接进读出源 —— PC 感知折中
    （感知 = 先验 + 证据）的原理性实现；风险 = S1/S2 输入保真回退（实验测）。
    """
    model.extra_mems[1] = PrototypeMemory(
        MemoryConfig(d=model.dims[1]), model.rng)


CONFIGS = [
    ("base", dict(dcfg=DeepConfig(), hook=None)),
    ("b1",   dict(dcfg=DeepConfig(ro_recon_mode="self"), hook=None)),
    ("b2",   dict(dcfg=DeepConfig(ro_recon_mode="dual"), hook=None)),
    ("a",    dict(dcfg=DeepConfig(), hook=hook_x1)),
    ("ab",   dict(dcfg=DeepConfig(ro_recon_mode="dual"), hook=hook_x1)),
]


def main() -> None:
    t0 = time.time()
    results = {}
    for name, spec in CONFIGS:
        dcfg, hook = spec["dcfg"], spec["hook"]
        mcfg = MemoryConfig(d=dcfg.dims[-1])
        mem_runs, no_runs = [], []
        for s in SEEDS:
            mem_runs.append(run_sequential(s, True, dcfg, mcfg,
                                           ArcConfig(), CLConfig(),
                                           model_hook=hook))
            # 对照 = 零记忆（a 路线的 no_mem 不挂 x2 记忆，保持干净配对）
            no_runs.append(run_sequential(s, False, dcfg, mcfg,
                                          ArcConfig(), CLConfig(),
                                          model_hook=None))
        combo = {k: float(np.mean([r["combo"][k] for r in mem_runs]))
                 for k in mem_runs[0]["combo"]}
        acc = evaluate_acceptance_b(mem_runs, no_runs, combo,
                                    AcceptanceBConfig())
        # S1/S2 符号保真（v2 阈值，mem 模型逐 seed）
        sym = [sym_fidelity(r["model"],
                            ArcLite(ArcConfig(),
                                    np.random.default_rng(s * 3000 + 2)),
                            CLConfig(), s)
               for r, s in zip(mem_runs, SEEDS)]
        row = dict(
            gain=acc["memory_gain"]["gain_frac"],
            gain_pass=acc["memory_gain"]["pass_"],
            gain_seeds=acc["memory_gain"]["gains"],
            retain_mem=acc["memory_gain"]["retain_mem"],
            retain_no=acc["memory_gain"]["retain_no"],
            forget=acc["forgetting_mem"]["forget_mean"],
            combo=acc["combination"]["gain_frac"],
            learn=acc["learning"]["pass_"],
            all_main=acc["all_pass"],
            s1_cell=min(s_["s1"]["cell_min"] for s_ in sym),
            s1_ambig=max(s_["s1"]["ambig_max"] for s_ in sym),
            s1_grid=float(np.mean([s_["s1"]["grid_mean"] for s_ in sym])),
            s2_pos=max(s_["s2"]["pos_cos_max"] for s_ in sym),
            s2_color=max(s_["s2"]["color_cos_max"] for s_ in sym),
            s2_div=float(np.mean([s_["s2"]["div_disc"] for s_ in sym])),
        )
        row["s1_pass"] = (row["s1_cell"] >= TH["cell"]
                          and row["s1_ambig"] <= TH["ambig"])
        row["s2_pass"] = (row["s2_pos"] <= TH["pos"]
                          and row["s2_color"] <= TH["color"]
                          and row["s2_div"] >= TH["div"])
        row["E0_PASS"] = (row["gain"] >= TH["gain"]
                          and row["s1_pass"] and row["s2_pass"])
        results[name] = row
        print(f"[{name}] gain={row['gain']*100:+.1f}% "
              f"(mem {row['retain_mem']:.4f} vs no {row['retain_no']:.4f}) "
              f"forget={row['forget']*100:.1f}% combo={row['combo']*100:+.0f}% "
              f"| S1 cell={row['s1_cell']:.3f} ambig={row['s1_ambig']:.3f} "
              f"grid={row['s1_grid']:.3f} | S2 pos={row['s2_pos']:.3f} "
              f"col={row['s2_color']:.3f} div={row['s2_div']:.3f} "
              f"| E0={'PASS' if row['E0_PASS'] else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open("/workspace/results_phaseB/e0_matrix.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== E0 矩阵判定（gain>={TH['gain']*100:.0f}% 且 S1/S2 v2 不回退）===")
    for name, row in results.items():
        why = []
        if row["gain"] < TH["gain"]:
            why.append("gain")
        if not row["s1_pass"]:
            why.append("S1")
        if not row["s2_pass"]:
            why.append("S2")
        print(f"  {name}: {'PASS' if row['E0_PASS'] else 'FAIL'}"
              + (f"  <- {'+'.join(why)}" if why else ""))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
