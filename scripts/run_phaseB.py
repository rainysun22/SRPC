#!/usr/bin/env python3
"""SR-PC 阶段 B 主入口：顺序学习（免遗忘）+ ARC-lite 组合泛化 + 验收报告。

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B：
    1. 能力随交互上升（任务内误差下降）
    2. 免遗忘（顺序学习后旧任务误差不回升，多时间尺度记忆对照）
    3. 组合泛化（保留组合零样本：组合嵌入 vs 随机条件）
    4. Pareto 回归（--regression：Phase-0 7.5 快速回归确认 A 指标不退化）

用法：
    python scripts/run_phaseB.py                 # 3 seeds
    python scripts/run_phaseB.py --seeds 0       # 单 seed 快速
    python scripts/run_phaseB.py --steps 200     # 冒烟（缩短每任务步数）

输出：results_phaseB/metrics.json + report.md + figB*.png
退出码：0 = 阶段 B 里程碑全过；1 = 存在未通过项（可证伪信号）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from srpc import (AcceptanceConfig, FieldConfig, ModelConfig, SlotConfig,
                  Track1Config, Track2Config, evaluate_acceptance, run_track1,
                  run_track2)
from srpc.arc import ArcLite
from srpc.config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                         MemoryConfig)
from srpc.plots_b import make_plots_b
from srpc.runner_b import evaluate_acceptance_b, run_sequential


def build_report(mem_runs, no_runs, combo, acc, figs, elapsed, regression=None):
    mf = acc["forgetting_mem"]["forget_mean"]
    ok = "PASS" if acc["all_pass"] else "FAIL"
    lines = [
        "# SR-PC 阶段 B 验收报告（自动生成）",
        "",
        f"总体结论：**{ok}** · 运行耗时 {elapsed:.0f}s · 全程 NumPy 局部规则、免反向传播",
        "",
        "对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B：",
        "",
        "| 里程碑 | 结果 | 关键证据 |",
        "|---|---|---|",
        f"| 1 能力随交互上升 | {'PASS' if acc['learning']['pass_'] else 'FAIL'} | "
        f"任务内误差斜率（跨 seed 均值）：{[f'{s:.2e}' for s in acc['learning']['slopes']]}，"
        f"最差 {acc['learning']['worst']:.2e}（全部 < 0） |",
        f"| 2 免遗忘（带记忆） | {'PASS' if acc['forgetting_mem']['pass_'] else 'FAIL'} | "
        f"末列相对回升均值 {mf*100:.1f}%（max {acc['forgetting_mem']['forget_max']*100:.1f}%） |",
        f"| 3 记忆增益 | {'PASS' if acc['memory_gain']['pass_'] else 'FAIL'} | "
        f"带记忆保留误差 {acc['memory_gain']['retain_mem']*100:.1f}% vs 无记忆 "
        f"{acc['memory_gain']['retain_no']*100:.1f}%（改善 {acc['memory_gain']['gain_frac']*100:.0f}%） |",
        f"| 4 组合零样本 | {'PASS' if acc['combination']['pass_'] else 'FAIL'} | "
        f"组合嵌入 {acc['combination']['combo_err']:.4f} vs 随机条件 "
        f"{acc['combination']['rand_err']:.4f}（增益 {acc['combination']['gain_frac']*100:.0f}%） |",
        "",
        "## 补充指标",
        "",
        f"- 保留组合逐项零样本误差：{ {k: f'{v:.4f}' for k, v in acc['combination']['per_combo'].items()} }",
        f"- 训练变换冻结误差：{ {k: f'{v:.4f}' for k, v in combo.items() if '_' not in k and not k.endswith('_rand')} }",
        f"- 事件驱动更新率（能量代理）：mem {np.mean([np.mean(r['event_rates']) for r in mem_runs]):.3f}"
        f" vs no-mem {np.mean([np.mean(r['event_rates']) for r in no_runs]):.3f}",
    ]
    if regression is not None:
        ok0 = "PASS" if regression["all_pass"] else "FAIL"
        lines += [
            "",
            "## Pareto 回归（4.3 单调性检查：阶段 A 指标不退化）",
            "",
            f"- Phase-0 7.5 快速回归（1 seed 冒烟）：**{ok0}**"
            + ("（A 指标未退化）" if regression["all_pass"] else "（A 指标回撤，需修复）"),
        ]
    lines += ["", "## 图表", ""] + [f"![{k}]({pathlib.Path(v).name})"
                                    for k, v in figs.items()]
    return "\n".join(lines) + "\n"


def forget_stats_help(r):
    from srpc.runner_b import forget_stats
    return forget_stats(r["R"], r["diag"])["forget_mean"]


def main():
    ap = argparse.ArgumentParser(description="SR-PC Phase B")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None, help="覆盖每任务步数（冒烟用）")
    ap.add_argument("--regression", action="store_true",
                    help="跑 Phase-0 快速回归（Pareto 检查）")
    ap.add_argument("--out", default="results_phaseB")
    args = ap.parse_args()

    t_start = time.time()
    dcfg = DeepConfig()
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    acfg = ArcConfig()
    clcfg = CLConfig()
    if args.steps:
        clcfg = CLConfig(steps_per_task=args.steps,
                         eval_samples=12, settle=3)

    seeds = args.seeds
    print(f"[1/4] 顺序学习（带记忆）× {len(seeds)} seeds ...")
    mem_runs = [run_sequential(s, True, dcfg, mcfg, acfg, clcfg) for s in seeds]
    print("[2/4] 顺序学习（无记忆对照）× %d seeds ..." % len(seeds))
    no_runs = [run_sequential(s, False, dcfg, mcfg, acfg, clcfg) for s in seeds]

    print("[3/4] 阶段 B 验收评估 ...")
    combo = {k: float(np.mean([r["combo"][k] for r in mem_runs]))
             for k in mem_runs[0]["combo"]}
    acc = evaluate_acceptance_b(mem_runs, no_runs, combo,
                                AcceptanceBConfig())

    regression = None
    if args.regression:
        print("[3.5/4] Phase-0 快速回归（Pareto 检查）...")
        m0 = ModelConfig()
        f0 = FieldConfig()
        t1 = Track1Config(steps=6000, perturb_step=3000, eps_decay_steps=2000)
        t1r = run_track1(seeds=seeds[:1], mcfg=m0, fcfg=f0, tcfg=t1)
        s0 = SlotConfig()
        t2r = run_track2(seeds=seeds[:1], mcfg=m0, scfg=s0)
        regression = evaluate_acceptance(t1r, t2r, AcceptanceConfig())

    print("[4/4] 绘图与报告 ...")
    out = pathlib.Path(args.out)
    figs = make_plots_b(mem_runs, no_runs, str(out))
    elapsed = time.time() - t_start
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            acceptance=acc,
            combo=combo,
            event_rate_mem=np.mean([np.mean(r["event_rates"]) for r in mem_runs]),
            event_rate_no=np.mean([np.mean(r["event_rates"]) for r in no_runs]),
            forget_mem_mean=np.mean([forget_stats_help(r) for r in mem_runs]),
            forget_no_mean=np.mean([forget_stats_help(r) for r in no_runs]),
            R_mem=[r["R"].tolist() for r in mem_runs],
            R_no=[r["R"].tolist() for r in no_runs],
            regression=regression,
            elapsed_sec=elapsed,
        ), f, ensure_ascii=False, indent=2, default=float)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(build_report(mem_runs, no_runs, combo, acc, figs, elapsed,
                             regression))
    # B 收尾 W1 节（人工整理的架构债务记录，含冻结证据）——存在则附加，重跑不丢失
    w1 = out / "report_w1.md"
    if w1.exists():
        with open(w1, encoding="utf-8") as fsrc:
            w1_txt = fsrc.read()
        with open(out / "report.md", "a", encoding="utf-8") as fdst:
            fdst.write("\n" + w1_txt)

    print("\n===== 阶段 B 验收结论 =====")
    for k, v in acc.items():
        if k == "all_pass":
            continue
        print(f"  [{'PASS' if v['pass_'] else 'FAIL'}] {k}")
    print(f"总体: {'PASS' if acc['all_pass'] else 'FAIL'}（{elapsed:.0f}s）")
    print(f"结果已写入 {out}/（metrics.json, report.md, figB*.png）")
    return 0 if acc["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
