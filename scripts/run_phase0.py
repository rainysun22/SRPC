#!/usr/bin/env python3
"""SR-PC Phase-0 主入口：运行全部实验并生成验收报告（docs/SRPC_DESIGN.md 第 7 节）。

用法：
    python scripts/run_phase0.py                 # 完整协议（3 seeds，~1-2 分钟）
    python scripts/run_phase0.py --seeds 0       # 单 seed 快速运行
    python scripts/run_phase0.py --steps 6000    # 缩短交互长度（仅冒烟）

输出：results/metrics.json + results/report.md + results/fig*.png
退出码：0 = 7.5 四条件全部通过；1 = 存在未通过项（可证伪信号）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from srpc import (AcceptanceConfig, CreditConfig, FieldConfig, ModelConfig,
                  SlotConfig, Track1Config, Track2Config, evaluate_acceptance,
                  run_credit_screen, run_track1, run_track2)
from srpc.plots import make_all


def build_report(t1, t2, cr, acc, figs, elapsed):
    a = acc
    on1, off1 = t1["on"]["metrics_mean"], t1["off"]["metrics_mean"]
    on2 = t2["on"]["metrics_mean"]
    ok = "PASS" if a["all_pass"] else "FAIL"
    lines = [
        "# SR-PC Phase-0 验收报告（自动生成）",
        "",
        f"总体结论：**{ok}** · 运行耗时 {elapsed:.0f}s · 全程无反向传播、在线增量",
        "",
        "对应 docs/SRPC_DESIGN.md 7.5 验收标准：",
        "",
        "| 条件 | 结果 | 关键证据 |",
        "|---|---|---|",
        f"| 1 自组织层级结构 | {'PASS' if a['formation']['pass_'] else 'FAIL'} | "
        f"感受野对齐 {on1['rf_pre_mean']:.3f}（初始 {on1['rf_init_mean']:.3f}），"
        f"捕获率 {a['formation']['rf_captured']*100:.0f}%；"
        f"概念层 NMI {a['formation']['nmi_zone']:.3f}（随机 {on1['nmi_zone_perm']:.3f}）；"
        f"组合流特征捕获率 {a['formation']['rf_feat_captured']*100:.0f}% |",
        f"| 2 误差随交互下降 | {'PASS' if a['correction']['pass_'] else 'FAIL'} | "
        f"EMA 斜率 {a['correction']['slope_pre']:.2e}；"
        f"首/末十分位误差 {a['correction']['decile_first']:.4f} → {a['correction']['decile_last']:.4f} |",
        f"| 3 自省环非零增益 | {'PASS' if a['self_reflection']['pass_'] else 'FAIL'} | "
        f"扰动重学窗口误差 ON {a['self_reflection']['relearn_on']:.4f} vs OFF "
        f"{a['self_reflection']['relearn_off']:.4f}"
        f"（低 {a['self_reflection']['relearn_gain_frac']*100:.0f}%）；"
        f"恢复步数 ON {a['self_reflection']['recovery_on']:.0f} vs OFF "
        f"{a['self_reflection']['recovery_off']:.0f}；"
        f"扰动后稳态误差 ON {a['self_reflection']['steady_post_on']:.4f} vs OFF "
        f"{a['self_reflection']['steady_post_off']:.4f} |",
        f"| 4 无反传·在线 | {'PASS' if a['no_backprop_online']['pass_'] else 'FAIL'} | "
        f"纯 NumPy 局部规则；逐样本在线更新 |",
        f"| 5 信用分配早筛（§8.5 承重墙） | {'PASS' if a['credit_screen']['pass_'] else 'FAIL'} | "
        f"延迟 Δ={cr['metrics_mean']['delay']:.0f}：误差驱动 {a['credit_screen']['acc_pcn']:.3f} vs "
        f"纯相关 {a['credit_screen']['acc_hebb']:.3f}（机会 0.5，分离 "
        f"{a['credit_screen']['acc_gap']:.3f}）；远端权重驱动 "
        f"{a['credit_screen']['distal_pcn']:.3f}（机会 0.20）；"
        f"Δ=1 对照：误差驱动 {a['credit_screen']['acc_pcn_d1']:.3f} vs 纯相关 "
        f"{a['credit_screen']['acc_hebb_d1']:.3f}（两者皆可学）|",
        "",
        "## 补充指标（7.4 组合 / 不变量 3 能量）",
        "",
        f"- 零样本组合泛化：SR-PC 保留组合误差 {on2['zs_srpc_novel']:.4f} "
        f"vs 查表基线 {on2['zs_lookup_novel']:.4f}（训练组合 {on2['zs_srpc_train']:.4f} "
        f"vs {on2['zs_lookup_train']:.4f}）",
        f"- 片段重组：保留组合 x1 重组余弦 {on2['recomb_x1_novel']:.3f}"
        f"（训练组合 {on2['recomb_x1_train']:.3f}），工作空间重组余弦 {on2['recomb_ws_novel']:.3f}",
        f"- 新概念新颖性（x2 模式距离比）：{on2['x2_novelty_ratio']:.2f}",
        f"- 少样本适应（保留组合）：ON 增益 {on2['fewshot_gain']:.4f} vs "
        f"OFF {t2['off']['metrics_mean']['fewshot_gain']:.4f}",
        f"- 事件驱动更新率：前期 {on1['event_rate_first']:.3f} → 末期 {on1['event_rate_last']:.3f}",
        "",
        "## 图表",
        "",
    ] + [f"![{k}]({pathlib.Path(v).name})" for k, v in figs.items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="SR-PC Phase-0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--slot-seeds", type=int, nargs="+", default=None,
                    help="Track-2 seeds（默认取 seeds 前 2 个）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 Track-1 步数（冒烟用）")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t_start = time.time()
    mcfg = ModelConfig()
    fcfg = FieldConfig()
    t1cfg = Track1Config()
    if args.steps:
        t1cfg = Track1Config(steps=args.steps,
                             perturb_step=args.steps // 2,
                             eps_decay_steps=max(1, args.steps // 3))
    scfg = SlotConfig()
    t2cfg = Track2Config()

    print("[1/4] Track-1 交互式导航（形成/修正/自省 A/B）...")
    t1 = run_track1(seeds=args.seeds, mcfg=mcfg, fcfg=fcfg, tcfg=t1cfg)
    print("[2/4] Track-2 组合流（组合泛化 + 基线 + 自省增益）...")
    slot_seeds = args.slot_seeds or args.seeds[:2]
    t2 = run_track2(seeds=slot_seeds, mcfg=mcfg, scfg=scfg, t2cfg=t2cfg)
    print("[3/4] 信用分配早筛（§8.5 承重墙：延迟关联，误差驱动 vs 纯相关）...")
    ccfg = CreditConfig()
    cr = run_credit_screen(seeds=args.seeds, ccfg=ccfg)

    print("[4/4] 验收评估与绘图...")
    acc = evaluate_acceptance(t1, t2, AcceptanceConfig(), cr, ccfg)
    out = pathlib.Path(args.out)
    figs = make_all(t1, t2, t1cfg.perturb_step, str(out))

    elapsed = time.time() - t_start
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            track1={k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                    for k, v in t1.items() if k != "seeds"},
            track2={k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                    for k, v in t2.items() if k != "seeds"},
            credit_screen=cr, acceptance=acc, elapsed_sec=elapsed,
        ), f, ensure_ascii=False, indent=2, default=float)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(build_report(t1, t2, cr, acc, figs, elapsed))

    print("\n===== 7.5 验收结论 =====")
    for k, v in acc.items():
        if k == "all_pass":
            continue
        print(f"  [{'PASS' if v['pass_'] else 'FAIL'}] {k}")
    print(f"总体: {'PASS' if acc['all_pass'] else 'FAIL'}（{elapsed:.0f}s）")
    print(f"结果已写入 {out}/（metrics.json, report.md, fig*.png）")
    return 0 if acc["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
