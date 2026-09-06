#!/usr/bin/env python3
"""SR-PC 阶段 C 主入口（软件版内在化）：结构稀疏核心 + 能力复验 + 能耗验收报告。

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 C（无专用硬件的软件版）：
    C1 能力无回撤：出生即稀疏核心（分块/扇入受限 + k-WTA）从头重训，
       阶段 B 四项验收全部复现（低功耗来自架构本身，非稠密裁剪）
    C2 能效：每样本推理有效 MACs 比大模型标尺（6.1b）低 >= 10^3 倍
    C3 结构由构造保证：权重密度 <= 35%，掩码自出生不变（学习只改已有突触）
    C4 部署就绪：训练后 int8 量化，冻结评估能力无回撤
       （事件驱动/神经形态芯片部署本体 deferred 至有专用硬件）

用法：
    python scripts/run_phaseC.py                 # 3 seeds
    python scripts/run_phaseC.py --seeds 0       # 单 seed 快速
    python scripts/run_phaseC.py --steps 200     # 冒烟（缩短每任务步数）

输出：results_phaseC/metrics.json + report.md + figC*.png
退出码：0 = 阶段 C 里程碑全过；1 = 存在未通过项（可证伪信号）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from srpc.config import (AcceptanceBConfig, AcceptanceCConfig, ArcConfig,
                         CLConfig, DeepConfig, MemoryConfig, PhaseCConfig)
from srpc.plots_c import make_plots_c
from srpc.runner_c import make_sparse_cfg, run_phase_c


def _fmt(v: float) -> str:
    return f"{v:.3e}" if abs(v) >= 1e4 or (0 < abs(v) < 1e-2) else f"{v:.3f}"


def build_report(res: dict, figs: dict, elapsed: float) -> str:
    acc = res["acceptance"]
    en = res["energy"]
    st = res["structure"]
    q = res["quant"]
    cap = res["capability"]
    b = acc["capability"]["phase_b"]
    ok = "PASS" if acc["all_pass"] else "FAIL"
    lines = [
        "# SR-PC 阶段 C 验收报告（软件版内在化，自动生成）",
        "",
        f"总体结论：**{ok}** · 运行耗时 {elapsed:.0f}s · 全程 NumPy 局部规则、免反向传播",
        "",
        "无专用硬件（CPU/GPU 软件验证）：验证**架构本身**的低功耗——出生即结构稀疏",
        "（分块/扇入受限权重 + k-WTA）的核心从头重训，能力与能效双验收。",
        "硬件部署本体（事件驱动/低比特/神经形态芯片）deferred 至有专用硬件。",
        "",
        "| 里程碑 | 结果 | 关键证据 |",
        "|---|---|---|",
        f"| C1 能力无回撤（阶段 B 复跑） | {'PASS' if acc['capability']['pass_'] else 'FAIL'} | "
        f"稀疏核心上 B 四项：学习 {'PASS' if b['learning']['pass_'] else 'FAIL'} / "
        f"免遗忘 {'PASS' if b['forgetting_mem']['pass_'] else 'FAIL'} / "
        f"记忆增益 {'PASS' if b['memory_gain']['pass_'] else 'FAIL'} / "
        f"组合零样本 {'PASS' if b['combination']['pass_'] else 'FAIL'}"
        f"（组合增益 {b['combination']['gain_frac']*100:.0f}%） |",
        f"| C2 能效（vs 大模型标尺） | {'PASS' if acc['energy']['pass_'] else 'FAIL'} | "
        f"每样本推理 {en['single']['event']:.2e} MACs（事件驱动）vs "
        + " / ".join(f"{k} {v:.2e}" for k, v in res['yardstick']['macs'].items())
        + f"（最低比率 {acc['energy']['ratio_min']:.0e}x，>= 10^3） |",
        f"| C3 结构由构造保证 | {'PASS' if acc['structure']['pass_'] else 'FAIL'} | "
        f"权重总密度 {st['total']['density']:.3f}（{st['total']['nnz']} / "
        f"{st['total']['size']} 突触），掩码自出生不变 = {st['masks_unchanged']} |",
        f"| C4 int8 部署就绪 | {'PASS' if acc['quantization']['pass_'] else 'FAIL'} | "
        f"量化后冻结误差 {q['train_err']:.4f} vs fp {acc['quantization']['fp_single_err']:.4f}；"
        f"组合 {q['combo_err']:.4f} vs fp {acc['quantization']['fp_combo_err']:.4f} |",
        "",
        "## 能耗明细（硬件无关三口径）",
        "",
        "| 口径 | 每样本推理 MACs | 说明 |",
        "|---|---|---|",
        f"| 事件驱动 | {en['single']['event']:.3e} | 活跃单元 × 已有突触（事件驱动硬件真实计算量） |",
        f"| 结构 | {en['single']['struct']:.3e} | 全单元 × 已有突触（结构稀疏硬件保守上界） |",
        f"| 稠密等价 | {en['single']['dense']:.3e} | 同规模稠密网络（稠密硬件真实计算量） |",
        "",
        "**同口径对照（稀疏核心 vs 稠密对照臂，每样本推理 MACs）**：",
        "",
        "| 口径 | 稀疏核心 | 稠密对照臂 | 节省 |",
        "|---|---|---|---|",
    ]
    endn = res["energy_dense"]["single"]
    for tier, nm in (("event", "事件驱动"), ("struct", "结构"), ("dense", "稠密等价")):
        sav = (1 - en["single"][tier] / max(endn[tier], 1e-12)) * 100
        lines.append(f"| {nm} | {en['single'][tier]:.3e} | {endn[tier]:.3e} | {sav:.0f}% |")
    act_sp = en["active_frac"]
    act_dn = res["energy_dense"]["active_frac"]
    struct_sav = (1 - en["single"]["struct"] / max(endn["struct"], 1e-12)) * 100
    dn_silent = [k for k, v in act_dn.items() if v < 1e-9]
    lines += [
        "",
        "口径解读（臂间对比的正确姿势）：",
        "",
        "- **结构口径是架构内在能耗的主度量**（全单元 × 已有突触，不受运行时激活幅值分布影响）："
        f"稀疏核心节省 {struct_sav:.0f}%，由出生即定型掩码保证。",
        f"- 事件口径臂间不可直接比：稠密对照臂内部层事件活跃率近 0（静默层 {dn_silent or '无'}，"
        f"激活幅值低于事件阈值、信息流微弱），而稀疏核心 k-WTA 保证内部层真实信息流"
        f"（活跃率 { {k: f'{v:.2f}' for k, v in act_sp.items()} }）——稀疏核心的事件 MACs 是"
        "\"真实计算\"的代价，稠密臂的低事件值是\"内部静默\"的结果。",
        f"- 事件口径的正确用法：与大模型标尺比（C2，{res['acceptance']['energy']['ratio_min']:.0e}x）"
        f"及与自身稠密等价比（{(1 - en['single']['event'] / max(en['single']['dense'], 1e-12)) * 100:.0f}% 节省）。",
        "",
        f"- 组合零样本（两次变换串联）：{en['combo']['event']:.3e} MACs/样本（事件驱动）",
        f"- 全程训练总能耗（4 任务顺序学习）：稀疏 {res['train_macs']['sparse']['event']:.3e}"
        f" vs 稠密对照 {res['train_macs']['dense']['event']:.3e} MACs",
    ]
    if res['train_macs']['sparse']['event'] > res['train_macs']['dense']['event']:
        lines.append(
            "  （注：训练段事件口径稀疏臂高于稠密臂，原因同上——k-WTA 结构性活跃下限 + 内部层真实传播误差，"
            "稠密臂内部层事件静默且误差幅值低；训练能耗是一次性成本，部署相关的稳态度量是冻结推理。"
            "**结构口径（架构属性）稀疏臂训练亦省 "
            f"{(1 - res['train_macs']['sparse']['struct'] / max(res['train_macs']['dense']['struct'], 1e-12)) * 100:.0f}%。**）")
    else:
        lines.append(
            f"  （架构节省 {(1 - res['train_macs']['sparse']['event'] / max(res['train_macs']['dense']['event'], 1e-12)) * 100:.0f}%）")
    lines += [
        f"- 每层平均活跃率：{ {k: f'{v:.2f}' for k, v in en['active_frac'].items()} }",
        f"- 事件驱动更新率：{cap['event_rates_sparse']:.3f}",
        "",
        "## 结构明细",
        "",
        "| 组件 | 密度 | 突触数（nnz/size） |",
        "|---|---|---|",
    ]
    for k, v in st["layers"].items():
        lines.append(f"| {k} | {v['density']:.3f} | {v['nnz']} / {v['size']}（fan-in {v['fan_in']}） |")
    lines.append(f"| 读出头（×任务） | {st['readout']['density']:.3f} | "
                 f"{st['readout']['nnz']} / {st['readout']['size']} |")
    lines.append(f"| Wdyn（自省环） | {st['dyn']['density']:.3f} | "
                 f"{st['dyn']['nnz']} / {st['dyn']['size']} |")
    lines += [
        "",
        "## 能力对照（稀疏核心 vs 稠密对照，同 seeds）",
        "",
        f"- 保留误差（末列）：稀疏 {[f'{v:.4f}' for v in cap['retain_sparse']]}"
        f" vs 稠密 {[f'{v:.4f}' for v in cap['retain_dense']]}",
        f"- 组合零样本：稀疏 { {k: f'{v:.4f}' for k, v in cap['combo_sparse'].items()} }"
        f" vs 稠密 { {k: f'{v:.4f}' for k, v in cap['combo_dense'].items()} }",
        "",
        "> 大模型标尺（6.1b）仅为比较基准，不进入系统构造（v1.4）。",
        "> MACs 口径：每参数每 token 1 次 MAC；SR-PC 每样本一次映射任务。",
        "",
        "## 图表",
        "",
    ]
    lines += [f"![{k}]({pathlib.Path(v).name})" for k, v in figs.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SR-PC Phase C (software)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖每任务步数（冒烟用）")
    ap.add_argument("--out", default="results_phaseC")
    args = ap.parse_args()

    t_start = time.time()
    dcfg = make_sparse_cfg(DeepConfig())
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    acfg = ArcConfig()
    clcfg = CLConfig()
    pcfg = PhaseCConfig()
    if args.steps:
        clcfg = CLConfig(steps_per_task=args.steps, eval_samples=12, settle=3)

    print(f"[1/3] 稀疏核心顺序学习 + 稠密对照 × {len(args.seeds)} seeds ...")
    res = run_phase_c(args.seeds, dcfg, mcfg, acfg, clcfg, pcfg,
                      AcceptanceCConfig(), AcceptanceBConfig())

    print("[2/3] 绘图 ...")
    out = pathlib.Path(args.out)
    figs = make_plots_c(res, str(out))

    print("[3/3] 报告 ...")
    elapsed = time.time() - t_start
    out.mkdir(parents=True, exist_ok=True)
    metrics = dict(
        acceptance=res["acceptance"],
        energy=dict(single=res["energy"]["single"], combo=res["energy"]["combo"],
                    per_task=res["energy"]["per_task"],
                    combo_per=res["energy"]["combo_per"],
                    single_err=res["energy"]["single_err"],
                    combo_err=res["energy"]["combo_err"],
                    active_frac=res["energy"]["active_frac"],
                    dense_control=dict(single=res["energy_dense"]["single"],
                                       active_frac=res["energy_dense"]["active_frac"])),
        structure=res["structure"],
        quant=res["quant"],
        yardstick=dict(res["yardstick"], ratios=res["acceptance"]["energy"]["ratios"]),
        train_macs=res["train_macs"],
        capability=res["capability"],
        elapsed_sec=elapsed,
    )
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(build_report(res, figs, elapsed))

    print("\n===== 阶段 C 验收结论（软件版内在化） =====")
    for k, v in res["acceptance"].items():
        if k == "all_pass":
            continue
        print(f"  [{'PASS' if v['pass_'] else 'FAIL'}] {k}")
    print(f"总体: {'PASS' if res['acceptance']['all_pass'] else 'FAIL'}（{elapsed:.0f}s）")
    print(f"结果已写入 {out}/（metrics.json, report.md, figC*.png）")
    return 0 if res["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
