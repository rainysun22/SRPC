#!/usr/bin/env python3
"""SR-PC 阶段 E1 主入口：字节级 UTF-8 词元前端 + 语言长程信用分配。

对应 docs/ROADMAP.md 阶段 E1 / docs/SRPC_DESIGN.md §9-2（自建词元前端）：
    E1a ByteTokenizer：text ⇄ 256 维字节一热，多语原生，无预训练词表；
    E1b 主验收（assoc 延迟文本关联）：零边际相关（承重墙判据），误差驱动
       在线可学、纯相关 Hebbian 贴机会；Δ=4 主判定 + Δ=8 跨度外推；
    E1c 边界记录（不计验收）：xorsum@{4,16} 在线组合信用分配边界
       （parity SQ-hard，BP 在线对照同样失败，batch 上界 1.0）、
       assoc@16 远端表征稀释边界。

用法：
    python scripts/run_e1.py                    # 3 seeds 全量
    python scripts/run_e1.py --seeds 0          # 单 seed 快速
    python scripts/run_e1.py --steps 300        # 冒烟（缩短步数）

输出：results_e1/metrics.json + report.md
退出码：0 = E1 主验收全过；1 = 存在未通过项（可证伪信号）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from srpc.config import LangConfig
from srpc.lang import ByteTokenizer, run_lang_screen

TOK_CHECKS = [
    "",                                    # 空串
    "hello world",                         # ASCII
    "预测编码与自省",                        # 中文（3 字节/字）
    "Привет мир",                          # 西里尔（2 字节/字）
    "🎉🚀UTF-8🧠",                          # emoji（4 字节）
    "a\0b\tc\n",                           # 控制字符
    "ARC→LM→世界",                          # 混合
]


def check_tokenizer() -> dict:
    """E1a：词元化器往返 + 维度协议质检。"""
    tk = ByteTokenizer()
    rows = []
    for s in TOK_CHECKS:
        x = tk.encode(s)
        r = tk.decode(x)
        ok = (r == s) and x.shape[1] == 256
        # 一热协议：每行恰一个 1
        onehot = bool((x.sum(axis=1) == 1.0).all()
                      and set(np.unique(x)) <= {0.0, 1.0})
        rows.append(dict(text=repr(s), n_bytes=x.shape[0],
                         roundtrip=ok, one_hot=onehot))
    return dict(vocab=256, checks=rows,
                pass_=all(r["roundtrip"] and r["one_hot"] for r in rows))


def build_report(res: dict, tok: dict, elapsed: float) -> str:
    v = res["verdict"]
    m = res["metrics_mean"]
    ok = "PASS" if v["pass_all"] else "FAIL"
    lines = [
        "# SR-PC 阶段 E1 验收报告（语言化起步，自动生成）",
        "",
        f"总体结论：**{ok}** · 运行耗时 {elapsed:.0f}s · 全程 NumPy 局部规则、免反向传播",
        "",
        "范围：字节级 UTF-8 词元前端（E1a）+ 语言长程信用分配（E1b 主验收）+",
        "在线信用分配边界记录（E1c）。v2 方案调整的完整证据链见文末。",
        "",
        "## E1a 词元化器质检",
        "",
        f"- 词表 = 256 字节值（UTF-8 万国码原生，无 OOV、无预训练词表）",
        f"- 往返 + 一热协议：{'PASS' if tok['pass_'] else 'FAIL'}"
        f"（{len(tok['checks'])} 组用例：ASCII / 中文 / 西里尔 / emoji / 控制字符 / 混合）",
        "",
        "## E1b 主验收：assoc 延迟文本关联（16 类，零边际相关）",
        "",
        "| 判据 | 阈值 | 实测（3 seeds 均值） | 结果 |",
        "|---|---|---|---|",
        f"| acc_error @Δ=4 | >= 0.80 | {v['acc_assoc_d4']:.3f} | {'PASS' if v['pass_acc'] else 'FAIL'} |",
        f"| acc_hebb @Δ=4 | <= 0.20 | {v['acc_hebb_d4']:.3f} | {'PASS' if v['pass_hebb'] else 'FAIL'} |",
        f"| gap @Δ=4 | >= 0.40 | {v['gap_d4']:.3f} | {'PASS' if v['pass_gap'] else 'FAIL'} |",
        f"| distal_error @Δ=4 | >= 0.25 | {v['distal_assoc_d4']:.3f} | {'PASS' if v['pass_distal'] else 'FAIL'} |",
        f"| acc_error @Δ=8（跨度外推） | >= 0.80 | {v['acc_assoc_d8']:.3f} | {'PASS' if v['pass_d8'] else 'FAIL'} |",
        "",
        "跨度扫描（error 臂 acc，括号 = hebb 臂）：",
        "",
        "| Δ | acc_error (hebb) | acc_cmp | BPC | distal_error | 机会 distal |",
        "|---|---|---|---|---|---|",
    ]
    for d in (1, 4, 8, 16):
        r = m[f"assoc_d{d}"]
        lines.append(f"| {d} | {r['acc_error']:.3f} ({r['acc_hebb']:.3f}) "
                     f"| {r['acc_cmp_error']:.3f} | {r['bpc_error']:.2f} "
                     f"| {r['distal_error']:.3f} | {1/(d+1):.3f} |")
    lines += [
        "",
        "判据说明：π 为双射且符号流均匀 => y 边际均匀、与任何单一输入符号",
        "零边际相关（与 §8.5 延迟 XOR 同判据）——输入一热线性空间的一阶",
        "统计无信号；误差驱动通过钳制-收敛-学习协议利用条件结构 P(y|x)",
        "在线可学。hebb 臂（浅迭代 + 纯相关）在长程（Δ>=4）下因传导不足",
        "贴机会、近程（Δ=1）可部分联想——与 §8.5 \"长程延迟下误差驱动",
        "显著优于纯相关\"判据一致。远端块（窗口最前块）为唯一任务相关块，",
        "distal_error 为其 W1 权重变化占比（机会水平 1/(Δ+1)）。",
        "",
        "## E1c 边界记录（不计入验收）",
        "",
        "| 任务 | Δ | acc_error (hebb) | distal_error | 归因 |",
        "|---|---|---|---|---|",
    ]
    for task, d in (("xorsum", 4), ("xorsum", 16), ("assoc", 16)):
        r = m[f"{task}_d{d}"]
        cause = ("在线组合信用分配边界（SQ-hard；修复排 F 阶段记忆回放）"
                 if task == "xorsum" else "远端表征稀释边界（容量，非原理性）")
        lines.append(f"| {task} | {d} | {r['acc_error']:.3f} ({r['acc_hebb']:.3f}) "
                     f"| {r['distal_error']:.3f} | {cause} |")
    lines += [
        "",
        "### xorsum 边界证据链（v2 方案调整依据，探针复现于探针脚本）",
        "",
        "1. **超参全排除**：credit 同款超参（h1=32/h2=16/32 iters/10k 步/full 能量）、",
        "   深收敛 64 iters、学习率 0.05-1.0、关 k-WTA / 掩码 / 事件门控、感受野",
        "   窗口 2-3 块——16 类 xorsum 全部贴机会（0.05-0.09）。",
        "2. **BP 在线对照同样失败**：同形状网络（1536-128-64-16，ReLU）+ Adam",
        "   单样本 20k 步 acc≈0.054-0.086（机会 0.0625）——瓶颈不在局部规则。",
        "3. **batch 上界 = 1.0**：mini-batch(512) plain SGD lr=0.3 1000 epoch /",
        "   Adam 300 epoch 均达 1.000——任务本身对同形状网络可解。",
        "4. **归因**：parity 为统计查询模型下不可学习函数（Kearns & Valiant 1989；",
        "   Blum et al. 1994），在线单样本更新的期望梯度≈0、对称无法破缺；batch",
        "   平均才使信号浮出。PCN+梯度累积/经验回放亦失败（k-WTA/clip 等结构",
        "   稀疏组件破坏收敛态≈BP 等效性，需 F 阶段与记忆回放协同设计）。",
        "5. **类数断点**：在线 PCN xorsum 2/4/8/16 类 acc = 贴机会/0.38/0.16/0.07",
        "   （机会 0.5/0.25/0.125/0.0625）——4 类（2-bit）有弱信号，gap 不足；",
        "   2 类 1-bit parity 为纯高频（所有低阶统计为零）反而最劣。",
        "6. **修复排期**：F 阶段记忆回放（阶段 B 慢记忆组件 -> 周期性重放 =",
        "   batch 等效，生物学对应睡眠重放/系统巩固），与 E3 语言流协同验证。",
        "",
        "### assoc@Δ=16 稀释边界",
        "",
        "层 1 每单元锚定 1 个时间块，span=17 块下远端覆盖仅 1/17（~7.5 个 h1",
        "单元），x2 随机扇入再稀释——容量边界而非原理性（扩 h1 或感受野覆盖",
        f"可推远）；distal_error {m['assoc_d16']['distal_error']:.3f} 仍为机会"
        f" 0.059 的 {m['assoc_d16']['distal_error']/0.059:.1f} 倍，远端驱动存在。",
        "",
        "## 与阶段 A 早筛（§8.5）的关系",
        "",
        "- 同判据（零边际相关 + 双臂 + distal），任务载体从 2 维 bit 窗口推广到",
        "  256 维字节一热语言符号流，跨度从 Δ=4 推到 Δ=8（2 倍）。",
        "- 阶段 A 的 2 类 XOR（交互对在单块内）保持通过；本阶段的组合交互对",
        "  （跨块）在线不可达已如实归因并排期修复，不降低阶段 A 结论的效力。",
        "",
        "> 结论：字节前端 + 长程信用分配（传导型）在语言符号流上成立；",
        "> 组合塑形型信用分配的在线边界已定位，修复路径（记忆回放）排入 F 阶段。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SR-PC Phase E1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖训练步数（冒烟用）")
    ap.add_argument("--out", default="results_e1")
    args = ap.parse_args()

    t_start = time.time()
    cfg = LangConfig()
    if args.steps:
        cfg.train_steps = args.steps
        cfg.eval_steps = min(cfg.eval_steps, 100)
        cfg.compare_steps = min(cfg.compare_steps, 20)

    print("[1/3] E1a 词元化器质检 ...")
    tok = check_tokenizer()
    print(f"  roundtrip+onehot: {'PASS' if tok['pass_'] else 'FAIL'}")

    print(f"[2/3] E1b/c 屏幕：assoc@Δ{{1,4,8,16}} + xorsum@Δ{{4,16}} × "
          f"{len(args.seeds)} seeds ...")
    res = run_lang_screen(cfg, seeds=args.seeds)

    print("[3/3] 报告 ...")
    elapsed = time.time() - t_start
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = dict(tokenizer=tok, verdict=res["verdict"],
                   metrics_mean=res["metrics_mean"],
                   per_probe=res["per_probe"], elapsed_sec=elapsed)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(build_report(res, tok, elapsed))

    print("\n===== 阶段 E1 验收结论 =====")
    v = res["verdict"]
    print(f"  [{'PASS' if v['pass_acc'] else 'FAIL'}] acc_error@Δ4 = {v['acc_assoc_d4']:.3f} (>= 0.80)")
    print(f"  [{'PASS' if v['pass_hebb'] else 'FAIL'}] acc_hebb@Δ4  = {v['acc_hebb_d4']:.3f} (<= 0.20)")
    print(f"  [{'PASS' if v['pass_gap'] else 'FAIL'}] gap@Δ4       = {v['gap_d4']:.3f} (>= 0.40)")
    print(f"  [{'PASS' if v['pass_distal'] else 'FAIL'}] distal@Δ4    = {v['distal_assoc_d4']:.3f} (>= 0.25)")
    print(f"  [{'PASS' if v['pass_d8'] else 'FAIL'}] acc_error@Δ8 = {v['acc_assoc_d8']:.3f} (>= 0.80)")
    print(f"总体: {'PASS' if v['pass_all'] else 'FAIL'}（{elapsed:.0f}s）")
    print(f"结果已写入 {out}/（metrics.json, report.md）")
    return 0 if v["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
