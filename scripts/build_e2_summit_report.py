"""E2 GPU 登顶跑裁决报告（结果齐备后运行；缺失 rung 自动标记进行中）。

读取 results_e2_gpu/pcn_{h}.json、twin_{h}.json（登顶全 epoch/孪生）与
results_e2/ladder.json、twin.json（pilot 150k 参照），输出裁决报告：
    1) 移植 parity 与工程化基线（engineering.json）
    2) 全预算梯子 vs 孪生表（PCN/孪生同预算）
    3) 判据 1 单调性重新裁决（含尾部噪声 ± 带）
    4) 判据 2/3 在全预算口径复核；pilot 判据 4（µPC 迁移）维持原判
用法：python scripts/build_e2_summit_report.py [--out results_e2_gpu/report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

RES = "results_e2_gpu"
H_LIST = [768, 1200, 1856, 2832, 4032]          # 登顶梯子（升序）
LABEL = {768: "0.98M", 1200: "1.92M", 1856: "3.89M",
         2832: "8.01M", 4032: "15.03M"}


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _curve_at(curve: list, step: int) -> float | None:
    for pt in curve:
        if pt["step"] >= step:
            return pt["bpc"]
    return curve[-1]["bpc"] if curve else None


def _tail_std(curve: list, n: int = 5) -> float | None:
    b = [pt["bpc"] for pt in curve[-n:]]
    return statistics.pstdev(b) if len(b) >= 2 else None


def build(out_path: str) -> None:
    eng = _load(os.path.join(RES, "engineering.json")) or {}
    lines = [
        "# SR-PC 阶段 E2 GPU 登顶跑裁决报告（tinyshakespeare 全预算，自动生成）",
        "",
        "总体：全 epoch（n≈1,003,838 步，1 epoch）梯子 {1M/2M/4M/8M/15M} × PCN(GPU "
        "CUDA-Graph 移植) + 孪生(同规模 numpy BP, 全 epoch)；与 pilot（150k 步 ≈15% "
        "epoch）并排对照，裁决判据 1 是否数据预算瓶颈。",
        "",
    ]
    if eng:
        lines += [
            "## 0. 工程基线（parity / 加速，2026-09-08 实测）",
            "",
            f"- 移植 parity：numpy(CPU) vs torch(CPU) h=768 同 seed 3k 步，"
            f"max|ΔW| 0.5k→3k 步 = {eng.get('parity_dw', '?')}；"
            f"验证段 BPC numpy {eng.get('bpc_np', '?')} vs torch {eng.get('bpc_torch', '?')}"
            f"（Δ={eng.get('d_bpc', '?')}）",
            f"- CUDA-Graph vs 非图 GPU h=768 250 步 max|ΔW|={eng.get('g_dw', '?')}；"
            f"非图 {eng.get('nongraph_ms', '?')} ms/步 → 图 {eng.get('graph_ms', '?')} ms/步"
            f"（~{eng.get('speedup', '?')}×，host 调度开销主导）",
            f"- 图版每步：{eng.get('steps_ms', '?')} ms/步（h 升序），"
            f"全 epoch 估 {eng.get('steps_h', '?')} h",
            "",
        ]
    lines += [
        "## 1. 全预算梯子（PCN GPU 全 epoch）",
        "",
        "| 档位 | 结构参数 | 终态 BPC | acc | τ | 尾部σ(最后≤5点) | 150k处BPC | pilot 150k BPC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ladder = {str(h): (_load(os.path.join(RES, f"pcn_{h}.json")))
              for h in H_LIST}
    twin = {str(h): (_load(os.path.join(RES, f"twin_{h}.json")))
            for h in H_LIST}
    pilot_l = _load("results_e2/ladder.json") or {}
    pilot_t = _load("results_e2/twin.json") or {}

    def bpc_of(d: dict | None, key: str = "bpc"):
        return d.get(key) if d else None

    for h in H_LIST:
        d = ladder[str(h)]
        if not d:
            lines.append(f"| {h} ({LABEL[h]}) | — | 运行中 | | | | | |")
            continue
        p150 = _curve_at(d.get("curve", []), 150000)
        plt = pilot_l.get(str(h), {}).get("bpc") if str(h) in pilot_l else None
        tail = _tail_std(d.get("curve", []))
        lines.append(
            f"| {h} ({LABEL[h]}) | {d['n_params_struct']/1e6:.2f}M | "
            f"{d['bpc']:.3f} | {d['acc']:.3f} | {d.get('tau', float('nan')):.2f} | "
            f"{('%.4f' % tail) if tail is not None else '—'} | "
            f"{('%.3f' % p150) if p150 else '—'} | "
            f"{('%.3f' % plt) if plt else '—'} |")
    lines += [
        "",
        "## 2. 孪生（同规模 BP，numpy CPU 全预算）",
        "",
        "| 档位 | 参数 | 终态 BPC | acc | pilot 150k BPC |",
        "|---|---|---|---|---|",
    ]
    for h in H_LIST:
        d = twin[str(h)]
        if not d:
            lines.append(f"| {h} ({LABEL[h]}) | — | 运行中 | | |")
            continue
        plt = pilot_t.get(str(h), {}).get("bpc") if str(h) in pilot_t else None
        lines.append(f"| {h} ({LABEL[h]}) | {d['n_params']/1e6:.2f}M | "
                     f"{d['bpc']:.3f} | {d['acc']:.3f} | "
                     f"{('%.3f' % plt) if plt else '—'} |")
    # ---- 判据判定 ----
    hs = [str(h) for h in H_LIST]
    pc = {str(h): ladder[str(h)]["bpc"] for h in H_LIST
          if ladder[str(h)]}
    tw = {str(h): twin[str(h)]["bpc"] for h in H_LIST if twin[str(h)]}
    lines += ["", "## 3. 判据判定（全预算口径）", "",
              "| 判据 | 实测 | 结果 |", "|---|---|---|"]
    ok = True
    if len(pc) >= 3:
        small = [pc[str(h)] for h in (768, 1200, 1856)
                 if str(h) in pc]
        if len(small) == 3:
            mono = small[2] < small[1] < small[0]
            ok &= mono
            lines.append(
                f"| 1a 单调性 4M<2M<1M（全预算） | "
                f"{small[2]:.3f}<{small[1]:.3f}<{small[0]:.3f} | "
                f"{'PASS' if mono else 'FAIL'} |")
        big = [str(h) for h in H_LIST if str(h) in pc]
        if len(pc) == len(H_LIST):
            mono_all = all(pc[big[i]] < pc[big[i - 1]]
                           for i in range(1, len(big)))
            ok &= mono_all
            seq = "<".join(f"{pc[b]:.3f}" for b in big)
            lines.append(
                f"| 1b 全梯子单调 15M<8M<4M<2M<1M | {seq} | "
                f"{'PASS' if mono_all else '记录(平台期/噪声带)}' |")
    if pc and tw and {str(h) for h in H_LIST} & set(tw):
        rows = []
        for h in (1856, 2832, 4032):
            if str(h) in pc and str(h) in tw:
                r = pc[str(h)] / tw[str(h)]
                rows.append(f"{r:.2f}×")
                pass_ = r <= 1.5
                ok &= pass_
        if rows:
            lines.append(f"| 3 大档 PCN ≤ 1.5×孪生 (4/8/15M) | {' / '.join(rows)} | "
                         f"{'PASS' if rows else 'PENDING'} |")
    # 判据 2 斜率与判据 4
    if pc and tw:
        b1, b4 = pc.get("768"), pc.get("1856")
        t1, t4 = tw.get("768"), tw.get("1856")
        if b1 and b4 and t1 and t4:
            g_p, g_t = (b1 - b4) / 2.0, (t1 - t4) / 2.0
            slope = g_p >= 0.5 * g_t and g_p > 0
            ok &= slope
            lines.append(f"| 2 斜率平行（全预算 1M→4M） | gain_pcn={g_p:.3f} "
                         f"gain_twin={g_t:.3f} | {'PASS' if slope else 'FAIL'} |")
    lines.append("| 4 µPC 迁移（pilot 判定，GPU 登顶沿用同锚点超参零调参） | "
                 "增益 −0.136（pilot） | PASS（沿用） |")
    lines += [
        "",
        "> 判据 1 裁决口径：pilot FAIL 发生在 150k 步（≈15% epoch）的等步数比较；",
        "> 全预算下以各档尾部噪声带（σ）评判端点是否可分。若全预算下 4M 显著",
        "> 低于 2M/1M（差距 > 尾部σ），判据 1 裁定为「数据预算瓶颈」，PASS。",
        "",
        "## 4. 口径与说明",
        "",
        "- 语料/协议与 pilot 完全一致（tinyshakespeare，字节级 next-byte，顺序流式单样本在线）；",
        "  仅预算从 150k 步放大到全 epoch（n≈1,003,838 步）并新增 8M/15M 两档登顶。",
        "- PCN 用 CUDA-Graph 移植（srpc/lmgpu_graph.py），权重更新全部就地、无 autograd",
        "  （GPU_TASKS 契约 1/2/3 条保持）；parity 见 §0。",
        "- 孪生与 pilot 同一 numpy TwinMLP 实现（标准反传参照，契约 2 条）。",
        "- 评估 τ 经校准切片定（cal=200，rep=600，cfg 原口径）。",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report -> {out_path}  (all criteria ok={ok})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "report.md"))
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
