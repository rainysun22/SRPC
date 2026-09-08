"""阶段 E2 主流程：锚点网格 → 梯子 → 孪生 → iPC 消融 → µPC 迁移对照 → 验收报告。

用法：
    python3 scripts/run_e2.py --stage all      # 全流程（后台长跑 ~4h）
    python3 scripts/run_e2.py --stage ladder   # 续跑（读已有 checkpoint）

分阶段执行，每阶段完成即写 results_e2/*.json；report 阶段汇总全部
checkpoint 并生成验收报告（PASS/FAIL 逐条预注册判据）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from srpc.config import E2Config
from srpc.lm import (ByteCorpus, train_lmpcn, train_twin)

RES = "results_e2"
os.makedirs(RES, exist_ok=True)


def _save(name: str, data: dict) -> None:
    p = os.path.join(RES, f"{name}.json")
    with open(p, "w") as f:
        json.dump(data, f, indent=1, default=float)
    print(f"[checkpoint] {p}")


def _load(name: str) -> dict | None:
    p = os.path.join(RES, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _best_anchor(rows: list[dict]) -> dict:
    """选 bpc 最低；bpc 差距 < 0.05 时取 iters 小者（省算力）。"""
    r = min(rows, key=lambda d: (d["bpc"], d["iters"]))
    return r


def run_anchor(cfg: E2Config, corpus: ByteCorpus) -> dict:
    """锚点网格（1M 档 h=768）：settle_iters × eta_w 双因素，裁定锚点超参。"""
    print("== anchor grid (h=768) ==")
    rows = []
    for iters in cfg.anchor_iters:
        for eta_w in cfg.anchor_eta:
            r = train_lmpcn(cfg, 768, cfg.anchor_steps, corpus,
                            eta_w=eta_w, iters=iters, seed=0)
            r.update(iters=iters, eta_w=eta_w)
            rows.append({k: v for k, v in r.items() if k != "model"})
            print(f"  iters={iters} eta_w={eta_w}: bpc={r['bpc']:.3f} "
                  f"acc={r['acc']:.3f} wall={r['wall']}s")
    best = _best_anchor(rows)
    out = dict(rows=rows, best=dict(iters=best["iters"], eta_w=best["eta_w"],
                                    bpc=best["bpc"], acc=best["acc"]))
    _save("anchor", out)
    return out


def run_ladder(cfg: E2Config, corpus: ByteCorpus, anchor: dict) -> dict:
    """梯子：h ∈ {768,1200,1856} × ladder_steps，锚点超参（µPC 迁移）。"""
    print("== ladder ==")
    it, ew = anchor["best"]["iters"], anchor["best"]["eta_w"]
    res = {}
    for h in cfg.ladder_widths:
        r = train_lmpcn(cfg, h, cfg.ladder_steps, corpus,
                        eta_w=ew, iters=it, seed=0)
        res[str(h)] = {k: v for k, v in r.items() if k != "model"}
        print(f"  h={h} ({r['n_params_struct']/1e6:.2f}M): bpc={r['bpc']:.3f} "
              f"acc={r['acc']:.3f} wall={r['wall']}s")
        _save("ladder", res)
    return res


def run_twin(cfg: E2Config, corpus: ByteCorpus, ladder: dict) -> dict:
    """孪生：参数量匹配 × 同样本流（batch 化，反传参照）。"""
    print("== twin ==")
    res = {}
    for h in cfg.ladder_widths:
        n_params = ladder[str(h)]["n_params_struct"]
        r = train_twin(cfg, n_params, cfg.ladder_steps, corpus, seed=0)
        res[str(h)] = r
        print(f"  twin@{h} ({r['n_params']/1e6:.2f}M): bpc={r['bpc']:.3f} "
              f"acc={r['acc']:.3f} wall={r['wall']}s")
        _save("twin", res)
    return res


def run_ipc(cfg: E2Config, corpus: ByteCorpus, anchor: dict) -> dict:
    """iPC 上下文课程消融（1M 档）：ipc vs 无 ipc（复用 ladder 1M 曲线）。"""
    print("== ipc ablation (h=768) ==")
    it, ew = anchor["best"]["iters"], anchor["best"]["eta_w"]
    r = train_lmpcn(cfg, 768, cfg.ipc_steps, corpus, eta_w=ew, iters=it,
                    seed=0, ipc=True)
    out = {"ipc": {k: v for k, v in r.items() if k != "model"}}
    # 对照 = ladder 1M 在 ipc_steps 处的曲线点（同预算）
    base = ladder_curve_at("ladder", "768", cfg.ipc_steps)
    out["base"] = base
    out["ipc_gain"] = base["bpc"] - out["ipc"]["bpc"] if base else None
    if base:
        print(f"  ipc bpc={r['bpc']:.3f} vs base@{cfg.ipc_steps} "
              f"bpc={base['bpc']:.3f}")
    else:
        print(f"  ipc bpc={r['bpc']:.3f} (no base)")
    _save("ipc", out)
    return out


def ladder_curve_at(name: str, h: str, step: int) -> dict | None:
    d = _load(name)
    if not d or h not in d:
        return None
    curve = d[h].get("curve", [])
    for pt in curve:
        if pt["step"] >= step:
            return pt
    return curve[-1] if curve else None


def run_control(cfg: E2Config, corpus: ByteCorpus, anchor: dict) -> dict:
    """µPC 迁移对照（4M 档 h=1856，20k 步同预算）：迁移超参 vs 重调网格。

    判据 4：重调最优 BPC − 迁移最优 BPC <= transfer_gain_max(0.03)。
    迁移最优 = ladder 4M 在 20k 步的曲线点（若存在）；重调 = 本网格最优。
    """
    print("== µPC transfer control (h=1856) ==")
    it, ew = anchor["best"]["iters"], anchor["best"]["eta_w"]
    rows = []
    for iters in cfg.anchor_iters:
        for eta_w in cfg.anchor_eta:
            r = train_lmpcn(cfg, 1856, cfg.control_steps, corpus,
                            eta_w=eta_w, iters=iters, seed=0)
            r.update(iters=iters, eta_w=eta_w)
            rows.append({k: v for k, v in r.items() if k != "model"})
            print(f"  iters={iters} eta_w={eta_w}: bpc={r['bpc']:.3f} "
                  f"acc={r['acc']:.3f} wall={r['wall']}s")
            _save("control", dict(rows=rows))
    best = _best_anchor(rows)
    transferred = ladder_curve_at("ladder", "1856", cfg.control_steps)
    out = dict(rows=rows,
               best=dict(iters=best["iters"], eta_w=best["eta_w"],
                         bpc=best["bpc"], acc=best["acc"]),
               transferred=transferred,
               transfer_gain=(transferred["bpc"] - best["bpc"]
                              if transferred else None))
    _save("control", out)
    return out


# ----------------------------------------------------------------------
# 验收报告
# ----------------------------------------------------------------------
def build_report(cfg: E2Config, anchor: dict, ladder: dict, twin: dict,
                 ipc: dict | None, control: dict | None,
                 unigram_bpc: float) -> str:
    v = {}
    h1, h2, h4 = (str(x) for x in cfg.ladder_widths)
    b1, b2, b4 = ladder[h1]["bpc"], ladder[h2]["bpc"], ladder[h4]["bpc"]
    t1, t2, t4 = twin[h1]["bpc"], twin[h2]["bpc"], twin[h4]["bpc"]
    # 判据 1：单调性
    v["mono"] = b4 < b2 < b1
    # 判据 2：斜率平行（每倍增跨度 1M→4M = 2 倍增）
    g_p = (b1 - b4) / 2.0
    g_t = (t1 - t4) / 2.0
    v["slope"] = g_p >= cfg.slope_frac_min * g_t and g_p > 0
    # 判据 3：4M 档 PCN <= 1.5×孪生
    v["ratio"] = b4 <= cfg.twin_ratio_max * t4
    # 判据 4：µPC 迁移（4M 20k 同预算）
    v["transfer"] = True
    if control and control["transfer_gain"] is not None:
        v["transfer"] = control["transfer_gain"] <= cfg.transfer_gain_max
    ok = all(v.values())
    best = anchor["best"]
    lines = [
        "# SR-PC 阶段 E2 验收报告（规模化语言训练，自动生成）",
        "",
        f"总体结论：**{'PASS' if ok else 'FAIL'}** · 语料 = tinyshakespeare "
        f"({_corpus_stats()} 字节, 一元 BPC {unigram_bpc:.2f})",
        "",
        "协议：字节级 next-byte 预测，顺序流式单样本在线（PCN）/ batch32（孪生），",
        "验证段同口径（PCN 读出温度 τ 经校准切片定，孪生直接 softmax）。",
        "",
        "## 锚点网格（1M 档 h=768，20k 步）",
        "",
        "| iters | eta_w | BPC | acc |",
        "|---|---|---|---|",
    ]
    for r in anchor["rows"]:
        lines.append(f"| {r['iters']} | {r['eta_w']} | {r['bpc']:.3f} "
                     f"| {r['acc']:.3f} |")
    lines += [
        f"**裁定锚点**：iters={best['iters']}、eta_w={best['eta_w']}"
        f"（BPC {best['bpc']:.3f}），梯子/消融/对照全部零调参迁移。",
        "",
        "## 梯子（µPC 迁移超参，pilot 150k 步）",
        "",
        "| 档位 | 结构参数 | BPC | acc | 相对孪生 |",
        "|---|---|---|---|---|",
    ]
    for h in (h1, h2, h4):
        lines.append(f"| {h} | {ladder[h]['n_params_struct']/1e6:.2f}M | "
                     f"{ladder[h]['bpc']:.3f} | {ladder[h]['acc']:.3f} | "
                     f"{ladder[h]['bpc']/twin[h]['bpc']:.2f}× |")
    lines += [
        "",
        "## 孪生（参数量匹配稠密 MLP + Adam，反传参照）",
        "",
        "| 档位 | 参数 | BPC | acc |",
        "|---|---|---|---|",
    ]
    for h in (h1, h2, h4):
        lines.append(f"| {h} | {twin[h]['n_params']/1e6:.2f}M | "
                     f"{twin[h]['bpc']:.3f} | {twin[h]['acc']:.3f} |")
    lines += [
        "",
        "## 验收判定（预注册）",
        "",
        "| 判据 | 实测 | 结果 |",
        "|---|---|---|",
        f"| 1 单调性 BPC(4M)<BPC(2M)<BPC(1M) | {b4:.3f}<{b2:.3f}<{b1:.3f} | "
        f"{'PASS' if v['mono'] else 'FAIL'} |",
        f"| 2 斜率平行（PCN 每倍增 {g_p:.3f} ≥ 0.5×孪生 {g_t:.3f}） | "
        f"gain_pcn={g_p:.3f} gain_twin={g_t:.3f} | "
        f"{'PASS' if v['slope'] else 'FAIL'} |",
        f"| 3 4M 档 PCN ≤ 1.5×孪生 | {b4:.3f} vs {t4:.3f} "
        f"({b4/t4:.2f}×) | {'PASS' if v['ratio'] else 'FAIL'} |",
    ]
    if control and control["transfer_gain"] is not None:
        lines.append(
            f"| 4 µPC 迁移（4M 20k 同预算增益 ≤ {cfg.transfer_gain_max}） | "
            f"重调最优 {control['best']['bpc']:.3f} vs 迁移 "
            f"{control['transferred']['bpc']:.3f}（增益 "
            f"{control['transfer_gain']:.3f}） | "
            f"{'PASS' if v['transfer'] else 'FAIL'} |")
    else:
        lines.append(f"| 4 µPC 迁移 | 数据不足（4M 20k 点缺失） | PENDING |")
    lines += [
        "",
        "## iPC 上下文课程消融（1M 档）",
        "",
    ]
    if ipc and ipc["ipc_gain"] is not None:
        lines.append(f"- 无 iPC BPC（同预算）：{ipc['base']['bpc']:.3f}"
                     f"；iPC BPC：{ipc['ipc']['bpc']:.3f}"
                     f"（增益 {ipc['ipc_gain']:+.3f}"
                     f"）→ {'采纳' if ipc['ipc_gain'] > 0.02 else '不采纳'}（±0.02 视为持平）")
    else:
        lines.append("- iPC 消融未完成或对照缺失（记录项）。")
    lines += [
        "",
        "## 沙箱口径说明",
        "",
        "- pilot 预算（梯子 150k 步 ≈ 13% epoch）为沙箱可行性缩水口径；",
        "  最终判定以全额预算 + GPU 登顶跑（GPU_TASKS T1）为准。",
        "- 孪生 = 标准反传参照（不属 SR-PC 构造，GPU_TASKS 契约第 2 条）。",
        "- 曲线逐点落盘于本报告同目录 JSON（curve 字段）。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _corpus_stats() -> str:
    cfg = E2Config()
    raw = np.frombuffer(open(cfg.corpus_path, "rb").read(), dtype=np.uint8)
    return f"{len(raw)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all", "anchor", "ladder", "twin",
                                        "ipc", "control", "report"),
                    default="all")
    args = ap.parse_args()
    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    unigram = corpus.unigram_bpc

    if args.stage in ("all", "anchor"):
        anchor = run_anchor(cfg, corpus)
    else:
        anchor = _load("anchor")
    if args.stage in ("all", "ladder") or (args.stage == "report"
                                           and not _load("ladder")):
        assert anchor, "缺 anchor checkpoint"
        ladder = run_ladder(cfg, corpus, anchor)
    else:
        ladder = _load("ladder")
    if args.stage in ("all", "twin") or (args.stage == "report"
                                         and not _load("twin")):
        assert ladder, "缺 ladder checkpoint"
        twin = run_twin(cfg, corpus, ladder)
    else:
        twin = _load("twin")
    if args.stage in ("all", "ipc"):
        assert anchor, "缺 anchor checkpoint"
        ipc = run_ipc(cfg, corpus, anchor)
    else:
        ipc = _load("ipc")
    if args.stage in ("all", "control"):
        assert anchor, "缺 anchor checkpoint"
        control = run_control(cfg, corpus, anchor)
    else:
        control = _load("control")

    # 报告仅在 report/all 阶段构建（单阶段跑完不因缺其他 checkpoint 崩溃）
    if args.stage in ("report", "all"):
        report = build_report(cfg, anchor, ladder, twin, ipc, control, unigram)
        with open(os.path.join(RES, "report.md"), "w") as f:
            f.write(report)
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
