"""阶段 C 可视化：结构稀疏 / 能耗对比（vs 大模型标尺）/ 能力对照 / 活跃率。"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots_c(res: dict, out_dir: str) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {}

    # ---- figC1：结构稀疏（各组件权重密度，稠密=1.0 参照线） ----
    plt.figure(figsize=(7, 4.5))
    st = res["structure"]
    names = list(st["layers"].keys()) + ["readout", "Wdyn"]
    dens = [st["layers"][k]["density"] for k in st["layers"]] + [
        st["readout"]["density"], st["dyn"]["density"]]
    bars = plt.bar(range(len(names)), dens, color="#4C72B0")
    plt.axhline(1.0, color="gray", ls="--", lw=1, label="dense (baseline)")
    plt.axhline(st["total"]["density"], color="#C44E52", ls=":",
                lw=1.5, label=f"overall {st['total']['density']:.3f}")
    for b, d in zip(bars, dens):
        plt.text(b.get_x() + b.get_width() / 2, d + 0.02, f"{d:.2f}",
                 ha="center", fontsize=8)
    plt.xticks(range(len(names)), names, rotation=20, fontsize=8)
    plt.ylabel("weight density (nnz / size)")
    plt.title("Stage C: born-sparse structure (masks fixed at init)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = out / "figC1_structure.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["structure"] = p

    # ---- figC2：每样本推理能耗（log 刻度）vs 稠密对照臂 vs 大模型标尺 ----
    plt.figure(figsize=(8.5, 4.8))
    en = res["energy"]["single"]
    endn = res["energy_dense"]["single"]
    yard = res["yardstick"]["macs"]
    labels = ["sparse\n(event)", "sparse\n(struct)", "sparse\n(dense-equiv)",
              "dense-arm\n(event)", "dense-arm\n(struct)", "dense-arm\n(dense)"] + list(yard.keys())
    vals = [en["event"], en["struct"], en["dense"],
            endn["event"], endn["struct"], endn["dense"]] + list(yard.values())
    colors = ["#55A868"] * 3 + ["#4C72B0"] * 3 + ["#C44E52"] * len(yard)
    bars = plt.bar(range(len(vals)), vals, color=colors)
    plt.yscale("log")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v * 1.4, f"{v:.1e}",
                 ha="center", fontsize=6.5, rotation=0)
    plt.xticks(range(len(vals)), labels, fontsize=7)
    plt.ylabel("MACs per sample (inference)")
    rmin = res["acceptance"]["energy"]["ratio_min"]
    plt.title(f"Stage C: per-sample inference energy (sparse vs dense arm, vs LLM yardsticks;"
              f" min ratio {rmin:.0e}x)")
    plt.tight_layout()
    p = out / "figC2_energy.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["energy"] = p

    # ---- figC3：能力对照（稀疏 vs 稠密：保留误差 + 组合零样本） ----
    plt.figure(figsize=(7.5, 4.8))
    cap = res["capability"]
    tasks = cap["task_names"]
    novel = list(cap["combo_sparse"].keys())
    cats = tasks + novel
    sp = cap["retain_sparse"] + [cap["combo_sparse"][k] for k in novel]
    dn = cap["retain_dense"] + [cap["combo_dense"][k] for k in novel]
    rd = [np.nan] * len(tasks) + [cap["combo_rand_sparse"][k + "_rand"]
                                  for k in novel]
    x = np.arange(len(cats))
    w = 0.27
    plt.bar(x - w, sp, width=w, label="sparse core (born-sparse)", color="#55A868")
    plt.bar(x, dn, width=w, label="dense control", color="#4C72B0", alpha=0.75)
    plt.bar(x + w, rd, width=w, label="random-cond baseline (sparse)",
            color="#C44E52", alpha=0.7)
    plt.xticks(x, cats, rotation=20, fontsize=8)
    plt.ylabel("frozen eval error (MSE)")
    plt.title("Stage C: capability no-regression (sparse vs dense, combo zero-shot)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = out / "figC3_capability.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["capability"] = p

    # ---- figC4：每层平均活跃率（k-WTA + 事件驱动，内在稀疏激活） ----
    plt.figure(figsize=(7, 4.2))
    act = res["energy"]["active_frac"]
    names_a = list(act.keys())
    plt.bar(range(len(names_a)), [act[k] for k in names_a], color="#55A868")
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="k-WTA cap (0.5)")
    for i, k in enumerate(names_a):
        plt.text(i, act[k] + 0.01, f"{act[k]:.2f}", ha="center", fontsize=8)
    plt.xticks(range(len(names_a)), names_a, fontsize=9)
    plt.ylabel("mean active fraction")
    plt.title("Stage C: event-driven activity (intrinsic sparse activation)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = out / "figC4_activity.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["activity"] = p

    return figs
