"""阶段 B 可视化（能力曲线 / 免遗忘曲线 / 组合零样本 / 事件率）。"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .runner_b import forget_stats


def _mean_std(list_of_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack(list_of_arrays)
    return a.mean(axis=0), a.std(axis=0)


def make_plots_b(mem_runs: list[dict], no_runs: list[dict],
                 out_dir: str) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_names = mem_runs[0]["task_names"]
    n_tasks = len(task_names)
    figs = {}

    # ---- figB1：能力曲线（每任务 EMA 误差，均值±std） ----
    plt.figure(figsize=(7, 4.5))
    per_task = len(mem_runs[0]["curves"][0])
    for j, name in enumerate(task_names):
        m, s = _mean_std([r["curves"][j] for r in mem_runs])
        xs = np.arange(per_task)
        plt.plot(xs, m, label=f"{name} (mem)")
        plt.fill_between(xs, m - s, m + s, alpha=0.15)
    plt.xlabel("steps within task")
    plt.ylabel("readout error (EMA)")
    plt.title("Stage B: ability grows with interaction (with memory)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = out / "figB1_ability.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["ability"] = p

    # ---- figB2：免遗忘曲线（R 矩阵：任务 i 的保留误差随已学任务数） ----
    plt.figure(figsize=(7, 4.5))
    Rm = np.mean([r["R"] for r in mem_runs], axis=0)
    Rn = np.mean([r["R"] for r in no_runs], axis=0)
    for i in range(n_tasks):
        xs = np.arange(i, n_tasks)
        plt.plot(xs, Rm[i, i:], "o-", label=f"{task_names[i]} (mem)")
        plt.plot(xs, Rn[i, i:], "s--", label=f"{task_names[i]} (no-mem)", alpha=0.6)
    plt.xlabel("tasks learned so far (index)")
    plt.ylabel("retained task error (frozen eval)")
    plt.title("Stage B: forgetting curve (backward transfer)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = out / "figB2_forgetting.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["forgetting"] = p

    # ---- figB3：组合零样本对比（训练 / 组合嵌入 / 随机条件） ----
    plt.figure(figsize=(7, 4.5))
    names_all = list(mem_runs[0]["combo"].keys())
    novel = list(mem_runs[0]["novel_names"])
    train = list(mem_runs[0]["task_names"])
    cats = train + novel
    vals, errs = [], []
    for k in cats:
        v = np.mean([r["combo"][k] for r in mem_runs])
        e = np.std([r["combo"][k] for r in mem_runs])
        vals.append(v)
        errs.append(e)
    rand_vals = [np.mean([r["combo"][k + "_rand"] for r in mem_runs]) for k in novel]
    plt.bar(np.arange(len(cats)), vals, yerr=errs, capsize=3, color="#4C72B0")
    for idx, (nm, rv) in enumerate(zip(novel, rand_vals)):
        plt.bar(len(train) + idx + 0.4, rv, width=0.35, color="#C44E52", alpha=0.8)
    plt.xticks(range(len(cats)), cats, rotation=20, fontsize=8)
    plt.ylabel("zero-shot readout error (MSE)")
    plt.title("Stage B: compositional zero-shot (blue=combo-embed, red=random-cond)")
    plt.tight_layout()
    p = out / "figB3_composition.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["composition"] = p

    # ---- figB4：事件率（不变量 3 能量代理，mem vs no-mem） ----
    plt.figure(figsize=(7, 4))
    ev_m = np.mean([r["event_rates"] for r in mem_runs], axis=0)
    ev_n = np.mean([r["event_rates"] for r in no_runs], axis=0)
    xs = np.arange(n_tasks)
    plt.plot(xs, ev_m, "o-", label="with memory")
    plt.plot(xs, ev_n, "s--", label="no memory")
    plt.xlabel("task index")
    plt.ylabel("mean event rate")
    plt.title("Stage B: event-driven sparsity (invariant 3)")
    plt.legend()
    plt.tight_layout()
    p = out / "figB4_events.png"
    plt.savefig(p, dpi=110)
    plt.close()
    figs["events"] = p

    return figs
