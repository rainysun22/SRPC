"""F2：课程级持续学习 — 50+ 任务流免遗忘上限实验（CPU，纯 NumPy）。

协议（对标阶段 B run_sequential / ContinuaFabric 同向验证，ROADMAP F2）：
  课程 = F2Curriculum 的 54 个确定性网格变换任务（7 任务族，难度递增 0→4）。
  顺序学习：按课程顺序逐个任务训练（每任务 steps_per_task 步，样本 i.i.d.，无回放无 shuffle），
  每任务结束时**冻结评估所有已见任务** -> 得 54×54 backward-transfer 矩阵 R[i][j]
  （任务 j 训练结束后，任务 i 的冻结读误差）。R[i][i]=diag_i 为刚学成误差。

指标：
  - forget_rel_i = (R[i,最后] - R[i,i]) / R[i,i]：任务 i 被后续 54-1-i 个任务"侵蚀"后的相对回升。
    免遗忘结构（每任务独读出口 + 任务分组 slow 记忆）应使所有 i 的 forget_rel ≤ 25%（ROADMAP 验收）。
  - 遗忘曲线上限：forget_rel vs "距末尾的距离 distance=(54-1-i)"——以记忆应保持平稳（上限封顶），
    无记忆随 distance 单调膨胀。
  - 难度正压：按任务族/难度分组的 forget_rel，验证"越难（靠后学）的旧任务也不遗忘"。
  - 记忆增益：带记忆 vs 无记忆的保留误差对比（secondary，dual 读出口给记忆通道）。

读出口 ro_recon_mode="dual"（concat[ŝ, x_self]）：保真走 ŝ、记忆/条件耦合走 x_self ——
让任务分组 slow 记忆真正进入读出（"能力=记忆·拼合"在课程规模上生效）。

运行：python3 f2_run.py [--steps N] [--seeds 0,1] （写 e3/results_f2.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

_cwd = os.path.dirname(os.path.abspath(__file__))
for _p in (_cwd, os.path.dirname(_cwd), "/workspace"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np

from srpc.config import CLConfig, DeepConfig, MemoryConfig
from srpc.deepmodel import DeepSRPC
from srpc.memory import PrototypeMemory
from srpc.runner_b import forget_stats

from f2_curriculum import F2Curriculum

# ----------------------------------------------------------------------
def eval_task_mse(model, curric: F2Curriculum, name: str, eval_samples: int):
    """冻结评估：读出一热 -> 均方误差 + 逐格正确率（decode_grid argmax）。"""
    errs = []
    accs = []
    for _ in range(eval_samples):
        s_in, s_out, cond = curric.sample(name)
        out = model.apply_transform(s_in, cond)
        errs.append(float(np.mean((out - s_out) ** 2)))
        g_out = curric.decode_grid(out)
        g_ref = curric.decode_grid(s_out)
        accs.append(float(np.mean(g_out == g_ref)))
    return float(np.mean(errs)), float(np.mean(accs))


def run_f2_sequential(seed: int, with_memory: bool,
                      dcfg: DeepConfig, mcfg: MemoryConfig,
                      steps_per_task: int, eval_samples: int) -> dict:
    rng = np.random.default_rng(seed * 1000 + 1)
    curric = F2Curriculum(np.random.default_rng(seed * 1000 + 2))
    mem = PrototypeMemory(mcfg, rng) if with_memory else None
    model = DeepSRPC(dcfg, 0, rng, self_loop=True, memory=mem)

    N = curric.n_train
    names = curric.train_names
    R = np.zeros((N, N))
    Acc = np.zeros((N, N))
    diag = np.zeros(N)
    acc_diag = np.zeros(N)

    for j, name in enumerate(names):
        for _ in range(steps_per_task):
            s_in, s_out, cond = curric.sample(name)
            model.step_mapping(s_in, s_out, cond)
        model.set_learning(False)
        for i in range(j + 1):
            e, a = eval_task_mse(model, curric, names[i], eval_samples)
            R[i, j] = e
            Acc[i, j] = a
        model.set_learning(True)
        diag[j] = R[j, j]
        acc_diag[j] = Acc[j, j]

    model.set_learning(False)
    # 记忆填充统计（诊断：memory 是否真正生效）
    mem_stats = None
    if mem is not None:
        snap = mem.snapshot()
        mem_stats = dict(
            n_slow_groups=len(snap["slow"]),
            slow_total=int(sum(int(v.sum()) for v in snap["n_slow"].values())),
            fast_total=int(snap["n_fast"].sum()),
        )
    f_mse = forget_stats(R, diag)
    f_acc = forget_stats(Acc, acc_diag)
    # 每任务相对回升 + 距离末尾距离（遗忘曲线上限轴）
    last = R[:, -1]
    per_task = []
    for i, name in enumerate(names):
        rel = (last[i] - diag[i]) / (diag[i] + 1e-8)
        per_task.append(dict(name=name, pos=i, family=name.split("_")[0],
                             difficulty=curric.difficulty[name],
                             distance=N - 1 - i,
                             diag=float(diag[i]), final=float(last[i]),
                             forget_rel=float(rel),
                             final_acc=float(Acc[i, -1]),
                             diag_acc=float(acc_diag[i])))
    return dict(seed=seed, with_memory=with_memory,
                R=R.tolist(), diag=diag.tolist(), acc_diag=acc_diag.tolist(),
                forget=dict(forget_mean=f_mse["forget_mean"],
                            forget_max=f_mse["forget_max"],
                            retain_mean=f_mse["retain_mean"]),
                forget_acc_mean=f_acc["forget_mean"],
                mem_stats=mem_stats,
                per_task=per_task,
                task_names=names,
                difficulty_counts=curric.difficulty_counts,
                families=curric.families)


def aggregate(mem_runs, no_runs, threshold=0.25):
    agg = {}
    # 判据主指标：带记忆每 seed 的 forget_mean/max 是否 ≤ 阈值
    fm = [r["forget"]["forget_mean"] for r in mem_runs]
    fx = [r["forget"]["forget_max"] for r in mem_runs]
    agg["forget_mean_mem"] = fm
    agg["forget_max_mem"] = fx
    agg["pass_forget_mean"] = all(v <= threshold for v in fm)
    agg["pass_forget_max"] = all(v <= threshold for v in fx)
    agg["forget_mean_avg"] = float(np.mean(fm))
    agg["forget_max_avg"] = float(np.mean(fx))
    # 无记忆对照（预期膨胀 -> 上限测试的对照组）
    no_fm = [r["forget"]["forget_mean"] for r in no_runs]
    no_fx = [r["forget"]["forget_max"] for r in no_runs]
    agg["forget_mean_no"] = no_fm
    agg["forget_max_no"] = no_fx
    agg["forget_mean_no_avg"] = float(np.mean(no_fm))
    agg["forget_max_no_avg"] = float(np.mean(no_fx))
    # 保留误差 + 记忆增益
    ret_m = np.mean([r["forget"]["retain_mean"] for r in mem_runs])
    ret_n = np.mean([r["forget"]["retain_mean"] for r in no_runs])
    agg["retain_mem_avg"] = float(ret_m)
    agg["retain_no_avg"] = float(ret_n)
    agg["memory_gain"] = float((ret_n - ret_m) / (ret_n + 1e-8))
    # 遗忘曲线上限：按 distance 分桶（均值相对回升）
    dist_bounds = [(0, 8), (9, 20), (21, 35), (36, 53)]
    for lb, ub in dist_bounds:
        def _dist(runs, key):
            vals = []
            for r in runs:
                for t in r["per_task"]:
                    if lb <= t["distance"] <= ub:
                        vals.append(t[key])
            return float(np.mean(vals)) if vals else float("nan")
        agg[f"forget_vs_dist_{lb}-{ub}_mem"] = _dist(mem_runs, "forget_rel")
        agg[f"forget_vs_dist_{lb}-{ub}_no"] = _dist(no_runs, "forget_rel")
    # 难度轴正压
    for d in (0, 1, 2, 3, 4):
        agg[f"forget_difficulty_{d}_mem"] = _by_difficulty(mem_runs, d, "forget_rel")
        agg[f"forget_difficulty_{d}_no"] = _by_difficulty(no_runs, d, "forget_rel")
    # 最差任务（forget_max 来源，带难度/距离/误差上下文诊断）
    agg["worst_mem"] = _worst(mem_runs)
    agg["worst_no"] = _worst(no_runs)
    agg["n_tasks"] = len(mem_runs[0]["task_names"])
    return agg


def _worst(runs):
    """聚合最差 forget_max 的任务名单（含 diag/final/distance，诊断 relative 放大）。"""
    worst = sorted((t for r in runs for t in r["per_task"]),
                   key=lambda t: t["forget_rel"], reverse=True)[:8]
    return [dict(name=t["name"], pos=t["pos"], difficulty=t["difficulty"],
                 distance=t["distance"], diag=round(t["diag"], 4),
                 final=round(t["final"], 4), forget_rel=round(t["forget_rel"], 3))
            for t in worst]


def _by_difficulty(runs, d, key):
    vals = []
    for r in runs:
        for t in r["per_task"]:
            if t["difficulty"] == d:
                vals.append(t[key])
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--eval", type=int, default=6)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--out", default=os.path.join(_cwd, "results_f2.json"))
    ap.add_argument("--ro", default="dual", choices=["dual", "self", "recon"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    dcfg = DeepConfig(dims=(256, 128, 112, 100, 80), ro_recon_mode=args.ro)
    mcfg = MemoryConfig(d=dcfg.dims[-1], n_slow_group=4, n_fast=16,
                        rate_fast=0.20, rate_slow=0.02, conf_thresh=0.20)

    _T0 = time.time()
    mem_runs, no_runs = [], []
    for seed in seeds:
        mem_runs.append(run_f2_sequential(seed, True, dcfg, mcfg,
                                          args.steps, args.eval))
        no_runs.append(run_f2_sequential(seed, False, dcfg, mcfg,
                                         args.steps, args.eval))
        print(f"[seed {seed}] mem forget_mean={mem_runs[-1]['forget']['forget_mean']:.3f}"
              f" max={mem_runs[-1]['forget']['forget_max']:.3f} | "
              f"no forget_mean={no_runs[-1]['forget']['forget_mean']:.3f}"
              f" max={no_runs[-1]['forget']['forget_max']:.3f}", flush=True)

    agg = aggregate(mem_runs, no_runs)
    meta = dict(n_tasks=len(mem_runs[0]["task_names"]),
                steps_per_task=args.steps, eval_samples=args.eval,
                seeds=seeds, dims=list(dcfg.dims), ro_recon_mode=dcfg.ro_recon_mode,
                n_slow_group=mcfg.n_slow_group,
                elapsed_s=round(time.time() - _T0, 1))
    out = dict(meta=meta, aggregate=agg, runs={"mem": mem_runs, "no": no_runs})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"== F2 50+ 任务免遗忘上限 == M={meta['n_tasks']} seeds={seeds} "
          f"elapsed={meta['elapsed_s']}s")
    print(f"  带记忆 forget_mean={agg['forget_mean_avg']:.3f} (≤0.25 "
          f"{'PASS' if agg['pass_forget_mean'] else 'FAIL'}) | "
          f"forget_max={agg['forget_max_avg']:.3f} "
          f"({'PASS' if agg['pass_forget_max'] else 'FAIL'})")
    print(f"  无记忆 forget_mean={agg['forget_mean_no_avg']:.3f} "
          f"forget_max={agg['forget_max_no_avg']:.3f}（对照，应显著高于带记忆 = 上限压力存在）")
    print(f"  保留误差 mem={agg['retain_mem_avg']:.4f} no={agg['retain_no_avg']:.4f} "
          f"记忆增益={agg['memory_gain']*100:.1f}%")
    for lb, ub in ((0, 8), (9, 20), (21, 35), (36, 53)):
        print(f"  distance[{lb:>2},{ub:<2}] mem={agg[f'forget_vs_dist_{lb}-{ub}_mem']:.3f}"
              f" | no={agg[f'forget_vs_dist_{lb}-{ub}_no']:.3f}")
    ms = mem_runs[0].get("mem_stats")
    if ms:
        print(f"  记忆填充(seed{seeds[0]}): slow_groups={ms['n_slow_groups']}"
              f" slow_writes={ms['slow_total']} fast_writes={ms['fast_total']}")
    print("  最差任务(带记忆): "
          + "; ".join(f"{w['name']}(d{w['difficulty']},dist{w['distance']})"
                      f"[diag{w['diag']}->final{w['final']}={w['forget_rel']:.2f}]"
                      for w in agg["worst_mem"][:5]))
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())