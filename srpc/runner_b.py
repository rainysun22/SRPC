"""阶段 B 实验运行器：顺序学习（免遗忘）+ ARC-lite 组合泛化 + 能力曲线 + 验收。

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B：
    1. 能力随交互上升（任务内预测误差下降）
    2. 免遗忘（顺序学习后旧任务误差不回升；多时间尺度记忆对照）
    3. 组合泛化（保留组合零样本：组合嵌入 vs 随机条件基线）
    4. Pareto 回归（Phase-0 7.5 验收不退化，见 scripts/run_phaseB.py）
"""
from __future__ import annotations

import numpy as np

from .arc import ArcLite
from .config import (AcceptanceBConfig, ArcConfig, CLConfig, DeepConfig,
                     MemoryConfig)
from .deepmodel import DeepSRPC
from .memory import PrototypeMemory
from .metrics import linreg_slope


# ----------------------------------------------------------------------
# 冻结评估
# ----------------------------------------------------------------------
def eval_task(model: DeepSRPC, arc: ArcLite, name: str,
              clcfg: CLConfig, seed: int) -> float:
    """冻结评估单任务：编码输入 -> 读出，返回均方误差（learning 已关闭）。"""
    rng = np.random.default_rng(seed * 977 + 13)
    errs = []
    for m in range(clcfg.eval_samples):
        s_in, s_out, cond = arc.sample(name)
        out = model.apply_transform(s_in, cond)
        errs.append(float(np.mean((out - s_out) ** 2)))
    return float(np.mean(errs))


def eval_combination(model: DeepSRPC, arc: ArcLite,
                     clcfg: CLConfig, seed: int) -> dict:
    """组合零样本评估：训练变换 + 保留组合（顺序复合 vs 随机条件基线）。

    零样本组合 = 跨时段片段重组：对输入先施加已学变换 t1（头 t1 读出），
    再把读出结果直接作为输入施加已学变换 t2（头 t2 读出）—— 复用两个
    已学变换片段重组出新变换（一热空间内各变换为线性映射，组合近精确）。
    随机条件基线：两次都用无信息均匀条件（不指向任何头，读出为多头混合）。
    """
    model.set_learning(False)
    res = {}
    for name in arc.train_names:
        res[name] = eval_task(model, arc, name, clcfg, seed)
    rng = np.random.default_rng(seed * 991 + 7)
    for nm in arc.novel_names:
        t1, t2 = arc.novel_combos[arc.novel_names.index(nm)]
        c1 = arc._cond(arc.train_names.index(t1))
        c2 = arc._cond(arc.train_names.index(t2))
        errs_c, errs_r = [], []
        for m in range(clcfg.eval_samples):
            s_in, s_out, _ = arc.sample(nm)
            # 顺序复合 = 先 t1 读出、解码回输入格式、再以 t2 读出（arc.decode_grid 契约 /
            # §7.7 写方向：中间读出在离散基底上做 argmax 转回网格再喂入下一段）。
            out1 = model.apply_transform(s_in, c1)
            out = model.apply_transform(
                arc._onehot(arc.decode_grid(out1)).astype(float), c2)
            errs_c.append(float(np.mean((out - s_out) ** 2)))
            # 无信息条件基线：均匀权重（不指向任何任务头），读出为多头混合
            r1 = np.full(arc.n_train, 1.0 / arc.n_train)
            r2 = np.full(arc.n_train, 1.0 / arc.n_train)
            out_r1 = model.apply_transform(s_in, r1)
            out_r = model.apply_transform(out_r1, r2)
            errs_r.append(float(np.mean((out_r - s_out) ** 2)))
        res[nm] = float(np.mean(errs_c))
        res[nm + "_rand"] = float(np.mean(errs_r))
    return res


# ----------------------------------------------------------------------
# 顺序学习（免遗忘）
# ----------------------------------------------------------------------
def run_sequential(seed: int, with_memory: bool,
                   dcfg: DeepConfig, mcfg: MemoryConfig,
                   acfg: ArcConfig, clcfg: CLConfig) -> dict:
    """按任务序列顺序训练 4 个变换，返回 backward-transfer 矩阵与能力曲线。"""
    rng = np.random.default_rng(seed * 3000 + 1)
    rng_arc = np.random.default_rng(seed * 3000 + 2)
    mem = PrototypeMemory(mcfg, rng) if with_memory else None
    model = DeepSRPC(dcfg, 0, rng, self_loop=True, memory=mem)
    arc = ArcLite(acfg, rng_arc)

    n_tasks = arc.n_train
    per_task = clcfg.steps_per_task
    # R[i][j] = 任务 j 训练结束后，任务 i 的冻结 readout 误差（backward transfer）
    R = np.zeros((n_tasks, n_tasks))
    diag = np.zeros(n_tasks)
    curves: list[np.ndarray] = []  # 每任务内 EMA 误差序列（能力上升）
    event_rates = []               # 每任务内事件率均值（不变量 3 能量代理）
    # 阶段 C：训练/评估分段能耗（trace 关闭时 ledger 恒为 0，零开销零影响）
    macs_keys = ("event", "struct", "dense")
    train_macs = {k: 0.0 for k in macs_keys}
    eval_macs = {k: 0.0 for k in macs_keys}

    for j, name in enumerate(arc.train_names):
        model.ledger.reset()
        ema_val = 0.0
        ev_accum = 0.0
        curve = np.empty(per_task)
        for t in range(per_task):
            s_in, s_out, cond = arc.sample(name)
            info = model.step_mapping(s_in, s_out, cond)
            e2 = info["e_readout"] ** 2
            ema_val = e2 if t == 0 else ema_val * 0.99 + e2 * 0.01
            curve[t] = ema_val
            ev_accum += float(np.mean(info.get("evs", [0.0])))
        curves.append(curve)
        event_rates.append(ev_accum / per_task)
        t_ = model.ledger.totals()
        for k in macs_keys:
            train_macs[k] += t_[k]
        # 冻结评估所有已见任务（backward transfer）
        model.set_learning(False)
        model.ledger.reset()
        for i in range(j + 1):
            R[i, j] = eval_task(model, arc, arc.train_names[i], clcfg, seed)
        t_ = model.ledger.totals()
        for k in macs_keys:
            eval_macs[k] += t_[k]
        model.set_learning(True)
        diag[j] = R[j, j]

    # 组合零样本评估（学完全部任务后）
    model.ledger.reset()
    combo = eval_combination(model, arc, clcfg, seed)
    t_ = model.ledger.totals()
    for k in macs_keys:
        eval_macs[k] += t_[k]
    return dict(R=R, diag=diag, curves=curves, event_rates=event_rates,
                combo=combo, model=model, mem=mem,
                task_names=arc.train_names, novel_names=arc.novel_names,
                train_macs=train_macs, eval_macs=eval_macs)


def forget_stats(R: np.ndarray, diag: np.ndarray) -> dict:
    """免遗忘指标：末列相对回升（均值/最大）与保留误差均值。"""
    n = R.shape[0]
    last = R[:, -1]
    rel = (last - diag) / (diag + 1e-8)
    return dict(
        forget_mean=float(np.mean(rel)),
        forget_max=float(np.max(rel)),
        retain_mean=float(np.mean(last)),
        last=last.tolist(),
        diag=diag.tolist(),
    )


# ----------------------------------------------------------------------
# 阶段 B 验收（可证伪里程碑，跨 seed 聚合：每个 seed 都必须通过）
# ----------------------------------------------------------------------
def _mean_rows(f_stats: list[dict]) -> dict:
    """把 per-seed forget_stats 聚合为均值（保留 same keys）。"""
    keys = ["forget_mean", "forget_max", "retain_mean", "last", "diag"]
    agg = {}
    for k in keys:
        if k in ("last", "diag"):
            agg[k] = np.mean([f[k] for f in f_stats], axis=0).tolist()
        else:
            agg[k] = float(np.mean([f[k] for f in f_stats]))
    return agg


def evaluate_acceptance_b(mem_runs: list[dict], no_mem_runs: list[dict],
                          combo: dict,
                          acfg: AcceptanceBConfig | None = None) -> dict:
    acfg = acfg or AcceptanceBConfig()
    crit = {}

    # 条件 1：能力随交互上升（每任务误差 EMA 下降；每个 seed 每个任务都须 < 0）
    slopes = [[linreg_slope(np.asarray(c, dtype=float)) for c in r["curves"]]
              for r in mem_runs]
    flat = [s for run in slopes for s in run]
    c1_pass = all(s < acfg.learn_slope_max for s in flat)
    crit["learning"] = dict(pass_=bool(c1_pass),
                            slopes=np.mean(slopes, axis=0).tolist(),
                            worst=min(flat))

    # 条件 2：免遗忘（带记忆；末列相对回升 <= 阈值；每个 seed 都须通过）
    f_mem = [forget_stats(r["R"], r["diag"]) for r in mem_runs]
    c2_pass = all(f["forget_mean"] <= acfg.forget_rel_max for f in f_mem)
    crit["forgetting_mem"] = dict(pass_=bool(c2_pass), **_mean_rows(f_mem))

    # 条件 3：记忆增益（带记忆 vs 无记忆的保留能力改善 >= 阈值；每个 seed 都须通过）。
    # 顺序学习后比较"全部已见任务的保留误差"：免遗忘结构 + 记忆先验应使旧任务
    # 能力保持显著优于无记忆（无记忆模型误差恒高 -> 相对回升小，用保留误差绝对
    # 比较才公平，对应里程碑"能力随交互上升且免遗忘"）。
    f_no = [forget_stats(r["R"], r["diag"]) for r in no_mem_runs]
    gains = [(fn["retain_mean"] - fm["retain_mean"]) / (fn["retain_mean"] + 1e-8)
             for fm, fn in zip(f_mem, f_no)]
    c3_pass = all(g >= acfg.retain_gain_min for g in gains)
    crit["memory_gain"] = dict(
        pass_=bool(c3_pass),
        retain_mem=float(np.mean([fm["retain_mean"] for fm in f_mem])),
        retain_no=float(np.mean([fn["retain_mean"] for fn in f_no])),
        gain_frac=float(np.mean(gains)), gains=gains)

    # 条件 4：组合泛化（组合嵌入 vs 随机条件，零样本；combo 已跨 seed 平均）
    novel = list(mem_runs[0]["novel_names"])
    e_c = float(np.mean([combo[k] for k in novel]))
    e_r = float(np.mean([combo[k + "_rand"] for k in novel]))
    gain_c = (e_r - e_c) / (e_r + 1e-8)
    c4_pass = gain_c >= acfg.combo_gain_min
    crit["combination"] = dict(pass_=bool(c4_pass), combo_err=e_c,
                               rand_err=e_r, gain_frac=gain_c,
                               per_combo={k: combo[k] for k in novel})

    crit["all_pass"] = all(v["pass_"] for k, v in crit.items() if k != "all_pass")
    return crit
