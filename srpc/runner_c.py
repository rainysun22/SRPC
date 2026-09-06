"""阶段 C（软件版内在化）实验运行器：结构稀疏核心 + 能力复验 + 能耗记账。

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 C（无专用硬件的软件版）：
    C1 能力无回撤：出生即稀疏核心（分块/扇入受限 + k-WTA）从头重训，
       阶段 B 四项验收全部复现 —— 低功耗来自架构本身，非稠密模型裁剪；
    C2 能效：每样本推理有效 MACs（事件驱动口径）比大模型标尺（6.1b）
       低至少 llm_ratio_min 倍（默认 10^3）；
    C3 结构由构造保证：权重密度 <= 阈值，掩码自出生不变（学习只改已有突触）；
    C4 部署就绪：训练后 int8 对称量化，冻结评估能力无回撤
       （低比特载体就绪；事件驱动/神经形态芯片本体 deferred 至有硬件）。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .arc import ArcLite
from .config import (AcceptanceBConfig, AcceptanceCConfig, ArcConfig,
                     CLConfig, DeepConfig, MemoryConfig, PhaseCConfig)
from .deepmodel import DeepSRPC
from .energy import llm_task_macs
from .runner_b import eval_combination, eval_task, evaluate_acceptance_b, run_sequential

MACS_KEYS = ("event", "struct", "dense")


def make_sparse_cfg(base: DeepConfig | None = None) -> DeepConfig:
    """阶段 C 稀疏核心配置：出生即结构稀疏（非训练后裁剪）。

    生成权重扇入 25%（层 1 连续感受野 + 内部层随机扇入）、读出头 40%、
    Wdyn 25%、k-WTA 保留 top-50% —— 权重总密度约 1/3，激活密度减半，
    全部由构造保证（局部学习只更新已有突触，掩码自出生不变）。
    """
    base = base or DeepConfig()
    return replace(base, fan_in_frac=0.25, fan_in_ro_frac=0.40,
                   fan_in_dyn_frac=0.25, kwta_frac=0.50, trace_energy=True)


def make_dense_cfg(base: DeepConfig | None = None) -> DeepConfig:
    """稠密对照臂：同超参、同 seeds、只开记账（能力与能耗对照）。"""
    base = base or DeepConfig()
    return replace(base, fan_in_frac=0.0, fan_in_ro_frac=0.0,
                   fan_in_dyn_frac=0.0, kwta_frac=0.0, trace_energy=True)


# ----------------------------------------------------------------------
# 每样本推理能耗 + 活跃率
# ----------------------------------------------------------------------
def measure_inference(model: DeepSRPC, arc: ArcLite, clcfg: CLConfig,
                      seed: int) -> dict:
    """冻结模型：每任务/组合的每样本推理 MACs（三口径）+ 每层平均活跃率。"""
    model.set_learning(False)
    n = clcfg.eval_samples
    per_task: dict[str, dict] = {}
    for name in arc.train_names:
        model.ledger.reset()
        err = eval_task(model, arc, name, clcfg, seed)
        t = model.ledger.totals()
        per_task[name] = dict(err=err, **{k: t[k] / n for k in MACS_KEYS})

    # 组合零样本（两次变换串联）能耗与误差
    combo_per: dict[str, dict] = {}
    combo_errs: dict[str, float] = {}
    for nm in arc.novel_names:
        t1, t2 = arc.novel_combos[arc.novel_names.index(nm)]
        c1 = arc._cond(arc.train_names.index(t1))
        c2 = arc._cond(arc.train_names.index(t2))
        model.ledger.reset()
        errs = []
        for _ in range(n):
            s_in, s_out, _ = arc.sample(nm)
            o1 = model.apply_transform(s_in, c1)
            o = model.apply_transform(o1, c2)
            errs.append(float(np.mean((o - s_out) ** 2)))
        t = model.ledger.totals()
        combo_per[nm] = {k: t[k] / n for k in MACS_KEYS}
        combo_errs[nm] = float(np.mean(errs))

    # 每层平均活跃率（单任务评估窗口内统计）
    model.ledger.reset()
    eval_task(model, arc, arc.train_names[0], clcfg, seed)
    act: dict[str, float] = {}
    for l in range(1, model.L + 1):
        k = f"act_{l}"
        if model.ledger.calls(k):
            act[f"x{l}"] = float(model.ledger.mean(k) / model.dims[l])

    single = {k: float(np.mean([per_task[t][k] for t in per_task]))
              for k in MACS_KEYS}
    combo = {k: float(np.mean([combo_per[nm][k] for nm in combo_per]))
             for k in MACS_KEYS}
    return dict(
        per_task=per_task, single=single,
        single_err=float(np.mean([per_task[t]["err"] for t in per_task])),
        combo=combo, combo_per=combo_per,
        combo_err=float(np.mean([combo_errs[nm] for nm in combo_errs])),
        combo_err_per=combo_errs, active_frac=act,
    )


# ----------------------------------------------------------------------
# 结构统计（C3：密度 + 掩码自出生不变）
# ----------------------------------------------------------------------
def structural_stats(model: DeepSRPC) -> dict:
    layers: dict[str, dict] = {}
    nnz_tot = size_tot = 0
    for l in range(1, model.L + 1):
        W = model.Ws[l]
        nnz, size = int(np.count_nonzero(W)), W.size
        layers[f"W{l}(x{l}->{l-1})"] = dict(
            density=nnz / size, nnz=nnz, size=size, fan_in=model.colfan[l])
        nnz_tot += nnz
        size_tot += size
    ro_nnz = ro_size = 0
    for W in model.W_outs:
        if W is None:
            continue
        ro_nnz += int(np.count_nonzero(W))
        ro_size += W.size
    dyn_nnz, dyn_size = int(np.count_nonzero(model.Wdyn)), model.Wdyn.size
    nnz_tot += ro_nnz + dyn_nnz
    size_tot += ro_size + dyn_size
    masks_ok = all(
        (m is None and s is None)
        or (m is not None and s is not None and hash(m.tobytes()) == s)
        for m, s in zip(model.masks, model.mask_sig))
    ro_ok = all(
        (m is None and s is None)
        or (m is not None and s is not None and hash(m.tobytes()) == s)
        for m, s in zip(model.ro_masks, model.ro_sig))
    return dict(
        layers=layers,
        readout=dict(density=ro_nnz / max(ro_size, 1), nnz=ro_nnz, size=ro_size),
        dyn=dict(density=dyn_nnz / dyn_size, nnz=dyn_nnz, size=dyn_size),
        total=dict(density=nnz_tot / size_tot, nnz=nnz_tot, size=size_tot),
        masks_unchanged=bool(masks_ok and ro_ok),
    )


# ----------------------------------------------------------------------
# C4：训练后 int8 对称量化检查（低比特部署就绪）
# ----------------------------------------------------------------------
def _quantize_int8(W: np.ndarray) -> None:
    """对称 int8 量化（原地）：scale = max|W| / 127。"""
    s = float(np.max(np.abs(W))) / 127.0
    if s < 1e-12:
        return
    W[:] = np.clip(np.round(W / s), -127, 127) * s


def int8_check(model: DeepSRPC, arc: ArcLite, clcfg: CLConfig, seed: int) -> dict:
    """训练后量化全部权重（Ws / W_outs / Wdyn）为 int8，冻结评估能力对比。

    记忆原型为内部状态（非权重），不量化。评估后从快照恢复 fp 权重。
    """
    snap = model.snapshot()
    mem_snap = model.memory.snapshot() if model.memory is not None else None
    for W in model.Ws[1:]:
        _quantize_int8(W)
    for W in model.W_outs:
        if W is not None:
            _quantize_int8(W)
    _quantize_int8(model.Wdyn)
    model.set_learning(False)
    err_q = float(np.mean([eval_task(model, arc, nm, clcfg, seed)
                           for nm in arc.train_names]))
    combo_q = eval_combination(model, arc, clcfg, seed)
    combo_err_q = float(np.mean([combo_q[k] for k in arc.novel_names]))
    rand_err_q = float(np.mean([combo_q[k + "_rand"] for k in arc.novel_names]))
    model.restore(snap)
    if mem_snap is not None:
        model.memory.restore(mem_snap)
    return dict(train_err=err_q, combo_err=combo_err_q, rand_err=rand_err_q,
                combo_gain=(rand_err_q - combo_err_q) / (rand_err_q + 1e-8))


# ----------------------------------------------------------------------
# 阶段 C 主流程与验收
# ----------------------------------------------------------------------
def run_phase_c(seeds: list[int], dcfg_sparse: DeepConfig, mcfg: MemoryConfig,
                acfg: ArcConfig, clcfg: CLConfig, pcfg: PhaseCConfig,
                accfg: AcceptanceCConfig | None = None,
                accb: AcceptanceBConfig | None = None) -> dict:
    accfg = accfg or AcceptanceCConfig()
    accb = accb or AcceptanceBConfig()

    # 1) 稀疏核心（出生即稀疏）顺序学习：mem + no-mem（trace 全开）
    sp_mem = [run_sequential(s, True, dcfg_sparse, mcfg, acfg, clcfg)
              for s in seeds]
    sp_no = [run_sequential(s, False, dcfg_sparse, mcfg, acfg, clcfg)
             for s in seeds]

    # 2) C1：阶段 B 四项验收在稀疏核心上复跑
    combo_sp = {k: float(np.mean([r["combo"][k] for r in sp_mem]))
                for k in sp_mem[0]["combo"]}
    acc_b = evaluate_acceptance_b(sp_mem, sp_no, combo_sp, accb)

    # 3) 稠密对照臂（同 seeds，同超参，只开记账）
    dcfg_dense = make_dense_cfg(dcfg_sparse)
    dn_mem = [run_sequential(s, True, dcfg_dense, mcfg, acfg, clcfg)
              for s in seeds]
    combo_dn = {k: float(np.mean([r["combo"][k] for r in dn_mem]))
                for k in dn_mem[0]["combo"]}

    # 4) 每样本推理能耗 + 活跃率（稀疏核心与稠密对照，seed 0）
    model0 = sp_mem[0]["model"]
    arc0 = ArcLite(acfg, np.random.default_rng(seeds[0] * 3000 + 2))
    energy = measure_inference(model0, arc0, clcfg, seeds[0])
    energy_dn = measure_inference(dn_mem[0]["model"], arc0, clcfg, seeds[0])

    # 5) C3：结构统计
    struct = structural_stats(model0)

    # 6) C4：int8 量化检查
    quant = int8_check(model0, arc0, clcfg, seeds[0])

    # 7) C2：大模型标尺与比率
    yard = {nm: llm_task_macs(p, pcfg.tokens_per_task)
            for nm, p in zip(pcfg.yardstick_names, pcfg.yardstick_params)}
    ratios = {nm: yard[nm] / max(energy["single"]["event"], 1e-12)
              for nm in yard}

    # 8) 验收汇总
    fp_single = energy["single_err"]
    fp_combo = energy["combo_err"]
    qmax = accfg.quant_err_ratio_max
    c1 = bool(acc_b["all_pass"])
    c2 = bool(min(ratios.values()) >= accfg.llm_ratio_min)
    c3 = bool(struct["total"]["density"] <= accfg.density_max
              and struct["masks_unchanged"])
    c4 = bool(quant["train_err"] <= max(fp_single * qmax, fp_single + 0.02)
              and quant["combo_err"] <= max(fp_combo * qmax, fp_combo + 0.02))
    acceptance = dict(
        capability=dict(pass_=c1, phase_b=acc_b),
        energy=dict(pass_=c2, per_sample_event=energy["single"]["event"],
                    ratios=ratios, ratio_min=float(min(ratios.values()))),
        structure=dict(pass_=c3, density=struct["total"]["density"],
                       masks_unchanged=struct["masks_unchanged"]),
        quantization=dict(pass_=c4, fp_single_err=fp_single,
                          quant_single_err=quant["train_err"],
                          fp_combo_err=fp_combo, quant_combo_err=quant["combo_err"]),
        all_pass=bool(c1 and c2 and c3 and c4),
    )

    # 训练总能耗（跨 seed 均值，稀疏 vs 稠密对照）
    train_macs_sp = {k: float(np.mean([r["train_macs"][k] for r in sp_mem]))
                     for k in MACS_KEYS}
    train_macs_dn = {k: float(np.mean([r["train_macs"][k] for r in dn_mem]))
                     for k in MACS_KEYS}
    # 能力对照（保留误差末列均值 + 组合零样本）
    retain_sp = np.mean([r["R"][:, -1] for r in sp_mem], axis=0).tolist()
    retain_dn = np.mean([r["R"][:, -1] for r in dn_mem], axis=0).tolist()

    return dict(
        acceptance=acceptance,
        energy=energy,
        energy_dense=energy_dn,
        structure=struct,
        quant=quant,
        yardstick=dict(macs=yard, tokens=pcfg.tokens_per_task),
        train_macs=dict(sparse=train_macs_sp, dense=train_macs_dn),
        capability=dict(
            task_names=sp_mem[0]["task_names"],
            retain_sparse=retain_sp, retain_dense=retain_dn,
            combo_sparse={k: combo_sp[k] for k in sp_mem[0]["novel_names"]},
            combo_dense={k: combo_dn[k] for k in sp_mem[0]["novel_names"]},
            combo_rand_sparse={k + "_rand": combo_sp[k + "_rand"]
                               for k in sp_mem[0]["novel_names"]},
            event_rates_sparse=float(np.mean(
                [np.mean(r["event_rates"]) for r in sp_mem])),
        ),
    )
