"""Phase-0 实验运行器：Track-1（形成/修正/自省 A/B）与 Track-2（组合泛化）。

对应 docs/SRPC_DESIGN.md：
    7.4 验证指标 —— 形成 / 修正 / 组合 / 自省
    7.5 验收标准 —— 自组织结构 / 误差单调下降 / 自省非零增益 / 无反传·在线
"""
from __future__ import annotations

import pathlib

import numpy as np

from .config import (AcceptanceConfig, CreditConfig, FieldConfig, ModelConfig,
                     SlotConfig, Track1Config, Track2Config)
from .env import N_FIELD_ACTIONS, SlotWorld, SourceFieldWorld
from .metrics import (best_cosine, cosine, ema, linreg_slope, nmi,
                      nmi_perm_pvalue, recovery_stats)
from .model import SRPCModel, FlatPCModel, LookupModel


# ----------------------------------------------------------------------
# Track 1：交互式导航（形成 / 修正 / 自省 A/B）
# ----------------------------------------------------------------------
def eps_schedule(t: int, cfg: Track1Config) -> float:
    if t >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = t / cfg.eps_decay_steps
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac


def run_field(seed: int, self_loop: bool,
              mcfg: ModelConfig | None = None,
              fcfg: FieldConfig | None = None,
              tcfg: Track1Config | None = None) -> dict:
    mcfg = mcfg or ModelConfig()
    fcfg = fcfg or FieldConfig()
    tcfg = tcfg or Track1Config()

    rng_env = np.random.default_rng(seed * 1000 + 1)
    rng_model = np.random.default_rng(seed * 1000 + 2)
    rng_act = np.random.default_rng(seed * 1000 + 3)

    world = SourceFieldWorld(fcfg, mcfg.d_obs, rng_env)
    model = SRPCModel(mcfg, N_FIELD_ACTIONS, rng_model, self_loop=self_loop)

    n = tcfg.steps
    p = tcfg.perturb_step
    e0sq = np.empty(n)
    ev = np.empty(n)
    boost = np.empty(n)
    eself = np.empty(n)
    zone = np.empty(n, dtype=int)
    x2arg = np.empty(n, dtype=int)

    rf_init = best_cosine(world.patterns, model.W10)
    patterns_pre = world.patterns.copy()
    snap_pre = None

    for t in range(n):
        s = world.observe()
        info = model.observe(s)
        e0sq[t] = info["e0"] ** 2
        ev[t] = (info["ev1"] + info["ev2"] + info["evs"]) / 3.0
        boost[t] = info["boost"]
        eself[t] = info["e_self"]
        zone[t] = world.zone()
        x2arg[t] = int(np.argmax(model.x2))
        a = model.select_action(eps_schedule(t, tcfg), rng_act)
        world.step(a)
        model.prepare_next(a)
        if t + 1 == p:
            world.perturb(rng_env)
            snap_pre = model.snapshot()

    err_ema = ema(e0sq, tcfg.ema_alpha)
    pre, post = err_ema[:p], err_ema[p:]

    # 形成：扰动前权重 vs 扰动前模式；修正后：最终权重 vs 重映射后的生效模式
    rf_pre = best_cosine(patterns_pre, snap_pre["W10"])
    patterns_final = world.effective_patterns()
    rf_final = best_cosine(patterns_final, model.W10)

    nz = nmi(zone[:p], x2arg[:p])
    nz_p, nz_perm = nmi_perm_pvalue(zone[:p], x2arg[:p], rng=np.random.default_rng(seed))

    rec = recovery_stats(err_ema, p, tcfg.recovery_base_win,
                         tcfg.recovery_mult, tcfg.recovery_sustain,
                         tcfg.relearn_lo, tcfg.relearn_hi)
    d10 = max(1, len(pre) // 10)

    metrics = dict(
        # 修正（7.4）：误差随交互下降 + 扰动恢复
        slope_pre=linreg_slope(pre),
        decile_first=float(pre[:d10].mean()),
        decile_last=float(pre[-d10:].mean()),
        steady_pre=float(pre[-max(1, len(pre) * 15 // 100):].mean()),
        steady_post=float(post[-max(1, len(post) * 15 // 100):].mean()),
        recovery_steps=float(rec["recovery_steps"]),
        relearn_error=float(rec["relearn_error"]),
        recovery_peak=float(rec["peak"]),
        recovery_excess=float(rec["excess_error"]),
        # 形成（7.4）：感受野/概念分组
        rf_init_mean=float(rf_init.mean()),
        rf_pre_mean=float(rf_pre.mean()),
        rf_final_mean=float(rf_final.mean()),
        rf_pre_captured=float((rf_pre >= 0.75).mean()),
        rf_final_captured=float((rf_final >= 0.75).mean()),
        nmi_zone=float(nz),
        nmi_zone_p=float(nz_p),
        nmi_zone_perm=float(nz_perm),
        # 能量代理（不变量 3）：事件驱动更新占比
        event_rate_mean=float(ev.mean()),
        event_rate_last=float(ev[-2000:].mean()),
        event_rate_first=float(ev[:2000].mean()),
        eself_post_mean=float(eself[p:].mean()),
        boost_mean=float(boost[p:].mean()),
    )

    logs = dict(err_ema=err_ema, e0sq=e0sq, ev=ev, boost=boost, eself=eself,
                zone=zone, x2arg=x2arg,
                rf_init=rf_init, rf_pre=rf_pre, rf_final=rf_final,
                patterns_pre=patterns_pre, patterns_final=patterns_final,
                W10_pre=snap_pre["W10"], W10_final=model.W10.copy(),
                action_counts=model.n_action.copy(),
                U_action=model.U_action.copy())
    return dict(metrics=metrics, logs=logs)


def _aggregate(runs: list[dict]) -> dict:
    keys = runs[0]["metrics"].keys()
    mean = {k: float(np.mean([r["metrics"][k] for r in runs])) for k in keys}
    std = {k: float(np.std([r["metrics"][k] for r in runs])) for k in keys}
    return dict(metrics_mean=mean, metrics_std=std,
                per_seed=[r["metrics"] for r in runs], runs=runs)


def run_track1(seeds=(0, 1, 2), mcfg=None, fcfg=None, tcfg=None) -> dict:
    on_runs = [run_field(s, True, mcfg, fcfg, tcfg) for s in seeds]
    off_runs = [run_field(s, False, mcfg, fcfg, tcfg) for s in seeds]
    return dict(on=_aggregate(on_runs), off=_aggregate(off_runs), seeds=list(seeds))


# ----------------------------------------------------------------------
# Track 2：组合流（组合泛化 / 基线对照 / 自省对组合的增益）
# ----------------------------------------------------------------------
def run_slot(seed: int, self_loop: bool,
             mcfg: ModelConfig | None = None,
             scfg: SlotConfig | None = None,
             t2cfg: Track2Config | None = None) -> dict:
    mcfg = mcfg or ModelConfig()
    scfg = scfg or SlotConfig()
    t2cfg = t2cfg or Track2Config()

    rng_env = np.random.default_rng(seed * 2000 + 1)
    rng_model = np.random.default_rng(seed * 2000 + 2)
    rng_flat = np.random.default_rng(seed * 2000 + 3)

    world = SlotWorld(scfg, mcfg.d_obs, rng_env)
    model = SRPCModel(mcfg, 0, rng_model, self_loop=self_loop)
    flat = FlatPCModel(mcfg.d_obs, mcfg.n_l1 + mcfg.n_l2, rng_flat)
    lookup = LookupModel(mcfg.d_obs)

    n = t2cfg.train_steps
    e0sq = np.empty(n)
    ctx = np.empty(n, dtype=int)
    x2arg = np.empty(n, dtype=int)

    rf_init = best_cosine(world.patterns, model.W10)

    for t in range(n):
        s, c = world.observe()
        info = model.observe(s)
        model.prepare_next(None)
        flat.observe(s)
        lookup.observe(s, c)
        e0sq[t] = info["e0"] ** 2
        ctx[t] = c
        x2arg[t] = int(np.argmax(model.x2))

    # 形成：x1 感受野对齐特征模式；x2 概念分组对齐上下文
    rf_feat = best_cosine(world.patterns, model.W10)
    nc = nmi(ctx, x2arg)
    nc_p, nc_perm = nmi_perm_pvalue(ctx, x2arg, rng=np.random.default_rng(seed))

    # ---- 冻结评估（零样本组合泛化） ----
    K = world.K
    n_train_pairs = len(world.train_pairs)
    n_ctx = len(world.ctx_feats)
    train_pair_ctx = list(range(K, K + n_train_pairs))
    novel_ctx = list(range(K + n_train_pairs, n_ctx))

    snap = model.snapshot()
    model.set_learning(False)
    srpc_err, srpc_x1, srpc_x2, srpc_ws = {}, {}, {}, {}
    settle = max(2, t2cfg.probe_samples // 3)
    for c in range(n_ctx):
        errs, x1s, x2s, wss = [], [], [], []
        for m in range(t2cfg.probe_samples):
            s = world.sample_ctx(c)
            info = model.observe(s)
            model.prepare_next(None)
            if m >= settle:
                errs.append(info["e0"] ** 2)
                x1s.append(model.x1.copy())
                x2s.append(model.x2.copy())
                g, _ = model.workspace()
                wss.append(g)
        srpc_err[c] = float(np.mean(errs))
        srpc_x1[c] = np.mean(x1s, axis=0)
        srpc_x2[c] = np.mean(x2s, axis=0)
        srpc_ws[c] = np.mean(wss, axis=0)

    flat.set_learning(False)
    flat_err = {}
    for c in range(n_ctx):
        errs = []
        for m in range(t2cfg.probe_samples):
            s = world.sample_ctx(c)
            errs.append(flat.observe(s) ** 2)
        flat_err[c] = float(np.mean(errs))

    lookup_err = {}
    for c in range(n_ctx):
        errs = []
        for m in range(t2cfg.probe_samples):
            s = world.sample_ctx(c)
            errs.append(float(np.sum((s - lookup.predict(c)) ** 2)))
        lookup_err[c] = float(np.mean(errs))

    # ---- 组合（7.4）：既有片段重组出新概念 ----
    def pair_recomb(pair_ctx_list, ws=False):
        vals = []
        for c in pair_ctx_list:
            i, j = world.ctx_feats[c]
            base_i = srpc_ws[i] if ws else srpc_x1[i]
            base_j = srpc_ws[j] if ws else srpc_x1[j]
            target = srpc_ws[c] if ws else srpc_x1[c]
            vals.append(cosine(target, base_i + base_j))
        return float(np.mean(vals))

    recomb_x1_novel = pair_recomb(novel_ctx)
    recomb_x1_train = pair_recomb(train_pair_ctx)
    recomb_ws_novel = pair_recomb(novel_ctx, ws=True)

    # 概念层新颖性：novel 对的 x2 模式与所有训练对模式的距离（相对训练对内部距离）
    tp = [srpc_x2[c] for c in train_pair_ctx]
    d_ref = float(np.mean([np.linalg.norm(a - b) for a in tp for b in tp
                           if not np.array_equal(a, b)]))
    novelty = float(np.mean([min(np.linalg.norm(srpc_x2[c] - a) for a in tp)
                             for c in novel_ctx]) / max(d_ref, 1e-8))

    zs = dict(
        srpc_single=float(np.mean([srpc_err[c] for c in range(K)])),
        srpc_train=float(np.mean([srpc_err[c] for c in train_pair_ctx])),
        srpc_novel=float(np.mean([srpc_err[c] for c in novel_ctx])),
        flat_train=float(np.mean([flat_err[c] for c in train_pair_ctx])),
        flat_novel=float(np.mean([flat_err[c] for c in novel_ctx])),
        lookup_train=float(np.mean([lookup_err[c] for c in train_pair_ctx])),
        lookup_novel=float(np.mean([lookup_err[c] for c in novel_ctx])),
    )

    # ---- 少样本在线适应（自省环对组合适应的增益） ----
    model.restore(snap)
    model.set_learning(True)
    world.set_phase("novel")
    fs = np.empty(t2cfg.fewshot_steps)
    for t in range(t2cfg.fewshot_steps):
        s, _ = world.observe()
        info = model.observe(s)
        model.prepare_next(None)
        fs[t] = info["e0"] ** 2
    fs_ema = ema(fs, 0.02)
    fewshot_first = float(fs_ema[: min(100, len(fs_ema) // 5)].mean())
    fewshot_last = float(fs_ema[-100:].mean())

    metrics = dict(
        rf_feat_init=float(rf_init.mean()),
        rf_feat_mean=float(rf_feat.mean()),
        rf_feat_captured=float((rf_feat >= 0.75).mean()),
        nmi_ctx=float(nc),
        nmi_ctx_p=float(nc_p),
        nmi_ctx_perm=float(nc_perm),
        recomb_x1_novel=recomb_x1_novel,
        recomb_x1_train=recomb_x1_train,
        recomb_ws_novel=recomb_ws_novel,
        x2_novelty_ratio=novelty,
        fewshot_first=fewshot_first,
        fewshot_last=fewshot_last,
        fewshot_gain=fewshot_first - fewshot_last,
        **{f"zs_{k}": v for k, v in zs.items()},
        train_err_last=float(ema(e0sq, 0.01)[-200:].mean()),
    )

    logs = dict(fs_ema=fs_ema, rf_feat=rf_feat, rf_init=rf_init,
                srpc_err=srpc_err, flat_err=flat_err, lookup_err=lookup_err,
                srpc_x2=srpc_x2, W10=model.W10.copy(),
                patterns=world.patterns.copy(), ctx=ctx)
    return dict(metrics=metrics, logs=logs)


def run_track2(seeds=(0, 1), mcfg=None, scfg=None, t2cfg=None) -> dict:
    on_runs = [run_slot(s, True, mcfg, scfg, t2cfg) for s in seeds]
    off_runs = [run_slot(s, False, mcfg, scfg, t2cfg) for s in seeds]
    return dict(on=_aggregate(on_runs), off=_aggregate(off_runs), seeds=list(seeds))


# ----------------------------------------------------------------------
# 7.5 验收标准
# ----------------------------------------------------------------------
def check_no_backprop() -> tuple[bool, list]:
    """静态检查：核心代码不依赖任何自动微分/反传框架。

    （特征串用拼接构造，避免检查器匹配到自身源码。）
    """
    import srpc
    bad = []
    root = pathlib.Path(srpc.__file__).parent
    tokens = ("tor" + "ch", "ja" + "x", "tensor" + "flow",
              ".back" + "ward(", "auto" + "grad")
    for py in root.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        for token in tokens:
            if token in src:
                bad.append((py.name, token))
    return len(bad) == 0, bad


def check_online_incremental() -> bool:
    """在线性：模型逐样本更新权重，且不持有任何数据缓冲。"""
    rng = np.random.default_rng(7)
    m = SRPCModel(ModelConfig(), 0, rng)
    w0 = m.W10.copy()
    m.observe(rng.uniform(0.0, 1.0, m.cfg.d_obs))
    m.prepare_next(None)
    changed = not np.allclose(w0, m.W10)
    no_buffer = not any(isinstance(v, list) for v in vars(m).values())
    return bool(changed and no_buffer)


def evaluate_acceptance(t1: dict, t2: dict, acfg: AcceptanceConfig | None = None,
                        cr: dict | None = None,
                        ccfg: CreditConfig | None = None) -> dict:
    """7.5 验收标准。cr 为信用分配早筛结果（§8.5，None 则跳过该条件）。"""
    acfg = acfg or AcceptanceConfig()
    crit = {}

    # ---- 条件 1：存在可观测的自组织层级结构 ----
    on1 = t1["on"]["metrics_mean"]
    nmi_sig_frac = float(np.mean([p < acfg.nmi_p_max
                                  for p in [r["nmi_zone_p"] for r in t1["on"]["per_seed"]]]))
    on2 = t2["on"]["metrics_mean"]
    nmi2_sig_frac = float(np.mean([p < acfg.nmi_p_max
                                   for p in [r["nmi_ctx_p"] for r in t2["on"]["per_seed"]]]))
    c1 = dict(
        rf_captured=on1["rf_pre_captured"],
        rf_improve=on1["rf_pre_mean"] - on1["rf_init_mean"],
        nmi_zone=on1["nmi_zone"], nmi_zone_sig_frac=nmi_sig_frac,
        rf_feat_captured=on2["rf_feat_captured"],
        rf_feat_improve=on2["rf_feat_mean"] - on2["rf_feat_init"],
        nmi_ctx=on2["nmi_ctx"], nmi_ctx_sig_frac=nmi2_sig_frac,
    )
    c1_pass = (c1["rf_captured"] >= acfg.rf_captured_frac
               and c1["rf_improve"] >= acfg.rf_improve_min
               and c1["nmi_zone_sig_frac"] >= 0.5
               and c1["rf_feat_captured"] >= acfg.rf_captured_frac
               and c1["nmi_ctx_sig_frac"] >= 0.5)
    crit["formation"] = dict(pass_=bool(c1_pass), **c1)

    # ---- 条件 2：预测误差随交互单调下降（EMA 趋势） ----
    c2 = dict(slope_pre=on1["slope_pre"],
              decile_first=on1["decile_first"], decile_last=on1["decile_last"])
    c2_pass = (c2["slope_pre"] < acfg.slope_max
               and c2["decile_first"] > c2["decile_last"])
    crit["correction"] = dict(pass_=bool(c2_pass), **c2)

    # ---- 条件 3：自省环对照显示非零增益 ----
    off1 = t1["off"]["metrics_mean"]
    gain_relearn = (off1["relearn_error"] - on1["relearn_error"]) / max(off1["relearn_error"], 1e-8)
    gain_steady = (off1["steady_post"] - on1["steady_post"]) / max(off1["steady_post"], 1e-8)
    on2m, off2m = t2["on"]["metrics_mean"], t2["off"]["metrics_mean"]
    gain_fewshot = (off2m["fewshot_first"] - on2m["fewshot_first"]) / max(off2m["fewshot_first"], 1e-8)
    c3 = dict(relearn_on=on1["relearn_error"], relearn_off=off1["relearn_error"],
              relearn_gain_frac=gain_relearn,
              recovery_on=on1["recovery_steps"], recovery_off=off1["recovery_steps"],
              steady_post_on=on1["steady_post"], steady_post_off=off1["steady_post"],
              steady_gain_frac=gain_steady,
              fewshot_first_on=on2m["fewshot_first"], fewshot_first_off=off2m["fewshot_first"],
              fewshot_gain_rel=gain_fewshot)
    c3_pass = (on1["relearn_error"] < off1["relearn_error"]
               and gain_relearn >= acfg.self_gain_min_frac)
    crit["self_reflection"] = dict(pass_=bool(c3_pass), **c3)

    # ---- 条件 4：全程无反向传播、可在线增量运行 ----
    nb, bad = check_no_backprop()
    c4 = dict(no_backprop=nb, bad_tokens=bad,
              online_incremental=check_online_incremental())
    crit["no_backprop_online"] = dict(pass_=bool(nb and c4["online_incremental"]), **c4)

    # ---- 条件 5：信用分配早筛（§2.4 / §7.5-2 / §8.5，承重墙） ----
    if cr is not None:
        ccfg = ccfg or CreditConfig()
        per = cr["per_seed"]
        m = cr["metrics_mean"]
        ok = dict(
            pcn=[p["acc_pcn"] >= ccfg.acc_pcn_min for p in per],
            hebb=[p["acc_hebb"] <= ccfg.acc_hebb_max for p in per],
            gap=[p["acc_gap"] >= ccfg.acc_gap_min for p in per],
            distal=[p["distal_pcn"] >= ccfg.distal_share_min for p in per],
        )
        c5 = dict(acc_pcn=m["acc_pcn"], acc_hebb=m["acc_hebb"], acc_gap=m["acc_gap"],
                  distal_pcn=m["distal_pcn"], distal_hebb=m["distal_hebb"],
                  acc_pcn_d1=m["acc_pcn_d1"], acc_hebb_d1=m["acc_hebb_d1"],
                  acc_pcn_min=ccfg.acc_pcn_min, acc_hebb_max=ccfg.acc_hebb_max,
                  acc_gap_min=ccfg.acc_gap_min, distal_share_min=ccfg.distal_share_min,
                  per_seed=[{k: p[k] for k in ("seed", "acc_pcn", "acc_hebb",
                                               "acc_gap", "distal_pcn")} for p in per],
                  seed_pass=ok)
        c5_pass = all(all(v) for v in ok.values())
        crit["credit_screen"] = dict(pass_=bool(c5_pass), **c5)

    crit["all_pass"] = all(v["pass_"] for k, v in crit.items() if k != "all_pass")
    return crit
