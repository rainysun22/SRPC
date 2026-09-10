"""H2：能量重锚记账 —— 同规模同任务 SR-PC vs BPTT 孪生，全链路累计 MAC @等能力。

口径（2026-09-10 用户裁定）：
  1. 能量 = 全链路累计 MAC @等能力（训练累计事件/结构 MAC + 部署前向）。
  2. 能力对齐 = 用 H1 已落地的深推断收缩修复（eta_inf_scl=0.5）+ 扫深迭代，
     把 SR-PC 端到端能力拉到其可达最高点；孪生按同能力在自身 acc-vs-step 曲线插值。
  3. 范围 = 有孪生的 3 档 h∈{768,1200,1856}。

三口径（与 phase-C EnergyLedger 一致）：
  struct = 结构稀疏硬件：保留突触全开；  event = 事件驱动硬件：×活跃率；
  dense  = 稠密等价（孪生即稠密，用作其原生硬件能耗）。

SR-PC per-字节前向（自由推断，N 迭代）：
  iter_struct = 2·(W·r·256·per)[W1c fwd/fbwd] + 2·m2s[W2 fwd/fbwd] + 2·m3s[W3 fwd/x3(do_out)]
  部署 per_byte_struct = N·iter_struct + C·h[W_out 读出]
SR-PC per-训练步（clamp 1 迭代 + 局部学习）：
  learn_struct 增量 ≈ 2·(W·r·256·per) + 2·m2s + dW 突触变更重算 ≈ 常数倍；记账用同一 iter_struct。
孪生 per-step = 4·n_params（fwd 1 + bwd 2 + Adam 更新 1，稠密）；per_byte(dense) = n_params。
累计 MAC(M) = steps_budget_to_能力(M) × per_step_M + eval_bytes × per_byte_M。

输出：能力拉升表 + 三口径累计 MAC 对账 + SR-PC/孪生 比值。
用法（4090, srpc_src）：python -B /root/srpc_e2/scripts/run_h2.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)
R = "/root/srpc_e2"
TWINS = {768: "results_e2_gpu/twin_768.json",
         1200: "results_e2_gpu/twin_1200.json",
         1856: "results_e2_gpu/twin_1856.json"}
ITERS = (12, 24, 32)
EVAL_N = 200


def ckpt(h: int) -> str:
    if h >= 1856:
        return f"{R}/srpc_src/results_e2_gpu_fix/pcn_{h}_fix.pt"
    return f"{R}/srpc_src/results_e2_gpu/pcn_{h}.pt"


def selfhead_scores(fx, Wt, bt, y):
    lgt = fx @ Wt + bt
    acc = float((lgt.argmax(1) == y).mean())
    z = lgt - lgt.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def mac_struct(m) -> dict:
    """三口径 per-推理迭代 / per-读出 / per-训练步（基于张量形状，精确）。"""
    W, C, h = m.W, m.C, m.h
    r, per = m.r, m.per
    w1_syn = W * (r * 256) * per                      # W1c 结构突触（块紧凑）
    m2_syn = int((m.W2 != 0).sum().item())
    m3_syn = int((m.W3 != 0).sum().item())
    iter_struct = 2 * w1_syn + 2 * m2_syn + 2 * m3_syn   # fwd/fbwd/x3 全链路
    readout = C * h                                    # W_out 读出
    # 事件：× 活跃率（用 kwta_frac 作 x1/x2 上界，x3 用 0.5）
    ev = cfg.kwta_frac
    iter_event = 2 * w1_syn * ev + 2 * m2_syn * ev + 2 * m3_syn * 0.5
    readout_ev = readout * ev
    # dense 等价
    iter_dense = 2 * (W * 256 * per) + 2 * (h * h) + 2 * (h * C)
    readout_dense = C * h
    return dict(iter_struct=iter_struct, iter_event=iter_event,
                iter_dense=iter_dense,
                readout=readout, readout_ev=readout_ev, readout_dense=readout_dense,
                w1_syn=w1_syn, m2_syn=m2_syn, m3_syn=m3_syn,
                n_struct=h * r * 256 + m2_syn + m3_syn,
                n_dense=W * 256 * h + h * h + h * C)


def twin_curve(twinj):
    """(acc, cum_dense_steps) 训练累计 dense MAC 表示，供等能力插值。n_params 稠密 per-step。"""
    n = int(twinj["n_params"])
    arr = sorted(twinj["curve"], key=lambda r: r["step"])
    # 每训练步 dense MAC ≈ 3×n_params（fwd+bwd，保守不含 Adam 常）；累计折到 token
    per_step_dense = 3 * n
    pts = [(r["acc"], r["step"] * per_step_dense) for r in arr]
    return n, per_step_dense, pts


def twin_steps_to_acc(pts, target_acc):
    """插值孪生达到某 acc 所需累计 dense MAC（线性，取最近区间）。"""
    pts = sorted(pts, key=lambda p: p[0])
    if target_acc <= pts[0][0]:
        return pts[0][1]
    if target_acc >= pts[-1][0]:
        return pts[-1][1]
    for (a0, m0), (a1, m1) in zip(pts, pts[1:]):
        if a0 <= target_acc <= a1:
            f = (target_acc - a0) / max(1e-9, (a1 - a0))
            return m0 + f * (m1 - m0)
    return pts[-1][1]


def main() -> int:
    cfg = E2Config()
    cfg.eta_inf_scl = 0.5
    corpus = ByteCorpus(cfg)
    ex = corpus.val_x[cfg.tau_cal_windows:][:EVAL_N]
    ey = corpus.val_y[cfg.tau_cal_windows:][:EVAL_N]
    exf = np.asarray([x.ravel() for x in ex], dtype=np.float32)
    n_bytes = EVAL_N

    rows = []
    for h in (768, 1200, 1856):
        m = LMPCNg(cfg, h, np.random.default_rng(5), eta_w=0.01, iters=12,
                   device="cuda")
        sd = torch.load(ckpt(h), map_location="cuda", weights_only=False)
        for k, v in sd["model"].items():
            if k in ("W1c", "W1cT", "W2", "W3", "W_out", "b_out") and hasattr(m, k):
                getattr(m, k).copy_(v)
        Wt = m.W_out.detach().cpu().numpy()
        bt = m.b_out.detach().cpu().numpy()
        mac = mac_struct(m)
        # 深推断能力扫描（收缩修复 + iters 梯）
        best = None
        for it in ITERS:
            m.learning = False
            fx = np.zeros((EVAL_N, m.h), np.float32)
            for i in range(EVAL_N):
                m._infer(exf[i], None, iters=it)
                fx[i] = m._x2.detach().cpu().numpy()
            m.learning = True
            a_, b_ = selfhead_scores(fx, Wt, bt, ey)
            if best is None or a_ > best[0]:
                best = (a_, b_, it)
        s_acc, s_bpc, s_it = best
        # 训练预算 = 同孪生 full budget 步（~1.04M token）；per-step struct/event/dense
        steps = 1_040_000
        struct_train = steps * 2 * mac["iter_struct"]   # clamp 1 iter + learn≈fwd 重算（×2 保守）
        event_train = steps * 2 * (mac["iter_event"])
        dense_train = steps * 3 * mac["n_dense"]
        # 部署 per_byte @ s_it
        dep_struct = s_it * mac["iter_struct"] + mac["readout"]
        dep_event = s_it * mac["iter_event"] + mac["readout_ev"]
        dep_dense = s_it * mac["iter_dense"] + mac["readout_dense"]
        sr_struct = struct_train + n_bytes * dep_struct
        sr_event = event_train + n_bytes * dep_event
        # ---- 孪生同能力 ----
        twinj = json.load(open(os.path.join(R1 := R, TWINS[h]))) if False else \
            {**json.load(open(TWINS[h])), "_confirmed": True}
        n, per_step_dense, pts = twin_curve(twinj)
        twin_mac_align = twin_steps_to_acc(pts, s_acc)
        twin_dep = n_bytes * n
        tw_struct = twin_mac_align + twin_dep
        rows.append({
            "h": h, "best_acc": round(s_acc, 4), "best_bpc": round(s_bpc, 4),
            "best_iters": s_it, "n_struct": mac["n_struct"], "n_dense": mac["n_dense"],
            "iter_struct": mac["iter_struct"], "iter_event": int(mac["iter_event"]),
            "iter_dense": mac["iter_dense"],
            "sr_cum_struct": int(sr_struct), "sr_cum_event": int(sr_event),
            "sr_perbyte_struct": int(dep_struct), "sr_perbyte_event": int(dep_event),
            "twin_acc_align": round(s_acc, 4),
            "twin_dense_cum": int(twin_mac_align),
            "twin_perbyte_dense": int(n),
            "ratio_struct": round(twin_mac_align / sr_struct, 2),
            "ratio_event": round(twin_mac_align / sr_event, 2),
        })

    print("H2 能量重锚记账（eta_inf_scl=0.5，深推断扫能力）\n")
    hdr = ["h", "best_acc", "best_bpc", "best_iters", "n_struct", "n_dense",
           "sr_perbyte_struct", "sr_perbyte_event", "twin_perbyte_dense",
           "sr_cum_struct", "sr_cum_event", "twin_dense_cum",
           "ratio_struct", "ratio_event"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[k]) for k in hdr))
    with open("/root/srpc_e2/results_e2_gpu_eta/h2_energy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nsaved -> results_e2_gpu_eta/h2_energy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())