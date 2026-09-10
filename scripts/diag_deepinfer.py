"""H1 深推断失稳机制诊断（纯轨迹，不重训、不改权重）。

承接 report_h1 §5.2b"越深越崩"（4032@48 自有头 acc 0.03、BPC 17.27）。本脚本在
固定权重上逐自由推断迭代（1..maxit）记录 x2/x3 轨迹量，定位退化形态到底是不是
"过冲顶到 x_max 饱和"、"振荡极限环"、还是"缓慢漂移"，并测每步自有头 acc/BPC
（x2@W_out 读出），找 acc 拐点 vs 饱和起点的先后关系——据此定方案级修复
（不动原理三条：局部规则/免反传/结构稀疏）。

轨迹量（每迭代步）：
  - x2 命中上界占比 sat_frac(x2==x_max)、x2∈[0,5] 均值/稀疏活跃比
  - x2 相对上一步的符号翻转率 flip（振荡指标，相对 u2 方向）
  - x3 均范/熵（自由输出变量的漂移）
  - 该步在用 x2 读自有头 W_out 的自有头 acc/BPC（收敛态质量快照）

用法（4090，srpc_src 目录，PYTHONPATH=src）：
  python -B scripts/diag_deepinfer.py --hs 4032,1856 --maxit 48 --eval-n 100
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)


def ckpt_path(h: int) -> str:
    R = "/root/srpc_e2"
    if h >= 2832:
        return os.path.join(R, "results_e2_gpu_eta", f"pcn_{h}_exp1.0.pt")
    if h >= 1856:
        return os.path.join(R, "srpc_src", "results_e2_gpu_fix",
                            f"pcn_{h}_fix.pt")
    return os.path.join(R, "srpc_src", "results_e2_gpu", f"pcn_{h}.pt")


def load(cfg, h):
    rng = np.random.default_rng(5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(ckpt_path(h), map_location="cuda", weights_only=False)
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    return m, sd


def trace_one(m, x0f, iters, q):
    """手动复刻自由推断体（与 _infer do_out=True 完全一致），逐迭代采样轨迹。"""
    cfg = m.cfg
    a, b_ = cfg.alpha, cfg.beta
    dev = m.device
    x0p = np.concatenate([x0f.ravel(), np.zeros(1, np.float32)])
    x0p_t = torch.as_tensor(x0p, device=dev)
    x0rf = x0p_t[m.idx_rf]
    x1g = torch.zeros((m.W, m.per), dtype=torch.float32, device=dev)
    x2 = torch.zeros(m.h, dtype=torch.float32, device=dev)
    x3 = torch.zeros(m.C, dtype=torch.float32, device=dev)
    th = cfg.theta_event
    W1c, W1cT, W2, W3 = m.W1c, m.W1cT, m.W2, m.W3
    x2_prev = x2.clone()
    traj = []
    for _ in range(iters):
        pred0 = torch.matmul(W1c, x1g.unsqueeze(-1)).squeeze(-1)
        e0c = x0rf - pred0
        x1 = x1g.reshape(-1)
        e1 = x1 - torch.mv(W2, x2)
        e2 = x2 - torch.mv(W3, x3)
        u1 = torch.matmul(W1cT, e0c.unsqueeze(-1)).squeeze(-1) * m.s1
        u1 = b_ * u1.reshape(m.W, m.per) - a * e1.reshape(m.W, m.per)
        u2 = b_ * torch.mv(W2.t(), e1) - a * e2
        g1 = u1.abs() > th
        g2 = u2.abs() > th
        x1g = (x1g + m.et1 * u1 * g1).clamp(0.0, cfg.x_max)
        x2 = (x2 + m.et2 * u2 * g2).clamp(0.0, cfg.x_max)
        e2b = x2 - torch.mv(W3, x3)
        x3 = (x3 + cfg.eta_out * torch.mv(W3.t(), e2b)).clamp(0.0, 1.0)
        # 轨迹量
        xn = x2.detach().cpu().numpy()
        q["sat"].append(float((xn >= cfg.x_max - 1e-6).mean()))
        q["act"].append(float((xn > 1e-3).mean()))
        q["meanz"].append(float(xn.mean()))
        q["x3n"].append(float(x3.detach().cpu().numpy().mean()))
        # 方向翻转率（相对上一步 x2 变化向量的余弦符号：<0 即反向，振荡指标）
        dcur = (x2 - x2_prev).detach().cpu().numpy()
        if len(q["dprev"]):
            Q = float(np.sum(dcur * q["dprev"][-1]))
            q["flip"].append(1.0 if Q < 0 else 0.0)
        x2_prev = x2.clone()
        q["dprev"].append(dcur)
    return traj


def selfhead(m, x2_np, y, Wt, bt):
    logit = x2_np @ Wt + bt
    acc = float((logit.argmax(1) == y).mean())
    z = logit - logit.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", default="4032,1856")
    ap.add_argument("--maxit", type=int, default=48)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--out", default="/root/srpc_e2/results_e2_gpu_eta/deepinfer_trace.json")
    args = ap.parse_args()

    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    vx, vy = corpus.val_x, corpus.val_y
    ex, ey = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    ex, ey = ex[:args.eval_n], ey[:args.eval_n]
    exf = np.asarray([x.ravel() for x in ex], dtype=np.float32)
    hs = [int(h) for h in args.hs.split(",")]

    out = {"config": dict(maxit=args.maxit, eval_n=args.eval_n), "rows": []}
    for h in hs:
        if not os.path.exists(ckpt_path(h)):
            print(f"[skip] {ckpt_path(h)}", flush=True)
            continue
        t0 = time.time()
        m, sd = load(cfg, h)
        Wt = m.W_out.detach().cpu().numpy()
        bt = m.b_out.detach().cpu().numpy()
        # 累积轨迹（逐样本运行，逐迭代只记一份聚合；样本间求均值）
        ACC = dict(sat=[], act=[], meanz=[], x3n=[], flip=[], dc=[], dprev=[])
        accs, bpcs = [], []
        for i in range(len(ex)):
            q = dict(sat=[], act=[], meanz=[], x3n=[], flip=[], dc=[], dprev=[])
            trace_one(m, exf[i], args.maxit, q)
            for k in ("sat", "act", "meanz", "x3n", "flip"):
                ACC.setdefault(k, []).append(q[k])        # 每样本一个迭代序列
        flip_mean = np.array(ACC["flip"]).mean(0).tolist()
        # 只保留 sat/act/meanz/x3n 做聚合矩阵，flip 单独算（长度 maxit-1）
        A = {k: np.array(v) for k, v in ACC.items()}       # (n, maxit)
        sat_mean = A["sat"].mean(0)
        act_mean = A["act"].mean(0)
        meanz = A["meanz"].mean(0)
        x3n = A["x3n"].mean(0)
        flip_mean = A["flip"].mean(0) if A["flip"].shape[1] else np.zeros(0)
        # 独立跑几档 iters 的自有头 acc/BPC（复用 _infer）
        iters_sel = sorted(set([maxit for maxit in range(1, args.maxit + 1)
                                if maxit in (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64)]))
        per_acc, per_bpc = [], []
        for it in iters_sel:
            m.learning = False
            fx = np.zeros((len(ex), m.h), np.float32)
            for i in range(len(ex)):
                m._infer(exf[i], None, iters=it)
                fx[i] = m._x2.detach().cpu().numpy()
            m.learning = True
            a, bpc = selfhead(m, fx, ey, Wt, bt)
            per_acc.append(a); per_bpc.append(bpc)
        ixs = iters_sel
        rows = dict(
            h=h, step=sd.get("step", "?"),
            w2_smax=round(float(torch.linalg.svdvals(m.W2.double())[0].item()), 2),
            x2_sat_frac=[round(float(x), 3) for x in sat_mean],
            x2_act_frac=[round(float(x), 3) for x in act_mean],
            x2_mean=[round(float(x), 3) for x in meanz],
            x3_mean=[round(float(x), 3) for x in x3n],
            x2_flip_frac=[round(float(x), 3) for x in flip_mean],
            iters_probe=ixs, selfhead_acc=per_acc, selfhead_bpc=per_bpc,
            wall=round(time.time() - t0, 1))
        out["rows"].append(rows)
        print(f"\nh={h} step={rows['step']} w2_smax={rows['w2_smax']} "
              f"wall={rows['wall']}s", flush=True)
        print("  sat_frac:", rows["x2_sat_frac"], flush=True)
        print("  acc (iters", ixs, "):", rows["selfhead_acc"], flush=True)
        print("  bpc:", rows["selfhead_bpc"], flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n== -> {args.out} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())