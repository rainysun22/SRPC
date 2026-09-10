"""H1 深推断稳定性修复——验证 W3 谱归一化（Meta-PCN 权重归一化式收缩修复）。

机制（report_h1 + 文献 Mali et al.; Ha et al. ICLR'26 Meta-PCN）：
  4032 档深推断进入符号翻转振荡 + x2_mean 单调漂升 = 能量上升发散（EVPE 爆发）；
  根因是 x2⇄W3⇄x3 闭环非收缩（σmax(W3)²≈25>>1）。修复 = W3 谱归一化压低闭环增益。

本脚本：不重训，冻结 4032 档，把 W3 谱归一化到 cap∈{1.5,2.0,3.0,5.0}，复跑深推断，
判定 acc/BPC@iters∈{12,24,48} 是否恢复（不再崩到 0.01/16.3），x2 是否不再饱和/漂升。

用法（4090，srpc_src）：python -B scripts/validate_w3cap.py --h 4032 --eval-n 100
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)
R = "/root/srpc_e2"


def ckpt(h: int) -> str:
    if h >= 2832:
        return f"{R}/results_e2_gpu_eta/pcn_{h}_exp1.0.pt"
    if h >= 1856:
        return f"{R}/srpc_src/results_e2_gpu_fix/pcn_{h}_fix.pt"
    return f"{R}/srpc_src/results_e2_gpu/pcn_{h}.pt"


def selfhead(x2_np, Wt, bt, y):
    logit = x2_np @ Wt + bt
    acc = float((logit.argmax(1) == y).mean())
    z = logit - logit.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def run_iters(cfg, m, exf, ey, iters, step_scl=1.0):
    """在给定权重下，accumulate iters 步自由推断，返回 x2 聚合 sat/mean_end + acc/bpc。
    step_scl!=1 时对推断期 x2/x3 更新步长（et2/eta_out）做缩放（收缩修复）。
    """
    orig_et2 = float(m.et2)
    orig_eta = float(cfg.eta_out)
    if step_scl != 1.0:
        m.et2 = orig_et2 * step_scl
        cfg.eta_out = orig_eta * step_scl
    m.learning = False
    fx = np.zeros((len(exf), m.h), np.float32)
    sats, meanend, flip = [], [], []
    for i in range(len(exf)):
        m._infer(exf[i], None, iters=iters)
        xn = m._x2.detach().cpu().numpy()
        fx[i] = xn
        sats.append(float((xn >= cfg.x_max - 1e-6).mean()))
        meanend.append(float(xn.mean()))
    m.learning = True
    if step_scl != 1.0:
        m.et2 = orig_et2
        cfg.eta_out = orig_eta
    Wt = m.W_out.detach().cpu().numpy()
    bt = m.b_out.detach().cpu().numpy()
    a_, bpc_ = selfhead(fx, Wt, bt, ey)
    return a_, bpc_, float(np.mean(sats)), float(np.mean(meanend))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=4032)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--caps", default="none")
    ap.add_argument("--step-scls", default="1.0")
    ap.add_argument("--iters", default="12,24,48")
    args = ap.parse_args()

    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    ex, ey = corpus.val_x[cfg.tau_cal_windows:], corpus.val_y[cfg.tau_cal_windows:]
    ex, ey = ex[:args.eval_n], ey[:args.eval_n]
    exf = np.asarray([x.ravel() for x in ex], dtype=np.float32)
    iters_list = [int(t) for t in args.iters.split(",")]
    caps = [1.5, 2.0, 3.0, 5.0] if args.caps == "default" else \
        [None if s == "none" else float(s) for s in args.caps.split(",")]
    step_scls = [None if s == "none" else float(s) for s in args.step_scls.split(",")]

    print(f"h={args.h} eval_n={args.eval_n} iters={iters_list}", flush=True)
    for cap in caps:
        for step_scl in step_scls:
            m = LMPCNg(cfg, args.h, np.random.default_rng(5), eta_w=0.01,
                       iters=12, device="cuda")
            sd = torch.load(ckpt(args.h), map_location="cuda", weights_only=False)
            for k, v in sd["model"].items():
                if k in ("W1c", "W1cT", "W2", "W3", "W_out", "b_out") and hasattr(m, k):
                    getattr(m, k).copy_(v)
            with torch.no_grad():
                s3_orig = torch.linalg.svdvals(m.W3.double())[0].item()
                if cap is not None:
                    cur = float(torch.linalg.svdvals(m.W3.double())[0])
                    m.W3.mul_(cap / cur)   # 谱归一化到 σmax=cap
                s3 = torch.linalg.svdvals(m.W3.double())[0].item()
            cells = []
            for it in iters_list:
                a_, bpc_, sat, meane = run_iters(cfg, m, exf, ey, it,
                                                 step_scl=step_scl or 1.0)
                cells.append(f"it{it}:acc{a_:.3f}/bpc{bpc_:.2f}/sat{sat:.3f}/mean{meane:.4f}")
            tag = f"cap={cap if cap is not None else 'NONE'},step={step_scl or 1.0}"
            print(f"  {tag}  s3:{s3_orig:.2f}->{s3:.2f}\n    " + "   ".join(cells),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())