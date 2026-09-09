"""H1 诊断：区分「线性探针欠拟合」vs「x2 表征坍缩」。

对每档冻结模型，在同批验证窗口上：
  1) 自由推断取收敛态 x2，统计活性率 / 幅度 / 方差（坍缩判据）；
  2) 模型**自有读出头** acc/bpc：
         x2 @ W_out + b_out  →  argmax vs y
     （自有 head 即该模型端到端 next-byte 精度；与孪生 json 的 acc 同口径）；
  3) 打印对照，供判陨：head高+探针低→探针欠拟合；head也低→x2 坍缩。
用法：env PYTHONPATH=. python scripts/diag_h1_x2.py [--include ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)
RES = "results_e2_gpu"
ALLOW = ["W1c", "W1cT", "W2", "W3", "W_out", "b_out"]


def load(cfg, h, ckpt):
    rng = np.random.default_rng(0 * 977 + 5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    for k, v in sd["model"].items():
        if k in ALLOW and hasattr(m, k):
            getattr(m, k).copy_(v)
    smax = float(torch.linalg.svdvals(m.W2.double())[0].item())
    return m, sd, smax


def ckpt_path(h):
    if h >= 1856:
        return f"{RES}_fix/pcn_{h}_fix.pt"
    return f"{RES}/pcn_{h}.pt"


def run_x2(cfg, m, Xf):
    m.learning = False
    n = len(Xf)
    feats = np.zeros((n, m.h), dtype=np.float32)
    for i in range(n):
        m._infer(Xf[i], None)
        feats[i] = m._x2.detach().cpu().numpy()
    m.learning = True
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="768,1200,1856,2832,4032")
    ap.add_argument("--n-windows", type=int, default=600)
    args = ap.parse_args()

    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    eval_x = eval_x[:args.n_windows]
    eval_y = eval_y[:args.n_windows]
    # 一热编成 (W,256) 平面向量（float32，_infer 输入口径）
    W_ = cfg.context
    Xf = np.zeros((len(eval_x), W_ * 256), dtype=np.float32)
    for i in range(len(eval_x)):
        Xf[i] = eval_x[i].ravel()

    print(f"n={len(eval_y)} eval windows")
    for h in [int(x) for x in args.include.split(",")]:
        ckpt = ckpt_path(h)
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt}", flush=True)
            continue
        m, sd, smax = load(cfg, h, ckpt)
        feats = run_x2(cfg, m, Xf)
        # 统计
        act = float((feats > 0).mean())
        nz = feats[feats > 0]
        amp = float(nz.mean()) if nz.size else 0.0
        std = float(feats.std())
        # 自有读出头
        Wt = m.W_out.detach().cpu().numpy()
        bt = m.b_out.detach().cpu().numpy()
        logit = feats @ Wt + bt
        pred = logit.argmax(1)
        acc = float((pred == eval_y).mean())
        z = logit - logit.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(eval_y)), eval_y], 1e-12, None))
        bpc = float(np.log2(np.e) * nll.mean())
        print(f"h={h}: step={sd.get('step')} smax={smax:.2f} x2 active={act:.3f} "
              f"amp(>0)={amp:.3f} std={std:.3f} | "
              f"own_head acc={acc:.4f} bpc={bpc:.3f}", flush=True)


if __name__ == "__main__":
    main()