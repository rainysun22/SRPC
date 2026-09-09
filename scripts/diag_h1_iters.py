"""H1 §5.2b：在固定权重上做自由推断 iters 加深消融，判断"端到端/探针不缩放"是否因欠迭代。

不重训、不读头拟合训练：仅改变自由推断步数，测
  - 自有头 acc / BPC（固定权重 → 深推断收敛态 x2 → 读自有线性头 W_out/b_out）
  - 再拟合岭探针 acc（在深推断 x2 上重训线性探针，排除表征本身不足）

对照：1856（必缩档，reference） vs 2832 / 4032（回落档）。
iterters ∈ {12, 24, 48}。cost：每档每 iters 一次 600 窗口推断 + 12000 窗口 fit。

用法（4090，srpc_src 目录）：
  python -B scripts/diag_h1_iters.py --hs 4032,2832,1856 --iters 12,24,48
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
RES = "results_e2_gpu"


def _ckpt(h: int) -> str:
    if h >= 1856:
        return os.path.join(RES + "_fix", f"pcn_{h}_fix.pt")
    return os.path.join(RES, f"pcn_{h}.pt")


def _load(cfg, h: int):
    rng = np.random.default_rng(5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(_ckpt(h), map_location="cuda", weights_only=False)
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    return m, sd


def _x2(cfg, m, X, iters: int) -> np.ndarray:
    m.learning = False
    feats = np.zeros((len(X), m.h), dtype=np.float32)
    for i in range(len(X)):
        m._infer(X[i].ravel(), None, iters=iters)   # 深推断覆盖步数
        feats[i] = m._x2.detach().cpu().numpy()
    m.learning = True
    return feats


def _ownhead(feats, m, y):
    Wt = m.W_out.detach().cpu().numpy()
    bt = m.b_out.detach().cpu().numpy()
    logit = feats @ Wt + bt
    acc = float((logit.argmax(1) == y).mean())
    z = logit - logit.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def _probe(fx, fy, ex, ey, lam=1e-3):
    Xb = np.hstack([fx.astype(np.float64), np.ones((len(fx), 1))])
    Y = np.zeros((len(fy), 256), dtype=np.float64)
    Y[np.arange(len(fy)), fy] = 1.0
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d)
    try:
        W = np.linalg.solve(A, Xb.T @ Y)
    except np.linalg.LinAlgError:
        W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Wl, bl = W[:-1], W[-1]
    lg = ex.astype(np.float64) @ Wl + bl
    acc = float((lg.argmax(1) == ey).mean())
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", default="1856,2832,4032")
    ap.add_argument("--iters", default="12,24,48")
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--eval-n", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(RES, "h1_iters.json"))
    args = ap.parse_args()
    hs = [int(x) for x in args.hs.split(",")]
    iters_list = [int(x) for x in args.iters.split(",")]

    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    eval_x, eval_y = eval_x[:args.eval_n], eval_y[:args.eval_n]
    W_ = cfg.context
    eval_xf = np.zeros((len(eval_x), W_ * 256), dtype=np.float32)
    for i in range(len(eval_x)):
        eval_xf[i] = eval_x[i].ravel()

    # probe fit 样本（独立于 eval）
    nf = args.probe_n
    stride = max(1, (len(corpus.train) - W_ - 1) // nf)
    pf_x = np.zeros((nf, W_, 256), dtype=np.float32)
    pf_y = np.zeros(nf, dtype=np.int64)
    for i in range(nf):
        t = (i * stride) % (len(corpus.train) - W_ - 1)
        b = corpus.train[t:t + W_]
        pf_x[i, np.arange(W_), b] = 1.0
        pf_y[i] = corpus.train[t + W_]

    # 续行：若已存在结果则跳过已完成 (h, iters)
    done = set()
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            rows = prev.get("rows", [])
            done = {(r["h"], r["iters"]) for r in rows}
            print(f"[resume] {len(rows)} existing rows", flush=True)
        except Exception:
            rows = []
    else:
        rows = []

    for h in hs:
        if not os.path.exists(_ckpt(h)):
            print(f"[skip] {_ckpt(h)}", flush=True)
            continue
        m, sd = _load(cfg, h)
        step = sd.get("step", "?")
        for it in iters_list:
            if (h, it) in done:
                print(f"  (skip done h={h} iters={it})", flush=True)
                continue
            t0 = time.time()
            fx = _x2(cfg, m, pf_x, it)
            ex = _x2(cfg, m, eval_xf, it)
            o_acc, o_bpc = _ownhead(ex, m, eval_y)
            p_acc = _probe(fx, pf_y, ex, eval_y)
            rows.append(dict(h=h, step=step, iters=it,
                             own_acc=round(o_acc, 4), own_bpc=round(o_bpc, 3),
                             probe_acc=round(p_acc, 4),
                             wall=round(time.time() - t0, 1)))
            # 该行即写盘（断点续跑/被杀不丢）
            with open(args.out, "w") as f:
                json.dump(dict(hs=hs, iters=iters_list, rows=rows,
                               note="固定权重深推断 iters 消融"), f, indent=1)
            print(f"h={h} iters={it}: own_acc={o_acc:.4f} own_bpc={o_bpc:.3f} "
                  f"probe_acc={p_acc:.4f} wall={rows[-1]['wall']}s", flush=True)

    print(f"== -> {args.out} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())