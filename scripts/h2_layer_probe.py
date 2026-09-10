"""H2 分层可读性诊断（冻结 checkpoint，不重训）：信息在哪一层丢失？

背景：h2_mech 已证 free_x2 表证上限 ≈0.28，且 it{1,3,6,12} 平展（深迭代不稀释
也不增益）——0.28 是真实 free 表证可读上限，读口头/评估层非瓶颈。剩下未定位：
类别信息在编码器哪一层丢失？本脚本测 x1 与 x2 在不同自由迭代下的岭可读 acc。

判据（分叉）：若 x1 ≫ x2，则塌缩发生在 W1→x2 的自由动力学(W2)；若 x1≈x2≈0.28，
塌缩发生在输入→x1 编码器(W1)；若 x1 随迭代明显下降，则 W1 的迭代推断稀释任务位。
这直接决定训练目标应改在哪一层（只改一层=最小干预）。

用法（4090 /root/srpc_e2/srpc_src）：
    python -B scripts/h2_layer_probe.py [--h 1856] [--probe-n 6000] [--ckpt auto]
输出：results_e2_gpu_eta/h2_layerprobe.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)
RES = "/root/srpc_e2/srpc_src/results_e2_gpu"
RES_FIX = "/root/srpc_e2/srpc_src/results_e2_gpu_fix"
OUT = "/root/srpc_e2/srpc_src/results_e2_gpu_eta/h2_layerprobe.json"


def ridge_probe(X, y, lam=1e-3, n_cls=256):
    n = X.shape[0]
    Xb = np.hstack([X.astype(np.float64), np.ones((n, 1), np.float64)])
    Y = np.zeros((n, n_cls), np.float64); Y[np.arange(n), y] = 1.0
    d = Xb.shape[1]
    if d > 8_000:                 # x1 展平可能超高维：先用 PCA 压到 512，防求解过慢/奇异
        from numpy.linalg import eigh
        Xc = X - X.mean(0, keepdims=True)
        cov = Xc.T @ Xc
        ev, V = eigh(cov)
        V = V[:, ::-1][:, :512]
        X = Xc @ V
        Xb = np.hstack([X.astype(np.float64), np.ones((n, 1), np.float64)])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float64)
    B = Xb.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W, *_ = np.linalg.lstsq(np.vstack([Xb, np.eye(Xb.shape[1]) * np.sqrt(lam)]),
                                np.vstack([Y, np.zeros((Xb.shape[1], n_cls))]),
                                rcond=None)
    return W[:-1].astype(np.float32), W[-1].astype(np.float32)


def probe_eval(W, b, X, y):
    lgt = X @ W + b
    acc = float((lgt.argmax(1) == y).mean())
    z = lgt - lgt.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def collect(cfg, m, X, y, it=None, clamp=False, layer="x2"):
    m.learning = False
    d = m.h if layer == "x2" else int(m.W * m.per)
    feats = np.zeros((len(y), d), np.float32)
    for i in range(len(y)):
        if clamp:
            yoh = np.zeros(256, np.float32); yoh[y[i]] = 1.0
            m._infer(X[i].ravel(), yoh, clamp=True, iters=it)
        else:
            m._infer(X[i].ravel(), None, clamp=False, iters=it)
        z = m._x2 if layer == "x2" else m._x1g
        feats[i] = z.detach().cpu().numpy().reshape(-1)
    m.learning = True
    return feats


def probe_samples(cfg, corpus, n):
    W_ = cfg.context
    X = np.zeros((n, W_, 256), np.float32); y = np.zeros(n, np.int64)
    stride = max(1, (len(corpus.train) - W_ - 1) // n)
    for i in range(n):
        t = (i * stride) % (len(corpus.train) - W_ - 1)
        b = corpus.train[t:t + W_]
        X[i, np.arange(W_), b] = 1.0; y[i] = corpus.train[t + W_]
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=1856)
    ap.add_argument("--probe-n", type=int, default=6000)
    ap.add_argument("--iters", type=str, default="1,4,8,12")
    args = ap.parse_args()
    cfg = E2Config()
    cfg.settle_iters = 12
    corpus = ByteCorpus(cfg)
    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    px, py = probe_samples(cfg, corpus, args.probe_n)
    rng = np.random.default_rng(5)
    m = LMPCNg(cfg, args.h, rng, eta_w=0.01, iters=12, device="cuda")
    ckpt = os.path.join(RES_FIX, f"pcn_{args.h}_fix.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(RES, f"pcn_{args.h}.pt")
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    print(f"loaded {ckpt} (step={sd.get('step','?')})", flush=True)

    out = {"h": args.h, "ckpt": os.path.basename(ckpt), "rows": {}}

    for layer in ("x1", "x2"):
        for it in [int(s) for s in args.iters.split(",")]:
            W_, b_ = ridge_probe(collect(cfg, m, px, py, it, False, layer), py)
            a, bp = probe_eval(W_, b_,
                               collect(cfg, m, eval_x, eval_y, it, False, layer),
                               eval_y)
            key = f"free_{layer}_it{it}"
            out["rows"][key] = dict(acc=round(a, 4), bpc=round(bp, 3))
            print(f"{key}: acc={a:.4f} bpc={bp:.3f}", flush=True)

    # clamp 上界（x2 已知=1.0；x1 看编码器是否也全保真）
    for layer, it in (("x1", 12), ("x2", 12)):
        W_, b_ = ridge_probe(collect(cfg, m, px, py, it, True, layer), py)
        a, bp = probe_eval(W_, b_,
                           collect(cfg, m, eval_x, eval_y, it, True, layer), eval_y)
        key = f"clamp_{layer}@it{it}"
        out["rows"][key] = dict(acc=round(a, 4), bpc=round(bp, 3))
        print(f"{key}: acc={a:.4f} bpc={bp:.3f}", flush=True)

    tj = json.load(open(f"{RES}/twin_{args.h}.json"))
    out["twin_acc"] = tj["acc"]; out["twin_bpc"] = tj["bpc"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())