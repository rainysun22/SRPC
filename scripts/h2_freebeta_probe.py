"""H2 对因探针：free 推断自底向上传导强度 β 对 free_x2 可读性的因果扫描。

根因（h2_layerprobe）：free x1 acc≈0.32 > free x2 acc≈0.24，clamp 全 1.0。
即 W2 的识别方向（u2 = β·W2ᵀe1 − α·e2 的 β 项）在自由推断中把类别信息
传导丢了。训练靠 clamp 注入标签(top-down 主导)，自由推断 top-down 无指引，
唯一类别信号 = β·W2ᵀe1。本探针在【冻结 checkpoint】上仅调评估侧 free 推断
的 β（不碰训练 β、不重训），测 free_x2 岭可读 acc 是否随 β 单调上升。

机制 → 预测 → 最小干预 → 可证伪判据：
  - 机制：free_x2 类别可读受 β·W2ᵀe1 传导强度限制，而非权重本身缺信息。
  - 预测：放大评估侧 β，free_x2 从 0.24 单调上升，向 free_x1(0.32)/孪生(0.49)靠拢。
  - 成功判据：free_x2 随 β∈{1,2,4,8} 单调升，且高点 ≥0.32。
  - 失败判据：free_x2 不随 β 升（→ 传导不是瓶颈，改取证：训练侧/权重本身）。

用法（4090 /root/srpc_e2/srpc_src）：
    python -B scripts/h2_freebeta_probe.py [--h 1856] [--probe-n 6000]
输出：results_e2_gpu_eta/h2_freebeta.json
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
OUT = "/root/srpc_e2/srpc_src/results_e2_gpu_eta/h2_freebeta.json"


def ridge_probe(X, y, lam=1e-3, n_cls=256):
    n = X.shape[0]
    Xb = np.hstack([X.astype(np.float64), np.ones((n, 1), np.float64)])
    Y = np.zeros((n, n_cls), np.float64); Y[np.arange(n), y] = 1.0
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


def collect_free_beta(cfg, m, X, y, it=12, beta=1.0):
    """临时替换 cfg.beta 做评估侧自由推断（冻结权重，不改训练 β/不重训）。"""
    m.learning = False
    dev = m.device
    base_beta = cfg.beta
    cfg.beta = beta
    feats = np.zeros((len(y), m.h), np.float32)
    for i in range(len(y)):
        m._infer(X[i].ravel(), None, clamp=False, iters=it)
        feats[i] = m._x2.detach().cpu().numpy()
    cfg.beta = base_beta
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
    ap.add_argument("--iters", type=int, default=8)   # 默认取不稀释档 it8
    ap.add_argument("--betas", type=str, default="1,2,4,8")
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

    # 基线（评估侧 β=1.0）探针先验
    W0, b0 = ridge_probe(collect_free_beta(cfg, m, px, py, args.iters, 1.0), py)
    a0, _ = probe_eval(W0, b0, collect_free_beta(cfg, m, eval_x, eval_y, args.iters, 1.0), eval_y)
    print(f"free_x2 β=1.0 (基线): acc={a0:.4f}", flush=True)

    out = {"h": args.h, "ckpt": os.path.basename(ckpt), "iters": args.iters,
           "twin_acc": None, "rows": {}}
    out.setdefault("rows", {})
    tg = json.load(open(f"{RES}/twin_{args.h}.json"))
    out["twin_acc"] = tg["acc"]

    for beta in [float(s) for s in args.betas.split(",")]:
        W_, b_ = ridge_probe(collect_free_beta(cfg, m, px, py, args.iters, beta), py)
        a, bp = probe_eval(W_, b_,
                           collect_free_beta(cfg, m, eval_x, eval_y, args.iters, beta),
                           eval_y)
        out["rows"][f"beta{beta}"] = dict(acc=round(a, 4), bpc=round(bp, 3))
        print(f"free_x2 β={beta}: acc={a:.4f} bpc={bp:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())