"""H2 能力差距机理诊断（冻结 checkpoint，不重训）。

判据：free x2 线性探针只到 ~0.28 vs 孪生 0.49。本脚本在单档（默认 1856）上
同时测量多路读出口，精确定位信息在哪一步丢失：
  1. free_x2  岭探针（复现基线 ~0.275）——自由收敛态 x2 的线性可读信息量。
  2. clamp_x2 岭探针——推断期把 x3 钳制到真实标签（clamp=True）得到的 x2 的
     线性可读信息量。若 >> free_x2：训练流形已含任务信息，是"自由推断丢失"，
     修复在推断/读出协议；若 ~同：训练流形本身缺乏判别结构，修复在训练目标。
  3. free_x3  模型原生日读出头：自由收敛态 x3 直接 argmax（= 教科书 supervised-PC
     输出层读出，config 文档记的口径），检验"换读出来源"能否解围。
  4. shallow  自由 x2 探针 @ iters ∈ {1,3,6}——检验深迭代是否主动稀释任务信息。
输出攒进 results_e2_gpu_eta/h2_mech.json。
用法（4090 /root/srpc_e2）：python -B scripts/h2_diag_mech.py [--h 1856] [--probe-n 12000]
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
RES = "/root/srpc_e2/srpc_src/results_e2_gpu"
RES_FIX = "/root/srpc_e2/srpc_src/results_e2_gpu_fix"


def ridge_probe(X, y, lam=1e-3, n_cls=256):
    n = X.shape[0]
    Xb = np.hstack([X.astype(np.float64), np.ones((n, 1), np.float64)])
    Y = np.zeros((n, n_cls), np.float64); Y[np.arange(n), y] = 1.0
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d, dtype=np.float64)
    B = Xb.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W, *_ = np.linalg.lstsq(np.vstack([Xb, np.eye(d) * np.sqrt(lam)]),
                                np.vstack([Y, np.zeros((d, n_cls))]), rcond=None)
    return W[:-1].astype(np.float32), W[-1].astype(np.float32)


def probe_eval(W, b, X, y):
    lgt = X @ W + b
    acc = float((lgt.argmax(1) == y).mean())
    z = lgt - lgt.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return acc, float(np.log2(np.e) * nll.mean())


def collect(cfg, m, X, y, clamp):
    m.learning = False
    feats = np.zeros((len(y), m.h), np.float32)
    for i in range(len(y)):
        if clamp:
            yoh = np.zeros(256, np.float32); yoh[y[i]] = 1.0
            m._infer(X[i].ravel(), yoh, clamp=True)
        else:
            m._infer(X[i].ravel(), None, clamp=False, iters=cfg.settle_iters)
        feats[i] = m._x2.detach().cpu().numpy()
    m.learning = True
    return feats


def collect_x3argmax(cfg, m, X, y):
    m.learning = False
    accs = 0.0
    for i in range(len(y)):
        m._infer(X[i].ravel(), None)          # 默认 free，iters=cfg.settle_iters
        x3 = m._x3.detach().cpu().numpy()
        accs += float(x3.argmax() == y[i])
    m.learning = True
    return accs / len(y)


def collect_shallow(cfg, m, X, y, it):
    m.learning = False
    feats = np.zeros((len(y), m.h), np.float32)
    for i in range(len(y)):
        m._infer(X[i].ravel(), None, clamp=False, iters=it)
        feats[i] = m._x2.detach().cpu().numpy()
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
    ap.add_argument("--probe-n", type=int, default=12000)
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

    out = {"h": args.h, "ckpt": ckpt, "rows": {}}

    # 1. free x2（基线）
    t0 = time.time()
    Wf, bf = ridge_probe(collect(cfg, m, px, py, clamp=False), py)
    af, bf_ = probe_eval(Wf, bf, collect(cfg, m, eval_x, eval_y, False), eval_y)
    out["rows"]["free_x2"] = dict(acc=round(af, 4), bpc=round(bf_, 3),
                                  wall=round(time.time() - t0, 1))
    print(f"free_x2 probe acc={af:.4f} bpc={bf_:.3f}", flush=True)

    # 2. clamp x2（训练流形信息上界）
    t0 = time.time()
    Wc, bc = ridge_probe(collect(cfg, m, px, py, clamp=True), py)
    ac, bc_ = probe_eval(Wc, bc, collect(cfg, m, eval_x, eval_y, True), eval_y)
    out["rows"]["clamp_x2"] = dict(acc=round(ac, 4), bpc=round(bc_, 3),
                                   wall=round(time.time() - t0, 1))
    print(f"clamp_x2 probe acc={ac:.4f} bpc={bc_:.3f}", flush=True)

    # 3. free x3 原生日读出
    t0 = time.time()
    a3 = collect_x3argmax(cfg, m, eval_x, eval_y)
    out["rows"]["free_x3_argmax"] = dict(acc=round(a3, 4),
                                         wall=round(time.time() - t0, 1))
    print(f"free_x3 argmax acc={a3:.4f}", flush=True)

    # 4. 浅迭代 free x2（深迭代是否稀释）
    for it in (1, 3, 6):
        t0 = time.time()
        Wi, bi = ridge_probe(collect_shallow(cfg, m, px, py, it), py)
        ai, _ = probe_eval(Wi, bi, collect_shallow(cfg, m, eval_x, eval_y, it), eval_y)
        out["rows"][f"free_x2_it{it}"] = dict(acc=round(ai, 4),
                                              wall=round(time.time() - t0, 1))
        print(f"free_x2 it={it} acc={ai:.4f}", flush=True)

    # 孪生参照
    tj = json.load(open(f"{RES}/twin_{args.h}.json"))
    out["twin_acc"] = tj["acc"]; out["twin_bpc"] = tj["bpc"]
    with open("/root/srpc_e2/results_e2_gpu_eta/h2_mech.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved -> results_e2_gpu_eta/h2_mech.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())