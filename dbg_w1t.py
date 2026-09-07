#!/usr/bin/env python3
"""Frozen 模型：recon 源上在线 delta 规则 vs 归一化 LMS 的收敛诊断。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
model.set_learning(False)

cond = arc._cond(0)
Xs, Ys = [], []
for _ in range(2000):
    g_in = arc.sample_input()
    model.apply_transform(arc._onehot(g_in).astype(float), cond)
    Xs.append((model.Ws[1] @ model.xs[1]).copy())
    Ys.append(arc._onehot(TRANSFORMS["flip_h"](g_in)).astype(float))


def run_online(eta, norm=False, gate=0.3, steps=2000, tag=""):
    rng = np.random.default_rng(5)
    W = rng.uniform(0.0, 0.5, (256, 256))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    es = []
    for x, y in zip(Xs, Ys):
        g = x > gate
        pred = W @ x
        e = y - pred
        es.append(np.linalg.norm(e))
        upd = eta * np.outer(e, x * g)
        if norm:
            denom = np.dot(x * g, x * g) + 1e-8
            upd = upd / denom
        W += upd
        n = np.linalg.norm(W, axis=0, keepdims=True)
        over = n[0] > 6.0
        if over.any():
            W[:, over] *= 6.0 / np.maximum(n[:, over], 1e-8)
    # eval
    ca = []
    for _ in range(150):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), cond)
        x = model.Ws[1] @ model.xs[1]
        g_pred = arc.decode_grid(W @ x)
        ca.append(float(np.mean(g_pred == TRANSFORMS["flip_h"](g_in).ravel())))
    print(f"{tag} eta={eta} norm={norm} gate={gate}: e_first={es[0]:.1f} "
          f"e_mid={es[len(es)//2]:.1f} e_last={es[-1]:.1f} cell={np.mean(ca):.4f}")


run_online(0.05, False, 0.3, 2000, "plain")
run_online(0.02, False, 0.3, 2000, "plain")
run_online(0.05, True, 0.3, 2000, "nlms")
run_online(0.3, True, 0.3, 2000, "nlms")


def run_online0(eta, norm=False, gate=0.3, steps=2000, tag=""):
    W = np.zeros((256, 256))
    es = []
    for x, y in zip(Xs, Ys):
        g = x > gate
        pred = W @ x
        e = y - pred
        es.append(np.linalg.norm(e))
        upd = eta * np.outer(e, x * g)
        if norm:
            denom = np.dot(x * g, x * g) + 1e-8
            upd = upd / denom
        W += upd
    ca = []
    for _ in range(150):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), cond)
        x = model.Ws[1] @ model.xs[1]
        g_pred = arc.decode_grid(W @ x)
        ca.append(float(np.mean(g_pred == TRANSFORMS["flip_h"](g_in).ravel())))
    print(f"{tag} eta={eta} norm={norm} gate={gate}: e_first={es[0]:.1f} "
          f"e_last={es[-1]:.1f} cell={np.mean(ca):.4f}")


run_online0(0.3, True, 0.3, 2000, "zero-init")
run_online0(0.3, True, 0.3, 800, "zero-init800")
run_online0(0.5, True, 0.3, 800, "zero-init800")
