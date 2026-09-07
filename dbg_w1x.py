#!/usr/bin/env python3
"""LS 上限测量：当前冻结特征 ŝ 上，最小二乘读出头能达到的符号级精度。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

clcfg = CLConfig(steps_per_task=800)
dcfg = replace(DeepConfig(), trace_energy=False)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                     acfg=ArcConfig(), clcfg=clcfg)
model = res["model"]
arc = ArcLite(ArcConfig(), np.random.default_rng(0 * 3000 + 2))
model.set_learning(False)

for name in arc.train_names:
    cond = arc._cond(arc.train_names.index(name))
    # 收集特征矩阵 X (n×256) 与目标 Y (n×256)
    rng = np.random.default_rng(100)
    X, Y = [], []
    for _ in range(3000):
        g_in = arc.sample_input()
        model.apply_transform(arc._onehot(g_in).astype(float), cond)
        X.append(model._ro_src().copy())
        Y.append(arc._onehot(TRANSFORMS[name](g_in)).astype(float))
    X = np.array(X); Y = np.array(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)   # LS 解
    # 特征条件数 + 表征区分度
    cond_n = np.linalg.cond(X)
    # LS 精度（全格 / 符号格）
    pred = X @ W
    g_pred = pred.reshape(-1, 8, 8, 4).argmax(-1)
    g_true = np.array([TRANSFORMS[name](g_in) for g_in in
                       [arc.decode_grid(x) for x in np.zeros(0)]])  # 占位
    # 重算 true
    g_true = np.array([TRANSFORMS[name](arc.decode_grid(y)) for y in []])  # 空
    # 直接用 Y 反推 true 网格
    g_true = np.array([arc.decode_grid(Y[i]).ravel() for i in range(len(Y))])
    g_pred = g_pred.reshape(-1, 64)
    allc = np.mean([np.mean(g_pred[i] == g_true[i]) for i in range(len(Y))])
    symc = np.mean([np.mean(g_pred[i][g_true[i] != 0] == g_true[i][g_true[i] != 0])
                    for i in range(len(Y))])
    gridc = np.mean([np.all(g_pred[i] == g_true[i]) for i in range(len(Y))])
    # ŝ 区分度
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cos = (Xn @ Xn.T)
    n_pair = 0; n_ok = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            n_pair += 1
            n_ok += (cos[i, j] < 0.9)
    print(f"{name:10s}: LS allcell={allc:.4f} sym={symc:.4f} grid={gridc:.4f} "
          f"cond={cond_n:.1e} div_disc={n_ok/n_pair:.3f}")
