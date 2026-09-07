#!/usr/bin/env python3
"""W1 诊断：符号级精度的错格模式 + readout 增强效果 + 顶层表征区分度。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential


def main():
    seed = 0
    dcfg = DeepConfig()
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    model, arc = res["model"], ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    rng = np.random.default_rng(7)

    cond = arc._cond(0)
    # --- 1. 错格模式：空格错 vs 符号格错（flip_h）---
    n_empty_err, n_sym_err, n_empty, n_sym = 0, 0, 0, 0
    for _ in range(200):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        g_pred = arc.decode_grid(v).ravel()
        g_true = TRANSFORMS["flip_h"](g_in).ravel()
        empt = g_true == 0
        n_empty += int(empt.sum()); n_sym += int((~empt).sum())
        n_empty_err += int(((g_pred != g_true) & empt).sum())
        n_sym_err += int(((g_pred != g_true) & ~empt).sum())
    print(f"[错格模式] 空格错 {n_empty_err}/{n_empty} ({n_empty_err/n_empty:.4f}) | "
          f"符号格错 {n_sym_err}/{n_sym} ({n_sym_err/n_sym:.4f})")

    # --- 2. readout 增强：额外训读出头（冻结内部），看 cell/grid 提升 ---
    model.set_learning(True)
    from srpc.deepmodel import _colnorm  # noqa
    for t in range(3000):
        s_in, s_out, c = arc.sample("flip_h")
        model.set_condition(c)
        model.reset_states()
        model.prepare_next(action=None)
        model.observe(s_in)
        model.learn_readout(s_out)
    model.set_learning(False)
    cell, grid = [], []
    for _ in range(200):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        g_pred = arc.decode_grid(v).ravel()
        g_true = TRANSFORMS["flip_h"](g_in).ravel()
        cell.append(float(np.mean(g_pred == g_true)))
        grid.append(float(np.all(g_pred == g_true)))
    print(f"[readout+3000] cell={np.mean(cell):.4f} grid={np.mean(grid):.4f}")

    # --- 3. 顶层表征区分度：2x2 块四角 / 三色（任务分布内输入）---
    def enc(g_int):
        model.apply_transform(arc._onehot(g_int).astype(float), cond)
        return model.xs[model.L].copy()
    g0, g1, g2, g3 = (np.zeros((8, 8), dtype=int) for _ in range(4))
    g0[0:2, 0:2] = 1; g1[0:2, 6:8] = 1; g2[6:8, 0:2] = 1; g3[6:8, 6:8] = 1
    z = [enc(g) for g in (g0, g1, g2, g3)]
    cos = [[float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            for b in z[i + 1:]] for i, a in enumerate(z)]
    print("[2x2 四角] 最大余弦:", max([c for row in cos for c in row]))
    c1, c2, c3 = (np.zeros((8, 8), dtype=int) for _ in range(3))
    c1[1:3, 1:3] = 1; c2[1:3, 1:3] = 2; c3[1:3, 1:3] = 3
    zc = [enc(g) for g in (c1, c2, c3)]
    cosc = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            for i, a in enumerate(zc) for b in zc[i + 1:]]
    print("[2x2 三色] 最大余弦:", max(cosc))
    # 同输入重复编码自一致（应≈1）
    zz = [enc(g0), enc(g0)]
    print("[同输入重复] 余弦:", float(np.dot(zz[0], zz[1]) / (
        np.linalg.norm(zz[0]) * np.linalg.norm(zz[1]) + 1e-9)))


if __name__ == "__main__":
    main()
