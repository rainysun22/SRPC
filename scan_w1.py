#!/usr/bin/env python3
"""W1 修复扫描：beta_cond / gamma_mem / kwta_frac 对符号级精度的影响。

对照实验发现 x[L] 被条件码支撑维主导（支撑内 32 非零，支撑外 0），
读出=W_out@x_self 退化为"平均输出"（全空），符号格全错。
扫描：降低 beta_cond（条件先验拉动强度）、gamma_mem（记忆先验）、
kwta_frac（顶层截断）是否能放行输入细节到 x[L]。
"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

N_SYM = 60


def sym_cell_acc(model, arc, name, seed):
    """单任务符号级逐格一致率（冻结）。"""
    model.set_learning(False)
    rng = np.random.default_rng(seed * 771 + 3)
    accs = []
    for _ in range(N_SYM):
        g_in = arc.sample_input()
        oh_in = arc._onehot(g_in).astype(float)
        cond = arc._cond(arc.train_names.index(name))
        v = model.apply_transform(oh_in, cond)
        g_pred = arc.decode_grid(v).ravel()
        g_true = TRANSFORMS[name](g_in).ravel()
        accs.append(float(np.mean(g_pred == g_true)))
    return float(np.mean(accs))


def one(beta_cond, gamma_mem, kwta_frac, seed=0):
    dcfg = replace(DeepConfig(), beta_cond=beta_cond,
                   gamma_mem=gamma_mem, kwta_frac=kwta_frac)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model = res["model"]
    model.set_learning(False)
    accs = [sym_cell_acc(model, arc, n, seed) for n in arc.train_names]
    # 输入敏感性：real vs zero 输出差
    rng = np.random.default_rng(7)
    g_real = arc.sample_input()
    oh_real = arc._onehot(g_real).astype(float)
    c0 = arc._cond(0)
    v_real = model.apply_transform(oh_real, c0)
    v_zero = model.apply_transform(np.zeros_like(oh_real), c0)
    d = np.linalg.norm(v_real - v_zero)
    return accs, float(np.mean(accs)), d


if __name__ == "__main__":
    print(f"{'beta_cond':>9} {'gamma_mem':>9} {'kwta':>5} | "
          f"{'flip_h':>6} {'flip_v':>6} {'rot90':>6} {'recolr':>6} | "
          f"{'mean':>6} {'|Δout|':>8}")
    for bc in (0.05, 0.1, 0.2, 0.4, 0.8):
        accs, mean, d = one(bc, 0.35, 0.5)
        print(f"{bc:9.2f} {0.35:9.2f} {0.5:5.1f} | "
              + " ".join(f"{a:6.3f}" for a in accs) + f" | {mean:6.3f} {d:8.4f}")
    for gm in (0.1, 0.2):
        accs, mean, d = one(0.1, gm, 0.5)
        print(f"{0.1:9.2f} {gm:9.2f} {0.5:5.1f} | "
              + " ".join(f"{a:6.3f}" for a in accs) + f" | {mean:6.3f} {d:8.4f}")
    for kf in (0.3, 0.7, 1.0):
        accs, mean, d = one(0.1, 0.35, kf)
        print(f"{0.1:9.2f} {0.35:9.2f} {kf:5.1f} | "
              + " ".join(f"{a:6.3f}" for a in accs) + f" | {mean:6.3f} {d:8.4f}")
