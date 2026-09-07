#!/usr/bin/env python3
"""W1 对照实验：核心是否真依赖输入？头是否真可区分？

对照 A：同 cond 下，真实输入 vs 全零输入 vs 随机噪声输入 -> 输出差异。
对照 B：同输入下，flip_h vs flip_v vs recolor cond -> 输出差异。
对照 C：xs[L] 中 cond 支撑维占比；k-WTA 是否截断输入细节。
"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite
from srpc.runner_b import run_sequential


def main():
    seed = 0
    dcfg = DeepConfig()
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), CLConfig())
    model, arc = res["model"], ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    rng = np.random.default_rng(7)

    g_real = arc.sample_input()
    oh_real = arc._onehot(g_real).astype(float)
    c0 = arc._cond(0)
    c1 = arc._cond(1)
    c3 = arc._cond(3)

    # ---- 对照 A：输入敏感性（同 cond=flip_h）----
    v_real = model.apply_transform(oh_real, c0)
    v_zero = model.apply_transform(np.zeros_like(oh_real), c0)
    v_noise = model.apply_transform(rng.uniform(0, 1, oh_real.size), c0)
    d_rz = np.linalg.norm(v_real - v_zero)
    d_rn = np.linalg.norm(v_real - v_noise)
    g_real_out = arc.decode_grid(v_real)
    g_zero_out = arc.decode_grid(v_zero)
    print(f"[对照A 输入敏感性] |out(real)-out(zero)|={d_rz:.4f} "
          f"|out(real)-out(noise)|={d_rn:.4f}")
    print(f"  real -> 非空格数 {np.count_nonzero(g_real_out)}/{np.count_nonzero(g_real)} | "
          f"zero -> 非空格数 {np.count_nonzero(g_zero_out)}")

    # ---- 对照 B：头区分度（同输入不同 cond）----
    v_a = model.apply_transform(oh_real, c0)
    v_b = model.apply_transform(oh_real, c1)
    v_c = model.apply_transform(oh_real, c3)
    d_ab = np.linalg.norm(v_a - v_b)
    d_ac = np.linalg.norm(v_a - v_c)
    print(f"[对照B 头区分度] |out(c0)-out(c1)|={d_ab:.4f} |out(c0)-out(c3)|={d_ac:.4f}")

    # ---- 对照 C：xs[L] cond 支撑维占比 + WTA 截断 ----
    model.apply_transform(oh_real, c0)
    x = model.xs[model.L].copy()
    Uc = model.Uc
    nc = Uc.shape[1]
    b = model.d_self // nc
    cond_support = np.zeros(model.d_self, dtype=bool)
    for k in range(nc):
        if c0[k] > 0:
            cond_support[k * b:(k + 1) * b] = True
    n_sup = int(cond_support.sum())
    x_on_sup = float(np.count_nonzero(x[cond_support]))
    x_off = float(np.count_nonzero(x[~cond_support]))
    print(f"[对照C 表征结构] d_self={model.d_self} cond支撑维={n_sup} "
          f"非零: 支撑内={x_on_sup:.0f} 支撑外={x_off:.0f}")
    print(f"  xs[L] 非零总数={np.count_nonzero(x)} "
          f"| x[支撑外] 均值={np.abs(x[~cond_support]).mean():.4f} "
          f"max={np.abs(x[~cond_support]).max():.4f}")


if __name__ == "__main__":
    main()
