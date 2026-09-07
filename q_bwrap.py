#!/usr/bin/env python3
"""B 收尾快速实验：① 空背景编码 on/off 对 S1/S2 影响；② eval 收敛迭代数对记忆增益影响。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential, eval_task, forget_stats
from srpc.arc import ArcLite, TRANSFORMS

ACFG = ArcConfig()
SEED = 0


def _onehot_nobg(self, g):
    """空(0) -> 全零（不占 channel0），颜色 1..3 -> 各自通道。"""
    oh = np.zeros((g.size, self.n_colors))
    for c in range(1, self.n_colors):
        oh[:, c] = (g.ravel() == c).astype(float)
    return oh.ravel()


def quick(seed, with_memory, bg=True, steps=400):
    if not bg:
        ArcLite._onehot = _onehot_nobg
    try:
        dcfg = DeepConfig()
        mcfg = MemoryConfig(d=dcfg.dims[-1])
        clcfg = CLConfig(steps_per_task=steps, eval_samples=12)
        res = run_sequential(seed, with_memory, dcfg, mcfg, ACFG, clcfg)
    finally:
        from srpc import arc as _arcmod
        ArcLite._onehot = _arcmod.ArcLite._onehot  # 恢复
    return res


def recompute_R(model, arc, clcfg, eval_iters):
    model.set_learning(False)
    model.cfg = replace(model.cfg, inner_iters=eval_iters)
    n = arc.n_train
    R = np.zeros((n, n))
    for j in range(n):
        for i in range(j + 1):
            R[i, j] = eval_task(model, arc, arc.train_names[i], clcfg, SEED)
    return R


def s1s2(model, arc, n_sym=40, n_div=12):
    """快速 S1 (flip_h) + S2。"""
    model.set_learning(False)
    rng = np.random.default_rng(7)
    cond = arc._cond(0)
    ca, ga = [], []
    for _ in range(n_sym):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        gp = arc.decode_grid(v).ravel()
        gt = TRANSFORMS["flip_h"](g_in).ravel()
        ca.append(np.mean(gp == gt)); ga.append(np.all(gp == gt))
    cell, grid = float(np.mean(ca)), float(np.mean(ga))
    g = arc.grid
    def enc(grid_int):
        model.apply_transform(arc._onehot(grid_int).astype(float), cond)
        return model._ro_src().copy()
    pos = []
    for (r, c) in ((0, 0), (0, g - 1), (g - 1, 0), (g - 1, g - 1)):
        gg = np.zeros((g, g), dtype=int); gg[r, c] = 1
        pos.append(enc(gg))
    pc = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
          for i, a in enumerate(pos) for b in pos[i + 1:]]
    div = [enc(arc.sample_input()) for _ in range(n_div)]
    ok = 0; np_ = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            cs = float(np.dot(div[i], div[j]) /
                       (np.linalg.norm(div[i]) * np.linalg.norm(div[j]) + 1e-9))
            np_ += 1; ok += (cs < 0.9)
    return dict(cell=cell, grid=grid, pos_cos=max(pc),
                div=ok / np_)


def main():
    for bg in (True, False):
        # S1/S2
        res_m = quick(SEED, True, bg=bg)
        arc = ArcLite(ACFG, np.random.default_rng(SEED * 3000 + 2))
        s = s1s2(res_m["model"], arc)
        # memory gain at eval_iters 3 / 6 / 9
        res_n = quick(SEED, False, bg=bg)
        model_m, model_n = res_m["model"], res_n["model"]
        for ei in (3, 6, 9):
            Rm = recompute_R(model_m, arc, CLConfig(eval_samples=12), ei)
            Rn = recompute_R(model_n, arc, CLConfig(eval_samples=12), ei)
            fm, fn = forget_stats(Rm, Rm.diagonal()), forget_stats(Rn, Rn.diagonal())
            # forget_stats needs diag; use last col vs diag
            fm = forget_stats(Rm, np.diag(Rm))
            fn = forget_stats(Rn, np.diag(Rn))
            g = (fn["retain_mean"] - fm["retain_mean"]) / (fn["retain_mean"] + 1e-8)
            print(f"bg={bg} eval_iters={ei}: mem={fm['retain_mean']:.4f} "
                  f"no={fn['retain_mean']:.4f} gain={g*100:+.1f}% "
                  f"forget_mem={fm['forget_mean']*100:.1f}%")
        print(f"bg={bg}: S1 cell={s['cell']:.4f} grid={s['grid']:.3f} "
              f"S2 pos_cos={s['pos_cos']:.3f} div={s['div']:.3f}")
        print()


if __name__ == "__main__":
    main()
