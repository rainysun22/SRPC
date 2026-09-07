#!/usr/bin/env python3
"""诊断重建质量与参数方向：ŝ vs s 相似度、x1 输入成分、S2 随 (alpha,beta,beta_cond)。"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
from dataclasses import replace
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import run_sequential
from srpc.arc import ArcLite, TRANSFORMS

def quick(seed, dcfg):
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    res = run_sequential(seed, with_memory=True, dcfg=dcfg, mcfg=mcfg,
                         acfg=ArcConfig(), clcfg=CLConfig())
    model = res["model"]
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    cond = arc._cond(0)
    # 重建质量：||ŝ - s|| / ||s||
    errs = []
    for _ in range(50):
        g_in = arc.sample_input()
        s = arc._onehot(g_in).astype(float)
        model.apply_transform(s, cond)
        shat = model._ro_src()
        errs.append(float(np.linalg.norm(shat - s) / (np.linalg.norm(s) + 1e-9)))
    # S2 div
    def enc(grid_int):
        model.apply_transform(arc._onehot(grid_int).astype(float), cond)
        return model._ro_src().copy()
    div = [enc(arc.sample_input()) for _ in range(20)]
    n_ok = n_pair = 0
    for i in range(len(div)):
        for j in range(i + 1, len(div)):
            cs = np.dot(div[i], div[j]) / (np.linalg.norm(div[i]) * np.linalg.norm(div[j]) + 1e-9)
            n_pair += 1; n_ok += (cs < 0.9)
    # S1 flip_h cell
    ca = []
    for _ in range(40):
        g_in = arc.sample_input()
        v = model.apply_transform(arc._onehot(g_in).astype(float), cond)
        gp = arc.decode_grid(v).ravel()
        gt = np.asarray(TRANSFORMS["flip_h"](g_in)).ravel()
        ca.append(float(np.mean(gp == gt)))
    # mem_gain
    res_n = run_sequential(seed, with_memory=False, dcfg=dcfg, mcfg=mcfg,
                           acfg=ArcConfig(), clcfg=CLConfig())
    fm = res["R"][:, -1]; fn = res_n["R"][:, -1]
    gain = (fn.mean() - fm.mean()) / (fn.mean() + 1e-8)
    return dict(recon=np.mean(errs), div=n_ok / n_pair, cell=np.mean(ca),
                mem_gain=gain, ret_mem=fm.mean(), ret_no=fn.mean())

for alpha, beta, bc in ((0.40, 0.25, 0.80), (0.15, 0.25, 0.80),
                        (0.40, 0.80, 0.80), (0.15, 0.80, 0.80),
                        (0.15, 0.40, 0.15), (0.10, 0.60, 0.15),
                        (0.05, 0.80, 0.15), (0.40, 0.25, 0.0)):
    dcfg = replace(DeepConfig(), trace_energy=False, alpha=alpha, beta=beta, beta_cond=bc)
    r = quick(0, dcfg)
    print(f"a={alpha} b={beta} bc={bc}: recon={r['recon']:.3f} div={r['div']:.3f} "
          f"cell={r['cell']:.4f} mem_gain={r['mem_gain']*100:.1f}% "
          f"ret_mem={r['ret_mem']:.4f} ret_no={r['ret_no']:.4f}")
