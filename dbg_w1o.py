#!/usr/bin/env python3
"""读出学习收敛方式：eta 衰减 vs 小批量累积（都是局部规则，免反传）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

seed = 0
arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))


def train_and_eval(mode, steps=1600, eta0=0.12, B=32):
    dcfg = replace(DeepConfig(), trace_energy=False, ro_norm="clip",
                   ro_norm_cap=6.0, eta_wout=eta0)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    cl = replace(CLConfig(), steps_per_task=steps)
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), cl)
    model = res["model"]
    model.set_learning(True)
    # 继续训练头：mode="decay" 线性衰减 eta；mode="batch" 小批量累积（B 样本平均后更新）
    for name in arc.train_names:
        cond = arc._cond(arc.train_names.index(name))
        idx = arc.train_names.index(name)
        acc_buf = np.zeros_like(model.W_outs[idx])
        n_acc = 0
        for t in range(2400):
            s_in, s_out, _ = arc.sample(name)
            model.set_condition(cond)
            model.reset_states()
            model.prepare_next(action=None)
            model.observe(s_in)
            if mode == "decay":
                model.cfg = replace(model.cfg, eta_wout=eta0 * max(0.02, 1.0 - t / 2400))
                model.learn_readout(s_out)
            elif mode == "batch":
                x1 = model.xs[model.ro_src]
                g = x1 > model.cfg.readout_gate
                e = s_out - (model.W_outs[idx] @ x1)
                acc_buf += np.outer(e, x1 * g)
                n_acc += 1
                if n_acc == B:
                    model.W_outs[idx] += (model.cfg.eta_wout * B / 32.0) * (acc_buf / B)
                    if model.ro_masks[idx] is not None:
                        model.W_outs[idx] *= model.ro_masks[idx]
                    n = np.linalg.norm(model.W_outs[idx], axis=0, keepdims=True)
                    over = n[0] > model.cfg.ro_norm_cap
                    if over.any():
                        model.W_outs[idx][:, over] *= model.cfg.ro_norm_cap / np.maximum(n[:, over], 1e-8)
                    acc_buf.fill(0.0)
                    n_acc = 0
    model.set_learning(False)
    accs = {}
    for nm in arc.train_names:
        ca = []
        for _ in range(60):
            g_in = arc.sample_input()
            v = model.apply_transform(arc._onehot(g_in).astype(float),
                                      arc._cond(arc.train_names.index(nm)))
            g_pred = arc.decode_grid(v).ravel()
            ca.append(float(np.mean(g_pred == TRANSFORMS[nm](g_in).ravel())))
        accs[nm] = float(np.mean(ca))
    print(f"{mode:6s} " + " ".join(f"{k}={v:.3f}" for k, v in accs.items()))


train_and_eval("decay", eta0=0.15)
train_and_eval("batch", eta0=0.06)
