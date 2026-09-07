#!/usr/bin/env python3
"""clip cap=6 读出扫描：eta × steps。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import (ArcConfig, CLConfig, DeepConfig, MemoryConfig)
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

CONFIGS = [
    ("cap6_eta08_s800",  dict(eta=0.08, steps=800)),
    ("cap6_eta12_s800",  dict(eta=0.12, steps=800)),
    ("cap6_eta08_s2400", dict(eta=0.08, steps=2400)),
    ("cap6_eta12_s2400", dict(eta=0.12, steps=2400)),
]

seed = 0


def one(name, eta, steps):
    dcfg = replace(DeepConfig(), trace_energy=False, eta_wout=eta)
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    cl = replace(CLConfig(), steps_per_task=steps)
    res = run_sequential(seed, True, dcfg, mcfg, ArcConfig(), cl)
    model = res["model"]
    arc = ArcLite(ArcConfig(), np.random.default_rng(seed * 3000 + 2))
    model.set_learning(False)
    accs = {}
    for nm in arc.train_names + arc.novel_names:
        is_novel = nm in arc.novel_names
        ca = []
        for _ in range(60):
            g_in = arc.sample_input()
            oh_in = arc._onehot(g_in).astype(float)
            if is_novel:
                i, j = arc.novel_combos[arc.novel_names.index(nm)]
                out1 = model.apply_transform(oh_in, arc._cond(arc.train_names.index(i)))
                v = model.apply_transform(out1, arc._cond(arc.train_names.index(j)))
                g_true = TRANSFORMS[j](TRANSFORMS[i](g_in))
            else:
                v = model.apply_transform(oh_in, arc._cond(arc.train_names.index(nm)))
                g_true = TRANSFORMS[nm](g_in)
            g_pred = arc.decode_grid(v).ravel()
            ca.append(float(np.mean(g_pred == np.asarray(g_true).ravel())))
        accs[nm] = float(np.mean(ca))
    print(f"{name:16s} " + " ".join(f"{k}={v:.3f}" for k, v in accs.items()))


for name, kw in CONFIGS:
    one(name, **kw)
