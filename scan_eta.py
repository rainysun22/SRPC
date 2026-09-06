#!/usr/bin/env python3
"""eta_w 扫参：Δ=1 与 Δ=4 双判据，3 seeds，10000 步。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import _run_seed

SEEDS = (0, 1, 2)


def ev(cfg, delay):
    rs = [_run_seed(cfg, s, delay=delay, train_steps=cfg.train_steps, eval_steps=500)
          for s in SEEDS]
    pcn = float(np.mean([r["acc_error"] for r in rs]))
    hebb = float(np.mean([r["acc_hebb"] for r in rs]))
    dist = float(np.mean([r["distal_error"] for r in rs]))
    lo = float(min(r["acc_error"] for r in rs))
    return pcn, hebb, dist, lo


base = CreditConfig(alpha=0.65)
for eta in (0.02, 0.03, 0.04, 0.05):
    cfg = replace(base, eta_w=eta)
    p4, h4, d4, lo4 = ev(cfg, 4)
    p1, h1b, d1, lo1 = ev(cfg, 1)
    ok4 = (lo4 >= 0.80 and p4 - h4 >= 0.30 and d4 >= 0.25 and h4 <= 0.68)
    ok1 = (lo1 >= 0.75 and p1 - h1b >= 0.15)
    print(f"eta={eta:.2f}: Δ4 pcn={p4:.3f}(min {lo4:.3f}) hebb={h4:.3f} gap={p4-h4:+.3f} "
          f"dist={d4:.3f} [{'PASS' if ok4 else 'fail'}] | "
          f"Δ1 pcn={p1:.3f}(min {lo1:.3f}) hebb={h1b:.3f} gap={p1-h1b:+.3f} "
          f"[{'PASS' if ok1 else 'fail'}]")
