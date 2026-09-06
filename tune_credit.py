#!/usr/bin/env python3
"""稳健性扫参：3 seeds，Δ=4（主判据）与 Δ=1（对照，应可学）。"""
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
    pcn = np.mean([r["acc_error"] for r in rs])
    hebb = np.mean([r["acc_hebb"] for r in rs])
    dist = np.mean([r["distal_error"] for r in rs])
    lo = min(r["acc_error"] for r in rs)
    return pcn, hebb, dist, lo


base = CreditConfig()
# (h1, h2, alpha, settle, eta_w, train)
cands = [
    (128, 64, 0.60, 15, 0.05, 10000),
    (128, 64, 0.55, 15, 0.05, 10000),
    (128, 64, 0.65, 15, 0.05, 10000),
    (128, 64, 0.60, 20, 0.05, 10000),
    (128, 64, 0.60, 15, 0.08, 10000),
    (128, 64, 0.60, 15, 0.05, 15000),
    (160, 80, 0.60, 15, 0.05, 10000),
]

for h1, h2, alpha, sit, eta_w, train in cands:
    cfg = replace(base, h1=h1, h2=h2, alpha=alpha, settle_iters=sit,
                  eta_w=eta_w, train_steps=train)
    p4, h4, d4, lo4 = ev(cfg, 4)
    p1, h1b, d1, lo1 = ev(cfg, 1)
    ok4 = (lo4 >= 0.80 and p4 - h4 >= 0.30 and d4 >= 0.25 and h4 <= 0.68)
    ok1 = (lo1 >= 0.75 and p1 - h1b >= 0.15)
    print(f"h={h1}/{h2} a={alpha:.2f} sit={sit} w={eta_w} tr={train}: "
          f"Δ4 pcn={p4:.3f}(min {lo4:.3f}) hebb={h4:.3f} gap={p4-h4:+.3f} "
          f"dist={d4:.3f} [{'PASS' if ok4 else 'fail'}] | "
          f"Δ1 pcn={p1:.3f}(min {lo1:.3f}) hebb={h1b:.3f} gap={p1-h1b:+.3f} "
          f"[{'PASS' if ok1 else 'fail'}]")
