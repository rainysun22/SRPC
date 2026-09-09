"""G4b 距离诊断：训练后自组织码的成对距离，找近重复（尤其入口到 A-in/B-in 目标）。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from g4.g4b_planner import make_selforg_model, train_selforg, stable_code

NAMES = {0: "root", 1: "A-in", 2: "A-lf0", 3: "A-lf1", 4: "B-in", 5: "B-n0", 6: "B-n1"}


def main():
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        task = TrapTree(rng=rng)
        model, d_obs = make_selforg_model(seed)
        patterns = encode_state_patterns(task, rng, d_obs)
        train_selforg(model, task, patterns, 4000, rng)
        codes = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
        norms = np.linalg.norm(codes, axis=1)
        print(f"\n== seed {seed}  norms={[round(float(x),2) for x in norms]}")
        pairs = sorted(
            ((float(np.linalg.norm(codes[i] - codes[j])), i, j)
             for i in range(7) for j in range(i + 1, 7)))
        print("  最小距离对:")
        for d, i, j in pairs[:8]:
            print(f"    {NAMES[i]:>6} <-> {NAMES[j]:>6}  L2={d:.3f}")
        # 转移目标 1 vs 4（入口 A 与 B）
        d14 = float(np.linalg.norm(codes[1] - codes[4]))
        print(f"  入口目标 A-in(1) vs B-in(4)  L2={d14:.3f}  (区分越好 Wdyn 若可表达越能分别预测)")


if __name__ == "__main__":
    main()