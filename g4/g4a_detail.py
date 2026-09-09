"""G4a 细节诊断：定位无预钉下塌缩的具体状态对。
复用 g4a_scan 的 block 局部受体野构造，打印逐状态误判目标 / 最近邻对 / norm / 支撑。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from g4.g4a_scan import make_model, train_quick, stable_xs


NAMES = {0: "root", 1: "A-in", 2: "A-leaf0", 3: "A-leaf1",
         4: "B-in", 5: "B-noise0", 6: "B-noise1"}


def detail(seed: int, win: float = 0.25, kwta: float = 0.5):
    task = TrapTree()
    model, rng = make_model(seed, "block", win, kwta, 0.5)
    patterns = encode_state_patterns(task, rng, model.cfg.d_obs)
    train_quick(model, task, patterns, 3000, rng)
    model.set_learning(False)
    codes = np.stack([stable_xs(model, patterns[v], v)
                      for v in range(task.n_states)])
    print(f"\n== seed {seed}: win={win} kwta={kwta}")
    print(f"   state norm    = {[round(np.linalg.norm(codes[v]),3) for v in range(7)]}")
    print(f"   support       = {[(codes[v]>1e-6).sum() for v in range(7)]}")
    # 逐状态最近邻（排除自身）
    print("   最近邻映射：")
    for v in range(task.n_states):
        d = np.linalg.norm(codes - codes[v].reshape(1, -1), axis=1)
        d[v] = np.inf
        nn = int(np.argmin(d))
        print(f"     {NAMES[v]:>9} -> {NAMES[nn]:>11}  L2={d[nn]:.3f}  cos={np.dot(codes[v],codes[nn])/max(np.linalg.norm(codes[v])*np.linalg.norm(codes[nn]),1e-9):.3f}")
    model.set_learning(True)


if __name__ == "__main__":
    for s in (0, 1, 2, 3, 4, 5):
        detail(s)