"""G4b 调参：扫 train_steps × freeze_frac，看是否恢复"陷阱"信号 + 让叶自环→0。

陷阱信号 ⇔ Err[root,1→B] < Err[root,0→A]（B 一步更干净，greedy 被骗进 B，
前瞻 H 看穿 A 而胜出）。叶自环 = Err[leaf,a]（应→0：持久→零误差）。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns, _reset, _learned_params
from g4.g4b_planner import make_selforg_model, train_selforg
from g1.g1_srpc_planner import run_policy


def probe(seed, steps, freeze):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed)
    patterns = encode_state_patterns(task, rng, d_obs)
    train_selforg(model, task, patterns, steps, rng, freeze_frac=freeze)
    Err, P = _learned_params(model, task, patterns, None)
    trap = float(Err[0][1]) < float(Err[0][0])   # 陷阱恢复（B 一步更干净）
    leaf = float(np.min([Err[2][a] for a in (0, 1)] + [Err[3][a] for a in (0, 1)]))
    # 短规划探针：3 episodes 平均 rel
    g, l = [], []
    for e in range(30):
        rg = np.random.default_rng(100000 + seed * 10000 + e)
        g.append(run_policy(model, task, patterns, None, "greedy", 1, 40, rg).sum())
        l.append(run_policy(model, task, patterns, None, "lookahead", 3, 40, rg).sum())
    gm, lm = float(np.mean(g)), float(np.mean(l))
    rel = (gm - lm) / (gm + 1e-9)
    return {"trap": trap, "E_A": round(float(Err[0][0]), 3),
            "E_B": round(float(Err[0][1]), 3), "leafmin": round(leaf, 3),
            "greedy": round(gm, 2), "lookahead": round(lm, 2), "rel": round(rel, 3)}


def main():
    print(f"{'seed':>4} {'steps':>6} {'freeze':>6} | {'trap':>5} {'E_A':>5} {'E_B':>5} "
          f"{'leaf':>5} {'greedy':>7} {'look':>7} {'rel':>7}")
    for seed in (0, 1):
        for steps, freeze in ((6000, 0.5), (12000, 0.5), (12000, 0.35), (20000, 0.5)):
            r = probe(seed, steps, freeze)
            print(f"{seed:>4} {steps:>6} {freeze:>6} | {str(r['trap']):>5} {r['E_A']:>5} "
                  f"{r['E_B']:>5} {r['leafmin']:>5} {r['greedy']:>7} {r['lookahead']:>7} "
                  f"{r['rel']:>7}")


if __name__ == "__main__":
    main()