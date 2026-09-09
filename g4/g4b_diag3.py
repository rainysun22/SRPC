"""G4b 诊断3：真实路径归因 —— 每 episode 根节点实际选 A(0) 还是 B(1)。
看 lookahead 是否因 DP 误差系统性选了 B（陷阱）而更差。附带打印叶自环误差。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns, _reset, _learned_params
from g4.g4b_planner import make_selforg_model, train_selforg, run_policy


def trace(model, task, patterns, policy, H, eps_steps, rng):
    from g1.g1_srpc_planner import _learned_params as lp
    Err, P = lp(model, task, patterns, None)
    model.set_learning(False)
    J = np.zeros((task.n_states, H + 1))
    for h in range(1, H + 1):
        for v in range(task.n_states):
            if task.is_leaf[v]:
                J[v, h] = 0.0
                continue
            vals = [Err[v][a] + P[v][a] * J[task.child[v][a], h - 1]
                    + (1.0 - P[v][a]) * J[v, h - 1] for a in (0, 1)]
            J[v, h] = min(vals)
    roots_A = roots_B = 0
    costs = []
    _reset(model)
    v = 0
    for _ in range(eps_steps):
        model.observe(patterns[v], v)
        if policy == "greedy":
            a = int(np.argmin(Err[v]))
        else:
            cand = [Err[v][aa] + P[v][aa] * J[task.child[v][aa], H - 1]
                    + (1.0 - P[v][aa]) * J[v, H - 1] for aa in (0, 1)]
            a = int(np.argmin(cand))
        if v == 0:
            roots_A += (a == 0)
            roots_B += (a == 1)
        model.prepare_next(a)
        pred = model.pred_self.copy()
        nxt = task.step(v, a, rng)
        model.observe(patterns[nxt], nxt)
        costs.append(float(np.linalg.norm(pred - model.xs)))
        v = nxt
    model.set_learning(True)
    return {"A": roots_A, "B": roots_B, "cost_sum": float(np.sum(costs))}


def main():
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        task = TrapTree(rng=rng)
        model, d_obs = make_selforg_model(seed)
        patterns = encode_state_patterns(task, rng, d_obs)
        train_selforg(model, task, patterns, 4000, rng)
        rg = np.random.default_rng(777)
        g = trace(model, task, patterns, "greedy", 1, 40, rg)
        rg = np.random.default_rng(777)
        l = trace(model, task, patterns, "lookahead", 3, 40, rg)
        print(f"seed {seed}: greedy rootA/B={g['A']}/{g['B']} cost={g['cost_sum']:.2f}"
              f" | lookahead rootA/B={l['A']}/{l['B']} cost={l['cost_sum']:.2f}")


if __name__ == "__main__":
    main()