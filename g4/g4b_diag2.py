"""G4b 诊断2：规划陷阱信号是否被自学 Wdyn 捕获（无预钉自组织码）。

直接检查 run_policy 实际使用的学习参数：(Err, P) 矩阵 + 多变上下文单发码可分性。
陷阱被捕获 ⇔ A 分支（1→2/3 干净叶）平均自误差 < B 分支（4/5/6 噪声井）。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns, _reset, _learned_params
from g4.g4b_planner import make_selforg_model, stable_code, train_selforg


def in_context_selfNN(model, task, patterns, rng, n=20):
    """带历史上下文的单发码对状态的可分性（同 run_policy 的采点方式）。"""
    model.set_learning(False)
    codes = np.zeros((task.n_states, model.cfg.n_self))
    # 制造不同上下文起点
    starts = []
    for _ in range(n):
        v = int(rng.integers(task.n_states))
        _reset(model)
        model.observe(patterns[v], v)
        starts.append(model.xs.copy())
    # 从每个上下文起点各自 observe 每个状态
    accs = np.zeros((task.n_states, model.cfg.n_self))
    cnt = np.zeros(task.n_states)
    for st in starts:
        for v in range(task.n_states):
            model.xs[:] = st
            model.observe(patterns[v], v)
            accs[v] += model.xs
            cnt[v] += 1
    codes = accs / np.maximum(cnt[:, None], 1e-9)
    hits = 0
    for i in range(task.n_states):
        d = np.linalg.norm(codes - codes[i].reshape(1, -1), axis=1)
        hits += int(np.argmin(d) == i)
    model.set_learning(True)
    return hits / task.n_states, codes


def main():
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        task = TrapTree(rng=rng)
        model, d_obs = make_selforg_model(seed)
        patterns = encode_state_patterns(task, rng, d_obs)
        train_selforg(model, task, patterns, 4000, rng)
        nn, codes = in_context_selfNN(model, task, patterns, rng)
        err, P = _learned_params(model, task, patterns, None)
        print(f"\n== seed {seed}")
        print(f"  in-context 单发码 NN 可分性: {nn:.3f}")
        print("  学习 Err[v][a]（A/B 两动作自误差）:")
        for v in range(task.n_states):
            print(f"    v{v}: {['%s' % round(float(err[v][a]),3) for a in (0,1)]}"
                  f"   P={['%.2f' % float(P[v][a]) for a in (0,1)]}")
        e0, e1 = float(err[0][0]), float(err[0][1])
        print(f"  root: act0->A Err={e0:.3f} vs act1->B Err={e1:.3f}  "
              f"(A 应<B 才对抗 '一步干净'陷阱: {'OK' if e0 < e1 else 'REVERSED'})")


if __name__ == "__main__":
    main()