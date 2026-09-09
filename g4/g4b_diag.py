"""G4b 诊断：单发(one-shot) x_self 码 vs 泛洪(flooded) 稳态码的可分性。

规划/训练用的是"单发 observe"（每步一次，inner_iters 局部推断后直接采 xs），
而参考码常用泛洪稳定码。若单发码对状态不可分，则 Wdyn 无可靠码可预测 → 规划失效。
本脚本对比：单发码展互性、单发 vs 泛洪一致性、inner_iters / settle 次数的影响。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns, _reset
from g4.g4b_planner import make_selforg_model, stable_code, train_selforg, \
    build_local_sparse_weights


def one_shot_codes(model, task, patterns, rng, n=20):
    """从零/随机初始重采样 n 次单发编码，看其对状态是否可分。"""
    model.set_learning(False)
    codes = []
    for v in range(task.n_states):
        acc = None
        for _ in range(n):
            _reset(model)                    # 清时序
            model.x1[:] = 0; model.x2[:] = 0; model.xs[:] = 0
            model.observe(patterns[v], v)     # 单发
            acc = model.xs if acc is None else acc + model.xs
        codes.append(acc / n)
    model.set_learning(True)
    return np.stack(codes)


def report(tag, codes):
    norms = np.linalg.norm(codes, axis=1)
    msep = min(np.linalg.norm(codes[i] - codes[j])
               for i in range(codes.shape[0]) for j in range(i + 1, codes.shape[0]))
    # 单发码内部稳定性：目标状态判读
    hits = 0
    for i in range(codes.shape[0]):
        d = np.linalg.norm(codes - codes[i].reshape(1, -1), axis=1)
        hits += int(np.argmin(d) == i)
    print(f"  {tag:<38} sep_min={msep:.4f} norm={[round(float(x),2) for x in norms]} "
          f"selfNN={hits}/{codes.shape[0]}")
    return msep


def diag(seed, n_l1=24, nself=20, inner_iters=3, win=0.25, kwta=0.5,
         train_steps=4000):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed, n_l1, nself, win, kwta, 0.5, 0.10, 0.50)
    # 覆盖 inner_iters（make_selforg_model 用默认 3，这里手动注入）
    model.cfg.inner_iters = inner_iters
    patterns = encode_state_patterns(task, rng, d_obs)
    print(f"\n== seed {seed}  inner_iters={inner_iters}")
    print("  — 未训练（随机初始权重）—")
    cc = one_shot_codes(model, task, patterns, rng)
    report("one-shot (untrained)", cc)
    print("  — 训练后（自组织码）—")
    train_selforg(model, task, patterns, train_steps, rng)
    cc = one_shot_codes(model, task, patterns, rng)
    report("one-shot (trained)", cc)
    fl = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
    report("flooded (trained)", fl)


if __name__ == "__main__":
    for it in (10, 20):
        diag(0, inner_iters=it)