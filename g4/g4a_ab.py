"""G4a-①c 推断权重扫描：alpha(顶层拉拢) × beta(底层证据) 对无预钉 x_self 码分离的作用。

诊断发现 collapse 对固定为 root(0)/A-in(1)：二者 norm 与支撑完全相同、cos≈1.0。
它们是"枢纽/入口"状态，底层证据 (Ws2.T@e2) 弱，被顶层 pred_self 先验拉成均值。
原理性修复：增强 beta（底层证据主导）、减弱 alpha（自省预测仅作弱先验）。

固定最优受体野结构：block, win=0.25, kwta=0.5, fdyn=0.5，扫 (alpha,beta)。
报告：7状态 NN 解码、root↔A-in 距离/cos、min_pairwise_dist。
目标：6/6 seed decode_acc=1.0 且 root/A-in 分离（min_dist>0、cos 显著 <1）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from g4.g4a_scan import make_model, train_quick, stable_xs


def evaluate(seed, alpha, beta, steps=3000):
    task = TrapTree()
    model, rng = make_model(seed, "block", 0.25, 0.5, 0.5, alpha, beta)
    patterns = encode_state_patterns(task, rng, model.cfg.d_obs)
    train_quick(model, task, patterns, steps, rng)
    model.set_learning(False)
    codes = np.stack([stable_xs(model, patterns[v], v)
                      for v in range(task.n_states)])
    model.set_learning(True)
    # NN 解码
    hits = 0
    for v in range(task.n_states):
        c = stable_xs(model, patterns[v], v)
        d = np.linalg.norm(codes - c.reshape(1, -1), axis=1)
        hits += int(np.argmin(d) == v)
    decode = hits / task.n_states
    # root(0) <-> A-in(1) 分离
    r01 = float(np.linalg.norm(codes[0] - codes[1]))
    c01 = float(np.dot(codes[0], codes[1]) / max(np.linalg.norm(codes[0]) * np.linalg.norm(codes[1]), 1e-9))
    # 全局最小成对距离（排除自身）
    mind = np.inf
    for i in range(task.n_states):
        for j in range(i + 1, task.n_states):
            mind = min(mind, float(np.linalg.norm(codes[i] - codes[j])))
    return {"decode": decode, "rootAin_L2": r01, "rootAin_cos": c01,
            "min_pairwise": mind, "norm01": [round(float(np.linalg.norm(codes[0])), 3),
                                              round(float(np.linalg.norm(codes[1])), 3)]}


def main() -> int:
    combos = [(0.05, 0.25), (0.10, 0.50), (0.05, 0.50), (0.10, 0.75),
              (0.15, 0.50), (0.05, 0.75), (0.10, 0.25), (0.02, 0.60)]
    seeds = list(range(6))
    results = []
    for alpha, beta in combos:
        dec, r01, c01, mind, n01 = [], [], [], [], []
        for s in seeds:
            r = evaluate(s, alpha, beta)
            dec.append(r["decode"]); r01.append(r["rootAin_L2"])
            c01.append(r["rootAin_cos"]); mind.append(r["min_pairwise"])
            n01.append(r["norm01"])
        dmean = float(np.mean(dec))
        flag = "  <== 6/6" if all(x == 1.0 for x in dec) else ""
        print(f"alpha={alpha:.2f} beta={beta:.2f} | NN解码 {dmean:.3f} "
              f"seeds{['%.2f' % x for x in dec]} | root/Ain L2 {['%.3f' % x for x in r01]} "
              f"cos {['%.2f' % x for x in c01]}{flag}")
        results.append({"alpha": alpha, "beta": beta, "mean_decode": dmean,
                        "per_seed": dec, "rootAin_L2": r01, "rootAin_cos": c01,
                        "min_pairwise": mind, "norm01": n01})
    os.makedirs("/workspace/g4", exist_ok=True)
    with open("/workspace/g4/results_g4a_ab.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n写入 /workspace/g4/results_g4a_ab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())