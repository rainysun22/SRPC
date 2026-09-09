"""G4b 动作条件化诊断：Wdyn 的 action 列是否被学到，以及 root 上两动作预测是否可分。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from g4.g4b_planner import make_selforg_model, train_selforg, stable_code


def act_sep(model, task, patterns):
    # Wdyn 最后 na 列 = action 条件列
    na = model.na
    acol = np.linalg.norm(model.Wdyn[:, -na:], axis=0)
    xcol = np.linalg.norm(model.Wdyn[:, :-na], axis=0)
    codes = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
    # root 上两动作预测
    model.set_learning(False)
    model.observe(patterns[0], 0)
    model.prepare_next(0); p0 = model.pred_self.copy()
    model.observe(patterns[0], 0)
    model.prepare_next(1); p1 = model.pred_self.copy()
    model.set_learning(True)
    def nearest(p):
        d = np.linalg.norm(codes - p.reshape(1, -1), axis=1)
        return int(np.argmin(d))
    # 预测到各状态码的距离
    d0 = np.linalg.norm(codes - p0.reshape(1, -1), axis=1)
    d1 = np.linalg.norm(codes - p1.reshape(1, -1), axis=1)
    return {
        "action_col_norm": [round(float(x), 4) for x in acol],
        "x_col_NN_avg": round(float(np.mean(xcol)), 4),
        "pred0->": nearest(p0), "pred1->": nearest(p1),
        "dist(p0,p1)": round(float(np.linalg.norm(p0 - p1)), 3),
        "d0_to_code": [round(float(x), 3) for x in d0],
        "d1_to_code": [round(float(x), 3) for x in d1],
    }


def main():
    from g4.g4b_planner import stable_code as _sc
    for seed in (0, 1):
        for cfg_i, (nself, n_l1, win, kwta) in enumerate(
                ((40, 48, 0.15, 0.4), (48, 48, 0.15, 0.33), (64, 48, 0.12, 0.33))):
            rng = np.random.default_rng(seed)
            task = TrapTree(rng=rng)
            model, d_obs = make_selforg_model(seed, n_l1=n_l1, nself=nself,
                                              win_frac=win, kwta=kwta,
                                              alpha=0.05, beta=0.50)
            model.cfg.eta_dyn = 0.12
            model.cfg.dyn_decay = 1e-4
            patterns = encode_state_patterns(task, rng, d_obs)
            train_selforg(model, task, patterns, 4000, rng)
            r = act_sep(model, task, patterns)
            codes = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
            d01 = float(np.linalg.norm(codes[0] - codes[1]))
            d04 = float(np.linalg.norm(codes[0] - codes[4]))
            print(f"\n== seed {seed}  nself={nself} win={win} kwta={kwta} "
                  f"root-Ain L2={d01:.3f} root-Bin L2={d04:.3f}")
            print(f"  Wdyn action 列范数 = {r['action_col_norm']}")
            print(f"  root: pred(act0)->状态{r['pred0->']}  pred(act1)->状态{r['pred1->']}"
                  f"   pred间距离 {r['dist(p0,p1)']}")


if __name__ == "__main__":
    main()