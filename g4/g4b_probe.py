"""G4b 综合探针：当前新配置（大码空间+强动力学）下 root 陷阱方向、叶自环、动作目标。
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import (TrapTree, encode_state_patterns, _learned_params,
                                run_policy)
from g4.g4b_planner import make_selforg_model, train_selforg, stable_code


def main():
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        task = TrapTree(rng=rng)
        model, d_obs = make_selforg_model(seed)
        patterns = encode_state_patterns(task, rng, d_obs)
        train_selforg(model, task, patterns, 6000, rng)
        codes = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
        Err, P = _learned_params(model, task, patterns, None)
        leaf = float(np.min([Err[2][a] for a in (0,1)] + [Err[3][a] for a in (0,1)]))
        # root 动作目标与 E
        model.set_learning(False)
        for a in (0, 1):
            model.observe(patterns[0], 0); model.prepare_next(a)
            d = np.linalg.norm(codes - model.pred_self.reshape(1, -1), axis=1)
            print(f"  seed{seed} root act{a}->状态{int(np.argmin(d))}  E={Err[0][a]:.3f}"
                  f"  d(->Ain/state1)={d[1]:.3f} d(->Bin/state4)={d[4]:.3f}")
        model.set_learning(True)
        print(f"  seed{seed}: E_A(->A)={Err[0][0]:.3f} E_B(->B)={Err[0][1]:.3f} "
              f"leafmin={leaf:.3f}  root trap(B<E_A)={Err[0][1] < Err[0][0]}")


if __name__ == "__main__":
    main()