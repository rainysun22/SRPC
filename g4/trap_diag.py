"""seed2 陷阱信号诊断：检查学到的 Err[root] 是否 E_B>E_A（陷阱成立）。"""
import sys, os
sys.path.insert(0, "/workspace")
import numpy as np
from g4.g4b_planner import (make_selforg_model, train_selforg,
                            _learned_params_c, TrapTree, encode_state_patterns)

SEED = 2
rng = np.random.default_rng(SEED)
task = TrapTree(rng=rng)
model, d_obs = make_selforg_model(SEED)
patterns = encode_state_patterns(task, rng, d_obs)
train_selforg(model, task, patterns, 8000, rng)
Err, P = _learned_params_c(model, task, patterns)
print("Err[row=state, col=action]:")
print(np.round(Err, 3))
print("真值 p:", task.p)
print("root Err[a0,a1] =", np.round(Err[0],3), " trap要求 Err[0,1]>Err[0,0]:", Err[0,1]>Err[0,0])
print("A-in Err[1] =", np.round(Err[1],3), " B-in Err[4] =", np.round(Err[4],3))