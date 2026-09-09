"""seed4 逐状态解码诊断：确认瓶颈是噪声转移还是编码塌缩。"""
import sys, os
sys.path.insert(0, "/workspace")
import numpy as np
from g4.g4b_planner import (make_selforg_model, train_selforg, _canon,
                            TrapTree, encode_state_patterns)

SEED = 4
rng = np.random.default_rng(SEED)
task = TrapTree(rng=rng)
model, d_obs = make_selforg_model(SEED)
patterns = encode_state_patterns(task, rng, d_obs)
train_selforg(model, task, patterns, 8000, rng)

def stable_code(model, pat, state_id, floods=100):
    model.set_learning(False)
    acc = None
    for _ in range(floods):
        model.observe(pat, state_id)
        acc = model.xs if acc is None else acc + model.xs
    model.set_learning(True)
    return acc / floods

codes = np.stack([stable_code(model, patterns[v], v) for v in range(task.n_states)])
print("code norms:", np.round(np.linalg.norm(codes, axis=1), 3))
print("sep_min:", round(float(np.min([np.linalg.norm(codes[i]-codes[j])
      for i in range(7) for j in range(i+1,7)])),4))
D = np.array([[np.linalg.norm(codes[i]-codes[j]) for j in range(7)] for i in range(7)])
np.fill_diagonal(D, np.inf)
for i in range(7):
    j = int(np.argmin(D[i]))
    print(f"state {i}: nearest={j} dist={D[i,j]:.3f} leaf={task.is_leaf[i]}")

model.set_learning(False)
er = np.random.default_rng(3)
n_meas = 400
per = np.zeros((7,2))
hit = np.zeros((7,2))
for vv in range(7):
    for a in (0,1):
        for _ in range(n_meas):
            _canon(model); model.observe(patterns[vv], vv); model.set_learning(False)
            model.prepare_next(a); pred=model.pred_self.copy()
            nxt = task.step(vv, a, er)
            d = np.linalg.norm(codes - pred.reshape(1,-1), axis=1)
            hit[vv,a] += int(np.argmin(d)==nxt); per[vv,a]+=1
model.set_learning(True)
for vv in range(7):
    for a in (0,1):
        print(f"v{vv}a{a}: decode={hit[vv,a]/per[vv,a]:.3f} p={task.p[vv][a]:.2f} leaf={task.is_leaf[vv]}")