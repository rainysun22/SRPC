"""seed4 Wdyn 动作条件化诊断：pred 落在哪个参考码附近 vs 真转移目标。"""
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
    model.set_learning(False); acc=None
    for _ in range(floods):
        model.observe(pat, state_id); acc = model.xs if acc is None else acc+model.xs
    model.set_learning(True); return acc/floods
codes = np.stack([stable_code(model, patterns[v], v) for v in range(7)])

print("Wdyn shape:", model.Wdyn.shape)  # (n_self, n_self + na)
cols = model.Wdyn.shape[1]
n_self = model.cfg.n_self
print("action-col norms:",
      np.round(np.linalg.norm(model.Wdyn[:, n_self:], axis=0), 3))
print("self-block norm:", round(float(np.linalg.norm(model.Wdyn[:, :n_self])),3))

model.set_learning(False)
for vv in range(7):
    for a in (0,1):
        _canon(model); model.observe(patterns[vv], vv)
        model.prepare_next(a); pred = model.pred_self.copy()
        c = task.child[vv][a]
        dd = np.linalg.norm(codes - pred.reshape(1,-1), axis=1)
        nrst = int(np.argmin(dd))
        # target = emix with p
        emix = task.p[vv][a]*codes[c] + (1-task.p[vv][a])*codes[vv]
        m = float(np.linalg.norm(pred-emix))
        print(f"v{vv}a{a}: child={c} p={task.p[vv][a]:.2f} | pred nearest={nrst} d={dd[nrst]:.3f} "
              f"(child dist={dd[c]:.3f}) | pred-vs-emix={m:.3f} | pred_norm={np.linalg.norm(pred):.3f}")
model.set_learning(True)