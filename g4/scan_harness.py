import sys, os, json
sys.path.insert(0, "/workspace")
import numpy as np
from srpc.config import ModelConfig
from srpc.env import make_patterns
from g4.g4b_planner import (make_selforg_model, train_selforg,
                            eval_decode_frozen_selforg, TrapTree,
                            encode_state_patterns, _learned_params_c,
                            run_policy_c)

def run(seed, n_l1, nself, win_frac, kwta, alpha, beta, eta_dyn, dyn_decay, steps):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed, n_l1=n_l1, nself=nself,
        win_frac=win_frac, kwta=kwta, alpha=alpha, beta=beta,
        eta_dyn=eta_dyn, dyn_decay=dyn_decay)
    patterns = encode_state_patterns(task, rng, d_obs)
    train_selforg(model, task, patterns, steps, rng)
    return eval_decode_frozen_selforg(model, task, patterns)

cfg = json.loads(sys.argv[1])
res = run(seed=cfg["seed"], n_l1=cfg["n_l1"], nself=cfg["nself"],
          win_frac=cfg["win_frac"], kwta=cfg["kwta"], alpha=cfg["alpha"],
          beta=cfg["beta"], eta_dyn=cfg["eta_dyn"], dyn_decay=cfg["dyn_decay"],
          steps=cfg["steps"])
print(json.dumps({k: round(v,4) if isinstance(v,float) else v
                  for k,v in res.items()}, ensure_ascii=False))
