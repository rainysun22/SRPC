"""G4 全判据快速扫描：解码分离 + 策略成本（确定性 DP，免 MC episode）。"""
import sys, os, json
sys.path.insert(0, "/workspace")
import numpy as np
import g4.g4b_planner as P
from g4.g4b_planner import (make_selforg_model, train_selforg,
                            eval_decode_frozen_selforg, _learned_params_c,
                            TrapTree, encode_state_patterns)

def policy_cost(Err, Pmat, task, H):
    ns = task.n_states
    # lookahead DP
    J = np.zeros((ns, H+1))
    for h in range(1, H+1):
        for v in range(ns):
            if task.is_leaf[v]:
                J[v,h] = 0.0; continue
            vals = []
            for a in (0,1):
                c = task.child[v][a]
                f = Pmat[v][a]*J[c,h-1] + (1-Pmat[v][a])*J[v,h-1]
                vals.append(Err[v][a] + f)
            J[v,h] = min(vals)
    # greedy: argmin 1-step Err each step (deterministic expected path)
    def step_cost(v, a):
        c = task.child[v][a]
        return Err[v][a] + Pmat[v][a]*0.0  # 1-step cost
    # simulate expected greedy walk
    def sim(policy):
        v=0; tot=0.0; seen=0
        while seen < 80:
            if task.is_leaf[v]:
                tot += Err[v][0]; continue  # stays in leaf
            if policy=="greedy":
                a = int(np.argmin(Err[v]))
            else:
                cand=[Err[v][aa]+Pmat[v][aa]*J[task.child[v][aa],H-1]+(1-Pmat[v][aa])*J[v,H-1] for aa in (0,1)]
                a=int(np.argmin(cand))
            c = task.child[v][a]
            tot += Err[v][a]
            v = c
            seen+=1
        return tot
    g = sim("greedy"); l = sim("lookahead")
    return g, l

def run(seed, cfg):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed,
        n_l1=cfg["n_l1"], nself=cfg["nself"], win_frac=cfg["win_frac"],
        kwta=cfg.get("kwta",0.33), alpha=cfg.get("alpha",0.05), beta=cfg.get("beta",0.5),
        eta_dyn=cfg.get("eta_dyn",0.8), dyn_decay=cfg.get("dyn_decay",1e-5))
    patterns = encode_state_patterns(task, rng, d_obs)
    P.train_selforg(model, task, patterns, cfg.get("steps",8000), rng,
                    freeze_frac=cfg.get("freeze_frac",0.5))
    dec = eval_decode_frozen_selforg(model, task, patterns)["decode_acc"]
    Err, Pp = _learned_params_c(model, task, patterns)
    g, l = policy_cost(Err, Pp, task, cfg.get("H",6))
    return dec, g, l, Err[0,0], Err[0,1]

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    res=[]
    for sd in range(6):
        dec,g,l,ea0,ea1 = run(sd, cfg)
        pass1 = dec>=0.55
        pass2 = l<g-1e-3
        res.append((sd, dec, g, l, ea0, ea1, pass1, pass2))
        print(f"seed{sd}: dec={dec:.3f} greedy={g:.2f} lookahead={l:.2f} "
              f"root_err[a0,a1]=({ea0:.3f},{ea1:.3f}) "
              f"dec={'P' if pass1 else 'F'} dyn={'P' if pass2 else 'F'} -> "
              f"{'PASS' if (pass1 and pass2) else 'FAIL'}", flush=True)
    p1 = sum(1 for r in res if r[6]); p2 = sum(1 for r in res if r[7])
    print(f"DECODE {p1}/6  DYNAMICS {p2}/6  BOTH {sum(1 for r in res if r[5] and r[7])}/6", flush=True)