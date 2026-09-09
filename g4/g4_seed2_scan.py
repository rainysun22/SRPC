"""G4 seed2 定向扫描：用规范 oracle(run_policy_c) 找让 E_B<E_A(陷阱成立) 且
前瞻>近视的配置。只扫 seed2，episodes 降为 60 提速。"""
from __future__ import annotations
import sys, os, time
import numpy as np
sys.path.insert(0, "/workspace")
from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from g4.g4b_planner import (make_selforg_model, train_selforg,
                            eval_decode_frozen_selforg, _learned_params_c,
                            run_policy_c)

def probe(seed, cfg, episodes=60):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed,
        n_l1=cfg["n_l1"], nself=cfg["nself"], win_frac=cfg["win_frac"],
        kwta=cfg.get("kwta",0.33), alpha=cfg.get("alpha",0.05),
        beta=cfg.get("beta",0.5), eta_dyn=cfg.get("eta_dyn",0.8),
        dyn_decay=cfg.get("dyn_decay",1e-5))
    patterns = encode_state_patterns(task, rng, d_obs)
    train_selforg(model, task, patterns, cfg.get("steps",8000), rng,
                  freeze_frac=cfg.get("freeze_frac",0.5))
    dec = eval_decode_frozen_selforg(model, task, patterns)
    Err, P = _learned_params_c(model, task, patterns)
    EA, EB = float(Err[0][0]), float(Err[0][1])
    leaf = float(np.min([Err[2][a] for a in (0,1)]+[Err[3][a] for a in (0,1)]))
    g, l = [], []
    for e in range(episodes):
        rg = np.random.default_rng(100000+seed*10000+e)
        g.append(run_policy_c(model, task, patterns, "greedy", 1, 40, rg).sum())
        l.append(run_policy_c(model, task, patterns, "lookahead",
                              cfg.get("H",6), 40, rg).sum())
    gm, lm = float(np.mean(g)), float(np.mean(l))
    rel = (gm-lm)/(gm+1e-9)
    return {"dec": dec["decode_acc"], "sep": dec["state_sep_min"],
            "EA": EA, "EB": EB, "trap": EB<EA, "leaf": leaf,
            "greedy": gm, "look": lm, "rel": rel,
            "pass": (dec["decode_acc"]>=0.55 and lm<gm-1e-3)}

def grid():
    base = dict(n_l1=60, nself=64, win_frac=0.15, kwta=0.33,
                alpha=0.05, beta=0.50, eta_dyn=0.80, dyn_decay=1e-5,
                steps=8000, freeze_frac=0.5, H=6)
    yield ("baseline", base)
    for ed in (0.9, 1.0, 1.2):
        c = dict(base); c["eta_dyn"]=ed; c["dyn_decay"]=5e-6
        yield (f"eta{ed}", c)
    for st, fr in ((4000,0.5),(8000,0.35),(8000,0.3),(12000,0.5),
                   (12000,0.35),(16000,0.35),(16000,0.25),(24000,0.25)):
        c = dict(base); c["steps"]=st; c["freeze_frac"]=fr
        yield (f"step{st}_fr{fr}", c)

if __name__ == "__main__":
    print(f"{'name':>14} {'dec':>5} {'sep':>5} {'E_A':>5} {'E_B':>5} {'trap':>5} "
          f"{'leaf':>5} {'greedy':>7} {'look':>7} {'rel':>7}")
    for name, cfg in grid():
        t0=time.time()
        r = probe(2, cfg)
        print(f"{name:>14} {r['dec']:.3f} {r['sep']:.3f} {r['EA']:.3f} {r['EB']:.3f} "
              f"{str(r['trap']):>5} {r['leaf']:.3f} {r['greedy']:.2f} {r['look']:.2f} "
              f"{r['rel']:+.1%}  [{time.time()-t0:.0f}s]", flush=True)