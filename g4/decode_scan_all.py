"""G4 解码扫描（解码-only 全种子）—— 目标是让 6/6 种子 dec≥0.60（越线余量熄灭边界方差）。"""
import sys, os, json
sys.path.insert(0, "/workspace")
import numpy as np
from g4.g4b_planner import (make_selforg_model, train_selforg,
                            eval_decode_frozen_selforg, TrapTree,
                            encode_state_patterns)

def run(seed, n_l1, nself, win_frac, kwta, alpha, beta, eta_dyn, dyn_decay, steps, floods, freeze_frac):
    rng = np.random.default_rng(seed)
    task = TrapTree(rng=rng)
    model, d_obs = make_selforg_model(seed, n_l1=n_l1, nself=nself,
        win_frac=win_frac, kwta=kwta, alpha=alpha, beta=beta,
        eta_dyn=eta_dyn, dyn_decay=dyn_decay)
    patterns = encode_state_patterns(task, rng, d_obs)
    train_selforg(model, task, patterns, steps, rng, freeze_frac=freeze_frac)
    # 用更多 flood 建参考码降低参考噪声
    import g4.g4b_planner as P
    P.stable_code = _stable_floods(floods)
    return eval_decode_frozen_selforg(model, task, patterns)

def _stable_floods(F):
    def inner(model, pat, state_id):
        model.set_learning(False)
        acc = None
        for _ in range(F):
            model.observe(pat, state_id)
            acc = model.xs if acc is None else acc + model.xs
        model.set_learning(True)
        return acc / F
    return inner

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    out = {}
    for sd in range(6):
        r = run(sd, cfg["n_l1"], cfg["nself"], cfg["win_frac"], cfg.get("kwta",0.33),
                cfg.get("alpha",0.05), cfg.get("beta",0.5), cfg.get("eta_dyn",0.8),
                cfg.get("dyn_decay",1e-5), cfg.get("steps",8000), cfg.get("floods",100),
                cfg.get("freeze_frac",0.6))
        out[sd] = {k: (round(v,4) if isinstance(v,float) else v) for k,v in r.items()}
        print(f"seed{sd}: {json.dumps(out[sd])}", flush=True)
    print("MIN_DEC", round(min(v["decode_acc"] for v in out.values()),4), flush=True)