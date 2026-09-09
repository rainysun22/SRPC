"""2832/4032 fix 全预算续跑：从 results_e2_gpu_fix/pcn_{h}_fix.pt 断点续跑到全 epoch。

与 fullb_fix.py 同口径（cfg.w2_cap=True, every=1, smax_cap=5.0, pow_iters=8，
seed=0，同 corpus/val），仅增加 --resume 从已有 fix checkpoint 续跑，保存目标路径一致，
使 2832/4032 补齐到 104 万步、消除 H1 大档预算混淆。

用例（4090）：
    python -B scripts/resume_fullb_fix.py 2832 1039838   # 续跑到 1039838
    python -B scripts/resume_fullb_fix.py 4032 1039838
"""
from __future__ import annotations
import json, os, sys, time
ROOT = "/root/srpc_e2/srpc_src"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu_graph import LMPCNgG

RES = "results_e2_gpu_fix"
os.makedirs(RES, exist_ok=True)

h = int(sys.argv[1]) if len(sys.argv) > 1 else 2832
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_039_838
eval_every = int(sys.argv[3]) if len(sys.argv) > 3 else 30_000
assert h in (768, 1200, 1856, 2832, 4032) and h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"pcn_{h}"
ckpt_p = os.path.join(RES, f"{name}_fix.pt")
jpath = os.path.join(RES, f"{name}_fix.json")
meta = {"h": h, "kind": "pcn+fix", "steps": full,
        "w2_cap": cfg.w2_cap, "w2_smax_cap": cfg.w2_smax_cap,
        "w2_cap_every": cfg.w2_cap_every}
if os.path.exists(jpath):
    meta = json.load(open(jpath))

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
curve = []
start_step = 0
if os.path.exists(ckpt_p):
    sd = torch.load(ckpt_p, map_location="cuda", weights_only=False)
    m.load_state(sd["model"])
    start_step = int(sd["step"])
    curve = list(sd.get("curve", []))
    print(f"[resume] {name} from step {start_step} of {full}")

t0 = time.time()
print(f"== {name} resume->{full} fix(cap={cfg.w2_smax_cap}/every={cfg.w2_cap_every}) "
      f"from {start_step} ==", flush=True)
for t in range(start_step, steps):
    x, y = corpus.train_window(t)
    m.train_step(x, y)
    if (t + 1) % cfg.eval_every == 0 or t == steps - 1:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_batch(rep_x, rep_y, tau)
        pt = dict(step=t + 1, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  event_rate=m.event_rate(), w2_smax=m.last_w2_smax,
                  wall=round(time.time() - t0, 1))
        curve.append(pt)
        meta.update(bpc=pt["bpc"], acc=pt["acc"], tau=pt["tau"],
                    event_rate=pt["event_rate"], wall=pt["wall"], curve=curve)
        with open(jpath, "w") as f:
            json.dump(meta, f, indent=1, default=float)
        torch.save({"step": t + 1, "curve": curve, "model": m.state_dict()}, ckpt_p)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.3f} "
              f"tau={tau:.2f} w2_smax={m.last_w2_smax:.3f} "
              f"wall={round(time.time()-t0,1)}s", flush=True)
print(f"== {name} fix resume done: bpc={meta.get('bpc')} step={start_step}->{steps} "
      f"-> {RES}", flush=True)