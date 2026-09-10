"""H2 有界验证 v11：局部 Adam + 软化正则（recog_adam，relu' 门控 credit）。

决定论据（h2_oraclebp_1856）：识别编码器【同结构】真 BP CE + Adam → 90k acc0.367
/180k 0.373 追平孪生；局部机制卡 0.33 是优化动力学而非架构/credit 形式。本脚本
保留 v10 的 relu' 门控两层 credit，权重更新换逐突触局部 Adam，软化破坏性正则。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_adam_validate.py 1856 300000 0.0003 1 2.5 30000
      argv: h steps lr rounds smax_cap eval_every
结果 results_e2_gpu_eta/h2_adam_{h}_lr{lr}_r{r}_c{smax}.json
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
from srpc.lmgpu import LMPCNg

RES = "results_e2_gpu_eta"
os.makedirs(RES, exist_ok=True)

h = int(sys.argv[1]) if len(sys.argv) > 1 else 1856
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
smax_cap = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5
eval_every = int(sys.argv[6]) if len(sys.argv) > 6 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = smax_cap
cfg.w2_pow_iters = 8
cfg.recog_on = True
cfg.recog_lr = 0.0005
cfg.recog_adam = True
cfg.recog_adam_lr = lr
cfg.recog_rounds = rounds
tag = f"lr{lr}_r{rounds}_c{smax_cap}"

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_adam_{h}_{tag}"
jpath = os.path.join(RES, f"{name}.json")

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
curve = []

twin_curve = {}
twin_json = f"results_e2_gpu/twin_{h}.json"
if os.path.exists(twin_json):
    for pt in json.load(open(twin_json))["curve"]:
        twin_curve[int(pt["step"])] = pt["acc"]
print(f"== {name}: {steps} steps, lr={lr} rounds={rounds} cap={smax_cap} ==", flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_recog_adam(x, y)
    if (t + 1) % cfg.eval_every == 0 or t == steps - 1:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_recog(rep_x, rep_y, tau)
        ref = None
        for k in sorted(twin_curve.keys()):
            if k >= t + 1:
                ref = twin_curve[k]; break
        pt = dict(step=t + 1, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  twin_acc=ref, gap=(float(acc) - ref) if ref is not None else None,
                  event_rate=m.recog_event_rate(), w2smax=m.last_w2_smax,
                  wall=round(time.time() - t0, 1))
        curve.append(pt)
        with open(jpath, "w") as f:
            json.dump(dict(h=h, steps=steps, lr=lr, rounds=rounds,
                           smax_cap=smax_cap, curve=curve), f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} twin@{t+1}={ref} "
              f"gap={pt['gap']} ev={pt['event_rate']:.3f} "
              f"w2smax={pt['w2smax']:.3f} wall={pt['wall']}s", flush=True)
print(f"== {name} done -> {jpath} ==", flush=True)