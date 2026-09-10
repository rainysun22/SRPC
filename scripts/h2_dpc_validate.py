"""H2 有界验证 v4：判别式 PC 能量训练（dpc_amp）对 free 能力爬升是否有效。

根因（h2_free_sweep，决定性）：自由沉降训练(free_nudge)三档逐字节同值 0.133 <<
clamp 基线 0.23——标签只在沉降【外】事后 nudge，沉降动力学全程无类别误差，W
学不到开环判别，自由态坍缩到固定点。CE耦合/batch累积/scheduled sampling/free_nudge
全失败。唯一文献支持且未试的对因机制 = 判别式 PC（Whittington&Bogacz / Salvatori
监督 PC）：把 CE 标签回归误差作为能量一项，在自由沉降【内部】每步软性推 x2 靠向
正确类（-dL_CE/dx2），让 W1/W2 在开环沉降路径下也学到判别编码（train_step_dpc）。

对照：online 基线（同超参 dpc_amp=0 → 走原 clamp train_step）@90k≈0.23、
@300k≈0.258；孪生@90k≈0.37、@300k≈0.42。若 dpc_amp>0 在 90k 明显高于基线 0.23
（>=+0.03）即方向有效，再烧全程。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_dpc_validate.py 1856 90000 1.0
结果攒进 results_e2_gpu_eta/h2_dpc_1856_amp{dpc_amp}.json
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
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 90_000
dpc_amp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
free_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 12
dpc_deep = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.dpc_amp = dpc_amp
cfg.free_iters = free_iters
cfg.dpc_deep = dpc_deep

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_dpc_{h}_amp{dpc_amp}_fi{free_iters}_df{dpc_deep}"
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
print(f"== {name}: {steps} steps, dpc_amp={dpc_amp}, "
      f"twin@end={twin_curve or 'NA'} ==", flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_dpc(x, y)
    if (t + 1) % cfg.eval_every == 0 or t == steps - 1:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_batch(rep_x, rep_y, tau)
        ref = None
        for k in sorted(twin_curve.keys()):
            if k >= t + 1:
                ref = twin_curve[k]; break
        pt = dict(step=t + 1, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  twin_acc=ref, gap=(float(acc) - ref) if ref is not None else None,
                  event_rate=m.event_rate(), w2smax=m.last_w2_smax,
                  wall=round(time.time() - t0, 1))
        curve.append(pt)
        with open(jpath, "w") as f:
            json.dump(dict(h=h, steps=steps, dpc_amp=dpc_amp, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)