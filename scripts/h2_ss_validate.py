"""H2 有界验证 v2：scheduled sampling（ss_eps）对 free 能力爬升方向是否有效。

根因（h2_mech）：clamp_x2 探针 1.0 vs free_x2 探针 0.28 —— teacher-forcing→free
泛化塌陷。CE 判别耦合与 batch 累积均失败（详见 h2_ce_* / h2_batch_*）。scheduled
sampling（Bengio 2015 / DAgger / vid-PC）用 {真值, 模型开环预测} 的凸混合做钳制
目标，使 x2 被拉向开环可达的判别态。
对照：online 基线（同超参 ss_eps=0）@90k≈0.23、@300k≈0.258；孪生@90k≈0.37、
@300k≈0.42。若 ss_eps>0 在 90k 明显高于基线 0.23（>=+0.03）即方向有效，再烧全程。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_ss_validate.py 1856 90000 0.3
结果攒进 results_e2_gpu_eta/h2_ss_1856_ep{k}.json
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
ss_eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.ss_eps = ss_eps

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_ss_{h}_ep{ss_eps}"
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
print(f"== {name}: {steps} steps, ss_eps={ss_eps}, twin@end={twin_curve or 'NA'} ==",
      flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_ss(x, y)
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
            json.dump(dict(h=h, steps=steps, ss_eps=ss_eps, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)