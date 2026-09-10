"""H2 有界验证 v3：自由沉降输出误差 PC（free_nudge）对 free 能力爬升是否有效。

根因（h2_layerprobe + h2_freebeta，均决定性）：硬 clamp 标签污染状态——训练时
x3=yoh 被钳死沉降，W1/W2 只学会解读"标签污染态"，自由沉降从 x1 起即塌陷
（clamp 探针 1.0 vs free x1 0.32 / free x2 0.24）；放大自底向上 β 探针单调降
（0.242→0.117）证伪"传导力度不足"。CE耦合/batch累积/scheduled sampling 全失败，
唯一没试的对因机制 = 不再硬钳制，让类别信号只在自由沉降态上经读头软误差局部进入
（train_step_free）。free_nudge 即软误差 nudge 强度。

对照：online 基线（同超参 free_nudge=0 → 走原 clamp train_step）@90k≈0.23、
@300k≈0.258；孪生@90k≈0.37、@300k≈0.42。若 free_nudge>0 在 90k 明显高于基线 0.23
（>=+0.03）即方向有效，再烧全程。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_free_validate.py 1856 90000 0.3
结果攒进 results_e2_gpu_eta/h2_free_1856_nu{free_nudge}.json
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
free_nudge = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.free_nudge = free_nudge

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_free_{h}_nu{free_nudge}"
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
print(f"== {name}: {steps} steps, free_nudge={free_nudge}, "
      f"twin@end={twin_curve or 'NA'} ==", flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_free(x, y)
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
            json.dump(dict(h=h, steps=steps, free_nudge=free_nudge, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)