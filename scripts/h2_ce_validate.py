"""H2 有界验证：CE 判别耦合进编码器（ce_amp）对 free 能力爬升方向是否有效。

对照孪生同 step：twin_1856 曲线 @300k≈0.42，@104万≈0.49。若 ce_amp>0 把 SR-PC
free-eval acc 从 ~0.28 拉向孪生曲线（同 step 越接近 >=+0.05 即爬升方向有效），
再烧全程；否则如实记录 FAIL 换方向。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_ce_validate.py 1856 300000 0.2
    # 参数：h steps ce_amp；默认 ce_amp=0 为基线对照
结果攒进 results_e2_gpu_eta/h2_ce_{h}_amp{ce_amp}.json
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

RES = "results_e2_gpu_eta"
os.makedirs(RES, exist_ok=True)

h = int(sys.argv[1]) if len(sys.argv) > 1 else 1856
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
ce_amp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.ce_amp = ce_amp

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_ce_{h}_amp{ce_amp}"
jpath = os.path.join(RES, f"{name}.json")

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
curve = []

# 孪生同档曲线（供同 step 对照）
twin_curve = {}
twin_json = f"results_e2_gpu/twin_{h}.json"
if os.path.exists(twin_json):
    tc = json.load(open(twin_json))["curve"]
    for pt in tc:
        twin_curve[int(pt["step"])] = pt["acc"]
print(f"== {name}: {steps} steps, ce_amp={ce_amp}, twin@end={twin_curve or 'NA'} ==",
      flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step(x, y)
    if (t + 1) % cfg.eval_every == 0 or t == steps - 1:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_batch(rep_x, rep_y, tau)
        # 孪生同 step acc（线性内插邻近点）
        ref = None
        ks = sorted(twin_curve.keys())
        for k in ks:
            if k >= t + 1:
                ref = twin_curve[k]; break
        pt = dict(step=t + 1, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  twin_acc=ref, gap=(float(acc) - ref) if ref is not None else None,
                  event_rate=m.event_rate(), wall=round(time.time() - t0, 1))
        curve.append(pt)
        with open(jpath, "w") as f:
            json.dump(dict(h=h, steps=steps, ce_amp=ce_amp, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)