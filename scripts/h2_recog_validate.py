"""H2 有界验证 v5：显式识别编码器 PC（recog_on）对 free 能力爬升是否有效。

根因（h2_free_sweep + v1-v4 全负）：所有"自由沉降中推类别"机制都受同一结构性
根因拖累——x2 由生成式沉降经 iters=12 从零自举、开环表征判别不足（DPC 最优也只
0.275 vs 孪生 0.42）。孪生(BP) 一次前馈即得判别特征；SR-PC 隐层依赖钳制标签污染态。
文献：识别/生成权重孪生绑定是 tPC-RTRL（Potter&Rhodes'26）、判别式 PC 标准结构——
识别方向 = 生成权重转置，无需沉降。

v5 修复（train_step_recog）：x2 不再沉降自举，而是一次自底向上前馈编码
x1r = relu(W1cT·x0rf)、x2r = relu(W2T·x1r)；读头在 x2r 上 LMS，再用 CE 梯度以
局部 outer 收紧编码器权重 W2/W1c，使生成转置承载判别编码。训练/评估同一路径 → 无
teacher-forcing→free 失配。结构稀疏+列归一+W2 谱截断全保留。评估走 eval_recog。

对照：online 基线（同超参 recog_on=0 → 原 clamp train_step）@90k≈0.23、
@300k≈0.258；孪生@90k≈0.37、@300k≈0.42。若 recog_on>0 在 90k 明显高于基线 0.23
（>=+0.03）即方向有效，再烧全程。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_recog_validate.py 768 90000 0.005
结果攒进 results_e2_gpu_eta/h2_recog_{h}_lr{recog_lr}.json
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

h = int(sys.argv[1]) if len(sys.argv) > 1 else 768
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 90_000
recog_lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.recog_on = True
cfg.recog_lr = recog_lr

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_recog_{h}_lr{recog_lr}"
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
print(f"== {name}: {steps} steps, recog_lr={recog_lr}, "
      f"twin@end={twin_curve or 'NA'} ==", flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_recog(x, y)
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
            json.dump(dict(h=h, steps=steps, recog_lr=recog_lr, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)