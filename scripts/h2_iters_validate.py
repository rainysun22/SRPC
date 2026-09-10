"""H2 有界验证 v4：训练沉降迭代数（settle iters）对 free 能力的影响。

文献依据（决定性）：
  - PCN-TA（arXiv:2510.25993，IROS'25）：PC 在训练中用 50/100 次推理迭代即可收窄
    甚至超越 BP 精度——"PCN-TAs（50 和 100 迭代）收窄与 BP 的差距并最终反超，
    100 迭代最接近 BP"；
  - Whittington & Bogacz / Rosenbaum：PC 权重更新方向对齐 BP 梯度以"充分沉降到不动点"
    为前提。基线只在 iters=12 下更新权重，远离不动点 → 更新方向≈粗糙 BP 近似。
  - 注意：此前"deep inference"实验改的是__评估侧__自由沉降迭代（12→24→48，自由输出
    振荡），并非训练侧__clamp__沉降迭代——参数正交，本次补测训练侧。

唯一变动：train_step_free(free_nudge=0) 退化为 clamp 训练路径（对照已验证惯性；
0.24@30k、0.258@300k），仅增大训练+一致评估的沉降 iters。若 iters>12 在 60k 明显高于
iters=12 对照 0.24（@30k），则方向有效，再烧全程 300k。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_iters_validate.py 1856 60000 50 30000
结果攒进 results_e2_gpu_eta/h2_iters_{h}_it{iters}.json
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
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.eta_inf_scl = 0.5   # 深沉降自由评估的已验证稳定值；训练 clamp 路径不受其影响（scl=1）

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_iters_{h}_it{iters}"
jpath = os.path.join(RES, f"{name}.json")

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=iters, device="cuda")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
curve = []

twin_curve = {}
twin_json = f"results_e2_gpu/twin_{h}.json"
if os.path.exists(twin_json):
    for pt in json.load(open(twin_json))["curve"]:
        twin_curve[int(pt["step"])] = pt["acc"]
print(f"== {name}: {steps} steps, iters={iters}, "
      f"twin@end={sorted(twin_curve.values())[-1] if twin_curve else 'NA'} ==",
      flush=True)

t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_free(x, y)   # free_nudge=0 -> 原 clamp train_step
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
            json.dump(dict(h=h, steps=steps, iters=iters, curve=curve),
                      f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t+1}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
print(f"== {name} done -> {RES}/{name}.json", flush=True)