"""H2 对因验证：SR-PC 批量累积局部更新（batch=32，同孪生学力）能力爬升方向。

根因（h2_conv）：SR-PC 在线单样本 Hebbian vs 孪生 batch=32+Adam，同样本数下
相差甚大（1856@300k：SR-PC acc 0.258/bpc 4.1，孪生 acc 0.42）。这是收敛效率
代差，非数据量差、非读出判别不足（CE 判别耦合 300k 验证无效已弃）。
修复：LMPCNg.train_step_batch 把 B 个样本的 PC 误差外积（∂F/∂W=e_l⊗pre_l）
批量累积取平均后应用一次——仍局部/免反传/结构稀疏+列归一+W2 谱截断全保留，
B=32 与孪生同预算、同样本流。
对照：twin_1856 曲线 @300k≈0.42、@104万≈0.49。若批量累积把 SR-PC free-eval acc
从 ~0.26 明显拉向/超过孪生曲线（同 step 变近，或 BPC 显著下降），方向即有效，
再烧全程；否则如实记录 FAIL 换方向。

用法（4090，/root/srpc_e2/srpc_src）：
    python -B scripts/h2_batch_validate.py 1856 300000
    # 参数：h steps；batch 固定 = twin_batch=32
结果攒进 results_e2_gpu_eta/h2_batch_{h}_{steps}.json
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
eval_every = int(sys.argv[3]) if len(sys.argv) > 3 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
B = cfg.twin_batch

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
name = f"h2_batch_{h}"
jpath = os.path.join(RES, f"{name}_{steps}.json")

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
print(f"== {name}: {steps} samples, B={B}, twin@end={twin_curve or 'NA'} ==",
      flush=True)

t0 = time.time()
t = 0
while t < steps:
    n = min(B, steps - t)
    X = np.zeros((n, cfg.context, 256), np.float32)
    ys = np.zeros(n, np.int64)
    for i in range(n):
        x, y = corpus.train_window(t + i)
        X[i] = x
        ys[i] = y
    m.train_step_batch(X, ys)
    t += n
    if t % cfg.eval_every == 0 or t >= steps:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_batch(rep_x, rep_y, tau)
        ref = None
        for k in sorted(twin_curve.keys()):
            if k >= t:
                ref = twin_curve[k]; break
        pt = dict(step=t, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  twin_acc=ref, gap=(float(acc) - ref) if ref is not None else None,
                  event_rate=m.event_rate(), w2smax=m.last_w2_smax,
                  wall=round(time.time() - t0, 1))
        curve.append(pt)
        with open(jpath, "w") as f:
            json.dump(dict(h=h, steps=t, batch=B, curve=curve),
                      f, indent=1, default=float)
        print(f"  sample {t}/{steps}: bpc={bpc:.3f} acc={acc:.4f} "
              f"twin@{t}={ref} gap={(pt['gap'] if ref is not None else None)} "
              f"w2smax={m.last_w2_smax:.3f} wall={round(time.time()-t0,1)}s",
              flush=True)
        if t >= steps:
            break
print(f"== {name} done -> {RES}/{name}_{steps}.json", flush=True)