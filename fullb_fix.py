"""h 档全预算重训（判据1重判），开启每步 W2 谱截断修复。

与 scripts/run_e2_summit.py run_pcn 同口径（同 seed、同语料、同 eval_batch/fit_tau），
仅：
  - cfg.w2_cap=True, w2_cap_every=1, w2_smax_cap=5.0（每步幂迭代截断，对因修复）
  - 保存到 RES="results_e2_gpu_fix"（不动原始崩溃跑 pcn_<h>.*）
用例：python -B fullb_fix.py <h> [max_steps] [eval_every]
  h ∈ {1856,2832,4032}；max_steps 缺省 = 全 epoch（len-ctx-1）；eval_every 缺省 30000。
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

h = 1856
if len(sys.argv) > 1:
    h = int(sys.argv[1])
assert h % 16 == 0 and h in (768, 1200, 1856, 2832, 4032), f"bad h={h}"

cfg = E2Config()
cfg.eval_every = int(sys.argv[3]) if len(sys.argv) > 3 else 30_000
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = int(sys.argv[2]) if len(sys.argv) > 2 else full
name = f"pcn_{h}"
ckpt_p = os.path.join(RES, f"{name}_fix.pt")
meta = {"h": h, "kind": "pcn+fix", "steps": steps,
        "w2_cap": cfg.w2_cap, "w2_smax_cap": cfg.w2_smax_cap,
        "w2_cap_every": cfg.w2_cap_every}

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
curve = []
t0 = time.time()
print(f"== {name} full-epoch steps={steps} fix(cap={cfg.w2_smax_cap}/every={cfg.w2_cap_every}) ==",
      flush=True)
for t in range(steps):
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
        with open(os.path.join(RES, f"{name}_fix.json"), "w") as f:
            json.dump(meta, f, indent=1, default=float)
        torch.save({"step": t + 1, "curve": curve, "model": m.state_dict()}, ckpt_p)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.3f} "
              f"tau={tau:.2f} w2_smax={m.last_w2_smax:.3f} "
              f"wall={round(time.time()-t0,1)}s", flush=True)
print(f"== {name} fix done: bpc={meta.get('bpc')} "
      f"wall={round(time.time()-t0,1)}s -> {RES}", flush=True)