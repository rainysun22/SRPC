"""H1 §5.2a：W2-cap 消融 —— cap=5.0 是否反向卡死大档容量。

从零全预算重训（seed=0、同语料/val，仅 cfg.w2_smax_cap 不同），保存到 RES 分离目录，
只动谱截断阈值这一方案级口径，不动 SR-PC 原理。

用例（4090，srpc_src）：
    python -B scripts/train_cap_ablate.py 4032 --cap 8.0 --out results_e2_gpu_cap8
结果：fin BPC/acc + 逐 eval 曲线；后续用 run_h1_probe.py 的探针协议读 x2 线性可读性。
"""
from __future__ import annotations
import argparse, json, os, sys, time
ROOT = "/root/srpc_e2/srpc_src"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu_graph import LMPCNgG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h", type=int)
    ap.add_argument("--cap", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=0, help="0=全 epoch")
    ap.add_argument("--eval-every", type=int, default=30000)
    ap.add_argument("--out", default="results_e2_gpu_cap8")
    args = ap.parse_args()
    h = args.h
    assert h in (768, 1200, 1856, 2832, 4032) and h % 16 == 0

    cfg = E2Config()
    cfg.eval_every = args.eval_every
    cfg.w2_cap = True
    cfg.w2_cap_every = 1
    cfg.w2_smax_cap = args.cap
    cfg.w2_pow_iters = 8

    RES = args.out
    os.makedirs(RES, exist_ok=True)
    corpus = ByteCorpus(cfg)
    full = len(corpus.train) - cfg.context - 1
    steps = full if args.steps <= 0 else args.steps
    name = f"pcn_{h}"
    ckpt_p = os.path.join(RES, f"{name}_cap{args.cap}.pt")

    seed = 0
    rng = np.random.default_rng(seed * 977 + 5)
    m = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    meta = {"h": h, "kind": "pcn", "cap": args.cap, "steps": steps,
            "w2_smax_cap": args.cap}
    curve = []
    t0 = time.time()
    print(f"== {name} cap={args.cap} full-epoch steps={steps} (fresh) ==", flush=True)
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
            with open(os.path.join(RES, f"{name}_cap{args.cap}.json"), "w") as f:
                json.dump(meta, f, indent=1, default=float)
            torch.save({"step": t + 1, "curve": curve, "model": m.state_dict()},
                       ckpt_p)
            print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.3f} "
                  f"tau={tau:.2f} w2_smax={m.last_w2_smax:.3f} "
                  f"wall={round(time.time()-t0,1)}s", flush=True)
    print(f"== {name} cap={args.cap} done: bpc={meta.get('bpc')} acc={meta.get('acc')} "
          f"-> {RES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())