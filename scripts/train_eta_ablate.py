"""W2 谱发散对因消融：µPC 宽度缩放指数 mu_pc_exp（1/√h → 1/h 档）。

背景：H1 cap 消融（cap=8.0）在 step≈450k 谱发散复现，否决"cap 卡容量"。
机制推理：dW2 = outer(e1, x2·g2)，|e1|,|x2| ∝ √h（k-WTA 稀疏激活）→ 谱增量 ∝ h。
当前 µPC 用 eta_w2 = ew·(h_ref/h)^0.5 只抵消 √h，剩余谱增量 ∝ √h（随宽度加速）。
本脚本以更强缩放指数 mu_pc_exp=1.0（eta_w2 ∝ 1/h）重训，验证：
  - σmax(W2) 是否自然稳定 <5（cap 不触发）
  - 端到端 BPC 是否随宽度恢复单调（对比 cap=5 基线 2832=3.993 / 4032=4.113）

用法：env PYTHONPATH=. python scripts/train_eta_ablate.py 4032 --exp 1.0
        [--steps 0(=全 epoch) --eval-every 30000 --out results_e2_gpu_eta]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lmgpu import LMPCNg
try:
    from srpc.lmgpu_graph import LMPCNgG
except Exception:
    LMPCNgG = None  # 旧版远端无图版时回退
from srpc.lm import ByteCorpus


def _full_epoch_steps(cfg: E2Config, corpus: ByteCorpus) -> int:
    return len(corpus.train) - cfg.context - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h", type=int)
    ap.add_argument("--exp", type=float, default=1.0, help="µPC 缩放指数（0.5=当前 1/√h；1.0=1/h）")
    ap.add_argument("--steps", type=int, default=0, help="0=全 epoch")
    ap.add_argument("--eval-every", type=int, default=30000)
    ap.add_argument("--out", default="results_e2_gpu_eta")
    ap.add_argument("--graph", type=int, default=1, help="0=LMPCNg / 1=LMPCNgG(CUDA Graph)")
    args = ap.parse_args()
    h = args.h
    assert h in (768, 1200, 1856, 2832, 4032) and h % 16 == 0

    cfg = E2Config()
    cfg.eval_every = args.eval_every
    cfg.w2_cap = True
    cfg.w2_cap_every = 1
    cfg.w2_smax_cap = 5.0      # 安全网（预期 exp=1.0 不触发）
    cfg.w2_pow_iters = 8
    # 语料路径解析：4090 沙箱固定位于 /root/srpc_e2/srpc_src/data（不依赖 cwd）；
    # 本地开发则用相对 data/
    for p in ["/root/srpc_e2/srpc_src/data/tinyshakespeare.txt"]:
        if os.path.exists(p):
            cfg.corpus_path = p
            break

    RES = args.out
    os.makedirs(RES, exist_ok=True)
    corpus = ByteCorpus(cfg)
    full = _full_epoch_steps(cfg, corpus)
    steps = full if args.steps <= 0 else args.steps
    name = f"pcn_{h}"
    ckpt_p = os.path.join(RES, f"{name}_exp{args.exp}.pt")

    seed = 0
    rng = np.random.default_rng(seed * 977 + 5)
    if args.graph and LMPCNgG is not None:
        m = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12, device="cuda",
                    mu_pc_exp=args.exp)
    else:
        m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda",
                   mu_pc_exp=args.exp)
    print(f"[init] h={h} exp={args.exp} eta_w1={m.eta_w1:.5f} "
          f"eta_w2={m.eta_w2:.5f} eta_w3={m.eta_w3:.5f}", flush=True)

    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    meta = {"h": h, "kind": "pcn", "mu_pc_exp": args.exp, "steps": steps,
            "w2_smax_cap": 5.0}
    curve = []
    t0 = time.time()
    print(f"== {name} exp={args.exp} full-epoch steps={steps} (fresh) ==", flush=True)
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
            with open(os.path.join(RES, f"{name}_exp{args.exp}.json"), "w") as f:
                json.dump(meta, f, indent=1, default=float)
            torch.save({"step": t + 1, "curve": curve, "model": m.state_dict()},
                       ckpt_p)
    print(f"== done h={h} exp={args.exp} final bpc={meta.get('bpc')} "
          f"w2_smax={m.last_w2_smax} wall={time.time()-t0:.0f}s ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
