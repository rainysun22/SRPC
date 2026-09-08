"""阶段 E2 GPU 登顶跑（GPU_TASKS T1/T2 落地，用户已拍板口径）：
    tinyshakespeare 全 epoch（n ≈ len(train)-W-1 步）× 梯子 {768,1200,1856,2832,4032}
    = {1M, 2M, 4M, 8M, 15M}（结构参数 ~0.98/1.92/3.89/8.01/15.03M）
    + 同规模孪生（numpy TwinMLP CPU，与 pilot 同一实现口径）。

    PCN = srpc.lmgpu.LMPCNg（PyTorch GPU，局部规则/免反传/结构稀疏三初衷不变）；
    孪生 = srpc.lm.TwinMLP（标准反传参照，契约第 2 条，CPU 即可）。
    超参 = 锚点迁移值（iters=12, eta_w=0.01, 读出 τ 校准），iPC 关——与 pilot
    梯子同协议，仅预算从 150k 步（≈15% epoch）放大到全 epoch。

用法（每 rung 一个进程，可并行；可断点续跑）：
    python scripts/run_e2_summit.py --kind pcn  --h 1856 --device cuda [--resume]
    python scripts/run_e2_summit.py --kind twin --h 1856

判据裁决见 report 阶段（scripts/build_e2_summit_report.py 或 runner --report）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus, train_twin
from srpc.lmgpu import LMPCNg
from srpc.lmgpu_graph import LMPCNgG

RES = "results_e2_gpu"
os.makedirs(RES, exist_ok=True)

# 登顶梯子：h -> (参数量标签, 理论结构参数)。与 E2Config.ladder_widths 同源推导：
# n_struct = h·W·r·256 块紧凑 W1 全结构  + k·h (W2 掩码) + k·256 (W3 掩码)，
# k = round(fan_in_frac·h)。h 需 16 整除。
LADDER = {
    768: 983040,     #  0.98M
    1200: 1924800,   #  1.92M
    1856: 3890176,   #  3.89M
    2832: 8008896,   #  8.01M
    4032: 15031296,  # 15.03M
}


def _struct_params(cfg: E2Config, h: int) -> int:
    k = int(round(cfg.fan_in_frac * h))
    return h * (cfg.context * cfg.rf_blocks * 256) + k * h + k * 256


def _full_epoch_steps(cfg: E2Config, corpus: ByteCorpus) -> int:
    return len(corpus.train) - cfg.context - 1


def _save_json(name: str, data: dict) -> None:
    p = os.path.join(RES, f"{name}.json")
    with open(p, "w") as f:
        json.dump(data, f, indent=1, default=float)
    print(f"[json] {p}")


# ----------------------------------------------------------------------
# PCN 全 epoch 跑（可续跑）
# ----------------------------------------------------------------------
def run_pcn(cfg: E2Config, corpus: ByteCorpus, h: int, device: str,
            resume: bool, steps_override: int | None = None,
            eval_every: int | None = None, seed: int = 0) -> None:
    name = f"pcn_{h}"
    ckpt_p = os.path.join(RES, f"{name}.pt")
    steps = (steps_override if steps_override
             else _full_epoch_steps(cfg, corpus))
    if eval_every:
        cfg.eval_every = eval_every
    meta = {"h": h, "kind": "pcn", "steps": steps,
            "n_params_struct": LADDER[h]}
    curve: list[dict] = []
    start_step = 0
    rng = np.random.default_rng(seed * 977 + 5)
    if device.startswith("cuda"):
        m: LMPCNg = LMPCNgG(cfg, h, rng, eta_w=0.01, iters=12,
                            device=device)
    else:
        m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device=device)

    if resume and os.path.exists(ckpt_p):
        sd = torch.load(ckpt_p, map_location=device, weights_only=False)
        m.load_state(sd["model"])
        start_step = int(sd["step"])
        curve = list(sd.get("curve", []))
        print(f"[resume] {name} from step {start_step} (of {steps})")

    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = (vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:])
    t0 = time.time()
    print(f"== {name} ({h}) full epoch steps={steps} device={device} ==", flush=True)
    for t in range(start_step, steps):
        x, y = corpus.train_window(t)
        m.train_step(x, y)
        if (t + 1) % cfg.eval_every == 0 or t == steps - 1:
            tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
            bpc, acc = m.eval_batch(rep_x, rep_y, tau)
            pt = dict(step=t + 1, bpc=bpc, acc=acc, tau=float(tau),
                      event_rate=m.event_rate(),
                      wall=round(time.time() - t0, 1))
            curve.append(pt)
            meta.update(bpc=bpc, acc=acc, tau=float(tau),
                        event_rate=pt["event_rate"],
                        wall=round(time.time() - t0, 1),
                        curve=curve)
            _save_json(name, meta)
            torch.save({"step": t + 1, "curve": curve, "model": m.state_dict()},
                       ckpt_p)
            print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.3f} "
                  f"tau={tau:.2f} wall={round(time.time()-t0,1)}s", flush=True)
    print(f"== {name} done: bpc={meta.get('bpc')} wall={round(time.time()-t0,1)}s")


# ----------------------------------------------------------------------
# 孪生全 epoch 跑（numpy CPU，与 pilot 同实现；可选 torch）
# ----------------------------------------------------------------------
def run_twin(cfg: E2Config, corpus: ByteCorpus, h: int, seed: int = 0) -> None:
    name = f"twin_{h}"
    steps = _full_epoch_steps(cfg, corpus)
    target = LADDER[h]
    r = train_twin(cfg, target, steps, corpus, seed=seed)
    r.update(h=h, kind="twin", steps=steps, n_params=target,
             target_params=target)
    print(f"== {name} done: bpc={r['bpc']:.3f} acc={r['acc']:.3f} "
          f"n_params={r['n_params']/1e6:.2f}M wall={r['wall']}s")
    _save_json(name, r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("pcn", "twin"), required=True)
    ap.add_argument("--h", type=int, required=True, choices=sorted(LADDER))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖步数（默认全 epoch ≈ 1003838）")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="评估间隔（默认沿用 cfg.eval_every=15000）")
    args = ap.parse_args()

    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    if args.kind == "pcn":
        run_pcn(cfg, corpus, args.h, args.device, args.resume,
                args.steps, args.eval_every, args.seed)
    else:
        run_twin(cfg, corpus, args.h, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
