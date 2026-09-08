"""E2 GPU 移植验证：numpy(CPU) vs torch(CPU) parity + torch(GPU) 冒烟基准。

parity：同 seed、同语料窗口、同超参下，torch 移植与 numpy 版逐 500 步
快照比对权重最大差（累计浮点漂移），并在 3k 步处用同一验证段窗口比 BPC。
GPU 冒烟：h=4032 测每步耗时，估算全 epoch（≈1.0M 步）GPU 预算。

用法：
    python scripts/e2_gpu_parity.py --parity        # torch CPU vs numpy
    python scripts/e2_gpu_parity.py --smoke         # GPU 冒烟基准（h=4032）
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import LMPCN, ByteCorpus
from srpc.lmgpu import LMPCNg

SEED = 0


def _max_wdiff(m_np: LMPCN, m_g: LMPCNg) -> float:
    d = 0.0
    for a, b in (("W1c", m_g.W1c), ("W2", m_g.W2), ("W3", m_g.W3),
                 ("W_out", m_g.W_out)):
        d = max(d, float(np.abs(getattr(m_np, a) - b.cpu().numpy()).max()))
    return d


def parity(cfg: E2Config, corpus: ByteCorpus, steps: int = 3000) -> None:
    print(f"== parity numpy vs torch(cpu), h=768, steps={steps} ==")
    m_np = LMPCN(cfg, 768, np.random.default_rng(SEED * 977 + 5),
                 eta_w=0.01, iters=12)
    m_g = LMPCNg(cfg, 768, np.random.default_rng(SEED * 977 + 5),
                 eta_w=0.01, iters=12, device="cpu")
    t0 = time.time()
    snaps = []
    for t in range(steps):
        x, y = corpus.train_window(t)
        m_np.train_step(x, y)
        m_g.train_step(x, y)
        if (t + 1) % 500 == 0:
            snaps.append((t + 1, _max_wdiff(m_np, m_g)))
            print(f"  step {t+1}: max|ΔW|={snaps[-1][1]:.3e}", flush=True)
    # 验证段同口径 BPC（τ 校准切片 + 剩余窗口）
    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = (vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:])
    best_np = best_g = (np.inf, None)
    for tau in cfg.tau_grid:
        bpc_np, _ = m_np.eval_batch(rep_x, rep_y, tau)
        bpc_g, _ = m_g.eval_batch(rep_x, rep_y, tau)
        if bpc_np < best_np[0]:
            best_np = (bpc_np, tau)
        if bpc_g < best_g[0]:
            best_g = (bpc_g, tau)
    print(f"  numpy bpc={best_np[0]:.4f} (tau={best_np[1]})")
    print(f"  torch bpc={best_g[0]:.4f} (tau={best_g[1]})")
    print(f"  |Δbpc|={abs(best_np[0]-best_g[0]):.4f}  wall={time.time()-t0:.0f}s")
    return abs(best_np[0] - best_g[0])


def smoke(cfg: E2Config, corpus: ByteCorpus, h: int = 4032,
          steps: int = 300) -> None:
    if not torch.cuda.is_available():
        print("!! no cuda"); return
    dev = "cuda"
    m = LMPCNg(cfg, h, np.random.default_rng(SEED * 977 + 5),
               eta_w=0.01, iters=12, device=dev)
    # 预热
    for t in range(50):
        x, y = corpus.train_window(t)
        m.train_step(x, y)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(50, 50 + steps):
        x, y = corpus.train_window(t)
        m.train_step(x, y)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    n = len(corpus.train) - cfg.context - 1
    print(f"== gpu smoke h={h} device={dev} ==")
    print(f"  {dt*1e3:.2f} ms/step -> full epoch ({n} 步) 约 "
          f"{dt*n/3600:.2f} h")
    # GPU 一次短 eval 冒烟
    vx, vy = corpus.val_x, corpus.val_y
    bpc, acc = m.eval_batch(vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows],
                            0.2)
    print(f"  smoke eval bpc={bpc:.3f} acc={acc:.3f} (GPU ok)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    if args.parity:
        parity(cfg, corpus)
    if args.smoke:
        smoke(cfg, corpus)
    if not args.parity and not args.smoke:
        print("usage: --parity | --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
