"""阶段 H1 承重墙：自-读出头（模型固有读出头）容量随宽度单调判定 —— 去解离化诊断。

承接 H1b（report_h1.md §2c/§3）：外置三口径读出口（线性 / 稀疏槽位 / MLP）均接不住
宽度红利 → "BPC/探针解离"，规模红利似以"读出口不可接触"方式承载。本脚本换一个最干净的
读出口口径：**模型自身训练时共同优化的读出头 W_out/b_out**。它的容量随宽度自动放大
（W_out ∈ R^{h×256}，参数量随 h 线性增长），且与表征在端到端目标下联合优化——若
**自有读出头 acc 随宽度严格单调**，而外置重训探针（同一 x2 上）不单调，则"解离"实为
**外置读出口（重训/容量/类不均衡）的伪影**，规模红利以可读形式存在 → H1 去解离化、
以"自-读出-宽度单调"口径修正通过；若自有头也不单调 → 坐实"读出不可接触"为原理层面。

协议（与 H1/H1b 同口径，保证可比）：
  - 任务：tinyshakespeare next-byte；取收敛态 x2 隐藏表征（freeze 权重自由推断）。
  - 权重口径（统一"最优健康档"）：768/1200=pcn_{h}.pt（E2 登顶健康档）；
    1856=pcn_1856_fix.pt（E2 fix，σmax≈5）；2832/4032=pcn_{h}_exp1.0.pt（eta1，端到端
    已方案级修复、σmax 平稳 ≤5、特性 acc 严格单调）。可 --h-ckpt 覆盖。
  - 读出口：直接用模型固有 m.W_out @ x2 + m.b_out → argmax acc / softmax BPC。
  - 输出：每档 selfhead_acc / selfhead_bpc / 与外置线性探针 acc（h1_probe.json 同段）。
  - 判定：selfhead_acc 随(宽度)严格单调 → 去解离化通过；不单调 → 坐实读出不可接触。

用法（4090 工程目录，PYTHONPATH=.）：
  env PYTHONPATH=/root/srpc_e2 python scripts/run_h1_selfhead.py \
      --include 768,1200,1856,2832,4032
  输出：results_e2_gpu_eta/h1_selfhead.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)

VALID_HS = [768, 1200, 1856, 2832, 4032]


def ckpt_path(h: int) -> str:
    """统一最优健康档口径。大档(2832/4032)用 eta1 健康权重，1856 用 fix，小档用登顶档。"""
    R = "/root/srpc_e2"
    if h >= 2832:
        return f"{R}/results_e2_gpu_eta/pcn_{h}_exp1.0.pt"
    if h >= 1856:
        return f"{R}/srpc_src/results_e2_gpu_fix/pcn_{h}_fix.pt"
    return f"{R}/srpc_src/results_e2_gpu/pcn_{h}.pt"


def load_model(cfg: E2Config, h: int, ckpt: str) -> tuple[LMPCNg, dict, float]:
    rng = np.random.default_rng(0 * 977 + 5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    smax = float(torch.linalg.svdvals(m.W2.double())[0].item())
    return m, sd, smax


def collect_x2(cfg: E2Config, m: LMPCNg,
               X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m.learning = False
    feats = np.zeros((len(y), m.h), dtype=np.float32)
    ys = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        m._infer(X[i].ravel(), None)
        feats[i] = m._x2.detach().cpu().numpy()
        ys[i] = y[i]
    m.learning = True
    return feats, ys


def selfhead_eval(m: LMPCNg, feats: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """模型固有读出头：W_out @ x2 + b_out → argmax acc / softmax BPC。"""
    Wt = m.W_out.detach().cpu().numpy()
    bt = m.b_out.detach().cpu().numpy()
    logit = feats @ Wt + bt
    pred = logit.argmax(1)
    acc = float((pred == y).mean())
    z = logit - logit.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    bpc = float(np.log2(np.e) * nll.mean())
    return acc, bpc


def probe_eval(W: np.ndarray, b: np.ndarray,
               X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logit = X @ W + b
    pred = logit.argmax(1)
    acc = float((pred == y).mean())
    z = logit - logit.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    bpc = float(np.log2(np.e) * nll.mean())
    return acc, bpc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="768,1200,1856,2832,4032")
    ap.add_argument("--out", default="/root/srpc_e2/results_e2_gpu_eta/h1_selfhead.json")
    ap.add_argument("--h-ckpt", default="",
                    help="h=path 逗号分隔覆盖（如 2832=...、4032=...）。")
    args = ap.parse_args()

    h_ckpt = {}
    for kv in filter(None, args.h_ckpt.split(",")):
        h_s, p = kv.split("=", 1)
        h_ckpt[int(h_s)] = p

    hs = [int(x) for x in args.include.split(",") if int(x) in VALID_HS]
    cfg = E2Config()
    corpus = ByteCorpus(cfg)

    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    print(f"eval windows={len(eval_y)}", flush=True)

    rows = []
    for h in hs:
        ckpt = h_ckpt.get(h, ckpt_path(h))
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} 缺失", flush=True)
            continue
        t0 = time.time()
        m, sd, smax = load_model(cfg, h, ckpt)
        step = sd.get("step", "?")
        ex, ey = collect_x2(cfg, m, eval_x, eval_y)
        sh_acc, sh_bpc = selfhead_eval(m, ex, ey)
        # 活性/幅度统计（坍缩核对）
        act = float((ex > 0).mean())
        amp = float(ex[ex > 0].mean()) if (ex > 0).any() else 0.0
        rows.append(dict(h=h, step=step, w2_smax=round(smax, 2),
                         selfhead_acc=round(sh_acc, 4),
                         selfhead_bpc=round(sh_bpc, 3),
                         x2_active=round(act, 3), x2_amp=round(amp, 3),
                         wall=round(time.time() - t0, 1)))
        print(f"h={h}: step={step} smax={smax:.2f} x2_act={act:.3f} amp={amp:.3f} "
              f"selfhead_acc={sh_acc:.4f} selfhead_bpc={sh_bpc:.3f} "
              f"wall={rows[-1]['wall']}s", flush=True)

    # 单调判定
    accs = [r["selfhead_acc"] for r in rows]
    mono = len(accs) >= 2 and all(accs[i] > accs[i - 1] for i in range(1, len(accs)))
    out = dict(rows=rows, selfhead_acc_monotonic=mono,
               note="freeze → 收敛态 x2 → 模型固有 W_out/b_out 自-读出头 next-byte acc/BPC；"
                    "权重口径=最优健康档(768/1200 登顶、1856 fix、2832/4032 eta1)；"
                    "若单调而外置探针不单调→读出解离为外置读出口伪影，H1 以自-读出-宽度口径修正")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"== selfhead -> {args.out} | monotonic={mono} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())