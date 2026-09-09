"""阶段 H1 承重墙：字节 LM 探针 —— 隐藏表征线性可读性随规模单调（PCN vs 孪生）。

协议（2026-09-09 拍板口径）：
  - 任务：tinyshakespeare next-byte 预测（真实且便宜的字节 LM）。
  - PCN 侧：直接复用 E2 已训 5 档 GPU checkpoint（results_e2_gpu/pcn_{h}.pt，含
    fix 版稳定档），**冻结权重**；对验证窗口做自由推断取收敛态 x2 隐藏表征；
    在其上训练一个**独立线性探针**（岭回归，从 x2 线性读出 next-byte），测 acc。
  - 探针 = 在表征上加线性读出头，衡量"表征里线性可读的 next-byte 信息量"；
    与孪生的 softmax-head 线性读出（json 端到端 acc）同口径可比。
  - 报告：每档（宽度）探针 acc / BPC，判断"随宽度单调"；并与孪生参照线
    （twin_{h}.json 里的 acc）算相对比例 ratio = probe_acc / twin_acc。
  - 验收 H1：同预算 acc 随(宽度)单调 且（预期）≥ 孪生 × 预设比例。

用法（需在含已训 checkpoint 的 GPU 工程目录运行）：
  python scripts/run_h1_probe.py [--probe-n 3072] [--ridge-lambda 1e-3]
                                 [--include 768,1200,1856,2832,4032]

输出：results_e2_gpu/h1_probe.json（每档探针 acc/BPC/ratio + 单调判定）。
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

RES = "results_e2_gpu"
VALID_HS = [768, 1200, 1856, 2832, 4032]


# ----------------------------------------------------------------------
# 特征收集：冻结权重，对 (x0 窗口, y) 自由推断取 x2
# ----------------------------------------------------------------------
def collect_x2(cfg: E2Config, m: LMPCNg,
               X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X: (n, W, 256)；y: (n,)。返回 (feat (n,h), y (n,))。"""
    m.learning = False                      # 冻结（仅推断，不改权重）
    feats = np.zeros((len(y), m.h), dtype=np.float32)
    ys = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        m._infer(X[i].ravel(), None)        # 自由推断收敛态 x2（自设 self._x2）
        feats[i] = m._x2.detach().cpu().numpy()
        ys[i] = y[i]
    m.learning = True
    return feats, ys


# ----------------------------------------------------------------------
# 线性探针（岭回归，多类 one-hot）：闭式解，无迭代超参
# ----------------------------------------------------------------------
def ridge_probe(X: np.ndarray, y: np.ndarray, lam: float = 1e-3,
                n_cls: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """W = (X̃ᵀX̃+λI)⁻¹X̃ᵀY_oh。返回 (W (d+1, C), b)。

    数值稳健：在 float64 下求正规方程（float32 累加会丢精度致近奇异），
    仍奇异则退 lstsq 伪逆（兜底，保闭式解语义不变）。
    """
    n = X.shape[0]
    Xb = np.hstack([X.astype(np.float64),
                    np.ones((n, 1), dtype=np.float64)])      # 加偏置列
    Y = np.zeros((n, n_cls), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d, dtype=np.float64)
    B = Xb.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W, _, _, _ = np.linalg.lstsq(np.vstack([Xb, np.eye(d) * np.sqrt(lam)]),
                                     np.vstack([Y, np.zeros((d, n_cls))]),
                                     rcond=None)
    return W[:-1].astype(np.float32), W[-1].astype(np.float32)


def probe_eval(W: np.ndarray, b: np.ndarray,
               X: np.ndarray, y: np.ndarray,
               n_cls: int = 256) -> tuple[float, float]:
    logit = X @ W + b
    pred = logit.argmax(1)
    acc = float((pred == y).mean())
    # BPC（离散分布交叉熵近似，取 top 温度=1 的 softmax NLL）
    z = logit - logit.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    bpc = float(np.log2(np.e) * nll.mean())
    return acc, bpc


# ----------------------------------------------------------------------
# 探针训练样本流：训练段窗口（探针fit用，独立于评估验证段）
# ----------------------------------------------------------------------
def probe_samples(cfg: E2Config, corpus: ByteCorpus, n_probe: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    W_ = cfg.context
    X = np.zeros((n_probe, W_, 256), dtype=np.float32)
    y = np.zeros(n_probe, dtype=np.int64)
    stride = max(1, (len(corpus.train) - W_ - 1) // n_probe)
    for i in range(n_probe):
        t = (i * stride) % (len(corpus.train) - W_ - 1)
        b = corpus.train[t:t + W_]
        X[i, np.arange(W_), b] = 1.0
        y[i] = corpus.train[t + W_]
    return X, y


def load_model(cfg: E2Config, h: int, ckpt: str) -> tuple[LMPCNg, dict]:
    """加载已训 checkpoint 权重到 GPU 模型实例，并返回加载后 σmax(W2)。"""
    rng = np.random.default_rng(0 * 977 + 5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    # 仅载入核心权重，跳过 m2/m3 掩码（用当前实例的一致构造）
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    smax = float(torch.linalg.svdvals(m.W2.double())[0].item())
    return m, sd, smax


def ckpt_path(h: int) -> str:
    """H1 权重点选择：≥1856 用 E2 修复版（谱截断），768/1200 用登顶健康档。"""
    if h >= 1856:
        return os.path.join(RES + "_fix", f"pcn_{h}_fix.pt")
    return os.path.join(RES, f"pcn_{h}.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--ridge-lambda", type=float, default=1e-3)
    ap.add_argument("--include", default="768,1200,1856,2832,4032")
    ap.add_argument("--out", default=os.path.join(RES, "h1_probe.json"))
    ap.add_argument("--h-ckpt", default="",
                    help="每档 checkpoint 覆盖，h=path 逗号分隔（如 "
                         "4032=/root/srpc_e2/results_e2_gpu_eta/pcn_4032_exp1.0.pt）")
    args = ap.parse_args()

    h_ckpt = {}
    for kv in filter(None, args.h_ckpt.split(",")):
        h_s, p = kv.split("=", 1)
        h_ckpt[int(h_s)] = p

    hs = [int(x) for x in args.include.split(",") if int(x) in VALID_HS]
    cfg = E2Config()
    corpus = ByteCorpus(cfg)

    # 验证窗口（固定，与 E2 评估同口径）：tau 校准后剩余作探针评估段
    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    print(f"probe fit windows={args.probe_n}, eval windows={len(eval_y)}", flush=True)

    probe_train_x, probe_train_y = probe_samples(cfg, corpus, args.probe_n)

    rows = []
    for h in hs:
        name = f"pcn_{h}"
        ckpt = h_ckpt.get(h, ckpt_path(h))
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} 缺失", flush=True)
            continue
        t0 = time.time()
        m, sd, smax = load_model(cfg, h, ckpt)
        step = sd.get("step", "?")
        # fit 探针（冻结 x2）
        fx, fy = collect_x2(cfg, m, probe_train_x, probe_train_y)
        W, b = ridge_probe(fx, fy, args.ridge_lambda)
        # eval 探针（验证段）
        ex, ey = collect_x2(cfg, m, eval_x, eval_y)
        p_acc, p_bpc = probe_eval(W, b, ex, ey)
        # 孪生参照线
        twin_path = os.path.join(RES, f"twin_{h}.json")
        twin_acc = None
        if os.path.exists(twin_path):
            try:
                twin_acc = float(json.load(open(twin_path)).get("acc"))
            except Exception:
                twin_acc = None
        ratio = (p_acc / twin_acc) if (twin_acc and twin_acc > 0) else None
        rows.append(dict(h=h, step=step, w2_smax=round(smax, 2),
                         probe_acc=round(p_acc, 4),
                         probe_bpc=round(p_bpc, 3),
                         twin_acc=(round(twin_acc, 4) if twin_acc is not None
                                   else None),
                         ratio=(round(ratio, 3) if ratio is not None else None),
                         wall=round(time.time() - t0, 1)))
        print(f"{name}: smax={smax:.2f} probe_acc={p_acc:.4f} "
              f"probe_bpc={p_bpc:.3f} twin_acc={twin_acc} ratio={ratio} "
              f"wall={rows[-1]['wall']}s", flush=True)

    # 单调判定（随宽度升序，probe_acc 应单调上升）
    accs = [r["probe_acc"] for r in rows]
    mono = len(accs) >= 2 and all(accs[i] > accs[i - 1] for i in range(1, len(accs)))
    out = dict(probe_n=args.probe_n, ridge_lambda=args.ridge_lambda,
               rows=rows, probe_acc_monotonic=mono,
               note="freeze 权重 → 自由推断 x2 → 岭回归线性探针 next-byte；"
                    "ratio=Pprobe_acc/twin_acc；≥1856 档用 E2 修复版(W2 谱截断后) "
                    "checkpoint，w2_smax 记账谱健康")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"== H1 probe -> {args.out} | monotonic={mono} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())