"""阶段 H1b：非线性/稀疏槽位探针 —— 验证"规模红利以非线性 k-WTA 稀疏集中承载"。

背景（report_h1.md §0/§2b）：eta_w 宽度缩放已修复端到端 BPC（2832=3.833、4032=3.848 反超
1856 的 3.884），但**线性探针 acc 不随宽度单调**（1856=0.275 > 2832=0.255 > 4032=0.252），
BPC/探针解离以干净形态复现。核心假设：x2 的 next-byte 信息以**非线性**方式承载，且集中在
**k-WTA 稀疏激活的少数强槽位**上——全维线性探针被大量弱激活列的噪声稀释。

本脚本对已有 checkpoint（含 eta1 版大档）自由推断取收敛态 x2（已 50% 稀疏），在 x2 上训练
三种读出探针对照：
  1. **linear**（基线）：全维岭回归线性读出，应与 run_h1_probe 一致（对照）。
  2. **topk-linear**（稀疏槽位）：对 x2 再做 top-k 掩码（k = topk_frac·h，扫描更紧稀疏度），
     在掩码后的稀疏槽位上线性读出 —— 若红利集中在少数强槽位，更紧 top-k 的线性 acc 应随宽度
     单调（且高于全维线性）。
  3. **mlp2**（非线性）：两层 MLP（h → M → ReLU → M → ReLU → 256）梯度读出 —— 若红利以非线性
     组合承载，MLP acc 应随宽度单调。

判定：三种探针随(宽度)的单调性 + 大档是否反超 1856。若 topk/MLP 单调而 linear 不单调 →
"非线性/稀疏集中承载"成立，H1 立式以非线性口径修正为成立。

用法（4090 工程目录）：
  env PYTHONPATH=. python scripts/run_h1_nlprobe.py \
      --include 768,1200,1856,2832,4032 \
      --h-ckpt \"2832=/root/srpc_e2/results_e2_gpu_eta/pcn_2832_exp1.0.pt,\
4032=/root/srpc_e2/results_e2_gpu_eta/pcn_4032_exp1.0.pt\"
  输出：results_e2_gpu_eta/h1_nlprobe.json
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
# 特征收集 / 采样（与 run_h1_probe 同接口）
# ----------------------------------------------------------------------
def collect_x2(cfg: E2Config, m: LMPCNg,
               X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m.learning = False                      # 冻结权重，仅自由推断
    feats = np.zeros((len(y), m.h), dtype=np.float32)
    ys = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        m._infer(X[i].ravel(), None)
        feats[i] = m._x2.detach().cpu().numpy()
        ys[i] = y[i]
    m.learning = True
    return feats, ys


def probe_samples(cfg: E2Config, corpus: ByteCorpus, n_probe: int,
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


def load_model(cfg: E2Config, h: int, ckpt: str) -> LMPCNg:
    rng = np.random.default_rng(0 * 977 + 5)
    m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    allow = {"W1c", "W1cT", "W2", "W3", "W_out", "b_out"}
    for k, v in sd["model"].items():
        if k in allow and hasattr(m, k):
            getattr(m, k).copy_(v)
    return m


def ckpt_path(h: int) -> str:
    if h >= 1856:
        return os.path.join(RES + "_fix", f"pcn_{h}_fix.pt")
    return os.path.join(RES, f"pcn_{h}.pt")


# ----------------------------------------------------------------------
# 探针 1：线性（岭回归，基线对照）
# ----------------------------------------------------------------------
def linear_probe(X: np.ndarray, y: np.ndarray, lam: float = 1e-3,
                 n_cls: int = 256) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    Xb = np.hstack([X.astype(np.float64), np.ones((n, 1), dtype=np.float64)])
    Y = np.zeros((n, n_cls), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d, dtype=np.float64)
    B = Xb.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W, _, _, _ = np.linalg.lstsq(
            np.vstack([Xb, np.eye(d) * np.sqrt(lam)]),
            np.vstack([Y, np.zeros((d, n_cls))]), rcond=None)
    return W[:-1].astype(np.float32), W[-1].astype(np.float32)


def linear_topkize(X: np.ndarray, frac: float) -> np.ndarray:
    """对特征逐样本做 top-k 掩码（保留前 frac 强激活，余清零）→ 稀疏槽位特征。"""
    k = max(1, int(round(frac * X.shape[1])))
    flat = X.copy()
    n = X.shape[0]
    idx = np.argpartition(X, -k, axis=1)[:, -k:]
    mask = np.zeros_like(flat)
    np.put_along_axis(mask, idx, 1.0, axis=1)
    return flat * mask


def probe_eval(W: np.ndarray, b: np.ndarray,
               X: np.ndarray, y: np.ndarray) -> float:
    logit = (X @ W + b).astype(np.float64)
    pred = logit.argmax(1)
    return float((pred == y).mean())


# ----------------------------------------------------------------------
# 探针 3：两层 MLP（非线性，torch 梯度）
# ----------------------------------------------------------------------
def mlp2_probe(X: np.ndarray, y: np.ndarray, h: int, seed: int = 0,
               M: int = 256, n_cls: int = 256, epochs: int = 20,
               lr: float = 1e-3, wd: float = 1e-2, bs: int = 256
               ) -> tuple[torch.nn.Module, list]:
    torch.manual_seed(seed)
    n = X.shape[0]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model = torch.nn.Sequential(
        torch.nn.Linear(h, M), torch.nn.ReLU(),
        torch.nn.Linear(M, M), torch.nn.ReLU(),
        torch.nn.Linear(M, n_cls))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    idx = torch.randperm(n)
    curve = []
    with torch.enable_grad():          # 覆盖模块顶部全局 set_grad_enabled(False)
        for ep in range(epochs):
            order = idx[torch.randperm(n)]
            for b0 in range(0, n, bs):
                bi = order[b0:b0 + bs]
                opt.zero_grad()
                l = lossf(model(Xt[bi]), yt[bi])
                l.backward()
                opt.step()
            curve.append(float(l.item()))
    return model, curve


def mlp2_eval(model: torch.nn.Module, X: np.ndarray,
              y: np.ndarray) -> float:
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32))
        pred = logit.argmax(1).numpy()
    return float((pred == y).mean())


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--ridge-lambda", type=float, default=1e-3)
    ap.add_argument("--include", default="768,1200,1856,2832,4032")
    ap.add_argument("--out", default="results_e2_gpu_eta/h1_nlprobe.json")
    ap.add_argument("--h-ckpt", default="",
                    help="每档 checkpoint 覆盖：h=path 逗号分隔")
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--mlp-epochs", type=int, default=20)
    ap.add_argument("--topk-frac", default="1.0,0.5,0.25,0.1",
                    help="稀疏槽位探针扫描的 top-k 占比（逗号分隔）")
    args = ap.parse_args()

    h_ckpt = {}
    for kv in filter(None, args.h_ckpt.split(",")):
        h_s, p = kv.split("=", 1)
        h_ckpt[int(h_s)] = p
    topk_fracs = [float(x) for x in args.topk_frac.split(",")]

    hs = [int(x) for x in args.include.split(",") if int(x) in VALID_HS]
    cfg = E2Config()
    corpus = ByteCorpus(cfg)
    vx, vy = corpus.val_x, corpus.val_y
    eval_x, eval_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    probe_train_x, probe_train_y = probe_samples(cfg, corpus, args.probe_n)
    print(f"probe fit={args.probe_n}, eval={len(eval_y)}, "
          f"topk_fracs={topk_fracs}", flush=True)

    rows = []
    for h in hs:
        ckpt = h_ckpt.get(h, ckpt_path(h))
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} 缺失", flush=True)
            continue
        t0 = time.time()
        m = load_model(cfg, h, ckpt)
        fx, fy = collect_x2(cfg, m, probe_train_x, probe_train_y)
        ex, ey = collect_x2(cfg, m, eval_x, eval_y)

        row = {"h": h, "linear": None, "topk": {}, "mlp2": None}
        # 1) 线性基线
        Wl, bl = linear_probe(fx, fy, args.ridge_lambda)
        row["linear"] = round(probe_eval(Wl, bl, ex, ey), 4)
        # 2) 稀疏槽位线性（扫描 top-k 稀疏度）
        for fr in topk_fracs:
            # frac=1.0 即全维（x2 本身已 50% 稀疏）
            frx = fx if fr >= 1.0 else linear_topkize(fx, fr)
            ere = ex if fr >= 1.0 else linear_topkize(ex, fr)
            Wk, bk = linear_probe(frx, fy, args.ridge_lambda)
            row["topk"][f"q{fr}"] = round(probe_eval(Wk, bk, ere, ey), 4)
        # 3) 两层 MLP（非线性）
        mdl, mcurve = mlp2_probe(fx, fy, h, M=args.mlp_hidden,
                                 epochs=args.mlp_epochs)
        row["mlp2"] = round(mlp2_eval(mdl, ex, ey), 4)
        row["wall"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"h={h}: linear={row['linear']} topk={row['topk']} "
              f"mlp2={row['mlp2']} wall={row['wall']}s", flush=True)

    # 单调判定（按 h 升序，各自探针口径）
    def mono(key_get):
        accs = [key_get(r) for r in rows]
        return all(accs[i] >= accs[i - 1] for i in range(1, len(accs)))

    out = {
        "probe_n": args.probe_n, "ridge_lambda": args.ridge_lambda,
        "topk_fracs": topk_fracs, "mlp_hidden": args.mlp_hidden,
        "rows": rows,
        "monotonic": {
            "linear": mono(lambda r: r["linear"]),
            "mlp2": mono(lambda r: r["mlp2"]),
            **{f"topk_q{f}": mono(lambda r, f=f: r["topk"][f"q{f}"])
               for f in topk_fracs},
        },
        "note": "三种读出探针对照；大档 2832/4032 用 eta1(exp=1.0) checkpoint；"
                "若 topk/mlp2 单调而 linear 不单调 → 规模红利以非线性/稀疏集中承载。",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"== nlprobe -> {args.out} ==", flush=True)
    print("monotonic:", out["monotonic"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())