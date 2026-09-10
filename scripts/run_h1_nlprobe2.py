"""阶段 H1b-2：升级读出口 —— 修复 MLP 坍缩 + 容量匹配宽度，排除"读出口容量不足"伪影。

承接 H1b（report_h1.md §2c）三口径不单调之判读，其中两层 MLP 全档恒定 0.1333 = 坍缩到
单类基线（高频字节霸榜），口径作废。本脚本针对两个可修的口径缺陷重做非线性支：

  1. **类不均衡 → 坍缩**：probe_n=12000 下 256 类平均仅 ~47 样本/类，未加权 CE 被高频
     类（空格/换行，P≈13%）主导 → 用**逐类逆频率加权 CE**（inv-freq weight），把低频
     类的梯度拉起来，杜绝"只猜高频类"的最优解。
  2. **容量不足 → 接不住宽度红利**：读出口参数量固定（hidden=256）远小于大档表征维度
     （4032）→ 用 **hidden ∝ h** 缩放（读出口容量与表征同尺度），并加 **hidden-max
     上限** 控制训练成本。

判据：若 **wCE-MLP acc** 随宽度单调（大档反超 1856）→ 原"非线性承载失败"实为**读出口
容量/C 不均衡伪影**，H1 去解离化（修口径后单调成立）；若仍不单调 → 坐实"读出口不可接触"
为原理层面，进调研循环。

用法（4090 工程目录，PYTHONPATH=.）：
  env PYTHONPATH=/root/srpc_e2 python scripts/run_h1_nlprobe2.py \
      --include 768,1200,1856,2832,4032 \
      --h-ckpt "2832=/root/srpc_e2/results_e2_gpu_eta/pcn_2832_exp1.0.pt,\
4032=/root/srpc_e2/results_e2_gpu_eta/pcn_4032_exp1.0.pt"
  输出：results_e2_gpu_eta/h1_nlprobe2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)
torch.manual_seed(0)

RES = "results_e2_gpu"
VALID_HS = [768, 1200, 1856, 2832, 4032]


# ----------------------------------------------------------------------
# 特征收集 / 采样（复用 H1b 同接口，保证口径连续）
# ----------------------------------------------------------------------
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
# 线性基线（岭回归，对照 H1）
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


def probe_eval(W: np.ndarray, b: np.ndarray,
               X: np.ndarray, y: np.ndarray) -> float:
    logit = (X @ W + b).astype(np.float64)
    pred = logit.argmax(1)
    return float((pred == y).mean())


# ----------------------------------------------------------------------
# 升级 MLP：类逆频率加权 CE + 容量随宽度缩放（hidden ∝ h）
# ----------------------------------------------------------------------
class ProbeMLP(nn.Module):
    def __init__(self, h: int, n_cls: int = 256):
        super().__init__()
        # 容量匹配宽度：hidden ∝ h（读出口与表征同尺度），上限控成本
        d = int(min(h, 1536))
        self.net = nn.Sequential(
            nn.Linear(h, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, n_cls))

    def forward(self, x):
        return self.net(x)


def mlp2_probe_wce(X: np.ndarray, y: np.ndarray, h: int, seed: int = 0,
                   n_cls: int = 256, epochs: int = 50,
                   lr: float = 2e-3, wd: float = 1e-3, bs: int = 512,
                   val_x: np.ndarray | None = None,
                   val_y: np.ndarray | None = None,
                   patience: int = 8) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    n = X.shape[0]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    # 逐类逆频率权重（1/count 再归一，防梯度爆炸）
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = 1.0 / np.clip(cnt, 1, None)
    w = w / w.sum() * n_cls
    ce_w = torch.tensor(w, dtype=torch.float32)

    model = ProbeMLP(h, n_cls).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def run_loader(xt, yt_):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xt, yt_), batch_size=bs, shuffle=True)

    trl = run_loader(Xt, yt)
    Xtv = ytv = None
    if val_x is not None:
        Xtv = torch.tensor(val_x, dtype=torch.float32)
        ytv = torch.tensor(val_y, dtype=torch.long)

    best_val, best_state, best_ep, bad = -1.0, None, -1, 0
    train_curve = []
    val_curve = []
    with torch.enable_grad():
        for ep in range(epochs):
            model.train()
            totl = 0.0
            for xb, yb in trl:
                opt.zero_grad()
                logit = model(xb.cuda())
                l = F.cross_entropy(logit, yb.cuda(), weight=ce_w.cuda())
                l.backward()
                opt.step()
                totl += float(l.item()) * xb.size(0)
            train_curve.append(totl / n)
            if val_x is not None:
                model.eval()
                with torch.no_grad():
                    logit = model(Xtv.cuda())
                    pv = float((logit.argmax(1).cpu().numpy() == ytv.numpy()).mean())
                val_curve.append(pv)
                if pv > best_val:
                    best_val, best_state, best_ep, bad = pv, \
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep, 0
                else:
                    bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        return model, dict(train_curve=train_curve, val_curve=val_curve,
                           best_val=best_val, best_ep=best_ep)
    return model, dict(train_curve=train_curve, val_curve=val_curve)


def mlp2_eval(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32).cuda())
        pred = logit.argmax(1).cpu().numpy()
    return float((pred == y).mean())


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--ridge-lambda", type=float, default=1e-3)
    ap.add_argument("--include", default="768,1200,1856,2832,4032")
    ap.add_argument("--out", default="results_e2_gpu_eta/h1_nlprobe2.json")
    ap.add_argument("--h-ckpt", default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
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
    probe_train_x, probe_train_y = probe_samples(cfg, corpus, args.probe_n)

    # 留一小段训练段作 early-stopping 验证（不触碰最终 eval 段）
    n_tr = len(probe_train_y)
    n_val = max(1500, n_tr // 8)
    te_x, te_y = probe_train_x[:n_tr - n_val], probe_train_y[:n_tr - n_val]
    vv_x, vv_y = probe_train_x[n_tr - n_val:], probe_train_y[n_tr - n_val:]

    rows = []
    for h in hs:
        ckpt = h_ckpt.get(h, ckpt_path(h))
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} 缺失", flush=True)
            continue
        t0 = time.time()
        m = load_model(cfg, h, ckpt)
        # 线性基线（对照）
        fx, fy = collect_x2(cfg, m, probe_train_x, probe_train_y)
        ex, ey = collect_x2(cfg, m, eval_x, eval_y)
        Wl, bl = linear_probe(fx, fy, args.ridge_lambda)
        lin = probe_eval(Wl, bl, ex, ey)

        # 升级 MLP（wCE + 容量匹配，epoch=args.epochs 早停在训练段内部验证）
        fx_te, fy_te = collect_x2(cfg, m, te_x, te_y)
        fv_x, fv_y = collect_x2(cfg, m, vv_x, vv_y)
        mdl, info = mlp2_probe_wce(
            fx_te, fy_te, h, n_cls=256, epochs=args.epochs,
            patience=args.patience, val_x=fv_x, val_y=fv_y)
        mlp_acc = mlp2_eval(mdl, ex, ey)

        rows.append(dict(
            h=h, linear=round(lin, 4), mlp2_wce=round(mlp_acc, 4),
            mlp_best_val=round(info.get("best_val", -1.0), 4),
            mlp_best_ep=info.get("best_ep", -1),
            wall=round(time.time() - t0, 1)))
        print(f"h={h}: linear={rows[-1]['linear']} mlp2_wce={rows[-1]['mlp2_wce']} "
              f"best_val={rows[-1]['mlp_best_val']}@ep{rows[-1]['mlp_best_ep']} "
              f"wall={rows[-1]['wall']}s", flush=True)

    def mono(key):
        accs = [r[key] for r in rows]
        return all(accs[i] >= accs[i - 1] for i in range(1, len(accs)))

    out = {
        "probe_n": args.probe_n, "ridge_lambda": args.ridge_lambda,
        "epochs": args.epochs, "patience": args.patience,
        "rows": rows,
        "monotonic": {"linear": mono("linear"), "mlp2_wce": mono("mlp2_wce")},
        "note": "升级读出口：类逆频率加权 CE（修 MLP 坍缩）+ 容量随宽度缩放（hidden ∝ h）。"
                "若 mlp2_wce 随宽度单调 → 原非线性承载失败实为读出口容量/类不均衡伪影，"
                "H1 去解离化；若仍不单调 → 坐实读出口不可接触，进调研循环。",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"== nlprobe2 -> {args.out} ==", flush=True)
    print("monotonic:", out["monotonic"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())