#!/usr/bin/env python3
"""E1 证据链探针（一）：xorsum@16类 的 BP 对照——在线 vs batch。

背景（v2 方案调整依据，见 srpc/config.py LangConfig 注释）：
16 类跨块 XOR 在 LangPCN 在线学习贴机会后，需区分"局部规则缺陷"与
"在线协议样本效率瓶颈"。本探针用同形状判别网络（1536-128-64-16，ReLU，
无掩码/k-WTA——纯上界口径）跑标准梯度：

    --mode online : 在线 Adam（batch=1）   => 贴机会（期望梯度≈0，对称不破缺）
    --mode batch  : mini-batch plain SGD   => 1.0（batch 平均使信号浮出）

结论：parity 为 SQ-hard 高频函数（Kearns & Valiant 1989；Blum et al. 1994），
在线单样本学习不可达与优化器无关；batch 可达证明任务本身可解。
注意：本探针走反传，仅作可学性上界对照，不属 SR-PC 构造（同 GPU_TASKS
T2 孪生契约）。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from srpc.config import LangConfig
from srpc.lang import make_task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("online", "batch"), default="online")
    ap.add_argument("--steps", type=int, default=20000,
                    help="online：训练样本数")
    ap.add_argument("--epochs", type=int, default=1000,
                    help="batch：epoch 数（固定 8192 训练集 + B=512）")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--delta", type=int, default=4)
    args = ap.parse_args()

    n_online = args.steps if args.mode == "online" else 8192
    cfg = LangConfig()
    rng = np.random.default_rng(7)
    n_eval = 500
    X, y, _ = make_task(cfg, rng, n_online + n_eval, args.delta, "xorsum")
    d0 = X.shape[1]
    W1 = rng.normal(0, 0.02, (d0, 128))
    W2 = rng.normal(0, 0.05, (128, 64))
    W3 = rng.normal(0, 0.1, (64, 16))
    b1, b2, b3 = (np.zeros(128), np.zeros(64), np.zeros(16))
    Y = np.eye(16)[y]

    def acc_eval() -> float:
        a1 = np.maximum(X[n_online:] @ W1 + b1, 0)
        a2 = np.maximum(a1 @ W2 + b2, 0)
        return float(((a2 @ W3 + b3).argmax(1) == y[n_online:]).mean())

    if args.mode == "online":
        lr = args.lr if args.lr is not None else 0.01
        params = [W1, W2, W3, b1, b2, b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for t in range(args.steps):
            x = X[t]
            a1 = np.maximum(x @ W1 + b1, 0)
            a2 = np.maximum(a1 @ W2 + b2, 0)
            logits = a2 @ W3 + b3
            p = np.exp(logits - logits.max())
            p /= p.sum()
            p[y[t]] -= 1.0
            dW3 = np.outer(a2, p)
            da2 = (W3 @ p) * (a2 > 0)
            dW2 = np.outer(a1, da2)
            da1 = (W2 @ da2) * (a1 > 0)
            dW1 = np.outer(x, da1)
            grads = [dW1, dW2, dW3, da1.copy(), da2.copy(), p.copy()]
            for i, (pp, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                pp -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                    np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
            if (t + 1) % (args.steps // 8) == 0:
                print(f"t={t+1}: acc={acc_eval():.3f}")
    else:
        lr = args.lr if args.lr is not None else 0.3
        B = 512
        for ep in range(1, args.epochs + 1):
            idx = rng.integers(0, n_online, B)
            Xb, Yb = X[idx], Y[idx]
            a1 = np.maximum(Xb @ W1 + b1, 0)
            a2 = np.maximum(a1 @ W2 + b2, 0)
            logits = a2 @ W3 + b3
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            P = p - Yb
            gW3 = a2.T @ P / B
            gb3 = P.mean(0)
            da2 = (P @ W3.T) * (a2 > 0)
            gW2 = a1.T @ da2 / B
            gb2 = da2.mean(0)
            da1 = (da2 @ W2.T) * (a1 > 0)
            gW1 = Xb.T @ da1 / B
            gb1 = da1.mean(0)
            W1 -= lr * gW1
            W2 -= lr * gW2
            W3 -= lr * gW3
            b1 -= lr * gb1
            b2 -= lr * gb2
            b3 -= lr * gb3
            if ep % max(1, args.epochs // 8) == 0:
                print(f"ep={ep}: acc={acc_eval():.3f}")
    print(f"final[{args.mode}]: acc={acc_eval():.3f} (chance=0.0625)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
