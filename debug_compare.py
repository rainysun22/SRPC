#!/usr/bin/env python3
"""对比 Δ=1 vs Δ=4：训练过程 acc 曲线 + 判别几何，定位 Δ=1 失败根因。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def train_curve(delay, alpha=0.65, kwta_frac=0.5, energy_mode="class",
                train_steps=10000, eval_n=300):
    cfg = replace(CreditConfig(alpha=alpha), delay=delay, kwta_frac=kwta_frac,
                  energy_mode=energy_mode)
    seed = 0
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_steps + 500, delay)
    Xtr, ytr = X[:train_steps], y[:train_steps]
    Xev, yev = X[train_steps:], y[train_steps:]
    m = CreditPCN(cfg, np.random.default_rng(seed * 3000 + 11), "error")

    # 训练集上取固定子集监控（与训练不同步的 mini-eval）
    ri = np.random.default_rng(123).choice(train_steps, size=eval_n, replace=False)
    Xm, ym = Xtr[ri], ytr[ri]

    for t in range(train_steps):
        m.train_step(Xtr[t], float(ytr[t]))
        if (t + 1) % 2000 == 0:
            m.set_learning(False)
            preds = np.array([m.predict(x) for x in Xm])
            m.set_learning(True)
            print(f"  [d={delay}] step {t+1}: acc={np.mean(preds == ym):.3f}")

    m.set_learning(False)
    preds = np.array([m.predict(x) for x in Xev])
    acc = float(np.mean(preds == yev))
    w0, w1 = m.W3[:, 0], m.W3[:, 1]
    wcos = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1) + 1e-9))
    # x2 类质心
    x2s = {0: [], 1: []}
    for t in range(0, train_steps, 5):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        x2s[c].append(m.x2.copy())
    c0 = np.mean(x2s[0], axis=0); c1 = np.mean(x2s[1], axis=0)
    ccos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]))
    print(f"[d={delay}] final acc={acc:.3f} | W3cos={wcos:.3f} x2cos={ccos:.3f} "
          f"distal={distal/total:.3f} (chance {1/(delay+1):.2f})")
    return acc


if __name__ == "__main__":
    print("== Δ=1 ==")
    train_curve(1)
    print("== Δ=4 ==")
    train_curve(4)
