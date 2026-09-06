#!/usr/bin/env python3
"""诊断 Δ=1 对照任务失败根因（分类偏置：总是预测类 1）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def main():
    cfg = CreditConfig(alpha=0.65)
    seed = 0
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10000 + 500, 1)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]

    m = CreditPCN(replace(cfg, delay=1), np.random.default_rng(seed * 3000 + 11), "error")
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)

    # 1) 训练/评估逐类准确率
    for tag, Xs, ys in (("train", Xtr, ytr), ("eval", Xev, yev)):
        preds = np.array([m.predict(x) for x in Xs])
        for c in (0, 1):
            idx = ys == c
            acc = float(np.mean(preds[idx] == c))
            print(f"{tag} class{c}: n={int(idx.sum())} acc={acc:.3f}")
        print(f"{tag} overall: {float(np.mean(preds == ys)):.3f}")

    # 2) 比较协议能量分布（前 200 个 eval 样本）
    e0s, e1s, cc = [], [], []
    for x in Xev[:200]:
        e_c = []
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            m.x1[:] = 0.0
            m.x2[:] = 0.0
            m._infer(x, yoh, free_out=False)
            e_c.append(m._energy)
        e0s.append(e_c[0]); e1s.append(e_c[1]); cc.append(int(np.argmin(e_c)))
    e0s, e1s, cc = np.array(e0s), np.array(e1s), np.array(cc)
    print(f"\nenergy E0 mean={e0s.mean():.4f} E1 mean={e1s.mean():.4f} "
          f"delta(E1-E0) mean={np.mean(e1s-e0s):.4f} std={np.std(e1s-e0s):.4f}")
    print(f"pred-class0 frac={np.mean(cc==0):.3f}")

    # 3) x2 类质心 vs W3 列（判别几何）
    m.x1[:] = 0.0; m.x2[:] = 0.0
    x2_0, x2_1 = [], []
    for t in range(0, 2000, 4):
        c = int(ytr[t])
        yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
        m.x1[:] = 0.0; m.x2[:] = 0.0
        m._infer(Xtr[t], yoh, free_out=False)
        (x2_0 if c == 0 else x2_1).append(m.x2.copy())
    c0 = np.mean(x2_0, axis=0)
    c1 = np.mean(x2_1, axis=0)
    n0, n1 = np.linalg.norm(c0), np.linalg.norm(c1)
    cos = float(np.dot(c0, c1) / (n0 * n1 + 1e-9))
    w0, w1 = m.W3[:, 0], m.W3[:, 1]
    print(f"\nW3 col cos={float(np.dot(w0,w1)/(np.linalg.norm(w0)*np.linalg.norm(w1))):.4f}")
    print(f"x2 centroids: |c0|={n0:.4f} |c1|={n1:.4f} cos(c0,c1)={cos:.4f}")
    print(f"dist(x2c0,W3c0)={np.linalg.norm(c0-w0):.4f} dist(x2c1,W3c1)={np.linalg.norm(c1-w1):.4f}")
    print(f"dist(x2c0,W3c1)={np.linalg.norm(c0-w1):.4f} dist(x2c1,W3c0)={np.linalg.norm(c1-w0):.4f}")

    # 4) 远端权重驱动（W1 第一块行）
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW))
    distal = float(np.linalg.norm(dW[:cfg.d_feat]))
    print(f"\ndistal W1 share={distal/(total+1e-12):.4f} (chance 0.5)")


if __name__ == "__main__":
    main()
