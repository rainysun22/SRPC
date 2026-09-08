"""单样本 settle 轨迹诊断：x2 是否对齐目标类 W3 列？"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax

cfg = E2Config()
corpus = ByteCorpus(cfg)
m = LMPCN(cfg, 768, np.random.default_rng(0), eta_w=0.005, iters=24)

x, y = corpus.train_window(100000)
yoh = np.zeros(256, np.float32); yoh[y] = 1.0

# 手动复刻 _infer（clamp=True）并逐迭代打印
cfg_ = cfg
a, b_ = cfg_.alpha, cfg_.beta
x0p = np.concatenate([x.ravel(), np.zeros(1, np.float32)])
x0rf = x0p[m.idx_rf]
x1g = np.zeros_like(m.x1g)
x2 = np.zeros_like(m.x2)
x3 = yoh.copy()
W1c, W1cT = m.W1c, m.W1cT
th = cfg_.theta_event
for it in range(24):
    pred0 = np.matmul(W1c, x1g[:, :, None])[:, :, 0]
    e0c = x0rf - pred0
    x1 = x1g.ravel()
    e1 = x1 - m.W2 @ x2
    e2 = x2 - m.W3 @ x3
    u1 = np.matmul(W1cT, e0c[:, :, None])[:, :, 0] * m.s1
    u1 = b_ * u1.reshape(16, 48) - a * e1.reshape(16, 48)
    u2 = b_ * (m.W2.T @ e1) - a * e2
    g2 = np.abs(u2) > th
    x2 = np.clip(x2 + m.et2 * u2 * g2, 0.0, cfg_.x_max)
    if it % 6 == 0 or it == 23:
        logit = m.W3.T @ x2
        argmax = logit.argmax()
        align = (x2 * np.maximum(m.W3[:, y], 0)).sum() / max(np.linalg.norm(x2) * np.linalg.norm(m.W3[:, y]), 1e-8)
        print(f"it={it:2d} e0={np.linalg.norm(e0c):.3f} e1={np.linalg.norm(e1):.3f} "
              f"e2={np.linalg.norm(e2):.3f} x2nnz={int((x2>0).sum()):4d} "
              f"argmax={argmax} y={y} align_cos={align:.3f} logit_self={logit[y]:.3f}")
print("W3 col norm target:", np.linalg.norm(m.W3[:, y]))
