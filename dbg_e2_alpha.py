"""α 消融：纯识别通道（无顶-下）x2 是否携带类信息？"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax

cfg = E2Config()
corpus = ByteCorpus(cfg)
m = LMPCN(cfg, 768, np.random.default_rng(0), eta_w=0.005, iters=24)
for t in range(8000):
    x, y = corpus.train_window(t)
    m.train_step(x.ravel(), y)

# 100 个训练窗口探针
n_probe = 100
px, py = [], []
for t in range(n_probe):
    x, y = corpus.train_window(100000 + t)
    px.append(x.ravel())
    py.append(y)
px = np.array(px); py = np.array(py)

def free_settle(x0p, alpha, iters, tau):
    x0rf = x0p[m.idx_rf]
    x1g = np.zeros_like(m.x1g)
    x2 = np.zeros_like(m.x2)
    x3 = _softmax(m.W3.T @ x2 / tau)
    for _ in range(iters):
        pred0 = np.matmul(m.W1c, x1g[:, :, None])[:, :, 0]
        e0c = x0rf - pred0
        x1 = x1g.ravel()
        e1 = x1 - m.W2 @ x2
        e2 = x2 - m.W3 @ x3
        u1 = np.matmul(m.W1cT, e0c[:, :, None])[:, :, 0] * m.s1
        u1 = cfg.beta * u1.reshape(16, 48) - alpha * e1.reshape(16, 48)
        u2 = cfg.beta * (m.W2.T @ e1) - alpha * e2
        x1g = np.clip(x1g + m.et1 * u1 * (np.abs(u1) > cfg.theta_event), 0.0, cfg.x_max)
        x2 = np.clip(x2 + m.et2 * u2 * (np.abs(u2) > cfg.theta_event), 0.0, cfg.x_max)
        x3 = _softmax(m.W3.T @ x2 / tau)
    return x2, x3

for tag, alpha, iters, tau in [("pure-rec(α=0)", 0.0, 60, 0.2),
                               ("full(α=1.5)", cfg.alpha, 60, 0.2),
                               ("full+τ=1.0", cfg.alpha, 60, 1.0)]:
    hits = 0
    pys = []
    for i in range(n_probe):
        x0p = np.concatenate([px[i], np.zeros(1, np.float32)])
        x2, x3 = free_settle(x0p, alpha, iters, tau)
        hits += int(x3.argmax() == py[i])
        pys.append(x3[py[i]])
    print(f"{tag:14s}: acc={hits/n_probe:.3f} P(y)={np.mean(pys):.4f}")
