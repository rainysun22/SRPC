"""自由评估 settle 轨迹：x1/x2 活动度与 x3 熵随迭代变化。"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax

cfg = E2Config()
corpus = ByteCorpus(cfg)
m = LMPCN(cfg, 768, np.random.default_rng(0), eta_w=0.005, iters=24)
for t in range(8000):
    x, y = corpus.train_window(t)
    m.train_step(x.ravel(), y)

x, y = corpus.train_window(100000)
x0p = np.concatenate([x.ravel(), np.zeros(1, np.float32)])
x0rf = x0p[m.idx_rf]
a, b_ = cfg.alpha, cfg.beta
x1g = np.zeros_like(m.x1g)
x2 = np.zeros_like(m.x2)
x3 = _softmax(m.W3.T @ x2 / m.tau)
for it in range(24):
    pred0 = np.matmul(m.W1c, x1g[:, :, None])[:, :, 0]
    e0c = x0rf - pred0
    x1 = x1g.ravel()
    e1 = x1 - m.W2 @ x2
    e2 = x2 - m.W3 @ x3
    u1 = np.matmul(m.W1cT, e0c[:, :, None])[:, :, 0] * m.s1
    u1 = b_ * u1.reshape(16, 48) - a * e1.reshape(16, 48)
    u2 = b_ * (m.W2.T @ e1) - a * e2
    g1 = np.abs(u1) > cfg.theta_event
    g2 = np.abs(u2) > cfg.theta_event
    x1g = np.clip(x1g + m.et1 * u1 * g1, 0.0, cfg.x_max)
    x2 = np.clip(x2 + m.et2 * u2 * g2, 0.0, cfg.x_max)
    x3 = _softmax(m.W3.T @ x2 / m.tau)
    if it % 4 == 0 or it == 23:
        ent = float(-(x3 * np.log(x3 + 1e-12)).sum() / np.log(256))
        logit = m.W3.T @ x2
        print(f"it={it:2d} |e0|={np.linalg.norm(e0c):.3f} |e1|={np.linalg.norm(e1):.3f} "
              f"|e2|={np.linalg.norm(e2):.3f} |x1|={np.linalg.norm(x1):.3f} "
              f"|x2|={np.linalg.norm(x2):.3f} x3ent={ent:.3f} "
              f"argmax={logit.argmax()} y={y}")
