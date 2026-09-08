"""E2 动力学诊断：训练集内部拟合 + 各层误差/活动度轨迹。"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax

cfg = E2Config()
corpus = ByteCorpus(cfg)
m = LMPCN(cfg, 768, np.random.default_rng(0), eta_w=0.005, iters=24)

# 固定 200 个训练窗口（内部拟合探针）
n_probe = 200
px, py = [], []
for t in range(n_probe):
    x, y = corpus.train_window(100000 + t)   # 训练段中部，避开开头
    px.append(x.ravel())
    py.append(y)
px = np.array(px)
py = np.array(py)


def train_acc():
    """训练集内部拟合：自由 LS 读出（x3 收敛态）argmax vs y。"""
    hits = 0
    probs = []
    for i in range(len(py)):
        m._infer(px[i], None, block_mask=None, clamp=False)
        x3 = m._x3
        hits += int(x3.argmax() == py[i])
        p = _softmax(x3)
        probs.append(p[py[i]])
    return hits / len(py), float(np.mean(probs))


print("step e0  e1  e2 | x1mean x2mean | train_acc P(y)")
for step in (100, 500, 1000, 2000, 4000, 8000):
    while m._ev_n < step:
        x, y = corpus.train_window(m._ev_n)
        m.train_step(x.ravel(), y)
        m._ev_n += 1
    e0 = float(np.linalg.norm(m._e0c))
    e1 = float(np.linalg.norm(m._e1))
    e2 = float(np.linalg.norm(m._e2))
    acc, pyprob = train_acc()
    print(f"{step:5d} {e0:.3f} {e1:.3f} {e2:.3f} | "
          f"{float(m._x1g.mean()):.4f} {float(m._x2.mean()):.4f} | "
          f"{acc:.3f} {pyprob:.3f}")
