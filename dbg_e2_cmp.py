"""钳制 vs 自由 settle 判分对比（训练 8k 步后）。"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax

cfg = E2Config()
corpus = ByteCorpus(cfg)
m = LMPCN(cfg, 768, np.random.default_rng(0), eta_w=0.005, iters=24)
for t in range(8000):
    x, y = corpus.train_window(t)
    m.train_step(x.ravel(), y)

n_probe = 200
px, py = [], []
for t in range(n_probe):
    x, y = corpus.train_window(100000 + t)
    px.append(x.ravel())
    py.append(y)
px = np.array(px); py = np.array(py)

def clamp_settle(x0p):
    """钳制 settle：x3=yoh，读 W3ᵀ·x2 判分（信用分配后 x2 的类可读性）。"""
    yoh = np.zeros(256, np.float32); yoh[0] = 1.0  # 占位（下面逐个真值用）
    return None

hits_c = hits_f = 0
margin_c = []
score_self_c = []
score_self_f = []
for i in range(n_probe):
    y = py[i]
    x0p = np.concatenate([px[i], np.zeros(1, np.float32)])
    # 钳制 settle
    yoh = np.zeros(256, np.float32); yoh[y] = 1.0
    m._infer(px[i], yoh, block_mask=None, clamp=True)
    x2c = m._x2
    logit_c = m.W3.T @ x2c
    self_c = logit_c[y]
    top_c = np.partition(logit_c, -2)[-2]
    margin_c.append(self_c - top_c)
    score_self_c.append(self_c)
    hits_c += int(logit_c.argmax() == y)
    # 自由 settle（新协议）
    m._infer(px[i], None, block_mask=None, clamp=False)
    x3 = m._x3
    self_f = x3[y]
    score_self_f.append(self_f)
    hits_f += int(x3.argmax() == y)

print(f"clamped acc={hits_c/n_probe:.3f}  margin mean={np.mean(margin_c):.4f} "
      f"self-score mean={np.mean(score_self_c):.4f}")
print(f"free    acc={hits_f/n_probe:.3f}  self-score mean={np.mean(score_self_f):.4f}")
# 自由 x3 的 top2 差值
x3all = []
for i in range(n_probe):
    m._infer(px[i], None, block_mask=None, clamp=False)
    x3all.append(m._x3)
x3all = np.array(x3all)
top1 = np.sort(x3all, axis=1)[:, -1]
top2 = np.sort(x3all, axis=1)[:, -2]
print(f"free x3: top1 mean={top1.mean():.4f} top2 mean={top2.mean():.4f} "
      f"max={x3all.max():.4f} nnz_per_sample={int((x3all>1e-3).sum(1).mean())}")
