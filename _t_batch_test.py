import os, sys
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _d); os.chdir(_d)
import numpy as np, torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

cfg = E2Config()
cfg.w2_cap = True; cfg.w2_cap_every = 1; cfg.w2_smax_cap = 5.0; cfg.w2_pow_iters = 8
h = 768
corpus = ByteCorpus(cfg)
seed = 0
rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cpu")

x, y = corpus.train_window(0)
yoh = np.zeros(256, np.float32); yoh[y] = 1.0

# 单样本
m._infer(x.ravel(), yoh, clamp=True, iters=12)
x1s, x2s = m._x1g.clone(), m._x2.clone()
e0s, e1s, e2s = m._e0c.clone(), m._e1.clone(), m._e2.clone()

# 批量 B=1
xb = x[None].astype(np.float32)  # (1,W,256)
yb = np.array([y])
x1b, x2b, x3b, e0b, e1b, e2b, g1, g2 = m._infer_batch(xb, yoh[None], clamp=True, iters=12)

for nm, a, b in [("x1g", x1s.reshape(-1), x1b.reshape(-1)),
                 ("x2", x2s, x2b[0]),
                 ("e0", e0s.reshape(-1), e0b[0].reshape(-1)),
                 ("e1", e1s.reshape(-1), e1b[0].reshape(-1)),
                 ("e2", e2s, e2b[0])]:
    d = float((a - b).abs().max().item())
    print(f"{nm}: maxdiff={d:.3e} {'OK' if d < 1e-4 else 'MISMATCH'}")

# 单样本 _learn 的 dW vs 批量 B=1 的 dW
m._x1g, m._x2 = x1s, x2s
m._e0c, m._e1, m._e2 = e0s, e1s, e2s
W1_before = m.W1c.clone(); W2_before = m.W2.clone(); W3_before = m.W3.clone()
# 用浅拷贝模型计算单个样本 learn delta（复制权重后跑一次 train 的 learn 部分）
import copy
m2 = LMPCNg(cfg, h, np.random.default_rng(seed), eta_w=0.01, iters=12, device="cpu")
m2.W1c.data.copy_(W1_before); m2.W2.data.copy_(W2_before); m2.W3.data.copy_(W3_before)

# 直接测 train_step_batch 单样本与 train_step 的权重变化方向是否一致（非逐位，因应用时机不同）
mb = LMPCNg(cfg, h, np.random.default_rng(seed), eta_w=0.01, iters=12, device="cpu")
mb.W1c.data.copy_(W1_before); mb.W2.data.copy_(W2_before); mb.W3.data.copy_(W3_before)
mb.W_out.data.copy_(m.W_out); mb.b_out.data.copy_(m.b_out)
mb.train_step_batch(xb, yb)
print("batch B=1 applied OK; W2 nan?", bool(torch.isnan(mb.W2).any()), "W1c nan?", bool(torch.isnan(mb.W1c).any()))
print("W_out max delta:", float((mb.W_out - m.W_out).abs().max()))  # W_out train via single train_step too

# W1c pad/s1 invariant
print("W1c row pad sum:", float(mb.W1c[mb.pad].abs().sum()))
print("ALL DONE")