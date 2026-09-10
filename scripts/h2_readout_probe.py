"""H2 决定性诊断：0.24 能力上限是表征天花板还是读头接口瓶颈？

先训练 clamp 基线（iters=12）得到 decent checkpoint，再以与评估一致的自由推断
（_infer clamp=False, iters=12）收集自由状态 (x2, x1, y)，离线拟合多档读头：

  (a) 当前线性 W_out（在线基线，参照 ~0.24）
  (b) x2 上的岭回归（线性上限）
  (c) x2 上的 2 层 MLP（非线性接口上限）
  (d) x1 上的 MLP（底层自由态上限，layer-probe 曾测 x1 0.32 > x2 0.24）
  (e) [x1; x2] 拼接 MLP（全接口上限）

若 (d)/(e) 明显 >0.25 冲向孪生 0.42 → 接口是瓶颈，可在不牺牲约束下设计更好的
局部读头；若全部平台在 ~0.24-0.28 → 表征天花板确证，则 H2『同能力更省』须改口径。

用途=判别方向，读头为离线探针（非最终局部实现）。用法（/root/srpc_e2/srpc_src）：
    python -B scripts/h2_readout_probe.py 1856 24000
"""
from __future__ import annotations
import os, sys, time
ROOT = "/root/srpc_e2/srpc_src"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg

torch.set_grad_enabled(False)

h = int(sys.argv[1]) if len(sys.argv) > 1 else 1856
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24_000
FEAT_TR = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 600

cfg = E2Config()
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.eta_inf_scl = 0.5

seed = 0
corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

print(f"== train {steps} steps (clamp, iters=12) ==", flush=True)
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_free(x, y)   # free_nudge=0 -> clamp 路径
m.learning = False

# 收集自由状态特征（评估口径：clamp=False, iters=12）
#   读头训练特征取自训练流（free 推断 2500 窗），测试用验证 rep 集
feat_tr = min(FEAT_TR, full - 100)
X2s, X1s, Ys = [], [], []
t0 = time.time()
for i in range(feat_tr):
    x, y = corpus.train_window(i)
    m._infer(x, None, iters=12)
    X2s.append(m._x2.cpu().numpy())
    X1s.append(m._x1g.cpu().numpy().reshape(-1))
    Ys.append(y)
vx, vy = corpus.val_x, corpus.val_y
cal = cfg.tau_cal_windows
N_TEST = min(N_TEST, len(vy) - cal)
for i in range(N_TEST):
    m._infer(vx[cal + i].ravel(), None, iters=12)
    X2s.append(m._x2.cpu().numpy())
    X1s.append(m._x1g.cpu().numpy().reshape(-1))
    Ys.append(vy[cal + i])
N_TR = feat_tr
X2 = np.array(X2s, np.float32)
X1 = np.array(X1s, np.float32)
Y = np.array(Ys, np.int64)
print(f"  collected {len(Y)} free states "
      f"(train {N_TR}, test {N_TEST}; x2 {X2.shape[1]}d, x1 {X1.shape[1]}d) "
      f"wall={round(time.time()-t0,1)}s", flush=True)

# 当前线性 W_out 在线参照：对同一批自由态，用模型自带读头打分（仅测试子集可比）
ref = 0
for i in range(N_TR, len(X2)):
    logit = torch.mv(m.W_out.t(), torch.as_tensor(X2[i], device="cuda")) + m.b_out
    ref += int(logit.argmax().item()) == Y[i]
print(f"(a) 在线线性 W_out 读头 acc(test) = {ref / N_TEST:.4f}", flush=True)

xtr, ytr = X2[:N_TR], Y[:N_TR]
xte, yte = X2[N_TR:], Y[N_TR:]

def acc_of_score(score_f):
    pred = score_f(xte).argmax(1)
    return float((pred == yte).mean())

# (b) 岭回归（线性上限）
d = xtr.shape[1]
lamb = 1.0
A = xtr.T @ xtr + lamb * np.eye(d, dtype=np.float32)
b = (xtr.T @ np.eye(256, dtype=np.float32)[ytr])
Wr = np.linalg.solve(A, b)
score = lambda Xx: Xx @ Wr
print(f"(b) x2 岭回归 acc = {acc_of_score(score):.4f}", flush=True)

def train_mlp(Xin_tr, Y_tr, Xin_te, hid=256, lr=0.01, epochs=15):
    # 单隐层 MLP CE，离线 torch GD（探针用，非 SR-PC 局部实现）
    D = Xin_tr.shape[1]
    W1 = torch.zeros((D, hid), device="cuda"); torch.nn.init.normal_(W1, 0, 0.05)
    b1 = torch.zeros(hid, device="cuda")
    W2 = torch.zeros((hid, 256), device="cuda"); torch.nn.init.normal_(W2, 0, 0.05)
    b2 = torch.zeros(256, device="cuda")
    Xt = torch.as_tensor(Xin_tr, device="cuda").float()
    Yt = torch.as_tensor(Y_tr, device="cuda").long()
    lr *= Xin_tr.shape[0] / 256.0   # 近似 batch 缩放
    for ep in range(epochs):
        idx = torch.randperm(Xt.shape[0])
        for s in range(0, Xt.shape[0], 256):
            bi = idx[s:s+256]
            xb, yb = Xt[bi], Yt[bi]
            t1 = torch.relu(xb @ W1 + b1)
            logit = t1 @ W2 + b2
            logsm = logit - torch.logsumexp(logit, 1, keepdim=True)
            loss = logsm.gather(1, yb[:, None]).mean()
            g2 = torch.exp(logsm); g2[torch.arange(len(yb)), yb] -= 1   # dL/dlogit (mean grad)
            g2 = g2 / len(yb)
            gw2 = t1.t() @ g2
            gb2 = g2.sum(0)
            gt1 = (g2 @ W2.t()) * (t1 > 0)
            gw1 = xb.t() @ gt1
            gb1 = gt1.sum(0)
            for p, g in ((W1, gw1), (b1, gb1), (W2, gw2), (b2, gb2)):
                p -= lr * g
    Xe = torch.as_tensor(Xin_te, device="cuda").float()
    score = lambda Xx: torch.relu(torch.as_tensor(Xx, device="cuda").float() @ W1 + b1) @ W2 + b2
    return score

def acc_mlp(Xin_tr, Xin_te, tag):
    s = train_mlp(Xin_tr, ytr, Xin_te)
    pred = s(Xin_te).argmax(1).cpu().numpy()
    print(f"({tag}) acc = {float((pred==yte).mean()):.4f}", flush=True)

acc_mlp(xtr, xte, "c) x2 MLP")
acc_mlp(X1[:N_TRAIN], X1[N_TRAIN:], "d) x1 MLP")
acc_mlp(np.concatenate([xtr, X1[:N_TRAIN]], 1), np.concatenate([xte, X1[N_TRAIN:]], 1),
        "e) [x1;x2] MLP")
print("== probe done ==", flush=True)