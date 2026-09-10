"""H2 诊断：识别编码器表征可读上限（v5 checkpoint + 充分训练本地读头）。

目的：定位 0.33(识别编码器) vs 0.42(孪生) 的 ~0.09 差距落在表征还是读头。
方法：从训练流收集 FEAT_TR 个样本的 x1r/x2r 特征（同 _recog_forward 路径），
用本地线性 softmax 读头在特征上充分训练（多 τ 校准），测验证集 acc/BPC →
即"该模型已学到的 x2r 能承到多少可读信号"。若上限≈0.33 ⇒ 表征不足，改编码器；
若上限更高 ⇒ 在线读头耦合不足，改读头。对照孪生隐藏层可读≈孪生 acc。

用法（4090）：
    python -B scripts/h2_recog_probe.py 1856 <ckpt.pt> [feat_tr]
结果打印 x1r/x2r 在校准 τ 下的验证 acc/bpc。
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

RES = "results_e2_gpu_eta"

h = int(sys.argv[1]) if len(sys.argv) > 1 else 768
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 90_000
feat_tr = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

cfg = E2Config()
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 5.0
cfg.w2_pow_iters = 8
cfg.recog_on = True
cfg.recog_lr = 0.0005

seed = 0
corpus = ByteCorpus(cfg)

rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
# ---- 先用识别编码器训练到目标步数（无 ckpt 可用，内嵌训练）----
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
print(f"training recog encoder h={h} steps={steps} ...", flush=True)
t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_recog(x, y)
print(f"trained in {time.time()-t0:.1f}s", flush=True)

# ---- 验证集 rep 样本（与 eval_recog 同口径）----
vx, vy = corpus.val_x, corpus.val_y
test_x, test_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
n_test = len(test_y)

# ---- 训练流特征样本 ----
feat_tr = min(feat_tr, len(corpus.train) - cfg.context - 1)
Xs1 = np.zeros((feat_tr, m.W * m.per), np.float32)
Xs2 = np.zeros((feat_tr, m.h), np.float32)
Ys = np.zeros(feat_tr, np.int64)
for t in range(feat_tr):
    x, y = corpus.train_window(t)
    m._recog_forward(x.ravel())
    Xs1[t] = m._x1r.cpu().numpy().reshape(-1)
    Xs2[t] = m._x2r.cpu().numpy()
    Ys[t] = y

TX1 = np.zeros((n_test, m.W * m.per), np.float32)
TX2 = np.zeros((n_test, m.h), np.float32)
TY = np.zeros(n_test, np.int64)
for i in range(n_test):
    m._recog_forward(test_x[i].ravel())
    TX1[i] = m._x1r.cpu().numpy().reshape(-1)
    TX2[i] = m._x2r.cpu().numpy()
    TY[i] = test_y[i]

def run(name, F, Y, TX, TY, C=256, epochs=300):
    dev = "cuda"
    Ft = torch.as_tensor(np.ascontiguousarray(F), device=dev)
    Yt = torch.as_tensor(np.ascontiguousarray(Y), device=dev)
    tx = torch.as_tensor(np.ascontiguousarray(TX), device=dev)
    ty = torch.as_tensor(np.ascontiguousarray(TY), device=dev)
    d = Ft.shape[1]
    yoh = torch.zeros(len(Yt), C, device=dev)
    yoh[torch.arange(len(Yt)), Yt] = 1.0
    best = (1e9, 0.0, None)
    for tau in (0.02, 0.05, 0.1, 0.2):
        W = torch.zeros(C, d, device=dev)
        b = torch.zeros(C, device=dev)
        for _ in range(epochs):
            p = torch.softmax(Ft @ W.t() + b, dim=1)
            err = p - yoh
            W -= 0.05 * (err.t() @ Ft) / len(Yt)
            b -= 0.05 * err.mean(0)
        lp = torch.softmax(tx @ W.t() + b, dim=1)
        py = lp[torch.arange(len(ty)), ty]
        bpc = float(-torch.log2(py.clamp_min(1e-12)).mean().item())
        acc = float((lp.argmax(1) == ty).float().mean().item())
        if bpc < best[0]:
            best = (bpc, acc, tau)
    print(f"[{name}] d={d} tau={best[2]:.2f} acc={best[1]:.4f} bpc={best[0]:.3f}",
          flush=True)
    return best

run("x1r", Xs1, Ys, TX1, TY)
run("x2r", Xs2, Ys, TX2, TY)
print("done", flush=True)