"""H2 决定性诊断：识别编码器同结构 + 真 BP(CE) 的可达上限 oracle。

目的：分离 0.33(多种局部 credit) vs 0.42(孪生) 的墙到底在
  (A) credit 算法（局部 vs 全局），还是 (B) 识别结构/协议（kWTA 截断、
      块紧凑 W1、单线性读头、relu' 门控 kill）。
方法与对照 - 单一变量：前馈路径与 train_step_recog 逐位一致
    x1r = relu(permute(W1c,0,2,1)@x0rf*s1) → kWTA(存活列 relu' 门控)
    x2r = relu(W2t@x1r_flat)             → kWTA
    logit = W_out.t@x2r + b_out  (线性读头, 与 eval_recog 同)
    评估走 eval_recog（写回 LMPCNg 后调用，完全一致编码路径）。
训练：真反传 softmax CE + Adam，跨层全局链式 credit（= 孪生口径）。
统计：
  - 若 oracle 仍卡 ~0.33 => 墙在结构(KWTA/块紧凑/单线性读头)，纯调 credit 无解，
    需放宽结构（损 MAC/bit 目标）或证明该架构本身 <0.42 上限。
  - 若 oracle 明显 >0.33 趋近 0.42 => 墙在局部 credit 实现，值得继续精换 credit。
忠实保留：relu、kWTA(forward)+straight-through(存活列 relu' 梯度)、块紧凑 W1、
s1 缩放、线性读头。不保留：列(col)norm / 谱界 / 结构稀疏 theta 掩码（这些是局部
稳定措施，孪生无，掺入会污染"真 BP 上限"）。

用法（4090）：
    python -B scripts/h2_oracle_bp.py 1856 180000 0.0003
      argv: h steps lr
写 results_e2_gpu_eta/h2_oraclebp_{h}_lr{lr}.json
"""
from __future__ import annotations
import json, os, sys, time
ROOT = "/root/srpc_e2/srpc_src"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu import LMPCNg   # noqa: E402  模块级 set_grad_enabled(False)，须在其后重开
torch.set_grad_enabled(True)          # oracle 用真反传（诊断用，非交付免反传产物）

RES = "results_e2_gpu_eta"
os.makedirs(RES, exist_ok=True)

h = int(sys.argv[1]) if len(sys.argv) > 1 else 1856
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 180_000
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
eval_every = int(sys.argv[4]) if len(sys.argv) > 4 else 30_000
assert h % 16 == 0, f"bad h={h}"

cfg = E2Config()
cfg.eval_every = eval_every
cfg.recog_on = True
cfg.recog_lr = 0.0005
tag = f"lr{lr}"
name = f"h2_oraclebp_{h}_{tag}"
jpath = os.path.join(RES, f"{name}.json")

# ---- 用 LMPCNg 拿初始结构张量（块紧凑 W1c、W2、W_out 同 seed 同初始）----
seed = 0
corpus = ByteCorpus(cfg)
rng = np.random.default_rng(seed * 977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")
W1c = m.W1c.clone().detach().requires_grad_(True)     # (W,per,per) 块
W2 = m.W2.clone().detach().requires_grad_(True)       # (h,h)  -> x2=relu(W2t@x1_flat)
W_out = m.W_out.clone().detach().requires_grad_(True)  # (h,C)
b_out = m.b_out.clone().detach().requires_grad_(True)
s1 = m.s1
dev = m.device

opt = torch.optim.Adam([W1c, W2, W_out, b_out], lr=lr)


def kwta_forward_gate(x: torch.Tensor, frac: float):
    """forward 取 topk(存活列保留原值)，梯度只在存活列上 relu' 门控(=1)。
    死亡列梯度为 0（straight-through，等同孪生在存活子空间的最优）。"""
    x = x.reshape(-1)
    n = x.numel()
    k = max(1, int(round(frac * n)))
    vals, idx = torch.topk(x, k)
    gate = torch.zeros_like(x)
    gate[idx] = 1.0
    return (x * gate).reshape(-1)  # topk 后置零；grad := gate（存活列 1）


def enforce_block(w_block: torch.Tensor, pad_idx: torch.Tensor, per: int):
    """块紧凑约束：pad 位置权重置零（在带 grad 图上 local_ 抹平）。"""
    with torch.no_grad():
        w_block.local_()[:, pad_idx, :] *= 0.0
    return w_block


def forward(x0_np: np.ndarray):
    x0p = np.concatenate([np.asarray(x0_np).ravel(), np.zeros(1, np.float32)])
    x0rf = torch.as_tensor(x0p, device=dev)[m.idx_rf].float()       # (W,per)
    # x1 = relu(W1cT@x0rf * s1)，W1c 存 (W,per,per)=W1c，用 perm(0,2,1)
    pre1 = torch.matmul(W1c.transpose(1, 2), x0rf.unsqueeze(-1)).squeeze(-1) * s1
    x1r = torch.clamp(pre1, 0.0, cfg.x_max)
    x1k = kwta_forward_gate(x1r, cfg.kwta_frac).reshape(m.W, m.per)
    x1flat = x1k.reshape(-1)
    pre2 = torch.mv(W2.t(), x1flat)
    x2r = torch.clamp(pre2, 0.0, cfg.x_max)
    x2k = kwta_forward_gate(x2r, cfg.kwta_frac)
    logit = torch.mv(W_out.t(), x2k) + b_out
    return logit, x2k


def acc_eval():
    """写回 LMPCNg 后走 eval_recog（与局部机制完全一致评估路径）。"""
    with torch.no_grad():
        m.W1c = W1c.detach()
        m.W1cT = W1c.detach().transpose(1, 2).contiguous()
        m.W2 = W2.detach()
        m.W_out = W_out.detach()
        m.b_out = b_out.detach()
    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
    tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
    bpc, acc = m.eval_recog(rep_x, rep_y, tau)
    return bpc, acc, tau


curve = []
twin_curve = {}
twin_json = f"results_e2_gpu/twin_{h}.json"
if os.path.exists(twin_json):
    for pt in json.load(open(twin_json))["curve"]:
        twin_curve[int(pt["step"])] = pt["acc"]

full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
print(f"== {name}: {steps} steps lr={lr} err=train_step_recog-forward + true-BP CE ==", flush=True)
t0 = time.time()
for t in range(steps):
    x, y = corpus.train_window(t)
    yoh = np.zeros(256, np.float32); yoh[y] = 1.0
    err_ = None
    opt.zero_grad()
    logit, _ = forward(x.ravel())
    p = torch.softmax(logit / cfg.readout_tau, dim=0)
    loss = -torch.log(p[y] + 1e-12)
    loss.backward()
    # 块紧凑：pad 行梯度抹零（结构约束，不参与学习）
    with torch.no_grad():
        if W1c.grad is not None:
            W1c.grad[m.pad] = 0.0
    opt.step()
    # 块紧凑就地抹平（防止 pad 行漂移）
    with torch.no_grad():
        W1c[m.pad] = 0.0
    if (t + 1) % eval_every == 0 or t == steps - 1:
        bpc, acc, tau = acc_eval()
        ref = None
        for k in sorted(twin_curve.keys()):
            if k >= t + 1:
                ref = twin_curve[k]; break
        pt = dict(step=t + 1, bpc=float(bpc), acc=float(acc), tau=float(tau),
                  twin_acc=ref, gap=(float(acc) - ref) if ref is not None else None,
                  w2smax="NA", wall=round(time.time() - t0, 1))
        curve.append(pt)
        with open(jpath, "w") as f:
            json.dump(dict(h=h, steps=steps, lr=lr, curve=curve), f, indent=1, default=float)
        print(f"  step {t+1}/{steps}: bpc={bpc:.3f} acc={acc:.4f} twin@{t+1}={ref} "
              f"gap={pt['gap']} tau={tau} wall={pt['wall']}s", flush=True)
print(f"== {name} done -> {jpath} ==", flush=True)