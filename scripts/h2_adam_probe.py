"""H2 v11 局部 Adam 探针：找出 acc 恒定 0.133/ev=0.0 的根因。

假设：累进列归一(colnorm) 与 Adam 逐坐标归一互斥，把 W2 方向推成负和 → x2r 塌缩。
本探针：小规模(h=256) 跑 train_step_recog_adam，每 1000 步打印
  ||x2r|| / W2 列均范数 / ev / acc，并做 controlA(colnorm 关闭，内联重训) 对照。

用法(远程): python -B scripts/h2_adam_probe.py
写入 results_e2_gpu_eta/h2_adam_probe.json
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
from srpc.lmgpu import LMPCNg, _colnorm_t

RES = "results_e2_gpu_eta"
os.makedirs(RES, exist_ok=True)
h = 256
steps = 5000
cfg = E2Config()
cfg.w2_cap = True
cfg.w2_cap_every = 1
cfg.w2_smax_cap = 2.5
cfg.w2_pow_iters = 8
cfg.recog_on = True
cfg.recog_lr = 0.0005
cfg.recog_adam = True
cfg.recog_adam_lr = 3e-4
cfg.recog_rounds = 1

corpus = ByteCorpus(cfg)
full = len(corpus.train) - cfg.context - 1
steps = min(steps, full)
vx, vy = corpus.val_x, corpus.val_y
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]

rng = np.random.default_rng(977 + 5)
m = LMPCNg(cfg, h, rng, eta_w=0.01, iters=12, device="cuda")

NOREG = os.environ.get("NOREG", "1") == "1"   # 1=v12 无归一器; 0=v11 原版


def step(x0_np, y):
    """v11/v12 切换：credit 同，正则阶段要不要 colnorm/w2cap/m2/s1。"""
    dev = m.device
    yoh = np.zeros(m.C, np.float32); yoh[y] = 1.0
    yoh_t = torch.as_tensor(yoh, device=dev)
    if getattr(m, "_adams", None) is None:
        m._adams = {}
        for k in ("W1c", "W2"):
            P = getattr(m, k)
            m._adams[k] = {"m": torch.zeros_like(P),
                           "v": torch.zeros_like(P), "t": 0}
    m._recog_forward(x0_np)
    x1r, x2r, x0rf = m._x1r, m._x2r, m._x1rf
    g2 = m._ro_train(x2r, yoh_t)
    # v12b: 幸存集 = kWTA+relu 实际保留的正单元 (x2r>0)，即真 BP 穿过 relu∘kWTA 的口径
    g2g = g2 * (x2r > 0.0).to(x2r.dtype)
    dW2 = torch.outer(x1r.reshape(-1), g2g)
    g1 = torch.mv(m.W2, g2g).reshape(m.W, m.per)
    pre1 = (torch.matmul(m.W1cT, x0rf.unsqueeze(-1))
            .squeeze(-1) * m.s1)
    g1g = g1 * (x1r > 0.0).to(g1.dtype)
    dW1c = torch.einsum("bi,bj->bij", x0rf, g1g) * m.s1
    b1, b2, eps, lr = (cfg.recog_adam_b1, cfg.recog_adam_b2,
                       cfg.recog_adam_eps, cfg.recog_adam_lr)
    for name, P, G in (("W2", m.W2, dW2), ("W1c", m.W1c, dW1c)):
        st = m._adams[name]; st["t"] += 1
        mh = st["m"].mul_(b1).add_(G, alpha=1 - b1) / (1 - b1 ** st["t"])
        vh = st["v"].mul_(b2).add_(G * G, alpha=1 - b2) / (1 - b2 ** st["t"])
        P -= lr * mh / (torch.sqrt(vh) + eps)
    if NOREG:
        # v12: 仅块紧凑 pad 掩码 + W1cT 同步（同 oracle；无 colnorm/w2cap/m2/s1 重标）
        m.W1c[m.pad] = 0.0
        m.W1cT = m.W1c.transpose(1, 2).contiguous()
    else:
        n = torch.linalg.vector_norm(m.W1c, dim=1, keepdim=True)
        scale = torch.where(n > 1.0, 1.0 / n.clamp_min(1e-12),
                            torch.ones_like(n))
        m.W1c = m.W1c * scale
        m.W1c[m.pad] = 0.0
        m.W1c *= m.s1
        m.W1cT = m.W1c.transpose(1, 2).contiguous()
        m.W2 *= m.m2
        m.W2 = _colnorm_t(m.W2)
        m._maybe_cap_w2()

def norm_stats():
    with torch.no_grad():
        w2c = torch.linalg.vector_norm(m.W2, dim=0)
        return (m.W2.numel(),
                float(w2c.mean()), float(w2c.max()),
                float(torch.linalg.vector_norm(m.W1c)),
                float(torch.linalg.vector_norm(m.W2)))

print(f"== probe start h={h} steps={steps} ==", flush=True)
t0 = time.time()
last = {}
for t in range(steps):
    x, y = corpus.train_window(t)
    m.train_step_recog_adam(x, y)
    if (t + 1) % 1000 == 0 or t == steps - 1:
        tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
        bpc, acc = m.eval_recog(rep_x, rep_y, tau)
        with torch.no_grad():
            m._recog_forward(rep_x[0].ravel())
            x2n = float(torch.linalg.vector_norm(m._x2r))
        _, w2c_mean, w2c_max, w1n, w2n = norm_stats()
        rec = dict(step=t + 1, bpc=float(bpc), acc=float(acc),
                   ev=m.recog_event_rate(), x2n=x2n,
                   w2_colmean=round(w2c_mean, 4), w2_colmax=round(w2c_max, 4),
                   w1n=round(w1n, 3), w2n=round(w2n, 3),
                   wall=round(time.time() - t0, 1))
        last = rec
        print(f"  {rec}", flush=True)
with open(os.path.join(RES, "h2_adam_probe.json"), "w") as f:
    json.dump(dict(h=h, steps=steps, final=last), f, indent=1, default=float)
print("== probe done ==", flush=True)