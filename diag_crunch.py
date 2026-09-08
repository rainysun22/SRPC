"""自由推断稳定性诊断（阶段 E2 GPU 登顶失稳）——E2 GPU SUMMIT §5.4 前置研究。

问题：h>=1856 档在超长预算（~33 万步）后，free-infer（无 yoh 引导）进入饱和死锁。
本脚本：训练 h=1856 至目标步，在关键步存 checkpoint，并在每个 key step 上运行
free-infer 逐迭代能量/收缩/饱和探针，记录 JSON，用于判别"能量不再单调下降 / 收缩性丢失"假说。

用法（远程）：python -B diag_crunch.py <h> <target_steps> [probe_steps_csv]
"""
import sys, os, json, time
ROOT = "/root/srpc_e2/srpc_src"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lm import ByteCorpus
from srpc.lmgpu_graph import LMPCNgG

h = int(sys.argv[1])
target = int(sys.argv[2])
probe_steps = ([int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
               else [200000, 250000, 280000, 300000, 315000, 325000, 335000, 350000])

cfg = E2Config()
cfg.eval_every = target + 1
corpus = ByteCorpus(cfg)
vx, vy = corpus.val_x, corpus.val_y
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
WIN = 40

def energy(m, x0_flat, k_iters=12):
    """逐迭代能量/收缩探针：复刻 free-infer（clamp=False, do_out=True）。返回逐迭代 dict。"""
    cfg0 = m.cfg
    a, b_ = cfg0.alpha, cfg0.beta
    th = cfg0.theta_event
    W1c, W1cT, W2, W3 = m.W1c, m.W1cT, m.W2, m.W3
    x0p = np.concatenate([np.asarray(x0_flat).ravel(), np.zeros(1, np.float32)])
    x0rf = torch.as_tensor(x0p, device=m.device)[m.idx_rf]
    x1g = torch.zeros((m.W, m.per), dtype=torch.float32, device=m.device)
    x2 = torch.zeros(m.h, dtype=torch.float32, device=m.device)
    x3 = torch.zeros(m.C, dtype=torch.float32, device=m.device)
    out = []
    for _ in range(k_iters):
        pred0 = torch.matmul(W1c, x1g.unsqueeze(-1)).squeeze(-1)
        e0c = x0rf - pred0
        e1 = x1g.reshape(-1) - torch.mv(W2, x2)
        e2 = x2 - torch.mv(W3, x3)
        u1 = torch.matmul(W1cT, e0c.unsqueeze(-1)).squeeze(-1) * m.s1
        u1 = (b_ * u1.reshape(m.W, m.per) - a * e1.reshape(m.W, m.per))
        u2 = b_ * torch.mv(W2.t(), e1) - a * e2
        g1 = u1.abs() > th
        g2 = u2.abs() > th
        x1g = (x1g + m.et1 * u1 * g1).clamp(0.0, cfg0.x_max)
        x2 = (x2 + m.et2 * u2 * g2).clamp(0.0, cfg0.x_max)
        x3 = (x3 + cfg0.eta_out * torch.mv(W3.t(), e2)).clamp(0.0, 1.0)
        F = 0.5 * (e0c.square().sum() + e1.square().sum() + e2.square().sum())
        out.append({
            "k": _ + 1,
            "F": float(F),
            "e0c": float(e0c.square().sum()),
            "e1": float(e1.square().sum()),
            "e2": float(e2.square().sum()),
            "u": float((u1.square().sum() + u2.square().sum())),
            "x2max": float(x2.max()),
            "x2sat": float((x2 > 4.99).sum().item()),
            "x1max": float(x1g.max()),
        })
    return out

def probe(m):
    rows = []
    for i in range(WIN):
        rows.append(energy(m, rep_x[i].ravel()))
    return rows

def bpc_fixed(m, tau=0.2):
    m.learning = False
    nll = acc = 0.0
    for i in range(WIN):
        m._infer(rep_x[i].ravel(), None)
        logit = torch.mv(m.W_out.t(), m._x2) + m.b_out
        p = torch.softmax(logit / tau, dim=0)
        py = float(p[rep_y[i]])
        nll -= np.log2(max(py, 1e-12))
        acc += float(p.argmax().item() == rep_y[i])
    m.learning = True
    return float(nll / WIN), float(acc / WIN)

m = LMPCNgG(cfg, h, np.random.default_rng(5), eta_w=0.01, iters=12, device="cuda")
print(f"cfg: x_max={cfg.x_max} alpha={cfg.alpha} beta={cfg.beta} iters={cfg.settle_iters} "
      f"eta_out={cfg.eta_out} et1={m.et1:.4f} et2={m.et2:.4f}", flush=True)

results = {}
t0 = time.time()
saved = set()
for t in range(target):
    x, y = corpus.train_window(t)
    m.train_step(x, y)
    s = t + 1
    if s in probe_steps:
        f = bpc_fixed(m)
        pr = probe(m)   # pr[win] = list of per-k dicts
        lastK = [win[-1] for win in pr]
        res = {"step": s, "bpc_tau02": f[0], "acc_tau02": f[1],
               "x2sat_final_max": max(d["x2sat"] for d in lastK),
               "x2max_final_max": max(d["x2max"] for d in lastK),
               "F_final": np.mean([d["F"] for d in lastK]),
               "F_last2_delta": np.mean([d["F"] for d in lastK]) -
                                np.mean([d["F"] for d in [win[-4] for win in pr]]),
               "diverged": any(d["x2sat"] > 0 for d in lastK),
               "w2_smax": float(torch.linalg.svd(m.W2, full_matrices=False)[1][0]),
               "w1_smax": float(torch.linalg.svd(
                   m.W1c.reshape(-1, m.W1c.shape[-1]),
                   full_matrices=False)[1][0])}
        results[str(s)] = res
        # 保存 checkpoint
        sd = {k: v.detach().cpu() for k, v in m.__dict__.items()
              if isinstance(v, torch.Tensor)}
        cp = f"/root/srpc_e2/srpc_src/results_e2_gpu/diag_{s}.pt"
        torch.save({"step": s, "model": sd}, cp)
        print(f"[{time.time()-t0:.0f}s] PROBE s={s} bpc={f[0]:.3f} acc={f[1]:.3f} "
              f"x2sat={results[str(s)]['x2sat_final_max']}/40 saved", flush=True)
        saved.add(s)
with open(f"/root/srpc_e2/results_e2_gpu_diag_h{h}.json", "w") as fp:
    json.dump({"h": h, "target": target, "probe": results,
               "wall_s": time.time() - t0}, fp, indent=1)
print("DEADCLOCK_DIAG_DONE h=%d target=%d saved=%s wall=%.0f"
      % (h, target, sorted(saved), time.time() - t0), flush=True)