"""W2 周期谱截断修复验证（阶段 E2 GPU 登顶失稳，对因修复 2026-09-08）。

对照：results_e2_gpu_diag_h1856.json（未修复控制，345k 处 W2 σmax 4.8->37.6，
BPC 崩溃到 39.9、x2 全饱和 deadlock）。

mode=recover：加载崩溃 checkpoint（diag_345000.pt，σmax≈37.6）为起点，
  开启 w2_cap=True 继续训练 N 步，周期探 W2 σmax / 自由推断 BPC / 发散计数，
  证明修复把已崩模型拉回健康（收缩性恢复）。
mode=fresh  ：与对照同 seed 全新 350k 全预算训练（w2_cap=True），在 330/335/
  340/345/350k 探针，证明修复后全程不崩（判据 1 重判 / 对照 A-B）。

用法（4090）：python -B val_fix_retrain.py recover|fresh
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

MODE = sys.argv[1] if len(sys.argv) > 1 else "recover"
h = 1856
cfg = E2Config()
cfg.eval_every = 10**9
cfg.w2_cap = True
cfg.w2_cap_every = 1      # 每步截断：失稳是 ~1000 步内 4.8->37 的快速正反馈，
                         # 每 2000 步截不住（复现：cap=5/2000 步仍在 335k 发散）。
                         # 每步 cap=5 把 σmax 钉在发散阈值以下，杜绝进入饱和死锁盆。
cfg.w2_smax_cap = 5.0     # 健康 σmax 峰值 ~4.8，cap=5 极低于发散阈值、高于健康 4.81
cfg.w2_pow_iters = 8      # 幂迭代次数（估计 W2 顶奇异值，O(n²) matvec，每步开销可忽略）
corpus = ByteCorpus(cfg)
vx, vy = corpus.val_x, corpus.val_y
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]
WIN = 40


def build():
    return LMPCNgG(cfg, h, np.random.default_rng(5), eta_w=0.01,
                   iters=12, device="cuda")


def free_stats(m):
    """自由推断整窗：BPC(τ=0.2)、acc、发散计数、末态 x2 饱和计数。"""
    m.learning = False
    nll = acc = diverge = sat = 0
    x2max = 0.0
    for i in range(WIN):
        m._infer(rep_x[i].ravel(), None)
        x2 = m._x2
        mx = float(x2.max())
        x2max = max(x2max, mx)
        if mx > 4.99:
            diverge += 1
        sat += int((x2 > 4.99).sum().item())
        logit = torch.mv(m.W_out.t(), x2) + m.b_out
        p = torch.softmax(logit / 0.2, dim=0)
        py = float(p[rep_y[i]])
        nll -= np.log2(max(py, 1e-12))
        acc += float(p.argmax().item() == rep_y[i])
    m.learning = True
    smax = float(torch.linalg.svd(m.W2, full_matrices=False)[1][0])
    return dict(bpc=nll / WIN, acc=acc / WIN, n_diverge=diverge,
                x2sat=sat, x2max=x2max, w2_smax=smax)


def probe_and_log(m, step, res, phase, t0):
    st = free_stats(m)
    st["step"] = step
    res[phase][str(step)] = st
    print(f"[{time.time()-t0:.0f}s] {phase} s={step} bpc={st['bpc']:.3f} "
          f"acc={st['acc']:.3f} n_diverge={st['n_diverge']} "
          f"x2sat={st['x2sat']} w2_smax={st['w2_smax']:.2f}", flush=True)


t0 = time.time()
res = {"h": h, "mode": MODE, "after": {}, "fresh": {}}

if MODE == "recover":
    m = build()
    sd = torch.load(f"{ROOT}/results_e2_gpu/diag_345000.pt",
                    map_location="cpu", weights_only=False)["model"]
    m.load_state(sd)            # 载入崩溃权重（W2 σmax≈37.6）
    res["start"] = free_stats(m)
    print("recover start(crashed):", res["start"], flush=True)
    CONT_START = 345000         # 崩溃位置，续训从该处继续消费语料
    for target_ext in (20000, 40000):
        while m._n_train < target_ext:
            x, y = corpus.train_window(CONT_START + m._n_train)
            m.train_step(x, y)
        probe_and_log(m, target_ext, res, "after", t0)
    probe_and_log(m, m._n_train, res, "after", t0)
else:  # fresh
    m = build()
    res["w2_smax_initial"] = res.get("w2_smax_initial", 0.0)
    target = 350000
    probe_at = {330000, 335000, 340000, 345000, 350000}
    for t in range(target):
        x, y = corpus.train_window(t)
        m.train_step(x, y)
        s = t + 1
        if s in probe_at:
            probe_and_log(m, s, res, "fresh", t0)
    probe_and_log(m, target, res, "fresh", t0)

res["wall_s"] = time.time() - t0
out = f"/root/srpc_e2/val_fix_retrain_{MODE}.json"
with open(out, "w") as fp:
    json.dump(res, fp, indent=1)
print(f"FIXVAL_{MODE.upper()}_DONE mode={MODE} wall={res['wall_s']:.0f} -> {out}",
      flush=True)
print("--- final ---", json.dumps({k: v for k, v in
      (res["after"] if MODE == "recover" else res["fresh"]).items()
      if k.isdigit()}, default=str), flush=True)