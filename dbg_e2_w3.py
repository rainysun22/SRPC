"""E2 诊断：W3 读出头学习速率 × 读出对比度 × τ 标定。

假设：eta_w=0.005 + 256 类分摊 + e2 幅度小 => W3 几乎不动，读出 logits
对比度低（argmax 0.185 但 softmax 均匀）。对比 eta_w ∈ {0.005,0.02,0.05}
的小预算训练，测 W3 移动量、x2 幅度、τ 标定后 BPC/acc。
"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax, train_lmpcn

cfg = E2Config()
corpus = ByteCorpus(cfg)
print(f"unigram BPC (val): {corpus.unigram_bpc:.3f}")

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]

# 钳制类原型分离度：对训练窗口子集，逐类收集钳制 x2，算类内/类间 cos
def class_prototype_analysis(m, n=400):
    px, py = [], []
    for t in range(n):
        x, y = corpus.train_window(90000 + t)
        px.append(x.ravel())
        py.append(y)
    # 逐类原型 = 该类的平均 x2（钳制）
    yoh = np.zeros(256, np.float32)
    cents, counts = {}, {}
    for i in range(n):
        y = py[i]
        yoh[:] = 0.0
        yoh[y] = 1.0
        m._infer(px[i], yoh, None, clamp=True)
        c = m._x2
        cents[y] = cents.get(y, 0.0) + c
        counts[y] = counts.get(y, 0) + 1
    # 用出现 >= 2 次的类算原型（不足则跳过）
    vecs, cs = [], []
    for y, s in cents.items():
        if counts[y] >= 2:
            vecs.append(s / counts[y])
            cs.append(y)
    if len(vecs) < 2:
        return None
    V = np.array(vecs)                       # (n_cls, h)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    S = Vn @ Vn.T                           # 类原型间 cos 矩阵
    i = np.arange(len(cs))
    S[i, i] = 0.0
    return dict(n_cls=len(cs),
                inter_mean=float(np.abs(S).mean()),
                inter_max=float(np.abs(S).max()),
                proto_norm=float(np.linalg.norm(V, axis=1).mean()))


for ew in (0.005, 0.02, 0.05):
    r = train_lmpcn(cfg, 768, 5000, corpus, eta_w=ew, iters=24, seed=0)
    m = r["model"]
    # W3 移动量：对照初始 W3（重新构造同 seed 初始）
    rng = np.random.default_rng(0 * 977 + 5)
    m0 = LMPCN(cfg, 768, rng, eta_w=ew, iters=24)
    dw3 = float(np.linalg.norm(m.W3 - m0.W3)) / float(np.linalg.norm(m0.W3))
    pa = class_prototype_analysis(m)
    pa_s = (f"cls={pa['n_cls']} inter_mean={pa['inter_mean']:.3f} "
            f"inter_max={pa['inter_max']:.3f} proto_norm={pa['proto_norm']:.3f}"
            if pa else "n/a")
    print(f"eta_w={ew}: bpc={r['bpc']:.3f} acc={r['acc']:.3f} "
          f"tau={r['tau']} | W3 move={dw3:.4f} "
          f"x2mean={float(m._x2.mean()):.4f} | {pa_s}")
    # τ 网格上的 BPC（看标定作用）
    bps = []
    for tau in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        b, a = m.eval_batch(rep_x, rep_y, tau)
        bps.append(f"{tau}:{b:.2f}/{a:.2f}")
    print("   τ(bpc/acc): " + " ".join(bps))
