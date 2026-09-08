"""E2 诊断 5：独立线性读出头（阶段 B §7.7 翻译器谱系）。

W3 LS 收敛读出在类非均匀下丢失 unigram 先验（列幅度被 LS 归一）。
改为：W_out (h,256) 线性头 + bias，从自由推断的 x2 读出（评估一致性，
E1 hebb_free 同思想）。W_out 列幅度 ∝ 更新次数 ∝ 频率 => 承载先验。
训练 5k 后冻结，读出头 LMS 再训 5k。
"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, train_lmpcn

cfg = E2Config()
corpus = ByteCorpus(cfg)
print(f"unigram BPC: {corpus.unigram_bpc:.3f}")
cnt = np.bincount(corpus.train, minlength=256).astype(np.float64)
p = cnt / cnt.sum()

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]

# 1) 核心训练（w3 unit 保留类原型塑形）
cfg.w3_norm = "unit"
cfg.w3_scale = 1.0
r = train_lmpcn(cfg, 768, 5000, corpus, eta_w=0.01, iters=24, seed=0)
m = r["model"]
print(f"core: bpc={r['bpc']:.3f} acc={r['acc']:.3f}")


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# 2) 读出头 LMS 训练（自由推断 x2，评估一致性）
W_out = (np.random.default_rng(0).normal(0, 0.01, (768, 256))
         ).astype(np.float32)
b_out = np.zeros(256, np.float32)
eta_ro = 0.05
m.learning = False
for t in range(5000):
    x, y = corpus.train_window(100000 + t)
    m._infer(x.ravel(), None, None, clamp=False)
    x2 = m._x2
    logit = W_out.T @ x2 + b_out
    p = softmax(logit / 0.1)
    # LMS：全列更新（读出头内部局部，无反传）
    err = p.copy()
    err[y] -= 1.0
    W_out -= eta_ro * np.outer(x2, err)
    b_out -= eta_ro * err
m.learning = True


def eval_ro(tau):
    nll = acc = 0.0
    for i in range(len(rep_y)):
        m._infer(rep_x[i].ravel(), None, None, clamp=False)
        logit = W_out.T @ m._x2 + b_out
        p = softmax(logit / tau)
        nll -= np.log2(max(p[rep_y[i]], 1e-12))
        acc += float(p.argmax() == rep_y[i])
    return float(nll / len(rep_y)), float(acc / len(rep_y))


for tau in (0.02, 0.05, 0.1, 0.2, 0.5):
    b, a = eval_ro(tau)
    print(f"  readout tau={tau}: bpc={b:.3f} acc={a:.3f}")
coln = np.linalg.norm(W_out, axis=0)
print(f"  W_out colnorm corr(freq)={np.corrcoef(coln, p)[0,1]:.3f} "
      f"hi={coln[np.argsort(p)[-10:]].mean():.4f} "
      f"lo={coln[np.argsort(p)[:10]].mean():.4f}")
