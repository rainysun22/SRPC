"""E2 诊断 3：unigram 先验缺失假设。

列归一化抹掉 W3 列幅度 => 读出 logits 不反映字节频率 => 连 unigram
都学不到（BPC 5.2+ > unigram 4.8）。测：
  A) eval 时 logit += lam·log_unigram（读出偏置，标准 LM 输出 bias）
  B) W3 归一化改 clip（列幅度自由生长，保留方向稳定）
"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, _softmax, train_lmpcn

cfg = E2Config()
corpus = ByteCorpus(cfg)
unigram = corpus.unigram_bpc
print(f"unigram BPC: {unigram:.3f}")
# 读出偏置源（训练段字节频率；与 unigram_bpc 同源）
cnt = np.bincount(corpus.train, minlength=256).astype(np.float64)
p = cnt / cnt.sum()
log_uni = np.log(np.maximum(p, 1e-12))

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]


def eval_with_bias(m, X, y, tau, lam):
    """收敛 x3 后加 log-unigram·lam 判分。"""
    m.learning = False
    nll = acc = 0.0
    for i in range(len(y)):
        m._infer(X[i].ravel(), None, None)
        logit = m._x3 + lam * log_uni
        p = _softmax(logit / tau)
        nll -= np.log2(max(p[y[i]], 1e-12))
        acc += float(p.argmax() == y[i])
    m.learning = True
    return float(nll / len(y)), float(acc / len(y))


# 训练 5k（w3s=0.5, iters=24, ew=0.01 为当前最优）
cfg.w3_scale = 0.5
r = train_lmpcn(cfg, 768, 5000, corpus, eta_w=0.01, iters=24, seed=0)
m = r["model"]
print(f"base: bpc={r['bpc']:.3f} acc={r['acc']:.3f}")
for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
    for tau in (0.02, 0.1, 0.5):
        b, a = eval_with_bias(m, rep_x, rep_y, tau, lam)
        if tau == 0.1 or (lam in (0.0, 0.5) and tau == 0.02):
            print(f"  lam={lam} tau={tau}: bpc={b:.3f} acc={a:.3f}")
