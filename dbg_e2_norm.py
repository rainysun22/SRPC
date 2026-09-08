"""E2 诊断 4：W3 列幅度自由生长（clip）vs 列归一（unit）。

假设：clip 模式下高频类列累积更大幅度 => 读出承载 unigram 先验，
BPC 应下穿 unigram 4.797。测量 W3 列范数与真实频率的相关。
"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, train_lmpcn, LMPCN

cfg = E2Config()
corpus = ByteCorpus(cfg)
print(f"unigram BPC: {corpus.unigram_bpc:.3f}")
cnt = np.bincount(corpus.train, minlength=256).astype(np.float64)
p = cnt / cnt.sum()
log_uni = np.log(np.maximum(p, 1e-12))

vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]

for w3_norm in ("unit", "clip"):
    cfg.w3_norm = w3_norm
    cfg.w3_scale = 0.5 if w3_norm == "unit" else 1.0
    r = train_lmpcn(cfg, 768, 6000, corpus, eta_w=0.01, iters=24, seed=0)
    m = r["model"]
    coln = np.linalg.norm(m.W3, axis=0)
    rho = float(np.corrcoef(coln, p)[0, 1])
    print(f"{w3_norm}: bpc={r['bpc']:.3f} acc={r['acc']:.3f} tau={r['tau']} "
          f"| colnorm mean={coln.mean():.3f} max={coln.max():.3f} "
          f"corr(colnorm, freq)={rho:.3f}")
    # 高频 10 字节与低频 10 字节的平均列范数
    hi = np.argsort(p)[-10:]
    lo = np.argsort(p)[:10]
    print(f"    colnorm hi={coln[hi].mean():.3f} lo={coln[lo].mean():.3f}")
