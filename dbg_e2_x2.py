"""E2 诊断 2：钳制收敛 x2 增长轨迹 × W3 列尺度 × 迭代数。

假设：x2 从 0 起步，24 迭代内只能长到 ~0.03（W3 列范数 1 下），
与 W3[:,y]（范数 1）失配 => e2 ≈ −W3[:,y] => dW3 学错方向。
打印 x2 范数随迭代轨迹，并测 W3 列尺度 {1.0, 0.5} × iters {24,48} 的
小预算训练 BPC。
"""
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, LMPCN, train_lmpcn

cfg = E2Config()
corpus = ByteCorpus(cfg)
print(f"unigram BPC: {corpus.unigram_bpc:.3f}")
vx, vy = corpus.val_x, corpus.val_y
cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
rep_x, rep_y = vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:]


def probe_x2_growth(m, n=8):
    """随机样本钳制：x2 范数随迭代轨迹（重开 _infer 循环，手动收集）。"""
    m.learning = False
    import numpy as _np
    from srpc.lm import _kwta2d, _kwta
    tracks = []
    for i in range(n):
        x, y = corpus.train_window(90000 + i)
        yoh = _np.zeros(256, _np.float32)
        yoh[y] = 1.0
        x0p = _np.concatenate([x.ravel(), _np.zeros(1, _np.float32)])
        x0rf = x0p[m.idx_rf]
        x1g = _np.zeros_like(m.x1g)
        x2 = _np.zeros_like(m.x2)
        W1c, W1cT = m.W1c, m.W1cT
        a, b_ = m.cfg.alpha, m.cfg.beta
        th = m.cfg.theta_event
        tr = []
        for _ in range(m.iters):
            pred0 = _np.matmul(W1c, x1g[:, :, None])[:, :, 0]
            e0c = x0rf - pred0
            x1 = x1g.ravel()
            e1 = x1 - m.W2 @ x2
            e2 = x2 - m.W3 @ yoh
            u1 = _np.matmul(W1cT, e0c[:, :, None])[:, :, 0] * m.s1
            u1 = b_ * u1.reshape(m.W, m.per) - a * e1.reshape(m.W, m.per)
            u2 = b_ * (m.W2.T @ e1) - a * e2
            g1 = _np.abs(u1) > th
            g2 = _np.abs(u2) > th
            x1g = _np.clip(x1g + m.et1 * u1 * g1, 0.0, m.cfg.x_max)
            x2 = _np.clip(x2 + m.et2 * u2 * g2, 0.0, m.cfg.x_max)
            tr.append(float(_np.linalg.norm(x2)))
        x1g = _kwta2d(x1g, m.cfg.kwta_frac)
        x2 = _kwta(x2, m.cfg.kwta_frac)
        tracks.append(tr)
    m.learning = True
    T = _np.array(tracks)
    return T.mean(0)


# 1) 随机初始化下的 x2 增长轨迹（it=24, W3 scale=1.0）
rng = np.random.default_rng(0 * 977 + 5)
m0 = LMPCN(cfg, 768, rng, eta_w=0.005, iters=24)
print("init  x2|| iter: " +
      " ".join(f"{v:.3f}" for v in probe_x2_growth(m0)))

# 2) 训练 3k 步后再测（0.005@24）
r = train_lmpcn(cfg, 768, 3000, corpus, eta_w=0.005, iters=24, seed=0)
print("t3000 x2|| iter: " +
      " ".join(f"{v:.3f}" for v in probe_x2_growth(r["model"])))

# 3) 尺度和迭代的交叉小实验（5k 步）
for w3s in (1.0, 0.5):
    for iters in (24, 48):
        cfg.w3_scale = w3s
        r = train_lmpcn(cfg, 768, 5000, corpus, eta_w=0.01, iters=iters,
                        seed=0)
        b, a = r["model"].eval_batch(rep_x, rep_y, 0.02)
        print(f"w3s={w3s} iters={iters}: bpc={r['bpc']:.3f} acc={r['acc']:.3f}"
              f" tau={r['tau']} | @0.02 bpc={b:.3f} acc={a:.3f} "
              f"x2mean={float(r['model']._x2.mean()):.4f}")
