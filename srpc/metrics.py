"""Phase-0 评估指标（对应 7.4 验证指标 / 7.5 验收标准）。"""
from __future__ import annotations

import numpy as np


def ema(values, alpha: float) -> np.ndarray:
    """前向指数滑动平均。"""
    out = np.empty(len(values), dtype=float)
    m = float(values[0])
    for i, v in enumerate(values):
        m += alpha * (float(v) - m)
        out[i] = m
    return out


def linreg_slope(y) -> float:
    """线性回归斜率（用于"误差随交互单调下降"检验）。"""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def best_cosine(patterns: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """每个真实模式在感受野集合中匹配到的最大余弦相似度。

    patterns: (D, M) 真实源/特征模式；rf: (D, N) 模型感受野（如 W10 列）。
    """
    pn = patterns / np.maximum(np.linalg.norm(patterns, axis=0, keepdims=True), 1e-8)
    rn = rf / np.maximum(np.linalg.norm(rf, axis=0, keepdims=True), 1e-8)
    return (pn.T @ rn).max(axis=1)


def nmi(a, b) -> float:
    """离散标签间的归一化互信息。"""
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    n = len(a)
    mi = 0.0
    for x in ua:
        pa = (a == x).mean()
        for y in ub:
            pb = (b == y).mean()
            pab = ((a == x) & (b == y)).mean()
            if pab > 0:
                mi += pab * np.log(pab / (pa * pb))
    ha = -sum(pa * np.log(pa) for pa in [(a == x).mean() for x in ua] if pa > 0)
    hb = -sum(pb * np.log(pb) for pb in [(b == y).mean() for y in ub] if pb > 0)
    if ha <= 0 or hb <= 0:
        return 0.0
    return float(mi / np.sqrt(ha * hb))


def nmi_perm_pvalue(a, b, n_perm: int = 200, rng=None) -> tuple[float, float]:
    """置换检验：概念层标签与真实上下文标签的 NMI 是否显著高于随机。

    返回 (p 值, 随机置换 NMI 均值)。
    """
    rng = rng or np.random.default_rng(0)
    obs = nmi(a, b)
    b = np.asarray(b)
    perms = [nmi(a, rng.permutation(b)) for _ in range(n_perm)]
    p = (1 + sum(1 for v in perms if v >= obs)) / (1 + n_perm)
    return float(p), float(np.mean(perms))


def recovery_stats(err_ema: np.ndarray, t0: int, base_win: int = 150,
                   mult: float = 1.30, sustain: int = 60,
                   relearn_lo: int = 300, relearn_hi: int = 2500) -> dict:
    """扰动后的恢复统计（修正指标）。

    - base: 扰动前 base_win 步的 EMA 误差均值；threshold = base*mult
    - peak: 扰动后峰值（考虑 EMA 滞后，在 peak 窗口内取最大）
    - recovery_steps: 从扰动时刻起，EMA 误差自峰值回落后持续 sustain 步
      低于 threshold 所需步数（若误差从未超过 threshold 则为 0）
    - relearn_error: 扰动后 [relearn_lo, relearn_hi) 窗口内的平均误差
      （A/B 主指标：衡量重学速度与深度，规避"各自基线不同"的不公平）
    - excess_error: 扰动后超出基线的累积误差
    """
    t0 = int(t0)
    base = float(err_ema[max(0, t0 - base_win):t0].mean())
    thr = base * mult
    post = err_ema[t0:]
    pw = min(max(relearn_hi, 1), len(post))
    t_peak = int(np.argmax(post[:pw]))
    peak = float(post[t_peak])
    if peak <= thr:
        steps = 0
    else:
        ok = (post[t_peak:] <= thr).astype(float)
        win = min(sustain, max(1, len(ok) // 2))
        c = np.convolve(ok, np.ones(win), mode="valid")
        good = np.nonzero(c == win)[0]
        steps = int(t_peak + good[0]) if len(good) else len(post)
    lo, hi = min(relearn_lo, len(post)), min(relearn_hi, len(post))
    relearn = float(post[lo:hi].mean()) if hi > lo else float("nan")
    return dict(
        base=base,
        threshold=thr,
        recovery_steps=steps,
        peak=peak,
        relearn_error=relearn,
        excess_error=float(np.maximum(post - base, 0.0).sum()),
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(a @ b / (na * nb))
