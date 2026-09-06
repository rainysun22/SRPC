#!/usr/bin/env python3
"""one-hot + 长程延迟：验证 Δ=4 延迟 XOR 误差驱动 vs 纯相关。"""
import sys
from dataclasses import replace
sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import _make_sequence


def _colnorm_c(w):
    return w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-8)


def _kwta(x, frac):
    k = max(1, int(round(frac * x.size)))
    if k >= x.size:
        return x
    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


class PCNOH:
    """x0(窗口) -> x1 -> x2(one-hot 双输出)。"""
    def __init__(self, cfg, rng, mode="error"):
        self.cfg = cfg; self.mode = mode
        d0 = (cfg.delay + 1) * cfg.d_feat
        self.W1 = _colnorm_c(rng.normal(0.0, 1.0, (d0, cfg.h1)))
        self.W2 = rng.normal(0.0, 0.3, (cfg.h1, 2))
        k1 = max(1, int(round(cfg.fan_in_frac * d0)))
        m1 = np.zeros((d0, cfg.h1), dtype=bool)
        for j in range(cfg.h1):
            st = int(rng.integers(0, d0 - k1 + 1))
            m1[st:st + k1, j] = True
        k2 = max(1, int(round(cfg.fan_in_frac * cfg.h1)))
        m2 = np.zeros((cfg.h1, 2), dtype=bool)
        for j in range(2):
            m2[rng.choice(cfg.h1, size=k2, replace=False), j] = True
        self.mask1, self.mask2 = m1, m2
        self.W1 = _colnorm_c(self.W1 * m1)
        self.W2 = self.W2 * m2
        self.W1_init = self.W1.copy(); self.W2_init = self.W2.copy()
        self.x1 = np.zeros(cfg.h1); self.learning = True

    def _infer(self, x0, yoh, free_out):
        cfg = self.cfg
        a, b = cfg.alpha, cfg.beta  # 顶层拉动（类） vs 底层误差（重建）
        x2 = np.zeros(2) if free_out else yoh.copy()
        for _ in range(cfg.settle_iters):
            e0 = x0 - self.W1 @ self.x1
            e1 = self.x1 - self.W2 @ x2
            u1 = b * (self.W1.T @ e0) - a * e1
            self.x1 = np.clip(self.x1 + u1 * (np.abs(u1) > cfg.theta_event),
                              0.0, cfg.x_max)
            if getattr(cfg, "kwta_on", True):
                self.x1 = _kwta(self.x1, cfg.kwta_frac)
            if free_out:
                e1 = self.x1 - self.W2 @ x2
                x2 = np.clip(x2 + getattr(cfg, "eta_out", 0.1) * (self.W2.T @ e1), 0.0, 1.0)
        self._e0 = x0 - self.W1 @ self.x1
        self._e1 = self.x1 - self.W2 @ x2
        self._x2 = x2
        self._energy = 0.5 * float(np.dot(self._e0, self._e0) + np.dot(self._e1, self._e1))

    def _learn(self, x0, yoh):
        if not self.learning: return
        cfg = self.cfg; lr = cfg.eta_w
        g1 = self.x1 > cfg.theta_syn
        if self.mode == "error":
            dW1 = np.outer(self._e0, self.x1 * g1)
            dW2 = np.outer(self._e1, yoh)
        else:
            dW1 = np.outer(x0, self.x1 * g1)
            dW2 = np.outer(self.x1, yoh)
        self.W1 += lr * dW1; self.W2 += lr * dW2
        self.W1 *= self.mask1; self.W2 *= self.mask2
        self.W1 = _colnorm_c(self.W1)
        self.W2 = _colnorm_c(self.W2)

    def train_step(self, x0, y):
        yoh = np.array([1.0, 0.0]) if y == 0 else np.array([0.0, 1.0])
        self.x1[:] = 0.0
        self._infer(x0, yoh, False)
        self._learn(x0, yoh)

    def predict(self, x0):
        """钳制-比较：分别钳制两个候选输出，取自由能更低者（经典 PCN 分类协议，
        训练/评估同域，消除 clamp 与 free 的分布失配）。"""
        best_c, best_e = 0, np.inf
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            self.x1[:] = 0.0
            self._infer(x0, yoh, False)
            if self._energy < best_e:
                best_e, best_c = self._energy, c
        return best_c

    def set_learning(self, f):
        self.learning = f


def run_seed(cfg, seed, delay, train, ev):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + ev, delay)
    rcfg = replace(cfg, delay=delay)
    out = {}
    for mode in ("error", "hebb"):
        m = PCNOH(rcfg, np.random.default_rng(seed * 3000 + 11), mode)
        for t in range(train):
            m.train_step(X[t], int(y[t]))
        m.set_learning(False)
        preds = np.array([m.predict(X[train + t]) for t in range(ev)])
        acc = float(np.mean(preds == y[train:train + ev].astype(int)))
        out[f"acc_{mode}"] = acc
        dW = m.W1 - m.W1_init
        total = float(np.linalg.norm(dW)) + 1e-12
        out[f"distal_{mode}"] = float(np.linalg.norm(dW[:cfg.d_feat])) / total
    return out


def run_curve(cfg, seed, delay, train, ev, every=1000):
    """训练中分段评估准确率，观察是否在学。"""
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + ev, delay)
    rcfg = replace(cfg, delay=delay)
    rows = []
    for mode in ("error", "hebb"):
        m = PCNOH(rcfg, np.random.default_rng(seed * 3000 + 11), mode)
        for t in range(train):
            m.train_step(X[t], int(y[t]))
            if (t + 1) % every == 0:
                m.set_learning(False)
                preds = np.array([m.predict(X[train + i]) for i in range(ev)])
                acc = float(np.mean(preds == y[train:train + ev].astype(int)))
                rows.append((mode, t + 1, acc))
                m.set_learning(True)
        m.set_learning(False)
        preds = np.array([m.predict(X[train + i]) for i in range(ev)])
        acc = float(np.mean(preds == y[train:train + ev].astype(int)))
        rows.append((mode, train, acc))
    return rows


def diagnose(cfg, seed, delay, train):
    """深挖 Δ=4 学不动的原因：重建误差、类质心分离、远端驱动。"""
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train + 400, delay)
    rcfg = replace(cfg, delay=delay)
    m = PCNOH(rcfg, np.random.default_rng(seed * 3000 + 11), "error")
    recon = []
    for t in range(train):
        m.train_step(X[t], int(y[t]))
        if (t + 1) % 2000 == 0:
            recon.append(float(np.linalg.norm(m._e0)))
    m.set_learning(False)
    # 类质心分离：free 推断后按类统计 x1 均值
    c0, c1, n0, n1 = 0.0, 0.0, 0, 0
    for t in range(400):
        m.x1[:] = 0.0
        m._infer(X[train + t], np.array([0.0, 0.0]), True)
        if y[train + t] == 0:
            c0 += m.x1; n0 += 1
        else:
            c1 += m.x1; n1 += 1
    sep = float(np.linalg.norm(c0 / n0 - c1 / n1))
    dW = m.W1 - m.W1_init
    total = float(np.linalg.norm(dW)) + 1e-12
    distal = float(np.linalg.norm(dW[:cfg.d_feat]))
    print(f"  recon_err={[f'{e:.3f}' for e in recon]} centroid_sep={sep:.3f} "
          f"distal_share={distal / total:.3f} (chance {1/(delay+1):.3f})")
    print(f"  W2 col0 norm={np.linalg.norm(m.W2[:,0]):.3f} col1 norm={np.linalg.norm(m.W2[:,1]):.3f} "
          f"cos={float(np.dot(m.W2[:,0], m.W2[:,1])):.3f}")


def bp_baseline(d_feat, delay, train=12000, ev=400, h=32, seed=0):
    """BP 训练的 10-32-32-2 MLP 基线：任务本身可学吗？"""
    rng = np.random.default_rng(seed)
    X, y = _make_sequence(CreditConfig(d_feat=d_feat), rng, train + ev, delay)
    d0 = (delay + 1) * d_feat
    W1 = rng.normal(0, 0.3, (d0, h)); W2 = rng.normal(0, 0.3, (h, h)); W3 = rng.normal(0, 0.3, (h, 2))
    def fwd(x):
        a1 = np.maximum(x @ W1, 0); a2 = np.maximum(a1 @ W2, 0)
        return a1, a2, np.exp(a2 @ W3); a1, a2, z = fwd(X[0])
    lr = 0.05
    for t in range(train):
        a1, a2, z = fwd(X[t]); p = z / z.sum()
        yh = np.array([1, 0]) if y[t] == 0 else np.array([0, 1])
        d3 = (p - yh) / train * train  # 保持与在线一致：用单样本梯度
        g3 = np.outer(a2, d3)
        d2 = (d3 @ W3.T) * (a2 > 0)
        g2 = np.outer(a1, d2)
        d1 = (d2 @ W2.T) * (a1 > 0)
        g1 = np.outer(X[t], d1)
        W3 -= lr * g3; W2 -= lr * g2; W1 -= lr * g1
    acc = 0
    for t in range(ev):
        _, _, z = fwd(X[train + t])
        acc += int(z.argmax() == int(y[train + t]))
    return acc / ev


def make_sequence_const_recent(cfg, rng, n, delay):
    """对照任务：远端块为双峰 bit（XOR），近期块全为常量 0.5（零干扰）。
    仅保留"延迟结构"，消除"干扰维稀释"。"""
    d = cfg.d_feat
    steps = n + delay
    feat = np.full((steps, d), 0.5)
    b0 = rng.random(steps) < 0.5
    b1 = rng.random(steps) < 0.5
    feat[:, 0] = np.where(b0, 0.8, 0.2)
    feat[:, 1] = np.where(b1, 0.8, 0.2)
    y = (b0[:-delay] != b1[:-delay]) if delay > 0 else (b0 != b1)
    X = np.empty((n, (delay + 1) * d))
    for t in range(n):
        X[t] = feat[t:t + delay + 1].ravel()
    return X, y.astype(float)


def run_const_recent():
    """近期块常量：检验失败源是"延迟结构"还是"干扰稀释"。"""
    from srpc.credit import _run_seed
    from dataclasses import replace
    base = CreditConfig(d_feat=2, h1=48, kwta_frac=0.75, fan_in_frac=1.0,
                        settle_iters=5, alpha=0.1, beta=0.1, eta_w=0.05)
    # 用 _run_seed 需要替换序列生成；直接在本地构造
    rng = np.random.default_rng(0 * 3000 + 7)
    X, y = make_sequence_const_recent(base, rng, 12000 + 400, 4)
    rcfg = replace(base, delay=4)
    for mode in ("error", "hebb"):
        m = PCNOH(rcfg, np.random.default_rng(0 * 3000 + 11), mode)
        for t in range(12000):
            m.train_step(X[t], int(y[t]))
        m.set_learning(False)
        acc = sum(m.predict(X[12000 + t]) == int(y[12000 + t]) for t in range(400)) / 400
        print(f"  [常量近期块] {mode:5s} acc={acc:.3f}")


sweeps = [
    # d_feat, h1, kwta_on, sit, alpha, beta, eta_w, train, fin, delay
    (2, 48, True, 5, 0.1, 0.1, 0.05, 12000, 1.0, 4),
]
for d_feat, h1, kf, sit, alpha, beta, eta_w, train, fin, delay in sweeps:
    cfg = CreditConfig(d_feat=d_feat, h1=h1, kwta_frac=0.75, fan_in_frac=fin,
                       settle_iters=sit, alpha=alpha, beta=beta, eta_w=eta_w)
    setattr(cfg, "kwta_on", bool(kf))
    print("BP 基线（任务可学性）:")
    for dly in (0, 1, 4):
        print(f"  Δ={dly}: bp_acc={bp_baseline(2, dly):.3f}")
    print("PCN 常量近期块对照:")
    run_const_recent()
    rows = run_curve(cfg, 0, delay, train, 400, every=2000)
    print(f"=== d_feat={d_feat} h1={h1} kwta={kf} sit={sit} a={alpha} b={beta} Δ={delay} ===")
    for mode, t, acc in rows:
        print(f"  {mode:5s} step={t:5d} acc={acc:.3f}")
    diagnose(cfg, 0, delay, train)
