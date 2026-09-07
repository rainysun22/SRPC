"""E1 证据链探针（二）：局部规则 + batch 等效协议（梯度累积 P1 / 回放 P2）。

背景：xorsum@16类 在线单样本学习不可达（BP 在线对照同样失败，见
e1_probe_bp.py），mini-batch 可达 1.0 => 瓶颈 = 在线样本效率。
本探针验证两条保持初衷（局部规则/免反传/结构稀疏）的协议修正：
    P1 梯度累积：局部 dW 逐样本计算，累积 M 步应用（时间平均 = batch 等效）
    P2 经验回放：滑动窗口回放旧样本（生物学对应：睡眠重放/系统巩固）

结论（v2 记录）：两条协议在当前组件配置下均失败（M=32/R=1024+p=0.5
以及 eta_w 0.05-1.0 / 关 k-WTA / 关掩码 / 关事件门控全排除）——
k-WTA/clip/列归一等结构稀疏组件破坏 PCN 收敛态≈BP 的等效性，
batch 等效需要与结构稀疏组件协同设计，排入 F 阶段（记忆回放 +
协议联合设计），不在 E1 解决。
"""
import time

import numpy as np

from srpc.config import LangConfig
from srpc.credit import _colnorm
from srpc.lang import LangPCN, make_task


class GradAccumPCN(LangPCN):
    """P1 梯度累积：局部 dW 逐样本计算，累积 M 步应用一次（batch 等效）。"""

    def __init__(self, *a, M: int = 32, **kw):
        super().__init__(*a, **kw)
        self.M = M
        self._acc = None
        self._cnt = 0

    def _learn(self, x0, yoh):
        if not self.learning:
            return
        cfg = self.cfg
        g1 = self.x1 > cfg.theta_syn
        g2 = self.x2 > cfg.theta_syn
        dW1 = np.outer(self._e0, self.x1 * g1)
        dW2 = np.outer(self._e1, self.x2 * g2)
        dW3 = np.outer(self._e2, yoh)
        if self._acc is None:
            self._acc = [np.zeros_like(w) for w in (self.W1, self.W2, self.W3)]
        for acc, d in zip(self._acc, (dW1, dW2, dW3)):
            acc += d
        self._cnt += 1
        if self._cnt >= self.M:
            for w, acc in zip((self.W1, self.W2, self.W3), self._acc):
                w += (cfg.eta_w * acc / self.M).astype(np.float32)
            self.W1 *= self.mask1
            self.W2 *= self.mask2
            self.W3 *= self.mask3
            self.W1 = _colnorm(self.W1)
            self.W2 = _colnorm(self.W2)
            self.W3 = _colnorm(self.W3)
            self._acc = None
            self._cnt = 0


def probe_p1(cfg, seed, task, delta, steps=20000, M=32, eta_w=0.05):
    n_distal = 1 if task == "assoc" else 2
    span = delta + n_distal
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, _ = make_task(cfg, rng, steps + 400, delta, task)
    m = GradAccumPCN(cfg, np.random.default_rng(seed * 5000 + 17), "error",
                     n_blocks=span, M=M)
    m.cfg_eta_w = eta_w
    curve = []
    for t in range(steps):
        m.train_step(X[t], int(y[t]))
        if (t + 1) % (steps // 8) == 0:
            a = float(np.mean([m.predict_free(X[steps + i])[0] == int(y[steps + i])
                               for i in range(200)]))
            curve.append(round(a, 2))
    acc = float(np.mean([m.predict_free(X[steps + i])[0] == int(y[steps + i])
                         for i in range(400)]))
    return acc, curve


def probe_p2(cfg, seed, task, delta, steps=20000, R=1024, p_replay=0.5):
    n_distal = 1 if task == "assoc" else 2
    span = delta + n_distal
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, _ = make_task(cfg, rng, steps + 400, delta, task)
    m = LangPCN(cfg, np.random.default_rng(seed * 5000 + 17), "error",
                n_blocks=span)
    buf: list = []
    curve = []
    for t in range(steps):
        if buf and rng.random() < p_replay:
            i = int(rng.integers(0, len(buf)))
            xt, yt = buf[i]
        else:
            xt, yt = X[t], int(y[t])
        m.train_step(xt, yt)
        buf.append((X[t], int(y[t])))
        if len(buf) > R:
            buf.pop(0)
        if (t + 1) % (steps // 8) == 0:
            a = float(np.mean([m.predict_free(X[steps + i])[0] == int(y[steps + i])
                               for i in range(200)]))
            curve.append(round(a, 2))
    acc = float(np.mean([m.predict_free(X[steps + i])[0] == int(y[steps + i])
                         for i in range(400)]))
    return acc, curve


if __name__ == "__main__":
    cfg = LangConfig()
    t0 = time.time()
    acc1, c1 = probe_p1(cfg, 0, "xorsum", 4)
    print(f"P1 梯度累积 M=32 ({time.time()-t0:.0f}s): acc={acc1:.3f} curve={c1}")
    t0 = time.time()
    acc2, c2 = probe_p2(cfg, 0, "xorsum", 4)
    print(f"P2 回放 R=1024 p=0.5 ({time.time()-t0:.0f}s): acc={acc2:.3f} curve={c2}")
