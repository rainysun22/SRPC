"""阶段 F1b-2b：PCN + 真·batch 等效回放的 xorsum 修复扫描。

原理（对 E1 探针失败的精确归因）：
  - xorsum@16 是 parity，SQ-hard：单样本期望梯度=0（Kearns&Valiant）；
  - "类均值原型回放"（f1b_replay_probe['proto']）失败是**原理性**的：给定 y，
    x_a 全部 16 值各等概（x_b=y⊕x_a 确定）=> 每类远端块 marginal≡全局 uniform，
    即类条件一阶均值零判别信息，batch 平均后二阶交互被抹掉；
  - mini-batch(512) SGD 可达 1.0 => 同形状网络任务可解，瓶颈是**真 batch
    （累积 M 个真实成对样本的梯度后一次性应用）**，而非在线点估计。

本探针：LangPCN 的**成对累积批次回放**。回放阶段每次取 B 个真实 (X,y)，
逐样本跑 clamp 推断、**累积局部 dW**，B 个样本后才做一次 mask+colnorm 应用
（= 真 mini-batch 的局部规则版）。对比：
  - online   ：纯在线（E1 基线，机会）
  - batch_into_online：对在线流的窗口做 decoder（模拟"睡眠重放"）——累积 B
    整批应用梯度
  - 关/开 k-WTA 与事件门控的子扫描，隔离稀疏组件对 batch 等效的破坏。

判据：xorsum@Δ4 回放后 acc ≥ 0.60（ROADMAP F1）。
"""
from __future__ import annotations
import time

import numpy as np

from srpc.config import LangConfig
from srpc.lang import LangPCN, make_task
from srpc.credit import _colnorm


class BatchReplayPCN(LangPCN):
    """在 LangPCN 上加"成对累积批次学习"：在线跑 clamp，周期用真实成对样本
    批量回流。learn_mode option 'batch_accum'：逐样本累积 dW，B 步一次性应用。"""

    def __init__(self, *a, B: int = 128, **kw):
        super().__init__(*a, **kw)
        self.B = int(B)
        self._acc = None
        self._cnt = 0

    def _learn_batch_push(self, x0, yoh):
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
        if self._cnt >= self.B:
            for w, acc in zip((self.W1, self.W2, self.W3), self._acc):
                w += (cfg.eta_w * acc / self.B).astype(np.float32)
            self.W1 *= self.mask1
            self.W2 *= self.mask2
            self.W3 *= self.mask3
            self.W1 = _colnorm(self.W1)
            self.W2 = _colnorm(self.W2)
            self.W3 = _colnorm(self.W3)
            self._acc = None
            self._cnt = 0

    def _learn(self, x0, yoh):
        # 正常在线用父类单样本学习；batch 回放用 _learn_batch_push
        super()._learn(x0, yoh)

    # 复用父类 _infer；额外暴露 clamp 推断（重置状态）
    def _infer_clamped(self, x0, y):
        yoh = np.zeros(self.C, dtype=np.float32); yoh[y] = 1.0
        self.x1[:] = 0.0; self.x2[:] = 0.0
        self._infer(x0, yoh, free_out=False)
        return yoh


def run(cfg, mode, seed=0, steps=20000, B=256, replay_every=500, kw=''):
    import copy
    c = copy.deepcopy(cfg)
    if kw == 'nokwta':
        c.kwta_on = False
    if kw == 'noevent':
        c.theta_event = 1e-9
    if kw == 'nokwta_nogate_cap':
        c.kwta_on = False; c.theta_event = 1e-9
    n_distal = 2
    span = 4 + n_distal
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, _ = make_task(c, rng, steps + 400, 4, "xorsum")
    m = BatchReplayPCN(c, np.random.default_rng(seed * 5000 + 17), "error",
                       n_blocks=span, B=B)
    curve = []
    for t in range(steps):
        m.train_step(X[t], int(y[t]))
        if mode == 'batch' and ((t + 1) % replay_every == 0 and t > 0):
            # 回放：从之前窗口取 B 个真实成对样本（闭环"睡眠重放"）
            lo = max(0, t - replay_every * 2)
            idx = rng.integers(lo, t, size=B)
            m._acc = None; m._cnt = 0
            for j in idx:
                yoh = np.zeros(c.n_classes, dtype=np.float32)
                yoh[int(y[j])] = 1.0
                m.x1[:] = 0.0; m.x2[:] = 0.0
                m._infer(X[j], yoh, free_out=False)
                m._learn_batch_push(X[j], yoh)
        if (t + 1) % (steps // 8) == 0:
            curve.append(_qa(m, X[steps:steps + 200], y[steps:steps + 200]))
    acc = _qa(m, X[steps:steps + 400], y[steps:steps + 400])
    return acc, curve


def _qa(m, X, y) -> float:
    n = sum(1 for t in range(len(y)) if m.predict_free(X[t])[0] == int(y[t]))
    return n / len(y)


def main() -> int:
    cfg = LangConfig()
    print(f"h1={cfg.h1} h2={cfg.h2} eta={cfg.eta_w} iters={cfg.settle_iters} "
          f"class={cfg.n_classes} B~256")
    res = {}
    for mode, kw in [("online", ''), ("batch", ''), ("batch", 'nokwta'),
                     ("batch", 'noevent'), ("batch", 'nokwta_nogate_cap')]:
        t0 = time.time()
        a, c = run(cfg, mode, kw=kw)
        key = f"{mode}" + (f"_{kw}" if kw else "")
        res[key] = round(a, 3)
        print(f"xorsum@{key}: acc={a:.3f} curve={c} ({time.time()-t0:.0f}s)")
    ok = max(res.get(k, 0) for k in res) >= 0.60
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())