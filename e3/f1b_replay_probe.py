"""阶段 F1b-2：VSA/HRR 结构化记忆回放协议 —— 修复 E1 xorsum 在线组合信用分配边界。

承接：E1 探针（e1_probe_protocol.py）证明 P1 梯度累积 / P2 朴素滑动窗口回放
在 xorsum@16 类上均失败（acc≈0.05-0.09）——结构稀疏组件（k-WTA/列归一/事件
门控）破坏"在线单样本点估计 ≈ batch 梯度期望"的等价性：每个样本单独过稀疏
算子，噪声一帧帧累计，对称不破缺。

F1b 协同设计：**类原型回放**。用叠加（VSA SOS，对稀疏一热即类均值）把每个
输出类的输入窗口聚合成**去噪原型 P_c**，在回放阶段**单次推断**这一原型——
k-WTA/列归一作用于平均（去噪）输入，等于把"逐样本噪声"在表示层先平均掉，
batch 等效性在表示层恢复而非梯度层。生物学对应：睡眠重放/系统巩固，把当天
经历固化成类典型再回放。

协议流程：
  1) 在线学习：每步 train_step(X_t, y_t)，同时把该样本压入 buf[y_t] 的类原型叠加；
  2) 每 cfg 步一次巩固：对每个类 c，P_c = buf[c]/cnt[c] 作为 x0，以 c 为标签
     train_step（把快照原型喂给同一网络，吃批量平均的信用信号）；
  3) 冻结评估：自由推断 acc（同 E1 主口径）。

对比臂：无回放（基线）/ F1b 类原型回放 / P2 朴素回放（复现 E1 失败）。
判据：xorsum→ 回放后 acc ≥ 0.60（ROADMAP F1）。
"""
from __future__ import annotations
import time

import numpy as np

from srpc.config import LangConfig
from srpc.lang import LangPCN, make_task


class ReplayPrototype:
    """类原型结构化记忆：逐类把窗口叠加成均值原型（VSA SOS）；供回放。"""

    def __init__(self, d: int, C: int):
        self.d = int(d)
        self.C = int(C)
        self.acc = np.zeros((C, d), dtype=np.float64)
        self.cnt = np.zeros(C, dtype=np.int64)

    def push(self, x: np.ndarray, y: int) -> None:
        self.acc[y] += x
        self.cnt[y] += 1

    def prototype(self, c: int) -> np.ndarray | None:
        if self.cnt[c] == 0:
            return None
        return (self.acc[c] / self.cnt[c]).astype(np.float32)


def train_with_replay(cfg, seed, task, delta, steps=20000,
                      replay_every=250, replay_mode="proto", R=1024,
                      p_replay=0.5) -> tuple[float, list]:
    """在线学习 + 间歇巩固回放。replay_mode:
        'none'    : 纯在线（E1 失败基线）
        'proto'   : F1b 类原型回放（去噪平均表示层恢复 batch 等效）
        'raw'     : P2 朴素滑动窗口回放（复现 E1 失败）
    """
    n_distal = 2 if task == "xorsum" else 1
    span = delta + n_distal
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, _ = make_task(cfg, rng, steps + 400, delta, task)
    m = LangPCN(cfg, np.random.default_rng(seed * 5000 + 17), "error",
                n_blocks=span)
    mem = ReplayPrototype(span * 256, cfg.n_classes)
    buf: list = []
    curve = []
    if replay_mode == "proto":
        replay_every = replay_every if replay_every else steps + 1
    for t in range(steps):
        # ---- 在线一步 ----
        m.train_step(X[t], int(y[t]))
        if replay_mode == "proto":
            mem.push(X[t], int(y[t]))
        elif replay_mode == "raw":
            buf.append((X[t], int(y[t])))
            if len(buf) > R:
                buf.pop(0)
            if buf and rng.random() < p_replay:
                i = int(rng.integers(0, len(buf)))
                m.train_step(buf[i][0], buf[i][1])
        # ---- 巩固回放（类原型）----
        if replay_mode == "proto" and (t + 1) % replay_every == 0:
            for c in range(cfg.n_classes):
                P = mem.prototype(c)
                if P is None:
                    continue
                m.train_step(P, c)
        if (t + 1) % (steps // 8) == 0:
            curve.append(_qa(m, X[steps:steps + 200], y[steps:steps + 200]))
    acc = _qa(m, X[steps:steps + 400], y[steps:steps + 400])
    return acc, curve


def _qa(m, X, y) -> float:
    n = sum(1 for t in range(len(y))
            if m.predict_free(X[t])[0] == int(y[t]))
    return n / len(y)


def main() -> int:
    cfg = LangConfig()
    print(f"h1={cfg.h1} h2={cfg.h2} fan_in={cfg.fan_in_frac} "
          f"theta={cfg.theta_event} kwta_frac={cfg.kwta_frac} "
          f"class={cfg.n_classes}")
    accs = {}
    for mode, kw in [("none", dict(replay_mode="none")),
                     ("proto_250", dict(replay_mode="proto", replay_every=250)),
                     ("proto_50", dict(replay_mode="proto", replay_every=50)),
                     ("raw_p2", dict(replay_mode="raw"))]:
        t0 = time.time()
        a, c = train_with_replay(cfg, 0, "xorsum", 4, **kw)
        accs[mode] = round(a, 4)
        print(f"xorsum@{mode}: acc={a:.3f} curve={c}  ({time.time()-t0:.0f}s)")
    return 0 if accs.get("proto_250", 0) >= 0.60 else 1


if __name__ == "__main__":
    raise SystemExit(main())