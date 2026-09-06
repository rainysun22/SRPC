"""阶段 B：多时间尺度原型记忆（免遗忘结构，不变量 2：能力=记忆·拼合）。

两组原型槽：
- fast（工作记忆）：学习率快，承载当前会话/任务的模式，可快速覆盖；
- slow（长期记忆）：学习率慢，按任务分组（group），只在"稳定/重复"时小幅巩固。
  顺序学习新任务只写新任务的 slow 组，旧任务原型不被覆盖 —— 结构上免遗忘
  （阶段 B 里程碑：能力随交互上升且免遗忘）。

写入为 WTA 稀疏：每次只更新与 x 最接近的 1 个槽（不变量 3 稀疏）。
读取（recall）返回最近激活原型，作为顶层的 top-down 先验。
全部显式局部规则（误差驱动加减），免反向传播。
"""
from __future__ import annotations

import numpy as np

from .config import MemoryConfig


class PrototypeMemory:
    """多时间尺度原型记忆。d 为原型维（= x_self 维）。

    group：任务标识（ARC 训练变换索引）。slow 槽按 group 分区，
    group=None 时只写 fast（工作记忆），recall 回退 fast。
    """

    def __init__(self, cfg: MemoryConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.d = cfg.d
        self.n_slow_group = cfg.n_slow_group
        self.fast = np.zeros((cfg.n_fast, cfg.d))      # 工作记忆原型
        self.n_fast = np.zeros(cfg.n_fast, dtype=int)  # 槽写入次数
        self.slow: dict[int, np.ndarray] = {}          # group -> (n_slow_group, d)
        self.n_slow: dict[int, np.ndarray] = {}        # group -> 写入次数
        self.last_macs = 0.0                           # 最近一次读/写的 MACs（记账）

    # ------------------------------------------------------------------
    # 读取：最近激活原型（slow 同组优先；否则 fast）
    # ------------------------------------------------------------------
    def recall(self, x: np.ndarray, group: int | None = None) -> np.ndarray:
        """返回最近激活原型作为 top-down 先验（无原型时返回零向量）。

        last_macs：本次调用的距离计算 MACs（槽数 × 维度，阶段 C 记账）。
        """
        self.last_macs = 0.0
        if group is not None and group in self.slow:
            protos, counts = self.slow[group], self.n_slow[group]
            self.last_macs += float(protos.shape[0] * self.d)
            if int(counts.sum()) > 0:
                d = np.linalg.norm(protos - x, axis=1)
                d[np.where(counts == 0)[0]] = np.inf
                return protos[int(np.argmin(d))].copy()
        if int(self.n_fast.sum()) > 0:
            self.last_macs += float(self.fast.shape[0] * self.d)
            d = np.linalg.norm(self.fast - x, axis=1)
            d[np.where(self.n_fast == 0)[0]] = np.inf
            return self.fast[int(np.argmin(d))].copy()
        return np.zeros(self.d)

    # ------------------------------------------------------------------
    # 写入：WTA 稀疏 + 快/慢两尺度巩固（slow 按任务分组）
    # ------------------------------------------------------------------
    def consolidate(self, x: np.ndarray, stable: float,
                    group: int | None = None) -> None:
        """把收敛信念 x 巩固进记忆。

        stable ∈ [0,1] 为置信（自省误差小则高）；低于 conf_thresh 不写入
        （避免噪声信念污染记忆）。WTA：只更新最近槽。
        last_macs：本次写入的距离计算 MACs（阶段 C 记账）。
        """
        cfg = self.cfg
        self.last_macs = 0.0
        if stable < cfg.conf_thresh:
            return

        # ---- fast：工作记忆（可快速覆盖，不分组） ----
        self.last_macs += float(self.fast.shape[0] * self.d)
        empty = np.where(self.n_fast == 0)[0]
        if len(empty) > 0:
            k = int(empty[0])
            self.fast[k] = x.copy()
        else:
            d = np.linalg.norm(self.fast - x, axis=1)
            k = int(np.argmin(d))
            # 误差驱动：Δp = rate * (x - p)（局部加减，免反传）
            self.fast[k] += cfg.rate_fast * (x - self.fast[k])
        self.n_fast[k] += 1

        # ---- slow：长期巩固（只写当前任务的组，旧任务原型结构上不受影响） ----
        if group is None:
            return
        if group not in self.slow:
            self.slow[group] = np.zeros((self.n_slow_group, self.d))
            self.n_slow[group] = np.zeros(self.n_slow_group, dtype=int)
        protos, counts = self.slow[group], self.n_slow[group]
        if self.n_fast[k] >= 2:
            self.last_macs += float(protos.shape[0] * self.d)
            empty2 = np.where(counts == 0)[0]
            if len(empty2) > 0:
                k2 = int(empty2[0])
                protos[k2] = self.fast[k].copy()
            else:
                d2 = np.linalg.norm(protos - self.fast[k], axis=1)
                k2 = int(np.argmin(d2))
                protos[k2] += cfg.rate_slow * (self.fast[k] - protos[k2])
            counts[k2] += 1

    def snapshot(self) -> dict:
        return dict(fast=self.fast.copy(), n_fast=self.n_fast.copy(),
                    slow={g: p.copy() for g, p in self.slow.items()},
                    n_slow={g: c.copy() for g, c in self.n_slow.items()})

    def restore(self, snap: dict) -> None:
        for k, v in snap.items():
            if k in ("slow", "n_slow"):
                setattr(self, k, {g: arr.copy() for g, arr in v.items()})
            else:
                setattr(self, k, v.copy())
