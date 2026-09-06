"""阶段 C：有效 MAC 能耗记账 + 大模型能效标尺（不变量 3 / 6.1b）。

硬件无关的"架构内在能耗"度量，三口径同时记账：
- event  事件驱动 MACs：只计活跃单元 x 已有突触（事件驱动硬件的真实计算量）；
- struct 结构 MACs：全单元激活 x 已有突触（结构稀疏硬件的保守上界）；
- dense  稠密等价 MACs：同规模稠密网络全连接计算（稠密硬件的真实计算量）。

大模型标尺（6.1b，仅作比较基准，不进系统）：每任务推理 MACs
≈ N_params × n_tokens（每参数每 token 1 次 MAC；FLOPs ≈ 2×MACs）。
"""
from __future__ import annotations

from collections import defaultdict

# 记账键（三口径共用前缀，_struct/_dense 为结构/稠密等价口径）
BASE_KEYS = ("fwd", "bwd", "learn_w", "learn_ro", "learn_dyn",
             "readout", "dyn", "cond", "mem")


class EnergyLedger:
    """有效 MAC 记账本（trace 关闭时零开销）。"""

    def __init__(self) -> None:
        self._c: defaultdict[str, float] = defaultdict(float)
        self._n: defaultdict[str, int] = defaultdict(int)

    def add(self, key: str, value: float) -> None:
        self._c[key] += float(value)
        self._n[key] += 1

    def reset(self) -> None:
        self._c.clear()
        self._n.clear()

    def get(self, key: str) -> float:
        return self._c.get(key, 0.0)

    def mean(self, key: str) -> float:
        n = self._n.get(key, 0)
        return self._c[key] / n if n else 0.0

    def calls(self, key: str) -> int:
        return self._n.get(key, 0)

    def totals(self) -> dict[str, float]:
        """三口径总 MACs（event / struct / dense）。"""
        out = {}
        for tier, suffix in (("event", ""), ("struct", "_struct"), ("dense", "_dense")):
            out[tier] = float(sum(self._c.get(k + suffix, 0.0) for k in BASE_KEYS))
        return out

    def breakdown(self) -> dict[str, float]:
        """按操作类型的事件驱动 MACs 分解。"""
        return {k: self._c.get(k, 0.0) for k in BASE_KEYS}


def llm_task_macs(n_params: float, n_tokens: int) -> float:
    """大模型单任务推理 MACs（解析值，仅作标尺）。

    每参数每 token 1 次 MAC（前向）；含提示（任务序列化 + few-shot 示例）
    与答案生成。训练能耗（大模型侧天文数字）不计，取最保守的推理口径。
    """
    return float(n_params * n_tokens)
