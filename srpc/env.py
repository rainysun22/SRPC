"""Phase-0 环境：两个共享同一模型与更新规则的合成世界。

Track 1 —— SourceFieldWorld（交互式导航，7.4 形成/修正/自省对照）：
    2-D 场地内分布若干"信息源"，每源携带一个稀疏非负感觉模式；
    观测 = 邻近源模式的高斯核加权和 + 噪声（空间上天然产生"组合"结构）。
    智能体以 5 个动作（保持/上/下/左/右）移动，主动推理选择动作。

Track 2 —— SlotWorld（组合流，7.4 组合，MVP 合成数据）：
    每个时段呈现"单特征"或"特征对"上下文，上下文持续一段时长（跨时段片段）；
    训练只见部分组合，保留 4 个组合用于验证"既有片段能否重组出新概念"。
"""
from __future__ import annotations

import numpy as np

from .config import FieldConfig, SlotConfig

# 动作表：[dx, dy]（stay/up/down/left/right）
FIELD_ACTIONS = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])
N_FIELD_ACTIONS = len(FIELD_ACTIONS)


def make_patterns(n: int, d: int, rng: np.random.Generator,
                  active: int, stride: int) -> np.ndarray:
    """生成 n 个稀疏非负感觉模式（D 维、列单位 L2 范数）。

    第 i 个模式的活跃维为块 [i*stride, i*stride+active)，相邻模式轻度重叠，
    保证"部件"可分但不过度平凡。
    """
    pats = np.zeros((d, n))
    for i in range(n):
        lo = i * stride
        hi = min(lo + active, d)
        pats[lo:hi, i] = rng.uniform(0.5, 1.5, size=hi - lo)
    pats /= np.linalg.norm(pats, axis=0, keepdims=True)
    return pats


class SourceFieldWorld:
    """Track-1：信息源场地（二维环面，周期边界，无"贴墙"退化策略）。"""

    def __init__(self, cfg: FieldConfig, d_obs: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.n_sources
        self.pos = np.array(cfg.source_pos, dtype=float)          # (M,2)
        self.patterns = make_patterns(self.n, d_obs, rng,
                                      cfg.pattern_active, cfg.pattern_stride)  # (D,M)
        self.agent = rng.uniform(0.25, 0.75, size=2)
        self.perm: np.ndarray | None = None   # 扰动后的感觉维度置换

    def _dist2(self) -> np.ndarray:
        """环面上的平方距离（周期边界）。"""
        d = np.abs(self.pos - self.agent)
        d = np.minimum(d, 1.0 - d)
        return (d ** 2).sum(axis=1)

    def observe(self) -> np.ndarray:
        w = np.exp(-self._dist2() / (2.0 * self.cfg.kernel_sigma ** 2))
        s = self.patterns @ w + self.rng.normal(0.0, self.cfg.obs_noise, self.patterns.shape[0])
        if self.perm is not None:
            s = s[self.perm]
        return np.maximum(s, 0.0)

    def step(self, action: int) -> None:
        move = FIELD_ACTIONS[action] * self.cfg.move_step
        noise = self.rng.normal(0.0, self.cfg.move_noise, size=2)
        self.agent = (self.agent + move + noise) % 1.0   # 环面：无墙

    def perturb(self, rng: np.random.Generator) -> None:
        """修正测试：全局感觉重映射（感知维度随机置换）。

        智能体无论身处何处都会立刻产生大规模预测误差，
        旧信念必须被新证据纠正 —— 不存在"回避扰动区"的捷径。
        """
        self.perm = rng.permutation(self.patterns.shape[0])

    def effective_patterns(self) -> np.ndarray:
        """当前生效的源模式（重映射后按行置换）。"""
        if self.perm is None:
            return self.patterns
        return self.patterns[self.perm, :]

    def zone(self) -> int:
        """最近源编号（作为"真实上下文"标签，用于结构化表征的 NMI 分析）。"""
        return int(np.argmin(self._dist2()))


class SlotWorld:
    """Track-2：特征/特征对组合流。

    上下文编号：0..K-1 为单特征；K..K+len(train)-1 为训练对；
    K+len(train).. 为保留（novel）对。
    """

    def __init__(self, cfg: SlotConfig, d_obs: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.K = cfg.n_features
        self.patterns = make_patterns(self.K, d_obs, rng,
                                      cfg.pattern_active, cfg.pattern_stride)
        self.train_pairs = [tuple(p) for p in cfg.train_pairs]
        self.novel_pairs = [tuple(p) for p in cfg.novel_pairs]
        # 上下文定义表
        self.ctx_feats: list[tuple[int, ...]] = [(i,) for i in range(self.K)]
        self.ctx_feats += self.train_pairs + self.novel_pairs
        self.n_train_ctx = self.K + len(self.train_pairs)
        self.phase = "train"
        self.ctx = 0
        self.left = 0

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.left = 0

    def _next_ctx(self) -> None:
        if self.phase == "train":
            if self.rng.random() < self.cfg.single_prob:
                self.ctx = int(self.rng.integers(self.K))
            else:
                self.ctx = self.K + int(self.rng.integers(len(self.train_pairs)))
        elif self.phase == "novel":
            base = self.n_train_ctx
            self.ctx = base + int(self.rng.integers(len(self.novel_pairs)))
        else:
            raise ValueError(f"unknown phase {self.phase}")
        self.left = int(self.rng.integers(self.cfg.ctx_min, self.cfg.ctx_max))

    def sample_ctx(self, ctx: int) -> np.ndarray:
        """给定上下文编号生成一次观测（用于冻结评估）。"""
        feats = self.ctx_feats[ctx]
        s = self.patterns[:, list(feats)].sum(axis=1)
        s = s + self.rng.normal(0.0, self.cfg.obs_noise, size=len(s))
        return np.maximum(s, 0.0)

    def observe(self) -> tuple[np.ndarray, int]:
        if self.left <= 0:
            self._next_ctx()
        self.left -= 1
        return self.sample_ctx(self.ctx), self.ctx
