"""阶段 B 组合基准：ARC-lite（8x8 网格变换任务 + 保留组合零样本）。

网格值 {0,1,2,3}（0=空，1-3 颜色），随机放置若干彩色块；
任务 = 网格变换（翻转/旋转/重着色）。训练集为基本变换（顺序学习任务序列），
保留集为两个已知变换的组合 —— 验证"既有片段重组出新概念"（阶段 B 里程碑）。

变换条件：训练变换 one-hot 向量（Uc 固定分块正交码，见 DeepSRPC.set_condition）；
保留组合的零样本评估 = 顺序复合两个已学变换（先 t1 读出、解码后以 t2 读出，
见 runner_b.eval_combination）—— 跨时段片段重组，不依赖嵌入相加假设。
"""
from __future__ import annotations

import numpy as np

from .config import ArcConfig


# ----------------------------------------------------------------------
# 原语变换（纯 NumPy，逐元素/逐轴操作）
# ----------------------------------------------------------------------
def tf_flip_h(g: np.ndarray) -> np.ndarray:
    return g[:, ::-1]


def tf_flip_v(g: np.ndarray) -> np.ndarray:
    return g[::-1, :]


def tf_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=1)          # 逆时针 90°


def tf_recolor(g: np.ndarray) -> np.ndarray:
    """颜色循环置换：1->2, 2->3, 3->1。"""
    out = np.zeros_like(g)
    out[g == 1] = 2
    out[g == 2] = 3
    out[g == 3] = 1
    return out


TRANSFORMS = {
    "flip_h": tf_flip_h,
    "flip_v": tf_flip_v,
    "rot90": tf_rot90,
    "recolor": tf_recolor,
}


def compose(t1: str, t2: str, g: np.ndarray) -> np.ndarray:
    """组合变换：先 t1 后 t2（保留组合的真值输出，用于零样本评估）。"""
    return TRANSFORMS[t2](TRANSFORMS[t1](g))


class ArcLite:
    """ARC-lite 任务生成器。"""

    def __init__(self, cfg: ArcConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.grid = cfg.grid
        self.n_colors = cfg.n_colors
        self.d_obs = cfg.grid * cfg.grid
        self.train_names = list(cfg.train_transforms)
        self.n_train = len(self.train_names)
        self.novel_names = ["_".join(c) for c in cfg.novel_combos]
        self.novel_combos = [tuple(c) for c in cfg.novel_combos]
        self.all_names = self.train_names + self.novel_names

    # ------------------------------------------------------------------
    # 样本生成
    # ------------------------------------------------------------------
    def sample_input(self) -> np.ndarray:
        """随机网格：1..n_blocks 个实心彩色块。"""
        g = np.zeros((self.grid, self.grid), dtype=int)
        n = int(self.rng.integers(1, self.cfg.n_blocks + 1))
        for _ in range(n):
            size = int(self.rng.integers(1, self.cfg.max_block + 1))
            r = int(self.rng.integers(self.grid - size + 1))
            c = int(self.rng.integers(self.grid - size + 1))
            color = int(self.rng.integers(1, self.n_colors))
            g[r:r + size, c:c + size] = color
        return g

    def sample(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """生成 (s_in, s_out, cond)。

        s_in / s_out：展平一热编码（D = grid^2 * n_colors = 256）。
        一热输入使翻转/旋转（像素置换）与重着色（颜色通道置换）在读出空间
        都是线性映射 —— 变换片段可被读出头精确表示，组合零样本 = 顺序复用
        已学头的近精确重组（阶段 B 里程碑）。
        一热输出使误差对"放错像素/颜色"给出满惩罚 —— 随机基线可区分。
        cond：训练变换 one-hot；组合任务为 None（零样本用顺序复合评估）。
        """
        g_in = self.sample_input()
        if name in self.train_names:
            g_out = TRANSFORMS[name](g_in)
            cond = self._cond(self.train_names.index(name))
        else:
            i, j = self.novel_combos[self.novel_names.index(name)]
            g_out = compose(i, j, g_in)
            cond = None
        return (self._onehot(g_in).astype(float),
                self._onehot(g_out).astype(float), cond)

    def _onehot(self, g: np.ndarray) -> np.ndarray:
        """网格值 -> 展平一热编码（grid^2 × n_colors）。"""
        oh = np.zeros((g.size, self.n_colors))
        for c in range(self.n_colors):
            oh[:, c] = (g.ravel() == c).astype(float)
        return oh.ravel()

    def decode_grid(self, oh_flat: np.ndarray) -> np.ndarray:
        """展平一热输出 -> 网格值（顺序复合时把中间读出解码回输入格式）。"""
        oh = oh_flat.reshape(-1, self.n_colors)
        return oh.argmax(axis=1).astype(float)

    def _cond(self, i: int) -> np.ndarray:
        c = np.zeros(self.n_train)
        c[i] = 1.0
        return c

    def novel_cond(self, name: str) -> np.ndarray:
        """保留组合的条件向量 = 两个成分 one-hot 相加（嵌入可加性假设）。"""
        i, j = self.novel_combos[self.novel_names.index(name)]
        return self._cond(self.train_names.index(i)) + self._cond(self.train_names.index(j))
