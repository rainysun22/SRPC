"""F2：课程级持续学习任务流构造器（8x8 网格变换，50+ 任务，难度递增）。

承接阶段 B 免遗忘结构（PrototypeMemory 任务分组 slow 记忆 + 每任务独立读出口），
把任务数从 4 扩到 **54 个**，验证"50+ 任务流免遗忘上限"（ROADMAP F2 验收）。

任务族（难度递增，difficulty=0/1/2/3/4）：
    F1 row_shift    k: 行循环平移（整体置换，易）          难度 0
    F2 col_shift    k: 列循环平移（整体置换，易）          难度 0
    F3 rot          k: 顺时针 k*90°（整体正交）            难度 1
    F4 flip           : 水平/垂直/主对角/反对角            难度 1
    F5 recolor      k: 颜色通道循环置换                    难度 2
    F6 fragment       : 仅子块(四象限)做内变换（局部化）   难度 3
    F7 composite      : 两原语顺序复合（片段重组，最难）   难度 4

每个任务是**确定性网格变换**：在 8x8×4 色一热空间内均为线性映射（像素置换/
颜色通道置换），读出口 RLS 可学到近精确解 —— 任务都"可学"，难度轴主要压迫
**训练速度与遗忘压力**（课程级应力测试协议）。

接口对齐 ArcLite（供 runner 复用 eval/forget 口径）：
    curric.train_names -> [str]（课程顺序，共 54 个）
    curric.n_train
    curric.sample(rng, name) -> (s_in, s_out, cond)   # 一热 s_in/s_out+
    curric.decode_grid / curric._cond(i)
    curric.families -> {fam_name: [task_names]}，curric.difficulty[name] -> int
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# 原语变换（确定性，纯 NumPy；全部为网格值置换）
# ----------------------------------------------------------------------
def _shift_rows(g: np.ndarray, k: int) -> np.ndarray:
    return np.roll(g, k, axis=0)


def _shift_cols(g: np.ndarray, k: int) -> np.ndarray:
    return np.roll(g, k, axis=1)


def _rot(g: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(g, -k)                    # 顺时针 k*90°


def _flip_h(g: np.ndarray) -> np.ndarray:
    return g[:, ::-1]


def _flip_v(g: np.ndarray) -> np.ndarray:
    return g[::-1, :]


def _transpose(g: np.ndarray) -> np.ndarray:
    return g.T


def _anti_diag(g: np.ndarray) -> np.ndarray:
    return np.rot90(g[::-1, :], -1)           # 反对角

def _recolor(g: np.ndarray, k: int) -> np.ndarray:
    """颜色循环置换：1->2, 2->3, 3->1 重复 k 次。"""
    out = np.zeros_like(g)
    for c in (1, 2, 3):
        out[g == c] = ((c - 1 + k) % 3) + 1
    return out


def compose(fn1, fn2):
    """组合变换：先 fn1 后 fn2（片段重组）。"""
    def _c(g):
        return fn2(fn1(g))
    return _c


# ----------------------------------------------------------------------
# 局部（fragment）变换：仅对指定子块做内变换，其余保持 —— 全局仍是置换
# ----------------------------------------------------------------------
QUADS = ("TL", "TR", "BL", "BR")


def _quad_bounds(quad: str):
    tr, br = (0, 4) if quad[0] == "T" else (4, 8)
    tc, bc = (0, 4) if quad[1] == "L" else (4, 8)
    return tr, br, tc, bc


def fragment_recolor(g, k, quad):
    out = g.copy()
    tr, br, tc, bc = _quad_bounds(quad)
    reg = out[tr:br, tc:bc].ravel()
    for c in (1, 2, 3):
        reg[reg == c] = ((c - 1 + k) % 3) + 1
    out[tr:br, tc:bc] = reg.reshape(4, 4)
    return out


def fragment_rot(g, k, quad):
    out = g.copy()
    tr, br, tc, bc = _quad_bounds(quad)
    out[tr:br, tc:bc] = np.rot90(out[tr:br, tc:bc], -k)
    return out


# ----------------------------------------------------------------------
# 课程任务表（54 个，难度递增；关闭闭包迟绑定坑）
# ----------------------------------------------------------------------
def _build_tasks():
    tasks = []   # [(name, difficulty, fn)]

    for k in range(1, 8):
        tasks.append((f"F1_rowshift_{k}", 0, (lambda g, k=k: _shift_rows(g, k))))
    for k in range(1, 8):
        tasks.append((f"F2_colshift_{k}", 0, (lambda g, k=k: _shift_cols(g, k))))
    for k in (1, 2, 3):
        tasks.append((f"F3_rot{k}", 1, (lambda g, k=k: _rot(g, k))))
    for nm, fn in (("F4_flip_h", _flip_h), ("F4_flip_v", _flip_v),
                   ("F4_transpose", _transpose), ("F4_antidiag", _anti_diag)):
        tasks.append((nm, 1, fn))
    for k in (1, 2, 3):
        tasks.append((f"F5_recolor{k}", 2, (lambda g, k=k: _recolor(g, k))))
    for q in QUADS:
        for k in (1, 2):
            tasks.append((f"F6_frag_recolor{q}_{k}", 3,
                          (lambda g, k=k, q=q: fragment_recolor(g, k, q))))
    for q in QUADS:
        for k in (1, 2):
            tasks.append((f"F6b_frag_rot{q}_{k}", 3,
                          (lambda g, k=k, q=q: fragment_rot(g, k, q))))
    # F7 复合（片段重组，最难）：14 个两原语顺序复合，全局双射 -> 读出口可学
    comps = [
        ((lambda g: _rot(g, 1)), (lambda g: _shift_rows(g, 3)), "F7_rot1_rows3"),
        ((lambda g: _rot(g, 1)), (lambda g: _shift_cols(g, 3)), "F7_rot1_cols3"),
        ((lambda g: _rot(g, 2)), (lambda g: _shift_rows(g, 5)), "F7_rot2_rows5"),
        ((lambda g: _rot(g, 3)), (lambda g: _shift_cols(g, 4)), "F7_rot3_cols4"),
        (_transpose, (lambda g: _shift_cols(g, 1)), "F7_trans_cols1"),
        (_transpose, (lambda g: _shift_rows(g, 2)), "F7_trans_rows2"),
        (_flip_h, (lambda g: _shift_rows(g, 1)), "F7_fliph_rows1"),
        (_flip_v, (lambda g: _shift_cols(g, 2)), "F7_flipv_cols2"),
        ((lambda g: _shift_rows(g, 1)), (lambda g: _recolor(g, 2)), "F7_rows1_recol2"),
        ((lambda g: _shift_cols(g, 2)), (lambda g: _recolor(g, 1)), "F7_cols2_recol1"),
        ((lambda g: _rot(g, 1)), (lambda g: _recolor(g, 3)), "F7_rot1_recol3"),
        (_flip_h, (lambda g: _recolor(g, 2)), "F7_fliph_recol2"),
        ((lambda g: _shift_rows(g, 4)), (lambda g: _shift_cols(g, 4)), "F7_rows4_cols4"),
        ((lambda g: _recolor(g, 3)), _flip_v, "F7_recol3_flipv"),
    ]
    for f1, f2, nm in comps:
        tasks.append((nm, 4, compose(f1, f2)))
    return tasks


class F2Curriculum:
    """54 任务课程流（任务族 + 难度递增 + 一热接口，对齐 ArcLite）。"""

    grid = 8
    n_colors = 4
    d_obs = 8 * 8

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        tasks = _build_tasks()
        self.train_names = [t[0] for t in tasks]
        self.n_train = len(self.train_names)
        self.difficulty = {t[0]: t[1] for t in tasks}
        self.fn = {t[0]: t[2] for t in tasks}
        self.families: dict[str, list[str]] = {}
        for nm in self.train_names:
            fam = nm.split("_")[0]
            self.families.setdefault(fam, []).append(nm)
        self._difficulty_counts = {}
        for nm in self.train_names:
            d = self.difficulty[nm]
            self._difficulty_counts[d] = self._difficulty_counts.get(d, 0) + 1
        self.difficulty_counts = self._difficulty_counts
        assert self.n_train >= 50, f"F2 需要 50+ 任务，实际 {self.n_train}"

    def sample_input(self) -> np.ndarray:
        g = np.zeros((8, 8), dtype=int)
        n = int(self.rng.integers(1, 4))
        for _ in range(n):
            size = int(self.rng.integers(1, 3))
            r = int(self.rng.integers(8 - size + 1))
            c = int(self.rng.integers(8 - size + 1))
            color = int(self.rng.integers(1, 4))
            g[r:r + size, c:c + size] = color
        return g

    def sample(self, name: str):
        """生成 (s_in, s_out, cond)：一热展平 + 任务 one-hot 条件。"""
        g_in = self.sample_input()
        g_out = self.fn[name](g_in)
        idx = self.train_names.index(name)
        return (self._onehot(g_in).astype(float), self._onehot(g_out).astype(float),
                self._cond(idx))

    def _onehot(self, g: np.ndarray) -> np.ndarray:
        oh = np.zeros((g.size, self.n_colors))
        for cc in range(self.n_colors):
            oh[:, cc] = (g.ravel() == cc).astype(float)
        return oh.ravel()

    def decode_grid(self, oh_flat: np.ndarray) -> np.ndarray:
        oh = np.reshape(oh_flat, (-1, self.n_colors))
        return oh.argmax(axis=1).astype(float)

    def _cond(self, i: int) -> np.ndarray:
        c = np.zeros(self.n_train)
        c[i] = 1.0
        return c