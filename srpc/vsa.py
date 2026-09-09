"""阶段 F1b：VSA/HRR 超维绑定层（循环卷积 bind/unbind + cleanup 记忆）。

动机（"能力=记忆·拼合"，不变量 2）：F1a 的 n-gram 结构化记忆做到"防遗忘+语言增益"，
但记忆载体（按字节前缀哈希的结构）**不可组合**——不支持概念-关系组合查询、无绑定/
泛化。F1b 引入 VSA（Vector Symbolic Architecture，Plate 1995 HRR）绑定层：

    bind(a,b)   = a ⊗ b   : 循环卷积（把 pair 绑成一个 d 维超维向量）
    superpose    = Σ bind   : 多个 pair 叠加（一张"组合场景"）
    unbind(q, r) = q ⊛ r   : 循环相关（用 r 解码 q 里与 r 绑定的 filler）
    cleanup      = 余弦最近   : 解出噪声向量 -> 词汇表里最近的标准向量

性质：
  - bind/unbind 互为近似逆（unbind(bind(a,b), b) ≈ a）。
  - 叠加近正交（不同随机对近似不干扰），容量 ~O(d/4-8)（Plate）。
  - 全部为 O(d log d)（FFT）线性算子，无反向传播、无全局 gradient
    （对齐 GPU_TASKS 契约第 2 条：读写是外积/卷积型局部运算）。
  - dim d 取 ≥512（ROADMAP 口径），更高维提升容量-精度。

组合查询任务（"概念-关系组合"）：
  - role（关系，如 形状/颜色/材质）与 filler（对象值，如 圆/红/木）各一大词表；
  - store_pairs 把 (role_i ⊗ filler_i) 叠加为 S；
  - query(role) = cleanup(S ⊛ role)  -> 应解回与其绑定的 filler；
  - recall = 解回命中真 filler 的比例；容量曲线 = recall ~ (k 对, d 维)。

F1b 验收对应该子项的"组合召回率 ≥ 阈值"与"容量-精度曲线"（组合口径）。
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# 超维算子（FFT 实现，O(d log d)）
# ----------------------------------------------------------------------
def gauss_vecs(d: int, n: int, seed: int = 0) -> np.ndarray:
    """n 个 d 维随机高斯向量，单位 L2 范数（确定性 seed）。"""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d))
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """循环卷积 a ⊗ b（返回实向量，d 维）。"""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))


def unbind(q: np.ndarray, key: np.ndarray) -> np.ndarray:
    """循环相关 q ⊛ key：解绑 key 在 q 里绑定的 filler ≈ Q^T interp。"""
    return np.fft.irfft(np.fft.rfft(q) * np.conj(np.fft.rfft(key)), n=len(q))


class VsaMemory:
    """HRR 绑定记忆：role-filler 对的叠加 + cleanup 范式解码头。

    组合查询：S = Σ_i role_i ⊗ filler_i；query(role_j) = clean(S ⊛ role_j)。
    cleanup = 与 filler 词汇表（此 memory 见过的所有 filler）的余弦最近。
    只存一个 d 维叠加向量 + 词汇常量表 —— 结构性记忆、线程安全、免遗忘
    （新 pair 是加性叠加，不覆盖旧 pair；除非超出噪声叠加容量）。
    """

    def __init__(self, d: int = 1024, seed: int = 0):
        self.d = int(d)
        self.S = np.zeros(self.d)          # 绑定叠加（组合记忆本体）
        self.n_pairs = 0
        self.roles: list[np.ndarray] = []
        self.fillers: list[np.ndarray] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 写入：绑定 (role, filler) 并叠加进场景记忆（加性、不相覆盖）
    # ------------------------------------------------------------------
    def store(self, role: np.ndarray, filler: np.ndarray) -> None:
        self.S += bind(role, filler)
        self.n_pairs += 1

    def store_pairs(self, roles: np.ndarray, fillers: np.ndarray) -> None:
        """批量绑定叠加。roles/fillers: (k, d)。"""
        for r, f in zip(roles, fillers):
            self.S += bind(r, f)
        self.n_pairs += len(fillers)

    # ------------------------------------------------------------------
    # 读取：组合查询（unbind + cleanup）
    # ------------------------------------------------------------------
    def query_raw(self, role: np.ndarray) -> np.ndarray:
        """解绑 -> 噪声 filler 向量（未 cleanup）。"""
        return unbind(self.S, role)

    def cleanup(self, vec: np.ndarray, vocab: np.ndarray | None = None
                ) -> int:
        """返回 vocab 中与 vec 余弦最近的向量索引（范式解码）。"""
        V = vocab if vocab is not None else np.array(self.fillers)
        if len(V) == 0:
            return -1
        nv = vec / (np.linalg.norm(vec) + 1e-12)
        pn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        return int((pn @ nv).argmax())

    def query(self, role: np.ndarray, vocab: np.ndarray | None = None) -> int:
        d = self.query_raw(role)
        return self.cleanup(d, vocab)

    def recall(self, roles: np.ndarray, fillers: np.ndarray, vocab=None
               ) -> float:
        """对每对 (role_i,filler_i)，query 解回是否命中其 filler。"""
        V = fillers if vocab is None else vocab
        pn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        k = len(roles)
        hit = 0.0
        for i in range(k):
            d = self.query_raw(roles[i])       # 解绑噪声向量
            d /= (np.linalg.norm(d) + 1e-12)
            q = int((pn @ d).argmax())
            # 命中 = 解回索引的向量 == 真 filler（按向量相等判定，规避对象身份比较）
            hit += 1.0 if np.allclose(V[q], fillers[i]) else 0.0
        return hit / k

    def kos(self, role: np.ndarray, vocab=None) -> np.ndarray:
        """解绑后与 vocab 各向量的内积（诊断 KOS 分布）。"""
        V = np.array(self.fillers) if vocab is None else vocab
        nv = self.query_raw(role)
        nv /= (np.linalg.norm(nv) + 1e-12)
        pn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        return pn @ nv