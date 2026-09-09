"""E3：语言版慢记忆原型读出（torch，按语种 group 存原型，注入读出，不动 srpc 核心）。

语义参照 srpc/memory.py 的 PrototypeMemory（slow 按 group 分区，recall 返回最近
激活原型作 top-down 先验），但这里做的是**语言版**：每语种 group 一个原型，原型维
= C = 256 ="该语种字节频率/分布"潜量（语种 unigram）。这与 E0 结论完全对齐——
在"当前输入的 16 字节窗口不含全部上下文（语种）信息"的语言流上，slow 记忆把
"当前语种的整体字节统计"作为 top-down 先验注入读出。

接口（torch/numpy 混合，4090 上跑 cuda）：
    mem = LangSlowMem(n_groups, C=256, device)
    mem.consolidate(y, group)            # 训练期：把目标字节巩固进该语种原型（计数累积）
    mem.recall(group)  -> (256,) 概率原型（归一 smooth）
    mem.mem_logit(group) -> (256,) log-prob 先验向量 或 None（该组无样本）
    mem.stable(group)   -> 该组累积计数（stable 度量）

记忆先验注入（本冒烟选用【log-空间贝叶斯先验乘】，替代线性混合，理由见 README）：
    logit_mem = logit_model + beta * tau * mem_logit
    p = softmax(logit_mem / tau)          # 等价 p ∝ p_model · proto^beta
其中 mem_logit = log(语种字节先验)。beta=0 即退化为无记忆读出头（== eval_batch）。
beta 语义清晰（0=关，1=完整对数先验），且对 unigram 型先验的注入比线性软混合
（p=αp_model+(1-α)p_mem）更稳（不稀释模型已会的近程结构）。

eval_batch_mem：逐窗口跑 m._infer 取 x2 → W_out 读出头 logit_model，注入 recalled
语种原型先验 → 计 BPC/acc。不动 lmgpu.py 内 eval_batch（无记忆对照走原版/或 beta=0）。
"""
from __future__ import annotations

import numpy as np
import torch


class LangSlowMem:
    """语言版慢记忆原型：每语种 group 一个 256 维字节频率原型（累积-归一）。"""

    def __init__(self, n_groups: int, C: int = 256, device: str = "cuda",
                 smooth: float = 1e-6):
        self.n_groups = int(n_groups)
        self.C = int(C)
        self.device = torch.device(device)
        self.smooth = float(smooth)         # 先验平滑（避免 log0）
        self.counts = np.zeros((n_groups, C), np.float64)   # 语种字节计数（CPU 累积）

    # ------------------------------------------------------------------
    # 写入：训练期逐样本巩固（该样本确属某语种，直接计数该语种字节出现）
    # ------------------------------------------------------------------
    def consolidate(self, y: int, group: int) -> None:
        self.counts[int(group), int(y)] += 1.0

    def stable(self, group: int) -> float:
        return float(self.counts[int(group)].sum())

    # 批量巩固一个语种段的全部目标字节
    def consolidate_segment(self, ys: np.ndarray, group: int) -> None:
        for y in ys:
            self.consolidate(int(y), group)

    # ------------------------------------------------------------------
    # 读取：recall 返回概率原型；mem_logit 返回 log-prob 先验
    # ------------------------------------------------------------------
    def _proto_prob(self, group: int) -> np.ndarray | None:
        cnt = self.counts[int(group)]
        if cnt.sum() <= 0:
            return None
        c = cnt + self.smooth
        return c / c.sum()

    def recall(self, group: int) -> torch.Tensor:
        p = self._proto_prob(group)
        if p is None:
            p = np.full(self.C, 1.0 / self.C)
        return torch.as_tensor(p, dtype=torch.float32, device=self.device)

    def mem_logit(self, group: int) -> torch.Tensor | None:
        """返回 (C,) log-prob 先验向量；该组无样本时返回 None（→ 不加先验）。"""
        p = self._proto_prob(group)
        if p is None:
            return None
        return torch.as_tensor(np.log(p), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # 自识别闭环：从输入窗口字节直方图判语种 → 取对应原型（无需标签）
    # 只用记忆自身固化的原型 + 模型输入，即可闭环"找到该注入哪个原型"。
    # ------------------------------------------------------------------
    def _proto_mat(self) -> np.ndarray:
        """(n_groups, C) 归一概率原型矩阵（含 smooth）；无样本组按均匀先验。"""
        protos = np.zeros((self.n_groups, self.C), np.float64)
        for g in range(self.n_groups):
            cnt = self.counts[g]
            if cnt.sum() <= 0:
                protos[g] = 1.0 / self.C
            else:
                c = cnt + self.smooth
                protos[g] = c / c.sum()
        return protos

    def selfid(self, X_win_np: np.ndarray) -> int:
        """单 16 字节窗口 -> 判语种 index：窗口字节直方图 距离最近 语种原型（余弦）。

        这是闭环接入的自识别开关：只用窗口输入字节 + 记忆原型，不取真实标签。
        窗口 insufficient 时（段首/边界）仍可能判错，恰好度量"自识别闭环"的代价。
        """
        hist = X_win_np.sum(axis=0).astype(np.float64)          # (256,) 窗口字节次数
        hn = hist / (np.linalg.norm(hist) + 1e-12)
        protos = self._proto_mat()
        pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - pn @ hn                                       # (n_groups,) 余弦距离
        return int(d.argmin())


# ----------------------------------------------------------------------
# 带记忆的逐样本判分（两臂共用），再子集化算 total/switch/within
# ----------------------------------------------------------------------
def eval_scores(m, X_np: np.ndarray, y_np: np.ndarray, tau: float,
                mem: LangSlowMem | None = None, groups=None, beta: float = 0.0,
                selfid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """返回 (nll_per, acc_per)：逐样本负对数似然(bits)与是否命中。

    无记忆（mem=None 或 beta=0）== 原 eval_batch 口径。
    有记忆：
      - selfid=False：每样本 recall groups[i]（真实语种 label，= oracle recall）
      - selfid=True ：每样本用窗口字节直方图自识别语种后再 recall
        （闭环接入：不依赖标签，只靠窗口输入 + 记忆固化的原型）。
    logit_mem = logit + beta*tau*logprotop。
    """
    m.learning = False
    n = len(y_np)
    nll = np.zeros(n, np.float64)
    acc = np.zeros(n, np.bool_)
    for i in range(n):
        m._infer(X_np[i].ravel(), None, None)
        logit = torch.mv(m.W_out.t(), m._x2) + m.b_out
        if mem is not None and beta != 0.0:
            g = mem.selfid(X_np[i]) if selfid else int(groups[i])
            lp = mem.mem_logit(g)
            if lp is not None:
                logit = logit + beta * tau * lp
        p = torch.softmax(logit / tau, dim=0)
        py = float(p[y_np[i]])
        nll[i] = -np.log2(max(py, 1e-12))
        acc[i] = bool(int(p.argmax().item()) == int(y_np[i]))
    m.learning = True
    return nll, acc


def selfid_acc(mem: LangSlowMem, X_np: np.ndarray, groups) -> np.ndarray:
    """闭环自识别判语种命中率：只用窗口 + 原型，对照真实语种。"""
    n = len(groups)
    hit = np.zeros(n, np.bool_)
    for i in range(n):
        hit[i] = bool(mem.selfid(X_np[i]) == int(groups[i]))
    return hit


def _agg(nll: np.ndarray, acc: np.ndarray, mask: np.ndarray | None = None
         ) -> tuple[float, float]:
    if mask is not None:
        nll, acc = nll[mask], acc[mask]
    if len(nll) == 0:
        return float("nan"), float("nan")
    return float(nll.mean()), float(acc.mean())


def eval_nomem(m, X_np, y_np, tau) -> tuple[float, float]:
    """无记忆对照（包装原版 eval_batch 口径，避免重复）。"""
    nll, acc = eval_scores(m, X_np, y_np, tau, mem=None, beta=0.0)
    return float(nll.mean()), float(acc.mean())


def eval_batch_mem(m, X_np, y_np, tau, mem, groups, beta=1.0
                   ) -> tuple[float, float]:
    """带记忆评估（任务要求的 eval_batch_mem，beta 可扫）。"""
    nll, acc = eval_scores(m, X_np, y_np, tau, mem, groups, beta)
    return float(nll.mean()), float(acc.mean())


# ----------------------------------------------------------------------
# F1a：上下文相关的结构化语言记忆（n-gram 条件字节先验）
# ----------------------------------------------------------------------
class NgramLangMem:
    """F1a 结构化慢记忆：把 E3 的"静态 unigram 原型"升级为"n-gram 条件字节先验"。

    动机（承接 E3 判据① FAIL）：unigram 原型只携带"整段语种字节频率"，重建不了旧语种
    被后训练覆盖的**词级/短程结构**。要在记忆里承载这种结构，先验必须**上下文相关**——
    给定窗口尾部 `order` 个字节 → 预测下一字节的条件分布。n-gram 把语种流的局部
    统计（"该语种的字/短序列长什么样"）固化进 slow 记忆，恰好是 unigram 缺的那层。

    记忆内容 = 该语种**训练段自身**的条件字节计数（语言自带的结构/统计固化成原型的
    "稳定的、重复的"部分），并离线从 train 段确定性重建（与 E0/E3 同款确定性口径）。

    参考实现细节：
      - 每语种 group 一个 dict：prefix_bytes(0..order-1) -> (256,) 计数；
      - 同时存 unigram 计数（用作**回退** + selfid 判语种的原型）；
      - 读出：p(y) = λ·p_cond(y|tail) + (1-λ)·p_unigram(y)；tail=窗口最后 order 字节；
        未见过的前缀 → 回退到 unigram（无信息则近均匀）。
      - 线性插值（λ 权）比硬 Katz 回退更稳，且 order=0 时自动退化为 unigram（复现 E3）。
      注入公式不变：logit_mem = logit + beta·tau·log p(y|tail,lang)。beta=0 即无记忆。
    """

    def __init__(self, n_groups: int, C: int = 256, order: int = 2,
                 device: str = "cuda", smooth: float = 1e-4, lam: float = 0.9):
        self.n_groups = int(n_groups)
        self.C = int(C)
        self.order = int(order)                 # n-gram 阶数（容量轴：0=unigram 基线）
        self.device = torch.device(device)
        self.smooth = float(smooth)             # 计数平滑（防 log0）
        self.lam = float(lam)                   # 条件与 unigram 的线性插值权
        self.unigram = np.zeros((n_groups, C), np.float64)
        self.cond: list[dict] = [None] * n_groups   # group -> {prefix: counts}

    # ------------------------------------------------------------------
    # 写入：从该语种训练段字节序列确定性重建计数
    # ------------------------------------------------------------------
    def build_from_segments(self, segs) -> None:
        for gi, (_lang, b) in enumerate(segs):
            self._build_group(gi, b)

    def _build_group(self, gi: int, b: np.ndarray) -> None:
        self.unigram[gi] += np.bincount(b, minlength=self.C).astype(np.float64)
        if self.order == 0:
            return
        d: dict = {}
        o = self.order
        for t in range(o, len(b) - 1):
            pref = b[t - o:t].tobytes()
            y = int(b[t])
            row = d.get(pref)
            if row is None:
                row = np.zeros(self.C, np.float64)
                d[pref] = row
            row[y] += 1.0
        self.cond[gi] = d

    # ------------------------------------------------------------------
    # 读取：上下文相关的 log-prob 先验
    # ------------------------------------------------------------------
    def mem_logit(self, win_bytes: np.ndarray, group: int) -> torch.Tensor:
        """返回 (C,) log-prob 先验向量（给定窗口尾部 + 语种）。

        win_bytes: (W,) uint8 窗口字节（含尾部 `order` 字节作为条件）。group: 语种索引。
        """
        sm = self.smooth
        C = self.C
        uni = self.unigram[group]
        p_uni = (uni + sm) / (uni.sum() + sm * C)
        if self.order > 0 and self.cond[group] is not None:
            pref = win_bytes[-self.order:].tobytes()
            row = self.cond[group].get(pref)
            if row is not None:
                p_cond = (row + sm) / (row.sum() + sm * C)
                p = self.lam * p_cond + (1.0 - self.lam) * p_uni
                p = p / p.sum()
                return torch.as_tensor(np.log(p), dtype=torch.float32,
                                       device=self.device)
        # 未见前缀 / order=0 → 回退 unigram
        return torch.as_tensor(np.log(p_uni), dtype=torch.float32,
                               device=self.device)

    # ------------------------------------------------------------------
    # 自识别闭环：窗口字节直方图 vs 各语种 marginal(unigram) 原型（余弦）
    # ------------------------------------------------------------------
    def _proto_mat(self) -> np.ndarray:
        sm = self.smooth
        protos = np.zeros((self.n_groups, self.C), np.float64)
        for g in range(self.n_groups):
            c = self.unigram[g] + sm
            protos[g] = c / c.sum()
        return protos

    def selfid(self, X_win_np: np.ndarray) -> int:
        hist = X_win_np.sum(axis=0).astype(np.float64)
        hn = hist / (np.linalg.norm(hist) + 1e-12)
        pn = self._proto_mat()
        pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - pn @ hn
        return int(d.argmin())

    def stable(self, group: int) -> float:
        return float(self.unigram[group].sum())