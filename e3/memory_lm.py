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


# ----------------------------------------------------------------------
# 带记忆的逐样本判分（两臂共用），再子集化算 total/switch/within
# ----------------------------------------------------------------------
def eval_scores(m, X_np: np.ndarray, y_np: np.ndarray, tau: float,
                mem: LangSlowMem | None = None, groups=None, beta: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """返回 (nll_per, acc_per)：逐样本负对数似然(bits)与是否命中。

    无记忆（mem=None 或 beta=0）== 原 eval_batch 口径。
    有记忆：每样本 recall groups[i] 语种原型先验，logit_mem = logit + beta*tau*logprotop。
    """
    m.learning = False
    n = len(y_np)
    nll = np.zeros(n, np.float64)
    acc = np.zeros(n, np.bool_)
    for i in range(n):
        m._infer(X_np[i].ravel(), None, None)
        logit = torch.mv(m.W_out.t(), m._x2) + m.b_out
        if mem is not None and beta != 0.0:
            lp = mem.mem_logit(int(groups[i]))
            if lp is not None:
                logit = logit + beta * tau * lp
        p = torch.softmax(logit / tau, dim=0)
        py = float(p[y_np[i]])
        nll[i] = -np.log2(max(py, 1e-12))
        acc[i] = bool(int(p.argmax().item()) == int(y_np[i]))
    m.learning = True
    return nll, acc


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