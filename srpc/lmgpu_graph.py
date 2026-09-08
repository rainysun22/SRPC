"""阶段 E2 GPU 登顶跑加速：LMPCNg 的 CUDA Graph 训练步封装。

动机（实测）：1660Ti 上 LMPCNg.train_step 受 Python/内核调度开销主导
（~30-37ms/步，与宽度几乎无关；拆解为 clamp-infer ~16ms + readout-infer
~13ms + learn ~1.3ms，且均为 ~15 个/迭代的小内核）。逐样本在线协议无法
合并时间步，故用 CUDA Graph 把整个训练步（12 次钳制推断 + 学习 + 8 次
读出推断 + 读出头 LMS + 事件 EMA）捕获为一张图，每步仅一次 replay，
去除逐内核 host 调度。

约束与保真：
    - 三条初衷不变：捕获图内只有局部规则算子（mv/bmm、逐元素、列归一、
      masked_fill、topk），无 autograd、无 global gradient、结构稀疏保持；
    - 权重更新全部改写为**就地**算子（add_/div_/mul_/masked_fill_/copy_），
      捕获图引用固定地址，replay 即执行完整学习步；
    - 运算顺序与 srpc/lmgpu.py 非图版一一对应，仅浮点归约顺序可引入
      ~1e-7 级差异（parity 容差范围，见报告）。
    - 权重/掩码/几何初始化沿用 numpy 真源（同 seed 逐位一致）。

评估路径（eval_batch/fit_tau）不经图，直接复用父类逐窗口自由推断——
评估仅每 eval 间隔发生一次，开销可接受。

用法：
    m = LMPCNgG(cfg, h, rng, device='cuda')   # 构造后即捕获（或 train_step
                                                # 首次调用时惰性捕获）
"""
from __future__ import annotations

import numpy as np
import torch

from .config import E2Config
from .lmgpu import LMPCNg, _kwta2d_t, _kwta_t


class LMPCNgG(LMPCNg):
    """CUDA Graph 版 LMPCNg：仅训练步用图；评估沿用父类。"""

    def __init__(self, cfg: E2Config, h: int, rng: np.random.Generator,
                 eta_w: float | None = None, iters: int | None = None,
                 device: str = "cuda"):
        if device == "cpu" or not torch.cuda.is_available():
            raise ValueError("LMPCNgG 仅支持 CUDA")
        super().__init__(cfg, h, rng, eta_w=eta_w, iters=iters, device=device)
        assert not cfg.kwta_every_iter, "图版未实现逐迭代 k-WTA 分支"
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_ready = False

    # ------------------------------------------------------------------
    # 就地版推断/学习体（与父类算子一致，仅在捕获与热身时运行）
    # ------------------------------------------------------------------
    def _step_body(self) -> None:
        cfg = self.cfg
        a, b_ = cfg.alpha, cfg.beta
        W1c, W2, W3 = self.W1c, self.W2, self.W3
        W1cT = self.W1c.transpose(1, 2)      # 视图（就地更新后自动同步）
        x0rf = self._x0b[self.idx_rf]
        # ---- clamp 推断（信用分配，12 迭代）----
        x1g = torch.zeros((self.W, self.per), dtype=torch.float32,
                          device=self.device)
        x2 = torch.zeros(self.h, dtype=torch.float32, device=self.device)
        th = cfg.theta_event
        for _ in range(self.iters):
            pred0 = torch.matmul(W1c, x1g.unsqueeze(-1)).squeeze(-1)
            e0c = x0rf - pred0
            e1 = x1g.reshape(-1) - torch.mv(W2, x2)
            e2 = x2 - torch.mv(W3, self._yohb)
            u1 = (torch.matmul(W1cT, e0c.unsqueeze(-1)).squeeze(-1)
                  * self.s1).reshape(self.W, self.per)
            u1 = b_ * u1 - a * e1.reshape(self.W, self.per)
            u2 = b_ * torch.mv(W2.t(), e1) - a * e2
            g1 = u1.abs() > th
            g2 = u2.abs() > th
            x1g = (x1g + self.et1 * u1 * g1).clamp(0.0, cfg.x_max)
            x2 = (x2 + self.et2 * u2 * g2).clamp(0.0, cfg.x_max)
        x1g = _kwta2d_t(x1g, cfg.kwta_frac)
        x2 = _kwta_t(x2, cfg.kwta_frac)
        self._ev.add_(((g1.float().mean() + g2.float().mean()) * 0.5
                       - self._ev) * 0.01)
        self._e0c, self._e1, self._e2 = e0c, e1, e2
        self._x1g, self._x2, self._x3 = x1g, x2, self._yohb
        # ---- 学习（误差驱动局部规则，全部就地）----
        g1 = (x1g > cfg.theta_syn).to(x1g.dtype)
        g2 = (x2 > cfg.theta_syn).to(x2.dtype)
        W1c.add_(self.eta_w1 * torch.einsum("bi,bj->bij", e0c, x1g * g1)
                 * self.s1)
        n1 = torch.linalg.vector_norm(W1c, dim=1, keepdim=True)
        if cfg.w1_norm == "unit":
            # 能量守恒：每列强制单位范数（赢者列会把弱输入行挤出至 0）
            W1c.div_(n1.clamp_min(cfg.w1_norm_eps))
        else:
            # clip：列范数仅截上限（≤1），允许弱列自由变弱、不把输入行挤出
            scale = torch.where(n1 > 1.0, 1.0 / n1.clamp_min(1e-12),
                                torch.ones_like(n1))
            W1c.mul_(scale)
        W1c.masked_fill_(self.pad.unsqueeze(-1), 0.0)
        W1c.mul_(self.s1)
        self.W1cT.copy_(W1c.transpose(1, 2))     # 同步连续转置（评估用）
        W2.add_(self.eta_w2 * torch.outer(e1, x2 * g2))
        W2.mul_(self.m2)
        W2.div_(torch.linalg.vector_norm(W2, dim=0, keepdim=True)
                .clamp_min(1e-8))
        W3.add_(self.eta_w3 * torch.outer(e2, self._yohb))
        W3.mul_(self.m3)
        W3.div_(torch.linalg.vector_norm(W3, dim=0, keepdim=True)
                .clamp_min(1e-8))
        if cfg.w3_norm == "unit":
            W3.mul_(cfg.w3_scale)
        # ---- 读出推断（自由 x3，8 迭代）+ 读出头 LMS ----
        x1g = torch.zeros((self.W, self.per), dtype=torch.float32,
                          device=self.device)
        x2 = torch.zeros(self.h, dtype=torch.float32, device=self.device)
        x3 = torch.zeros(self.C, dtype=torch.float32, device=self.device)
        for _ in range(cfg.readout_iters):
            pred0 = torch.matmul(W1c, x1g.unsqueeze(-1)).squeeze(-1)
            e0c = x0rf - pred0
            e1 = x1g.reshape(-1) - torch.mv(W2, x2)
            e2 = x2 - torch.mv(W3, x3)
            u1 = (torch.matmul(W1cT, e0c.unsqueeze(-1)).squeeze(-1)
                  * self.s1).reshape(self.W, self.per)
            u1 = b_ * u1 - a * e1.reshape(self.W, self.per)
            u2 = b_ * torch.mv(W2.t(), e1) - a * e2
            g1 = u1.abs() > th
            g2 = u2.abs() > th
            x1g = (x1g + self.et1 * u1 * g1).clamp(0.0, cfg.x_max)
            x2 = (x2 + self.et2 * u2 * g2).clamp(0.0, cfg.x_max)
            x3 = (x3 + cfg.eta_out * torch.mv(W3.t(), e2)).clamp(0.0, 1.0)
        x1g = _kwta2d_t(x1g, cfg.kwta_frac)
        x2 = _kwta_t(x2, cfg.kwta_frac)
        self._ev.add_(((g1.float().mean() + g2.float().mean()) * 0.5
                       - self._ev) * 0.01)
        self._x2 = x2
        logit = torch.mv(self.W_out.t(), x2) + self.b_out
        p = torch.softmax(logit / cfg.readout_tau, dim=0)
        err = p - self._yohb          # == p − one_hot(y)，可捕获（无标量索引）
        self.W_out.add_(-(cfg.readout_lr * torch.outer(x2, err)))
        self.b_out.add_(-(cfg.readout_lr * err))

    def _capture(self) -> None:
        n_in = self.W * 256 + 1
        self._x0b = torch.zeros(n_in, dtype=torch.float32, device=self.device)
        self._yohb = torch.zeros(self.C, dtype=torch.float32,
                                 device=self.device)
        self._y = torch.tensor(0, dtype=torch.long, device=self.device)
        # 捕获过程会真实更新权重（warm + capture 各跑 body），先快照、后恢复
        snap = {k: getattr(self, k).detach().clone()
                for k in ("W1c", "W1cT", "W2", "W3", "W_out", "b_out")}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):            # 热身：分配器/cuBLAS workspace
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step_body()
        # 恢复初值（图引用同一批地址，replay 从初值继续）
        for k, v in snap.items():
            getattr(self, k).copy_(v)
        self._ev.zero_()
        self._graph_ready = True

    def train_step(self, x0_np: np.ndarray, y: int) -> float:
        """图版训练步：填输入缓冲 + replay（无每步 python 计算开销）。"""
        if not self._graph_ready:
            self._capture()
        x0_flat = np.asarray(x0_np).ravel()
        x0p = np.concatenate([x0_flat, np.zeros(1, np.float32)])
        self._x0b.copy_(torch.from_numpy(x0p), non_blocking=True)
        yoh = np.zeros(self.C, np.float32)
        yoh[y] = 1.0
        self._yohb.copy_(torch.from_numpy(yoh), non_blocking=True)
        self._graph.replay()
        # 周期 W2 谱截断（对因修复）：在两次 graph replay 之间就地写回，
        # 保持图对 W2 同一内存地址的绑定不被破坏。
        self._maybe_cap_w2()
        return 0.0
