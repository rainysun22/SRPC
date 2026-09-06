"""阶段 B：深层预测编码网络 DeepSRPC（规模化：Phase-0 的 3 层 -> 可配置 L 层）。

层级（自上而下生成、自下而上只传误差，方向严格分开）：

    s -> x1 -> x2 -> ... -> x_{L-1} -> x_L(=x_self) --Wdyn--> pred_self
                                              |--- 记忆先验（多时间尺度）
                                              |--- 变换条件（ARC-lite）

方向分离（对应"forward / backward_error 两个方向独立"骨架）：
    forward()         自上而下生成预测：preds[L] = pred_self，preds[l] = W_{l+1} @ x_{l+1}
    backward_error()  自下而上只传误差：e_0 = s - ŝ；e_l = x_l - preds[l]；up[l] = W_l^T @ e_{l-1}
    update_states()   事件驱动稀疏更新：|Δ| > θ 的节点才更新（不变量 3）

更新规则（7.3，全部局部、在线、免反传）：
    规则 1 推断：x_l += alpha*(pred_l - x_l) + beta*up_l
    规则 2 学习：ΔW_l ∝ e_{l-1} ⊗ x_l（突触前活跃门控，列归一化）
    规则 3 自省：e_self = x_self - pred_self -> 精度调制 boost + Wdyn 局部 LMS

阶段 B 新增（第 8 节里程碑）：
    - 多时间尺度记忆先验：x_L 更新额外 += gamma_mem*(recall - x_L)，按任务分组免遗忘
    - ARC 映射：编码输入 -> 注入变换条件（固定分块正交码，直接拉动 x_L）-> 条件门控读出
      ŝ = Σ_i cond_i * W_out_i @ x_self（每个任务一个头，只更新当前条件头 -> 片段知识免遗忘）
    - 零样本组合：顺序复合两个已学变换（先 t1 读出、再以 t2 读出）—— 跨时段片段重组
"""
from __future__ import annotations

import numpy as np

from .config import DeepConfig
from .memory import PrototypeMemory


def _colnorm(w: np.ndarray) -> np.ndarray:
    """列单位 L2 归一化（局部操作，防止 Hebbian 权重发散）。"""
    n = np.linalg.norm(w, axis=0, keepdims=True)
    return w / np.maximum(n, 1e-8)


class DeepSRPC:
    """深层 SR-PC。self_loop=False 为自省环关闭的 A/B 对照。"""

    def __init__(self, cfg: DeepConfig, n_actions: int = 0,
                 rng: np.random.Generator | None = None,
                 self_loop: bool = True,
                 memory: PrototypeMemory | None = None):
        self.cfg = cfg
        self.L = cfg.n_layers = len(cfg.dims) - 1
        self.dims = list(cfg.dims)
        self.d_self = self.dims[-1]
        self.na = n_actions
        self.self_loop = self_loop
        self.learning = True
        self.memory = memory
        rng = rng or np.random.default_rng(0)
        self.rng = rng

        # 生成权重（非负部件字典，列单位 L2）
        self.Ws: list[np.ndarray | None] = [None] + [
            _colnorm(rng.uniform(0.5, 1.0, (self.dims[l - 1], self.dims[l])))
            for l in range(1, self.L + 1)
        ]
        # 条件门控读出头（ARC：每个变换任务一个头，ŝ = Σ_i cond_i * W_out_i @ x_self）。
        # 训练任务 i 只更新头 i -> 变换片段知识结构上免遗忘（不变量 2：能力=记忆·拼合）；
        # 组合零样本 = 顺序复用已学头（先头 t1 后头 t2），随机条件 = 多头加权混合（无信息基线）。
        self.W_outs: list[np.ndarray | None] = []   # W_outs[i]: d_out × d_self
        self.d_out: int = 0
        self.last_e_out: float = 0.0
        # 变换条件嵌入（d_self, n_train）：固定分块正交码（非负、不相交支撑）
        self.Uc: np.ndarray | None = None
        # 自省环动力学（x_self 的时间预测器）
        nz = self.d_self + max(n_actions, 0)
        self.Wdyn = rng.normal(0.0, 0.05, (self.d_self, nz))

        # 状态（非负稀疏）
        self.xs = [np.zeros(dim) for dim in self.dims]   # xs[0] 输入缓冲（不更新）
        self.pred_self = np.zeros(self.d_self)
        self.last_z: np.ndarray | None = None
        self.last_action: int | None = None
        self.ema_self = 1e-3
        self.U_action = np.ones(max(n_actions, 1))
        self.n_action = np.zeros(max(n_actions, 1), dtype=int)
        self.cond: np.ndarray | None = None   # 当前变换条件

    # ------------------------------------------------------------------
    # 方向 1：forward —— 自上而下生成预测
    # ------------------------------------------------------------------
    def forward(self) -> list[np.ndarray | None]:
        """自上而下生成预测：preds[L] = pred_self；preds[l] = W_{l+1} @ x_{l+1}。"""
        L, xs, Ws = self.L, self.xs, self.Ws
        preds: list[np.ndarray | None] = [None] * (L + 1)
        preds[L] = self.pred_self
        for l in range(L - 1, 0, -1):
            preds[l] = Ws[l + 1] @ xs[l + 1]      # 上层对下层的生成预测
        preds[0] = Ws[1] @ xs[1]                  # 底层对输入的生成预测 ŝ
        return preds

    # ------------------------------------------------------------------
    # 方向 2：backward_error —— 自下而上只传误差
    # ------------------------------------------------------------------
    def backward_error(self, s: np.ndarray,
                       preds: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """自下而上只传误差。

        errs[0] = s - ŝ；errs[l] = x_l - preds[l]；
        up[l] = W_l^T @ errs[l-1]（经权重转置的误差信号，预测编码标准局部操作）。
        """
        L, xs, Ws = self.L, self.xs, self.Ws
        errs: list[np.ndarray] = [np.zeros(self.dims[0])] * (L + 1)
        up: list[np.ndarray] = [np.zeros(self.dims[0])] * (L + 1)
        errs[0] = s - preds[0]
        for l in range(1, L + 1):
            errs[l] = xs[l] - preds[l]
            up[l] = Ws[l].T @ errs[l - 1]
        return errs, up

    # ------------------------------------------------------------------
    # 规则 1：事件驱动稀疏状态更新
    # ------------------------------------------------------------------
    def update_states(self, preds: list, up: list) -> list[float]:
        """x_l += alpha*(pred_l - x_l) + beta*up_l；|Δ|>θ 才更新（不变量 3）。

        顶层额外两路先验拉动：
        - 记忆先验 gamma_mem*(recall - x_L)：多时间尺度记忆（按任务分组，免遗忘）；
        - 条件先验 beta_cond*(Uc@cond - x_L)：ARC 变换任务标识。
        条件码为固定分块正交码（非负、不相交支撑），任务编码天然分离
        （组合任务的零样本能力来自"变换片段的顺序复合"，见 eval_combination）。
        """
        cfg = self.cfg
        evs = []
        cond_mask = None
        if self.cond is not None and self.Uc is not None:
            cond_target = self.Uc @ self.cond
            cond_mask = cond_target > 1e-8   # 只拉条件码支撑维，其余维留给输入细节
        for l in range(1, self.L + 1):
            u = cfg.alpha * (preds[l] - self.xs[l]) + cfg.beta * up[l]
            if l == self.L:
                if cond_mask is not None:
                    u = u + cfg.beta_cond * cond_mask * (cond_target - self.xs[l])
                if self.memory is not None:
                    u = u + cfg.gamma_mem * (self.memory.recall(self.xs[l], self.group()) - self.xs[l])
            m = np.abs(u) > cfg.theta_event
            self.xs[l] = np.clip(self.xs[l] + u * m, 0.0, cfg.x_max)
            evs.append(float(m.mean()))
        return evs

    # ------------------------------------------------------------------
    # 感知/编码：收敛内部状态
    # ------------------------------------------------------------------
    def observe(self, s: np.ndarray) -> dict:
        cfg = self.cfg
        evs = []
        for _ in range(cfg.inner_iters):
            preds = self.forward()
            errs, up = self.backward_error(s, preds)
            evs = self.update_states(preds, up)
        self.last_evs = evs
        preds = self.forward()
        errs, _ = self.backward_error(s, preds)
        return dict(e0=float(np.linalg.norm(errs[0])), ev1=evs[-1],
                    ev_self=evs[-1], evs=evs)

    # ------------------------------------------------------------------
    # 规则 2：局部 Hebbian 学习（活跃门控，免反传）
    # ------------------------------------------------------------------
    def learn(self, s: np.ndarray, boost: float = 1.0) -> None:
        if not self.learning:
            return
        cfg = self.cfg
        preds = self.forward()
        errs, _ = self.backward_error(s, preds)
        lr = cfg.eta_w * boost
        for l in range(1, self.L + 1):
            g = self.xs[l] > cfg.theta_syn
            self.Ws[l] += lr * np.outer(errs[l - 1], self.xs[l] * g)
            np.clip(self.Ws[l], 0.0, None, out=self.Ws[l])
            self.Ws[l] = _colnorm(self.Ws[l])

    # ------------------------------------------------------------------
    # 规则 3：自省环（e_self -> 精度调制 -> Wdyn LMS -> 记忆巩固）
    # ------------------------------------------------------------------
    def reflect(self, action: int | None = None) -> tuple[float, float]:
        """自省环。返回 (boost, ||e_self||)。"""
        cfg = self.cfg
        boost = 1.0
        e_self = np.zeros(self.d_self)
        if self.self_loop:
            e_self = self.xs[self.L] - self.pred_self
            nrm = float(np.linalg.norm(e_self))
            surprise = nrm / (self.ema_self + 1e-8)
            self.ema_self += cfg.ema_self_rate * (nrm - self.ema_self)
            boost = float(np.clip(1.0 + cfg.kappa_boost * max(surprise - 1.0, 0.0),
                                  1.0, cfg.boost_max))
            if self.learning and self.last_z is not None:
                gate = self.last_z > cfg.theta_syn
                self.Wdyn *= (1.0 - cfg.dyn_decay)
                self.Wdyn += cfg.eta_dyn * np.outer(e_self, self.last_z * gate)
            if self.memory is not None:
                # 置信 = 读出误差的负指数（任务执行得好 -> 信念可信 -> 巩固进记忆）
                stable = float(np.exp(-self.last_e_out / max(self.d_out, 1.0) ** 0.5))
                self.memory.consolidate(self.xs[self.L], stable, self.group())
        return boost, float(np.linalg.norm(e_self))

    # ------------------------------------------------------------------
    # ARC 映射：条件码 + 读出 + 读出学习
    # ------------------------------------------------------------------
    def group(self) -> int | None:
        """当前条件的任务组（one-hot 条件的 argmax；无/组合条件时 None）。"""
        if self.cond is None or self.Uc is None:
            return None
        i = int(np.argmax(self.cond))
        return i if self.cond[i] > 0.5 else None

    def set_condition(self, cond: np.ndarray) -> None:
        """注入变换条件（训练 one-hot 或组合相加向量）。

        Uc 为固定分块正交码：每个任务占一段不相交的维（块内归一），
        非负、正交 —— 任务编码天然分离，无需嵌入学习（变换知识在 W_out）。
        组合任务条件 = 成分码相加（两个块的并集），见 eval_combination 的顺序复合。
        """
        n_cond = len(cond)
        if self.Uc is None or self.Uc.shape[1] != n_cond:
            b = max(1, self.d_self // n_cond)
            Uc = np.zeros((self.d_self, n_cond))
            for i in range(n_cond):
                Uc[i * b:(i + 1) * b, i] = 1.0 / np.sqrt(b)
            self.Uc = Uc
        self.cond = cond.copy()

    def head_idx(self) -> int | None:
        """当前条件的读出头索引（one-hot 条件 -> 对应头；非 one-hot -> None）。"""
        if self.cond is None or self.Uc is None:
            return None
        i = int(np.argmax(self.cond))
        return i if self.cond[i] > 0.5 else None

    def readout(self) -> np.ndarray:
        """ŝ = Σ_i cond_i * (W_out_i @ x_self)（条件门控读出）。

        one-hot 条件退化为对应单头；组合/随机条件（非 one-hot）为多头加权
        混合 —— 无信息条件的零样本基线，验证"正确片段重组"优于无信息混合。
        """
        idx = self.head_idx()
        if idx is not None and idx < len(self.W_outs) and self.W_outs[idx] is not None:
            return self.W_outs[idx] @ self.xs[self.L]
        out = np.zeros(self.d_out)
        if self.cond is not None:
            for i, w in enumerate(self.W_outs):
                if w is not None and i < len(self.cond):
                    out = out + self.cond[i] * (w @ self.xs[self.L])
        return out

    def learn_readout(self, s_out: np.ndarray) -> tuple[float, np.ndarray]:
        """读出学习：只更新当前条件对应的头（ΔW_out_i = lr * e ⊗ x_L，局部外积，免反传）。

        训练任务 i 只更新头 i -> 变换片段知识结构上免遗忘（阶段 B 里程碑）。
        """
        cfg = self.cfg
        idx = self.head_idx()
        if idx is None:
            e = s_out - self.readout()
        else:
            while len(self.W_outs) <= idx:
                self.W_outs.append(None)
            if self.W_outs[idx] is None:
                self.W_outs[idx] = _colnorm(self.rng.uniform(
                    0.0, 0.5, (len(s_out), self.d_self)))
                self.d_out = len(s_out)
            e = s_out - (self.W_outs[idx] @ self.xs[self.L])
            if self.learning:
                g = self.xs[self.L] > cfg.readout_gate
                self.W_outs[idx] += cfg.eta_wout * np.outer(e, self.xs[self.L] * g)
                self.W_outs[idx] = _colnorm(self.W_outs[idx])
        self.last_e_out = float(np.linalg.norm(e))
        return self.last_e_out, e

    def step_mapping(self, s_in: np.ndarray, s_out: np.ndarray,
                     cond: np.ndarray | None = None) -> dict:
        """映射任务一步：注入条件 -> 重置 -> 生成先验 -> 编码 -> 读出学习 -> 自省。

        ARC 任务样本为 i.i.d.（每步独立输入-输出对，无时序依赖），
        每步从干净状态收敛，避免上一步状态污染本步编码。
        """
        if cond is not None:
            self.set_condition(cond)
        self.reset_states()
        self.prepare_next(action=None)   # 用当前条件生成 top-down 先验
        self.observe(s_in)
        self.learn(s_in)
        e_out, e_out_v = self.learn_readout(s_out)
        boost, e_self = self.reflect(action=None)
        return dict(e_readout=e_out, e_self=e_self, boost=boost,
                    evs=self.last_evs)

    def apply_transform(self, s_in: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """对输入施加一个已学变换（编码 + 读出），用于顺序复合评估。

        零样本组合 = 依次调用两次 apply_transform（先 t1 后 t2）：
        复用两个已学变换片段，重组出新变换 —— 跨时段片段重组（阶段 B 里程碑）。
        """
        self.set_condition(cond)
        self.reset_states()
        self.prepare_next(action=None)
        self.observe(s_in)
        return self.readout()

    def reset_states(self) -> None:
        """重置内部状态（评估时从干净状态收敛）。"""
        for l in range(1, self.L + 1):
            self.xs[l] = np.zeros_like(self.xs[l])

    # ------------------------------------------------------------------
    # 自省时间对齐
    # ------------------------------------------------------------------
    def prepare_next(self, action: int | None = None) -> None:
        """生成对当前步的自我预测（时间对齐）。

        pred_self = Wdyn @ z + beta_cond * (Uc @ cond)：
        条件作为 top-down 生成先验注入（ARC 变换）；组合条件 c1+c2
        对应两个嵌入方向相加 —— 组合泛化由"表征空间可加性"涌现（可证伪）。
        """
        cfg = self.cfg
        if self.na > 0 and action is not None:
            oh = np.zeros(self.na)
            oh[action] = 1.0
            z = np.concatenate([self.xs[self.L], oh])
        else:
            z = self.xs[self.L].copy()
        self.last_z = z
        if self.self_loop:
            p = self.Wdyn @ z
            if self.cond is not None and self.Uc is not None:
                p = p + cfg.beta_cond * (self.Uc @ self.cond)
            self.pred_self = np.clip(p, 0.0, cfg.x_max)
        else:
            self.pred_self = self.xs[self.L].copy()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def set_learning(self, flag: bool) -> None:
        self.learning = flag

    def snapshot(self) -> dict:
        return dict(
            Ws=[None] + [w.copy() for w in self.Ws[1:]],
            W_outs=[None if w is None else w.copy() for w in self.W_outs],
            d_out=self.d_out,
            Uc=None if self.Uc is None else self.Uc.copy(),
            Wdyn=self.Wdyn.copy(),
            xs=[x.copy() for x in self.xs],
            pred_self=self.pred_self.copy(),
            ema_self=self.ema_self,
            cond=None if self.cond is None else self.cond.copy(),
        )

    def restore(self, snap: dict) -> None:
        for k, v in snap.items():
            if k == "Ws":
                setattr(self, k, [None] + [x.copy() for x in v[1:]])
            elif k == "xs":
                setattr(self, k, [x.copy() for x in v])
            elif k == "W_outs":
                setattr(self, k, [None if w is None else w.copy() for w in v])
            else:
                setattr(self, k, v.copy() if hasattr(v, "copy") else v)
