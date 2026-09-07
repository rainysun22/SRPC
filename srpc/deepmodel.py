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

阶段 C 新增（软件版内在化，不变量 3：能量内生·结构稀疏）：
    - 出生即结构稀疏：层 1 连续感受野窗口（网格局部性）、内部层/读出头/Wdyn 随机扇入
      受限——掩码在初始化定型，局部学习只更新已有突触（*= mask），结构由构造保证；
    - k-WTA 激活：每层保留 top-k，其余置零（结构性稀疏激活）；
    - 三口径有效 MAC 记账（trace_energy）：事件驱动（活跃单元×已有突触）/
      结构（全单元×已有突触）/ 稠密等价（全连接），硬件无关的内在能耗度量。
"""
from __future__ import annotations

import numpy as np

from .config import DeepConfig
from .energy import EnergyLedger
from .memory import PrototypeMemory


def _colnorm(w: np.ndarray) -> np.ndarray:
    """列单位 L2 归一化（局部操作，防止 Hebbian 权重发散）。"""
    n = np.linalg.norm(w, axis=0, keepdims=True)
    return w / np.maximum(n, 1e-8)


def _kwta(x: np.ndarray, frac: float) -> np.ndarray:
    """k-WTA：保留 top-k 激活，其余置零（结构性稀疏激活，不变量 3）。"""
    k = max(1, int(round(frac * x.size)))
    if k >= x.size:
        return x
    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


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
        # E0-a：层挂载式附加记忆 {层号: PrototypeMemory}（默认空 = 行为零变化）。
        # 顶层记忆只拉 x_L、读出源在底层 -> 中间迭代动力学不下传（B 收尾债务 #1）；
        # 把记忆下移至读出源邻近层（如 x2），先验经 1 层即可抵达读出路径。
        # 与海马-皮层多层投射同构（内嗅皮层/浅层皮层均受海马回投射）。
        self.extra_mems: dict[int, PrototypeMemory] = {}
        rng = rng or np.random.default_rng(0)
        self.rng = rng

        # 生成权重（非负部件字典，列单位 L2）
        self.Ws: list[np.ndarray | None] = [None] + [
            _colnorm(rng.uniform(0.5, 1.0, (self.dims[l - 1], self.dims[l])))
            for l in range(1, self.L + 1)
        ]
        # --- 阶段 C：结构稀疏掩码（出生即定型；fan_in_frac=0 为稠密旧路径） ---
        # masks[l][i, j]：层 l 单元 j 的生成野是否触及下层维 i。
        # 层 1 = 连续感受野窗口（网格一热局部性，分块稀疏）；内部层 = 随机扇入。
        self.masks: list[np.ndarray | None] = [None] * (self.L + 1)
        self.colfan: list[int] = [0] * (self.L + 1)      # 每列非零数（前向记账）
        self.rownnz: list[np.ndarray] = [None] * (self.L + 1)  # 每行非零数（反向记账）
        for l in range(1, self.L + 1):
            if cfg.fan_in_frac > 0:
                k = max(1, int(round(cfg.fan_in_frac * self.dims[l - 1])))
                m = np.zeros((self.dims[l - 1], self.dims[l]), dtype=bool)
                if l == 1:
                    for j in range(self.dims[l]):
                        st = int(rng.integers(0, self.dims[0] - k + 1))
                        m[st:st + k, j] = True
                else:
                    for j in range(self.dims[l]):
                        m[rng.choice(self.dims[l - 1], size=k, replace=False), j] = True
                self.masks[l] = m
                self.colfan[l] = k
                self.rownnz[l] = m.sum(axis=1).astype(float)
                self.Ws[l] = _colnorm(self.Ws[l] * m)
            else:
                self.colfan[l] = self.dims[l - 1]
                self.rownnz[l] = np.full(self.dims[l - 1], float(self.dims[l]))
        # 掩码签名（阶段 C 验收：结构自出生不变，学习只改已有突触）
        self.mask_sig: list = [None if m is None else hash(m.tobytes())
                               for m in self.masks]
        # 条件门控读出头（ARC：每个变换任务一个头，ŝ = Σ_i cond_i * W_out_i @ x_{ro_src}）。
        # 训练任务 i 只更新头 i -> 变换片段知识结构上免遗忘（不变量 2：能力=记忆·拼合）；
        # 组合零样本 = 顺序复用已学头（先头 t1 后头 t2），随机条件 = 多头加权混合（无信息基线）。
        # ro_src：读出源层 = 1（底层 x1，输入保真表征）；顶层 x_self 承载自省/条件/记忆。
        self.ro_src: int = 1
        self.W_outs: list[np.ndarray | None] = []   # W_outs[i]: d_out × dims[ro_src]
        self.d_out: int = 0
        self.last_e_out: float = 0.0
        # 变换条件嵌入（d_self, n_train）：固定分块正交码（非负、不相交支撑）
        self.Uc: np.ndarray | None = None
        # 读出头稀疏掩码（阶段 C：每 self 维扇入受限，头创建时定型）
        self.ro_masks: list[np.ndarray | None] = []
        self.ro_fan: list[int] = []
        self.ro_sig: list = []
        # 读出头 RLS 协方差逆（ro_alg="rls" 时每头一个 P；翻译器内部状态，局部在线）
        self.ro_Ps: list[np.ndarray | None] = []
        # 自省环动力学（x_self 的时间预测器）
        nz = self.d_self + max(n_actions, 0)
        self.dyn_nz = nz
        self.Wdyn = rng.normal(0.0, 0.05, (self.d_self, nz))
        self.dyn_mask: np.ndarray | None = None
        self.dynfan = self.d_self
        if cfg.fan_in_dyn_frac > 0:
            kd = max(1, int(round(cfg.fan_in_dyn_frac * self.d_self)))
            md = np.zeros((self.d_self, nz), dtype=bool)
            for j in range(nz):
                md[rng.choice(self.d_self, size=kd, replace=False), j] = True
            self.dyn_mask = md
            self.Wdyn = self.Wdyn * md
            self.dynfan = kd
        # 三口径 MAC 记账（阶段 C；trace 关闭时零开销）
        self.ledger = EnergyLedger()
        self.trace = bool(cfg.trace_energy)

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
    # 阶段 C：三口径 MAC 记账辅助
    # ------------------------------------------------------------------
    def _mac3(self, key: str, event: float, struct: float, dense: float) -> None:
        """三口径 MAC 记账：事件驱动（活跃×突触）/ 结构（全单元×突触）/ 稠密等价。"""
        if self.trace:
            self.ledger.add(key, event)
            self.ledger.add(key + "_struct", struct)
            self.ledger.add(key + "_dense", dense)

    def _fwd_macs(self, l: int) -> None:
        """层 l -> l-1 生成预测记账（preds[l-1] = Ws[l] @ xs[l]）及活跃单元数。"""
        d_up, d_lo = self.dims[l], self.dims[l - 1]
        n = float(np.count_nonzero(self.xs[l]))
        fan = self.colfan[l]
        self._mac3("fwd", n * fan, d_up * fan, float(d_up * d_lo))
        self.ledger.add(f"act_{l}", n)

    def _ro_src(self) -> np.ndarray:
        """读出源表征（cfg.ro_recon_mode，E0 矩阵）：
        recon = 核心重建 ŝ=W1@x1（符号空间，变换=精确线性映射，读出头逼近置换矩阵）；
        self  = 顶层 x_L（记忆/条件先验直接拉动层，耦合零传播延迟）；
        dual  = concat[ŝ, x_L]（保真走 ŝ、记忆耦合走 x_L）；
        x1    = 原始 x1（旧对照路径）。
        """
        m = self.cfg.ro_recon_mode
        if m == "recon":
            return self.Ws[1] @ self.xs[1]
        if m == "self":
            return self.xs[self.L]
        if m == "dual":
            return np.concatenate([self.Ws[1] @ self.xs[1], self.xs[self.L]])
        return self.xs[self.ro_src]

    def _ro_dsrc(self) -> int:
        """读出源维数：recon=dims[0]；self=dims[L]；dual=dims[0]+dims[L]；x1=dims[ro_src]。"""
        m = self.cfg.ro_recon_mode
        if m == "recon":
            return self.dims[0]
        if m == "self":
            return self.dims[self.L]
        if m == "dual":
            return self.dims[0] + self.dims[self.L]
        return self.dims[self.ro_src]

    def _ro_macs(self, idx: int, n_heads: float = 1.0) -> None:
        """读出头记账（ŝ = W_out @ src；src = 重建 ŝ 或 x1）。"""
        d_src = self._ro_dsrc()
        fan = self.ro_fan[idx] if idx < len(self.ro_fan) else d_src
        n = float(np.count_nonzero(self._ro_src()))
        self._mac3("readout", n_heads * n * fan, n_heads * d_src * fan,
                   n_heads * float(self.d_out * d_src))

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
            if self.trace:
                self._fwd_macs(l + 1)
        preds[0] = Ws[1] @ xs[1]                  # 底层对输入的生成预测 ŝ
        if self.trace:
            self._fwd_macs(1)
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
            if self.trace:
                nz_err = errs[l - 1] != 0
                self._mac3("bwd", float(self.rownnz[l] @ nz_err),
                           float(self.rownnz[l].sum()),
                           float(self.dims[l - 1] * self.dims[l]))
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
            if self.trace:
                nc = self.Uc.shape[1]
                b = self.d_self // nc
                self._mac3("cond", float(np.count_nonzero(self.cond)) * b,
                           float(nc) * b, float(self.d_self * nc))
            cond_mask = cond_target > 1e-8   # 只拉条件码支撑维，其余维留给输入细节
        for l in range(1, self.L + 1):
            u = cfg.alpha * (preds[l] - self.xs[l]) + cfg.beta * up[l]
            if l == self.L:
                if cond_mask is not None:
                    u = u + cfg.beta_cond * cond_mask * (cond_target - self.xs[l])
                if self.memory is not None:
                    proto = self.memory.recall(self.xs[l], self.group())
                    if self.trace:
                        m_ = float(self.memory.last_macs)
                        self._mac3("mem", m_, m_, m_)
                    u = u + cfg.gamma_mem * (proto - self.xs[l])
            elif l in self.extra_mems:
                # E0-a：中间层附加记忆先验（拉动下移，绕开深层传播延迟）
                proto = self.extra_mems[l].recall(self.xs[l], self.group())
                u = u + cfg.gamma_mem * (proto - self.xs[l])
            u = u * cfg.eta_inf   # 推断步长（阻尼，防 W1^TW1 大特征值振荡）
            m = np.abs(u) > cfg.theta_event
            self.xs[l] = np.clip(self.xs[l] + u * m, 0.0, cfg.x_max)
            if cfg.kwta_frac > 0:
                self.xs[l] = _kwta(self.xs[l], cfg.kwta_frac)
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
            act = int(g.sum())
            self.Ws[l] += lr * np.outer(errs[l - 1], self.xs[l] * g)
            if self.masks[l] is not None:
                self.Ws[l] *= self.masks[l]      # 结构由构造保证：只更新已有突触
            np.clip(self.Ws[l], 0.0, None, out=self.Ws[l])
            self.Ws[l] = _colnorm(self.Ws[l])
            if self.trace:
                fan = self.colfan[l]
                self._mac3("learn_w", act * fan, self.dims[l] * fan,
                           float(self.dims[l - 1] * self.dims[l]))

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
                if self.dyn_mask is not None:
                    self.Wdyn *= self.dyn_mask     # 结构由构造保证
                if self.trace:
                    self._mac3("learn_dyn", float(gate.sum()) * self.dynfan,
                               float(self.dyn_nz) * self.dynfan,
                               float(self.d_self * self.dyn_nz))
            if self.memory is not None or self.extra_mems:
                # 置信 = 读出误差的负指数（任务执行得好 -> 信念可信 -> 巩固进记忆）
                stable = float(np.exp(-self.last_e_out / max(self.d_out, 1.0) ** 0.5))
                if self.memory is not None:
                    self.memory.consolidate(self.xs[self.L], stable, self.group())
                    if self.trace:
                        m_ = float(self.memory.last_macs)
                        self._mac3("mem", m_, m_, m_)
                # E0-a：中间层附加记忆同步巩固（同一置信、同组免遗忘结构）
                for l_, mem_ in self.extra_mems.items():
                    mem_.consolidate(self.xs[l_], stable, self.group())
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
        """ŝ = Σ_i cond_i * (W_out_i @ src)（条件门控读出）。

        src = 读出源（cfg.ro_recon_mode，见 _ro_src）。默认 recon = 核心重建
        ŝ=W1@x1（符号空间，翻转/旋转/重着色为精确线性映射，读出头逼近置换矩阵 ->
        符号保真上限由重建质量决定，W1 实证 LS 可达 0.99）。x1 旧路径（随机扇入
        掩码与置换结构冲突，上限 0.74）。E0：self/dual 为记忆-读出耦合实验模式。
        顶层 x_self 保留为自省/条件/记忆承载层（不变量 4）。
        """
        idx = self.head_idx()
        if idx is not None and idx < len(self.W_outs) and self.W_outs[idx] is not None:
            if self.trace:
                self._ro_macs(idx)
            return self.W_outs[idx] @ self._ro_src()
        out = np.zeros(self.d_out)
        if self.cond is not None:
            n_heads = 0
            for i, w in enumerate(self.W_outs):
                if w is not None and i < len(self.cond):
                    out = out + self.cond[i] * (w @ self._ro_src())
                    n_heads += 1
            if self.trace and n_heads:
                self._ro_macs(0, float(n_heads))
        return out

    def learn_readout(self, s_out: np.ndarray) -> tuple[float, np.ndarray]:
        """读出学习：只更新当前条件对应的头（ΔW_out_i = lr * e ⊗ src，局部外积，免反传）。

        src = 读出源（cfg.ro_recon_mode）。训练任务 i 只更新头 i ->
        变换片段知识结构上免遗忘（阶段 B 里程碑）。recon/self/dual 源头为稠密零初始化
        （噪声列不参与 -> 不干扰置换 argmax；随机扇入掩码与置换结构冲突，
        W1 实证 LS+掩码 0.74 vs 稠密 0.99），MAC 以事件驱动记账。
        """
        cfg = self.cfg
        src = self._ro_src()
        idx = self.head_idx()
        if idx is None:
            e = s_out - self.readout()
        else:
            while len(self.W_outs) <= idx:
                self.W_outs.append(None)
                self.ro_masks.append(None)
                self.ro_fan.append(0)
                self.ro_sig.append(None)
                self.ro_Ps.append(None)
            if self.W_outs[idx] is None:
                if cfg.ro_recon_mode in ("recon", "self", "dual"):
                    # recon/self/dual 源：零初始化（噪声列不参与 -> 不干扰置换 argmax）
                    # + 稠密（随机扇入掩码与置换结构冲突，W1 实证 LS+掩码 0.74 vs 稠密 0.99）。
                    self.W_outs[idx] = np.zeros((len(s_out), self._ro_dsrc()))
                else:
                    W = self.rng.uniform(0.0, 0.5, (len(s_out), self._ro_dsrc()))
                    if cfg.fan_in_ro_frac > 0:
                        kro = max(1, int(round(cfg.fan_in_ro_frac * len(s_out))))
                        m = np.zeros(W.shape, dtype=bool)
                        for j in range(self._ro_dsrc()):
                            m[self.rng.choice(len(s_out), size=kro, replace=False), j] = True
                        self.ro_masks[idx] = m
                        self.ro_fan[idx] = kro
                        self.ro_sig[idx] = hash(m.tobytes())
                        W = _colnorm(W * m)
                    else:
                        self.ro_fan[idx] = self._ro_dsrc()
                        W = _colnorm(W)
                    self.W_outs[idx] = W
                if cfg.ro_alg == "rls":
                    # RLS 协方差逆初始化为 δ·I（δ 大 = 信任少，快速起步）
                    self.ro_Ps[idx] = 10.0 * np.eye(self._ro_dsrc())
                self.d_out = len(s_out)
            e = s_out - (self.W_outs[idx] @ src)
            if self.learning:
                if cfg.ro_alg == "rls":
                    # 递归最小二乘（RLS）：逐样本、逐输出维独立、免反传。
                    # 增益 g = P@x/(λ + xᵀP@x) 对所有输出维共享（同一输入回归量），
                    # 每输出维仅用自己的标量误差 e_k 更新 w_k -> 输出维间不互传误差。
                    # 病态 x1 特征上收敛远快于 NLMS（B 收尾 W1 实证，见 config 注释）。
                    P = self.ro_Ps[idx]
                    z = P @ src
                    g = z / (cfg.ro_rls_lam + float(np.dot(src, z)))
                    self.ro_Ps[idx] = (P - np.outer(g, z)) / cfg.ro_rls_lam
                    self.W_outs[idx] += np.outer(e, g)
                else:
                    g = src > cfg.readout_gate
                    # 归一化 LMS（NLMS）：局部、在线、免反传；按 ||src*g||² 归一化
                    # 学习率 -> 收敛对 eta 鲁棒（recon 源 256 维回归条件数差，固定 eta 易发散）
                    denom = float(np.dot(src * g, src * g)) + 1e-8
                    self.W_outs[idx] += (cfg.eta_wout / denom) * np.outer(e, src * g)
                    if self.ro_masks[idx] is not None:
                        self.W_outs[idx] *= self.ro_masks[idx]
                    if cfg.ro_norm == "clip":
                        # 列范数超 ro_norm_cap 时投影回 cap 球面：保稳定性，允许达到
                        # LS 解所需的大尺度（argmax 对尺度不敏感，单位球 cap 会绑死精度）
                        n = np.linalg.norm(self.W_outs[idx], axis=0, keepdims=True)
                        over = n[0] > cfg.ro_norm_cap
                        if over.any():
                            self.W_outs[idx][:, over] *= cfg.ro_norm_cap / np.maximum(n[:, over], 1e-8)
                    elif cfg.ro_norm:
                        self.W_outs[idx] = _colnorm(self.W_outs[idx])
                if self.trace:
                    fan = self.ro_fan[idx]
                    self._mac3("learn_ro", float(g.sum()) * fan if cfg.ro_alg != "rls" else float(self._ro_dsrc()) * fan,
                               float(self._ro_dsrc()) * fan,
                               float(self.d_out * self._ro_dsrc()))
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
            if self.trace:
                self._mac3("dyn", float(np.count_nonzero(z)) * self.dynfan,
                           float(self.dyn_nz) * self.dynfan,
                           float(self.d_self * self.dyn_nz))
            if self.cond is not None and self.Uc is not None:
                p = p + cfg.beta_cond * (self.Uc @ self.cond)
                if self.trace:
                    nc = self.Uc.shape[1]
                    b = self.d_self // nc
                    self._mac3("cond", float(np.count_nonzero(self.cond)) * b,
                               float(nc) * b, float(self.d_self * nc))
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
            ro_Ps=[None if p is None else p.copy() for p in self.ro_Ps],
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
            elif k in ("W_outs", "ro_Ps"):
                setattr(self, k, [None if w is None else w.copy() for w in v])
            else:
                setattr(self, k, v.copy() if hasattr(v, "copy") else v)
