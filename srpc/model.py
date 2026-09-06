"""SR-PC（自省式预测编码）核心模型 —— Phase-0 实现（docs/SRPC_DESIGN.md 第 7 节）。

层级结构（7.2 模块 1/2，自上而下生成、自下而上只传误差）：

    [自省环]  Wdyn: (x_self, a) -> 预测 x_self(t+1)
        x_self --Ws2--> x2 --W21--> x1 --W10--> s_hat ≈ s
        （Wdyn 坐在顶层之上：自我预测器；不变量 4：自我模型从一开始就在）

三条更新规则（7.3，全部局部、在线、免反传）：

    规则 1（推断，稀疏/事件驱动）：
        x_l += alpha*(上层预测 - x_l)          # "来自上层的预测误差"拉动
              + beta * W^T @ 下层误差           # "来自下层的（误差）信号"
        只有 |更新| > theta_event 的节点参与（不变量 3：能量内生·稀疏）。

    规则 2（学习，局部 Hebbian）：
        ΔW ∝ 突触前活动 × 接收层误差
        仅突触前活跃（> theta_syn）的列更新；权重列归一化保持有界。

    规则 3（自省环）：
        e_self = x_self(t) - pred_self(t-1 -> t)
        - Wdyn 以局部 LMS（Widrow-Hoff）在线修正自我模型；
        - ||e_self|| 高于基线（surprise）时提升学习精度（快速修正信念）；
        - 每动作维护"预期自省误差"EMA，用于主动推理动作选择。

动作选择（7.2 模块 5）：mu = argmin_a E[e_self | a]（ε-探索/利用均衡）。
全局工作空间（7.2 模块 4，MVP）：高置信收敛信念子集（x2 活跃度/(1+|误差|)
的 top-k），为阶段 D 的全局广播留接口。
"""
from __future__ import annotations

import numpy as np

from .config import ModelConfig


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


class SRPCModel:
    """SR-PC 认知核心。self_loop=False 为自省环关闭的 A/B 对照：
    结构与参数量完全一致，仅停用自我预测器/精度调制/主动推理动作选择。"""

    def __init__(self, cfg: ModelConfig, n_actions: int,
                 rng: np.random.Generator, self_loop: bool = True):
        self.cfg = cfg
        self.na = n_actions
        self.self_loop = self_loop
        self.learning = True

        # 生成权重（非负部件字典；Wdyn 为有符号动力学模型）
        self.W10 = _colnorm(rng.uniform(0.5, 1.0, (cfg.d_obs, cfg.n_l1)))
        self.W21 = _colnorm(rng.uniform(0.5, 1.0, (cfg.n_l1, cfg.n_l2)))
        self.Ws2 = _colnorm(rng.uniform(0.5, 1.0, (cfg.n_l2, cfg.n_self)))
        nz = cfg.n_self + max(n_actions, 0)
        self.Wdyn = rng.normal(0.0, 0.05, (cfg.n_self, nz))
        # 结构性稀疏掩码（不变量 3：出生即定型，学习只更新已有突触）
        self.mask10 = self.mask21 = self.mask2s = self.mask_dyn = None
        self.mask_sig: tuple = ()
        self._born_sparse(cfg, rng)

        # 状态（非负稀疏）
        self.x1 = np.zeros(cfg.n_l1)
        self.x2 = np.zeros(cfg.n_l2)
        self.xs = np.zeros(cfg.n_self)

        # 自省环状态
        self.pred_self = np.zeros(cfg.n_self)   # 对当前步的自我预测（上一步末生成）
        self.last_z: np.ndarray | None = None   # 上一步的 Wdyn 输入上下文
        self.last_action: int | None = None
        self.ema_self = 1e-3                    # ||e_self|| 基线
        self.U_action = np.ones(max(n_actions, 1))
        self.n_action = np.zeros(max(n_actions, 1), dtype=int)

    # ------------------------------------------------------------------
    # 结构性稀疏（不变量 3：出生即定型掩码，学习只改已有突触）
    # ------------------------------------------------------------------
    def _born_sparse(self, cfg: ModelConfig, rng: np.random.Generator) -> None:
        """层 1 连续感受野窗口（感知局部性，分块稀疏）+ 内部层/自省随机扇入。

        fan_in_frac=0 为稠密旧路径；kwta_frac 在 observe 中生效。
        """
        if cfg.fan_in_frac > 0:
            f = cfg.fan_in_frac
            k1 = max(1, int(round(f * cfg.d_obs)))
            m1 = np.zeros((cfg.d_obs, cfg.n_l1), dtype=bool)
            for j in range(cfg.n_l1):
                st = int(rng.integers(0, cfg.d_obs - k1 + 1))
                m1[st:st + k1, j] = True
            k2 = max(1, int(round(f * cfg.n_l1)))
            m2 = np.zeros((cfg.n_l1, cfg.n_l2), dtype=bool)
            for j in range(cfg.n_l2):
                m2[rng.choice(cfg.n_l1, size=k2, replace=False), j] = True
            ks = max(1, int(round(f * cfg.n_l2)))
            ms = np.zeros((cfg.n_l2, cfg.n_self), dtype=bool)
            for j in range(cfg.n_self):
                ms[rng.choice(cfg.n_l2, size=ks, replace=False), j] = True
            self.mask10, self.mask21, self.mask2s = m1, m2, ms
            self.W10 = _colnorm(self.W10 * m1)
            self.W21 = _colnorm(self.W21 * m2)
            self.Ws2 = _colnorm(self.Ws2 * ms)
        if cfg.fan_in_dyn_frac > 0:
            kd = max(1, int(round(cfg.fan_in_dyn_frac * cfg.n_self)))
            md = np.zeros(self.Wdyn.shape, dtype=bool)
            for j in range(self.Wdyn.shape[1]):
                md[rng.choice(cfg.n_self, size=kd, replace=False), j] = True
            self.mask_dyn = md
            self.Wdyn = self.Wdyn * md
        self.mask_sig = tuple(
            hash(m.tobytes()) for m in (self.mask10, self.mask21,
                                        self.mask2s, self.mask_dyn)
            if m is not None)

    # ------------------------------------------------------------------
    # 一步在线处理：局部推断 -> 误差 -> 自省 -> 局部学习
    # ------------------------------------------------------------------
    def observe(self, s: np.ndarray) -> dict:
        cfg = self.cfg

        # ---------- 规则 1：局部推断（事件驱动稀疏更新） ----------
        ev1 = ev2 = evs = 0.0
        for _ in range(cfg.inner_iters):
            e0 = s - self.W10 @ self.x1
            x1_hat = self.W21 @ self.x2
            u1 = cfg.alpha * (x1_hat - self.x1) + cfg.beta * (self.W10.T @ e0)
            m1 = np.abs(u1) > cfg.theta_event
            self.x1 = np.clip(self.x1 + u1 * m1, 0.0, cfg.x_max)
            if cfg.kwta_frac > 0:
                self.x1 = _kwta(self.x1, cfg.kwta_frac)

            e1 = self.x1 - x1_hat
            x2_hat = self.Ws2 @ self.xs
            u2 = cfg.alpha * (x2_hat - self.x2) + cfg.beta * (self.W21.T @ e1)
            m2 = np.abs(u2) > cfg.theta_event
            self.x2 = np.clip(self.x2 + u2 * m2, 0.0, cfg.x_max)
            if cfg.kwta_frac > 0:
                self.x2 = _kwta(self.x2, cfg.kwta_frac)

            e2 = self.x2 - x2_hat
            us = cfg.alpha * (self.pred_self - self.xs) + cfg.beta * (self.Ws2.T @ e2)
            ms = np.abs(us) > cfg.theta_event
            self.xs = np.clip(self.xs + us * ms, 0.0, cfg.x_max)
            if cfg.kwta_frac > 0:
                self.xs = _kwta(self.xs, cfg.kwta_frac)

            ev1, ev2, evs = m1.mean(), m2.mean(), ms.mean()

        # 收敛后的误差
        e0 = s - self.W10 @ self.x1
        e1 = self.x1 - self.W21 @ self.x2
        e2 = self.x2 - self.Ws2 @ self.xs

        # ---------- 规则 3：自省环 ----------
        boost = 1.0
        e_self = np.zeros(cfg.n_self)
        if self.self_loop:
            e_self = self.xs - self.pred_self
            nrm = float(np.linalg.norm(e_self))
            surprise = nrm / (self.ema_self + 1e-8)
            self.ema_self += cfg.ema_self_rate * (nrm - self.ema_self)
            # 精度调制：只在 surprise 高于基线时放大学习率（不降低，保证对照公平）
            boost = float(np.clip(1.0 + cfg.kappa_boost * max(surprise - 1.0, 0.0),
                                  1.0, cfg.boost_max))
            # 每动作"预期自省误差"EMA（主动推理的 G(a)，7.2 模块 5）。
            # 久未执行的动作 U 向均值回归 —— 对其后果的预测不确定性随时间回升，
            # 防止一次坏经验导致动作被永久回避（保持探索/利用均衡）。
            if self.na > 0:
                mu = float(self.U_action.mean())
                self.U_action += cfg.u_regress * (mu - self.U_action)
                if self.last_action is not None:
                    a = self.last_action
                    self.U_action[a] += cfg.u_action_rate * (nrm - self.U_action[a])
            # 自我预测器局部 LMS 修正（ΔW ∝ 局部误差 × 突触前活动，免反传）
            if self.learning and self.last_z is not None:
                gate = self.last_z > cfg.theta_syn
                self.Wdyn *= (1.0 - cfg.dyn_decay)
                self.Wdyn += cfg.eta_dyn * np.outer(e_self, self.last_z * gate)
                if self.mask_dyn is not None:
                    self.Wdyn *= self.mask_dyn     # 结构由构造保证

        # ---------- 规则 2：局部 Hebbian 学习（活跃门控） ----------
        evw = 0.0
        if self.learning:
            lr = cfg.eta_w * boost
            g1 = self.x1 > cfg.theta_syn
            self.W10 += lr * np.outer(e0, self.x1 * g1)
            g2 = self.x2 > cfg.theta_syn
            self.W21 += lr * np.outer(e1, self.x2 * g2)
            gs = self.xs > cfg.theta_syn
            self.Ws2 += lr * np.outer(e2, self.xs * gs)
            if self.mask10 is not None:
                self.W10 *= self.mask10          # 结构由构造保证：只更新已有突触
            if self.mask21 is not None:
                self.W21 *= self.mask21
            if self.mask2s is not None:
                self.Ws2 *= self.mask2s
            np.clip(self.W10, 0.0, None, out=self.W10)
            np.clip(self.W21, 0.0, None, out=self.W21)
            np.clip(self.Ws2, 0.0, None, out=self.Ws2)
            self.W10 = _colnorm(self.W10)
            self.W21 = _colnorm(self.W21)
            self.Ws2 = _colnorm(self.Ws2)
            evw = float((g1.mean() + g2.mean() + gs.mean()) / 3.0)

        return dict(
            e0=float(np.linalg.norm(e0)),
            e1=float(np.linalg.norm(e1)),
            e2=float(np.linalg.norm(e2)),
            e_self=float(np.linalg.norm(e_self)),
            boost=boost,
            ev1=float(ev1), ev2=float(ev2), evs=float(evs), evw=evw,
        )

    # ------------------------------------------------------------------
    # 动作选择（7.2 模块 5）
    # ------------------------------------------------------------------
    def select_action(self, eps: float, rng: np.random.Generator) -> int | None:
        """主动推理：mu = argmin 预期自省误差的期望，ε-探索均衡。

        G(a) = U(a) - curiosity * mean(U) / sqrt(1 + n_a)：
        第二项是认识价值（epistemic value）—— 少试的动作不确定性高、
        信息增益大，按 1/sqrt(n) 折价鼓励系统性探索（探索/利用均衡）。
        自省环关闭（A/B 对照）时退化为均匀随机动作（无自我模型可用）。
        """
        if self.na == 0:
            return None
        if not self.self_loop:
            a = int(rng.integers(self.na))
        elif rng.random() < eps:
            a = int(rng.integers(self.na))
        else:
            mean_u = float(self.U_action.mean())
            g = self.U_action - self.cfg.curiosity * mean_u / np.sqrt(1.0 + self.n_action)
            a = int(np.argmin(g + 1e-9 * rng.random(self.na)))
        self.last_action = a
        self.n_action[a] += 1
        return a

    # ------------------------------------------------------------------
    # 生成对 t+1 的自我预测（自省环的时间对齐）
    # ------------------------------------------------------------------
    def prepare_next(self, action: int | None) -> None:
        cfg = self.cfg
        if self.na > 0 and action is not None:
            oh = np.zeros(self.na)
            oh[action] = 1.0
            z = np.concatenate([self.xs, oh])
        else:
            z = self.xs.copy()
        self.last_z = z
        if self.self_loop:
            self.pred_self = np.clip(self.Wdyn @ z, 0.0, cfg.x_max)
        else:
            # 对照：无自我预测器，用零阶持续性先验
            self.pred_self = self.xs.copy()

    # ------------------------------------------------------------------
    # 全局工作空间 MVP（7.2 模块 4，为阶段 D 留接口）
    # ------------------------------------------------------------------
    def workspace(self) -> tuple[np.ndarray, np.ndarray]:
        """高置信收敛信念子集：置信度 = x2 活跃度 / (1 + |误差|) 的 top-k。"""
        cfg = self.cfg
        e2 = self.x2 - self.Ws2 @ self.xs
        conf = self.x2 / (1.0 + np.abs(e2))
        idx = np.argsort(conf)[-cfg.ws_k:]
        g = np.zeros(cfg.n_l2)
        g[idx] = self.x2[idx]
        return g, idx

    # ------------------------------------------------------------------
    # 工具：冻结评估 / 快照
    # ------------------------------------------------------------------
    def set_learning(self, flag: bool) -> None:
        self.learning = flag

    def snapshot(self) -> dict:
        return dict(W10=self.W10.copy(), W21=self.W21.copy(), Ws2=self.Ws2.copy(),
                    Wdyn=self.Wdyn.copy(), x1=self.x1.copy(), x2=self.x2.copy(),
                    xs=self.xs.copy(), pred_self=self.pred_self.copy(),
                    ema_self=self.ema_self, U_action=self.U_action.copy())

    def restore(self, snap: dict) -> None:
        for k, v in snap.items():
            setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)


# ----------------------------------------------------------------------
# 对照基线（Track-2 组合泛化实验）
# ----------------------------------------------------------------------
class FlatPCModel:
    """扁平单层预测编码基线：同样的局部规则，但无层级 / 无概念层 / 无自省。"""

    def __init__(self, d: int, h: int, rng: np.random.Generator,
                 beta: float = 0.25, x_max: float = 5.0,
                 inner_iters: int = 3, theta: float = 1e-2):
        self.W = _colnorm(rng.uniform(0.5, 1.0, (d, h)))
        self.x = np.zeros(h)
        self.beta, self.x_max, self.inner_iters, self.theta = beta, x_max, inner_iters, theta
        self.learning = True

    def observe(self, s: np.ndarray, lr: float = 0.06) -> float:
        for _ in range(self.inner_iters):
            e0 = s - self.W @ self.x
            u = self.beta * (self.W.T @ e0)
            m = np.abs(u) > 1e-4
            self.x = np.clip(self.x + u * m, 0.0, self.x_max)
        e0 = s - self.W @ self.x
        if self.learning:
            g = self.x > self.theta
            self.W += lr * np.outer(e0, self.x * g)
            np.clip(self.W, 0.0, None, out=self.W)
            self.W = _colnorm(self.W)
        return float(np.linalg.norm(e0))

    def set_learning(self, flag: bool) -> None:
        self.learning = flag


class LookupModel:
    """查表记忆基线：上下文 ID -> 观测均值。

    验证"记忆 != 组合"：对训练上下文记忆完好，但对保留组合只能退回全局均值。
    """

    def __init__(self, d: int):
        self.d = d
        self.mean: dict[int, np.ndarray] = {}
        self.gsum = np.zeros(d)
        self.n = 0

    def observe(self, s: np.ndarray, ctx: int, rate: float = 0.05) -> float:
        """在线更新并返回预测误差（先预测后更新）。"""
        pred = self.predict(ctx)
        err = float(np.linalg.norm(s - pred))
        if ctx in self.mean:
            self.mean[ctx] += rate * (s - self.mean[ctx])
        else:
            self.mean[ctx] = s.copy()
        self.gsum += s
        self.n += 1
        return err

    def predict(self, ctx: int) -> np.ndarray:
        if ctx in self.mean:
            return self.mean[ctx]
        if self.n > 0:
            return self.gsum / self.n
        return np.zeros(self.d)
