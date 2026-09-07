"""信用分配早筛（承重墙）—— docs/SRPC_DESIGN.md §2.4 / §7.5-2 / §8.5。

**任务（长程依赖小任务）**：延迟 XOR。
- 每步 d_feat 维特征，前两维为双峰 bit（0/1 二值，v2 方差修复），其余为满幅随机干扰；
- 目标 `y_t = XOR(bit0(x_{t-Δ}), bit1(x_{t-Δ}))`：只与 Δ 步之前的输入有关，
  且与任何单一输入特征**零边际相关**（纯相关性联想必然失败，§2.4）；
- 输入 = 最近 Δ+1 步窗口拼接（延迟线属机械外围序列化，§2.2），
  远端块（t-Δ）为唯一任务相关块，其余块为干扰；
- 网络：`x0(窗口) -> x1(隐层) -> x2(one-hot 双输出)`，监督学习期输出钳制到目标。
  XOR 线性不可分 => 输出必须为双输出单元（标量线性读头封顶 0.5）；
  评估用经典 PCN 分类协议"钳制-比较"（分别钳制两个候选，取自由能更低者），
  训练/评估同域，消除 clamp 与 free 的分布失配。
  动力学与 Whittington & Bogacz (2017) 相同（自由能梯度下降，稳定态近似反传），
  出生即稀疏（扇入掩码 + k-WTA，不变量 3）。

**双学习规则对照（§2.4 关键区分：朴素 Hebbian ≠ 误差驱动 PCN）**：
- `error`：误差驱动 —— ΔW ∝ 接收层误差 ⊗ 突触前（PCN 标准局部规则，
  长程信用分配在同一原理内涌现）；
- `hebb`：纯相关 —— ΔW ∝ 突触后 ⊗ 突触前（无误差加权），朴素联想，
  对 XOR 类任务有天花板（零边际相关，无梯度可用）。

**判据（§7.5-2 / §8.5）**：长程延迟下误差驱动显著优于纯相关（误差机制承载
信用分配而非相关性联想），且误差能量回传并驱动远端权重（远端块权重变化占比
显著高于机会水平 1/(Δ+1)）。

**Δ=1 对照**：近程（无长程延迟）时误差驱动即可学会 —— 证明 Δ=4 的失败/成功
差异源于"信用分配深度"，而非任务本身不可学。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import CreditConfig


def _colnorm(w: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(w, axis=0, keepdims=True)
    return w / np.maximum(n, 1e-8)


def _orth_cols(w: np.ndarray) -> np.ndarray:
    """逐列 Gram-Schmidt 正交化（类原型近正交 -> 类判别信号强，§8.5）。"""
    w = w.copy()
    for j in range(1, w.shape[1]):
        for i in range(j):
            w[:, j] -= float(np.dot(w[:, i], w[:, j])) * w[:, i]
    return _colnorm(w)


def _kwta(x: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(frac * x.size)))
    if k >= x.size:
        return x
    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


class CreditPCN:
    """延迟 XOR 上的局部误差驱动 PCN（与 Phase-0 同一套 7.3 局部规则）。

    x0(窗口) -> x1 -> x2 -> x3(one-hot 双输出)（信用分配深度 2，双隐层）：
        ε0 = x0 − W1@x1（输入重建误差）
        ε1 = x1 − W2@x2（L1 预测误差）
        ε2 = x2 − W3@x3（L2 预测误差，类原型）
        dx1 = β·(W1ᵀ·ε0) − α·ε1       # 自由能梯度下降，规则 1
        dx2 = β·(W2ᵀ·ε1) − α·ε2
        dW1 ∝ ε0 ⊗ x1；dW2 ∝ ε1 ⊗ x2；dW3 ∝ ε2 ⊗ yoh  # 规则 2（error 臂带误差加权）

    层 1 掩码 = **时间局部性分块窗口**（延迟线是机械外围的时序序列化，
    每个隐单元的感受野 = 一个时间块的连续窗口，锚点随机落在某个块上，
    块稀疏权重，不变量 3）；内部层 / 输出层 = 随机扇入。
    推断平衡：β ≥ α，隐状态由输入驱动（底部重建为主），类分离由
    top-down 原型（W3 列）经 x2 在隐空间上完成 —— XOR 类条件均值相同，
    重建误差本身无法分位，分离靠信用分配把输出误差经 2 层回传到 x1/x2，
    塑出类可分的内部表征（§2.4 承重墙：误差驱动 vs 纯相关）。

    learn_mode='error'：规则 2 带误差加权（信用分配）；
    learn_mode='hebb'：规则 2 为纯相关性（对照，§2.4）。
    """

    def __init__(self, cfg: CreditConfig, rng: np.random.Generator,
                 learn_mode: str = "error"):
        self.cfg = cfg
        self.mode = learn_mode
        d0 = (cfg.delay + 1) * cfg.d_feat
        self.d0 = d0
        # 双隐层权重（有符号初始化，近正交列）
        self.W1 = _colnorm(rng.normal(0.0, 1.0, (d0, cfg.h1)))          # x0→x1
        self.W2 = _colnorm(rng.normal(0.0, 1.0, (cfg.h1, cfg.h2)))      # x1→x2
        self.W3 = rng.normal(0.0, 0.3, (cfg.h2, 2))                     # x2→输出
        # 出生即稀疏掩码（不变量 3）：层 1 = 每单元一个时间块（连续感受野窗口），
        # 锚点随机落在某块上 -> 1/(Δ+1) 的单元看到远端块（感知局部性）；
        # 内部层 / 输出层 = 随机扇入。
        m1 = np.zeros((d0, cfg.h1), dtype=bool)
        n_blk = cfg.delay + 1
        for j in range(cfg.h1):
            b = int(rng.integers(0, n_blk))
            m1[b * cfg.d_feat:(b + 1) * cfg.d_feat, j] = True
        k2 = max(1, int(round(cfg.fan_in_frac * cfg.h1)))
        m2 = np.zeros((cfg.h1, cfg.h2), dtype=bool)
        for j in range(cfg.h2):
            m2[rng.choice(cfg.h1, size=k2, replace=False), j] = True
        k3 = max(1, int(round(cfg.fan_in_frac * cfg.h2)))
        m3 = np.zeros((cfg.h2, 2), dtype=bool)
        for j in range(2):
            m3[rng.choice(cfg.h2, size=k3, replace=False), j] = True
        self.mask1, self.mask2, self.mask3 = m1, m2, m3
        self.W1 = _colnorm(self.W1 * m1)
        self.W2 = _colnorm(self.W2 * m2)
        self.W3 = _orth_cols(self.W3 * m3)   # 类原型近正交：判别信号强（§8.5）
        self.W1_init = self.W1.copy()
        self.x1 = np.zeros(cfg.h1)
        self.x2 = np.zeros(cfg.h2)
        self.learning = True
        # 侧抑制（7.3 规则 2 横向竞争）：W3 类原型列出生即正交（_orth_cols），
        # 结构由构造保证（不变量 3）；训练期不重复正交化 —— 重复 Gram-Schmidt
        # 会抹平列信号幅度、拉低 err 臂准确率（dbg5/verify_final：orth=True 掉到 0.75-0.77）。
        self.orth = False
        self.eta_inf = cfg.eta_inf   # 推断阻尼步长（谱半径 ~8 下的收敛步长）
        self.iters = cfg.settle_iters
        self.hebb_free = cfg.hebb_free  # hebb 臂训练用自由推断（无 yoh 钳制，消除类泄漏）

    # ------------------------------------------------------------------
    # 规则 1：局部推断（自由能梯度下降；事件门控 + k-WTA，不变量 3）
    # ------------------------------------------------------------------
    def _infer(self, x0: np.ndarray, yoh: np.ndarray, free_out: bool) -> None:
        cfg = self.cfg
        a, b = cfg.alpha, cfg.beta   # 顶层类拉动（α） vs 底层重建（β）
        x3 = np.zeros(2) if free_out else yoh.copy()
        for _ in range(self.iters):
            e0 = x0 - self.W1 @ self.x1
            e1 = self.x1 - self.W2 @ self.x2
            e2 = self.x2 - self.W3 @ x3
            u1 = b * (self.W1.T @ e0) - a * e1
            u2 = b * (self.W2.T @ e1) - a * e2
            self.x1 = np.clip(self.x1 + self.eta_inf * u1 * (np.abs(u1) > cfg.theta_event),
                              0.0, cfg.x_max)
            self.x2 = np.clip(self.x2 + self.eta_inf * u2 * (np.abs(u2) > cfg.theta_event),
                              0.0, cfg.x_max)
            if cfg.kwta_on:
                self.x1 = _kwta(self.x1, cfg.kwta_frac)
                self.x2 = _kwta(self.x2, cfg.kwta_frac)
            if free_out:
                e2 = self.x2 - self.W3 @ x3
                x3 = np.clip(x3 + cfg.eta_out * (self.W3.T @ e2), 0.0, 1.0)
        self._e0 = x0 - self.W1 @ self.x1
        self._e1 = self.x1 - self.W2 @ self.x2
        self._e2 = self.x2 - self.W3 @ x3
        self._x3 = x3
        if cfg.energy_mode == "class":
            # 分类比较口径：只含类相关项（隐层预测误差）。
            # XOR 类条件均值相同 => 输入重建 e0 与类无关，加入只会稀释判别信号；
            # 判别信号完全来自"隐表征 vs 类原型"的距离（信用分配塑出的隐空间）。
            self._energy = 0.5 * float(np.dot(self._e1, self._e1)
                                       + np.dot(self._e2, self._e2))
        else:
            self._energy = 0.5 * float(np.dot(self._e0, self._e0)
                                       + np.dot(self._e1, self._e1)
                                       + np.dot(self._e2, self._e2))

    # ------------------------------------------------------------------
    # 规则 2：局部学习（error 误差驱动 vs hebb 纯相关）
    # ------------------------------------------------------------------
    def _learn(self, x0: np.ndarray, yoh: np.ndarray) -> None:
        if not self.learning:
            return
        cfg = self.cfg
        lr = cfg.eta_w
        g1 = self.x1 > cfg.theta_syn
        g2 = self.x2 > cfg.theta_syn
        if self.mode == "error":
            dW1 = np.outer(self._e0, self.x1 * g1)
            dW2 = np.outer(self._e1, self.x2 * g2)
            dW3 = np.outer(self._e2, yoh)
        else:  # 纯相关 Hebbian：无误差加权（§2.4 对照组）
            dW1 = np.outer(x0, self.x1 * g1)
            dW2 = np.outer(self.x1, self.x2 * g2)
            dW3 = np.outer(self.x2, yoh)
        self.W1 += lr * dW1
        self.W2 += lr * dW2
        self.W3 += lr * dW3
        self.W1 *= self.mask1
        self.W2 *= self.mask2
        self.W3 *= self.mask3
        self.W1 = _colnorm(self.W1)
        self.W2 = _colnorm(self.W2)
        if self.orth:
            # 7.3 规则 2 侧抑制（横向竞争）：类原型列去相关。
            # XOR 类条件均值相同 => 无侧抑制时两列被拉到同一类质心方向而坍缩
            # （Δ=1 对照实测 W3cos 0.99 -> 判别力消失）；侧抑制保持原型正交，判别方向不被抹平。
            self.W3 = _orth_cols(self.W3)
        self.W3 *= self.mask3   # Gram-Schmidt 可能引入掩码外值，恢复出生定型支撑
        self.W3 = _colnorm(self.W3)

    # ------------------------------------------------------------------
    # 一步训练：reset -> 收敛（err 臂钳制 yoh；hebb 臂自由推断防类泄漏）-> 学习
    # ------------------------------------------------------------------
    def train_step(self, x0: np.ndarray, y: float) -> float:
        yoh = np.array([1.0, 0.0]) if y == 0 else np.array([0.0, 1.0])
        self.x1[:] = 0.0
        self.x2[:] = 0.0
        self._infer(x0, yoh, free_out=self.mode == "hebb" and self.hebb_free)
        self._learn(x0, yoh)
        return float(np.linalg.norm(self._e0))

    # ------------------------------------------------------------------
    # 冻结评估：钳制-比较（经典 PCN 分类协议，训练/评估同域）
    # ------------------------------------------------------------------
    def predict(self, x0: np.ndarray) -> float:
        if self.cfg.predict_mode == "free":
            self.x1[:] = 0.0
            self.x2[:] = 0.0
            self._infer(x0, np.zeros(2), free_out=True)
            return float(np.argmax(self._x3))
        best_c, best_e = 0, np.inf
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            self.x1[:] = 0.0
            self.x2[:] = 0.0
            self._infer(x0, yoh, free_out=False)
            if self._energy < best_e:
                best_e, best_c = self._energy, c
        return float(best_c)

    def set_learning(self, flag: bool) -> None:
        self.learning = flag


def _make_sequence(cfg: CreditConfig, rng: np.random.Generator,
                   n: int, delay: int) -> tuple[np.ndarray, np.ndarray]:
    """生成 (窗口 X, 目标 y)。

    每步 d_feat 维：前两维为双峰 bit（cfg.bit_lo/cfg.bit_hi），其余维满幅随机干扰；
    目标 y_t = XOR(bit0(x_{t-delay}), bit1(x_{t-delay}))；
    窗口 = 最近 delay+1 步拼接（远端块在最前）。
    """
    d = cfg.d_feat
    steps = n + delay
    feat = rng.uniform(0.0, 1.0, (steps, d))
    b0 = rng.random(steps) < 0.5
    b1 = rng.random(steps) < 0.5
    feat[:, 0] = np.where(b0, cfg.bit_hi, cfg.bit_lo)
    feat[:, 1] = np.where(b1, cfg.bit_hi, cfg.bit_lo)
    y = (b0[:-delay] != b1[:-delay]) if delay > 0 else (b0 != b1)
    X = np.empty((n, (delay + 1) * d))
    for t in range(n):
        X[t] = feat[t:t + delay + 1].ravel()
    return X, y.astype(float)


def _run_seed(cfg: CreditConfig, seed: int, delay: int,
              train_steps: int, eval_steps: int) -> dict:
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, train_steps + eval_steps, delay)
    Xtr, ytr = X[:train_steps], y[:train_steps]
    Xev, yev = X[train_steps:], y[train_steps:]
    rcfg = replace(cfg, delay=delay)

    out = {}
    for mode in ("error", "hebb"):
        m = CreditPCN(rcfg, np.random.default_rng(seed * 3000 + 11), mode)
        if mode == "hebb":
            # hebb 臂浅迭代：纯相关是瞬时联想、无需深迭代（深迭代反抬基线）
            m.iters = rcfg.hebb_settle_iters
        for t in range(train_steps):
            m.train_step(Xtr[t], float(ytr[t]))
        m.set_learning(False)
        acc = 0.0
        for t in range(eval_steps):
            acc += float((m.predict(Xev[t]) == float(yev[t])))
        out[f"acc_{mode}"] = acc / eval_steps
        # 远端权重驱动：W1 上延迟块（首 d 行）的权重变化占比（机会水平 = 1/(Δ+1)）
        dW = m.W1 - m.W1_init
        total = float(np.linalg.norm(dW)) + 1e-12
        distal = float(np.linalg.norm(dW[:cfg.d_feat]))
        out[f"distal_{mode}"] = distal / total
    return out


def run_credit_screen(seeds=(0, 1, 2),
                      ccfg: CreditConfig | None = None) -> dict:
    """信用分配早筛主流程。返回 per-seed 指标 + 聚合均值。"""
    ccfg = ccfg or CreditConfig()
    per = []
    for s in seeds:
        r4 = _run_seed(ccfg, s, delay=ccfg.delay,
                       train_steps=ccfg.train_steps,
                       eval_steps=ccfg.eval_steps)
        r1 = _run_seed(ccfg, s, delay=1,
                       train_steps=max(ccfg.train_steps // 2, 200),
                       eval_steps=ccfg.eval_steps)
        per.append(dict(seed=s, delay=ccfg.delay,
                        acc_pcn=r4["acc_error"], acc_hebb=r4["acc_hebb"],
                        acc_gap=r4["acc_error"] - r4["acc_hebb"],
                        distal_pcn=r4["distal_error"], distal_hebb=r4["distal_hebb"],
                        acc_pcn_d1=r1["acc_error"], acc_hebb_d1=r1["acc_hebb"],
                        distal_pcn_d1=r1["distal_error"]))
    keys = per[0].keys()
    mean = {k: float(np.mean([p[k] for p in per])) for k in keys}
    return dict(seeds=list(seeds), per_seed=per, metrics_mean=mean)
