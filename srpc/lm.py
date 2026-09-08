"""阶段 E2：规模化语言训练（µPC 适配 + 1/2/4M 梯子 + BP 孪生对照）。

对应 docs/ROADMAP.md 阶段 E2。任务 = 真实语料的字节级 next-byte 预测
（E1 ByteTokenizer 前端直接复用：256 维一热 = 正交基底），主指标 BPC。
协议 = 顺序流式单样本在线（无 batch / 无回放 / 无 shuffle）。

**LMPCN**（LangPCN 的 LM 化，同一套 7.3 局部规则）：
    x0(W×256 一热) -> x1(h, 时间块感受野) -> x2(h, 随机扇入) -> x3(256 logits)
    e0 = x0rf − s1·W1·x1（块紧凑存储）；e1 = x1 − W2·x2；e2 = x2 − W3·x3
    Δx1 = et1·(s1·W1ᵀ·e0) − α·e1；Δx2 = β·(W2ᵀ·e1) − α·e2   # 规则 1
    ΔW ∝ 误差 ⊗ 突触前（误差驱动，逐层局部）                    # 规则 2
训练钳制 x3=目标 one-hot；评估自由推断读收敛态 x3（LS 读出），
softmax(x3/τ) 计 BPC。

**µPC 适配**（宽度规则锚定 h_ref，推导见 E2Config）：s1 能量前乘子 +
x1 活动步长补偿 + η2/η3 缩放，锚点处超参 = E1 谱系值；迁移有效性由
4M 重调对照裁决（µPC 原文 Fig.5 的 weight/activity lr 零成本迁移协议）。

**W1 块紧凑存储**：每单元感受野 = 相邻 r 个时间块（出生即定型），
按锚点分组存 (W, r·256, per)，einsum 批量前向/反向——计算量 = 结构
参数量（稠密存取的 ~1/8），不变量 3 的算力兑现。

**TwinMLP**：参数量匹配的稠密 MLP + Adam（标准反传参照，不属 SR-PC
构造；GPU_TASKS 契约第 2 条）。

**iPC**：远端上下文块掩码 4→8→16 课程（Salvatori et al. ICLR 2024
增量调度的上下文课程化），1M 档消融。
"""
from __future__ import annotations

import time

import numpy as np

from .config import E2Config
from .credit import _colnorm, _kwta


# ----------------------------------------------------------------------
# 语料与流式窗口（顺序单样本在线协议）
# ----------------------------------------------------------------------
class ByteCorpus:
    """字节语料：位置顺序切分 train/val，滑窗流式供给。

    训练步 t 取窗口 bytes[t : t+W) → 目标 bytes[t+W]（t 顺序递增，
    无 shuffle——在线增量协议）；验证窗口按固定间隔铺满验证段。
    """

    def __init__(self, cfg: E2Config):
        raw = np.frombuffer(open(cfg.corpus_path, "rb").read(), dtype=np.uint8)
        n_val = int(len(raw) * cfg.val_frac)
        self.train = raw[:-n_val]
        self.val = raw[-n_val:]
        self.W = cfg.context
        # 验证窗口（固定，覆盖验证段全程）
        v = self.val
        n_win = min(cfg.eval_windows, max(1, (len(v) - self.W - 1)
                                          // max(1, cfg.eval_stride)))
        starts = np.arange(n_win) * max(1, cfg.eval_stride)
        self.val_x = np.zeros((n_win, self.W, 256), dtype=np.float32)
        self.val_y = np.zeros(n_win, dtype=np.int64)
        for i, t in enumerate(starts):
            self.val_x[i, np.arange(self.W), v[t:t + self.W]] = 1.0
            self.val_y[i] = v[t + self.W]
        # 一元基线（训练段字节频率）
        cnt = np.bincount(self.train, minlength=256).astype(np.float64)
        p = cnt / cnt.sum()
        self.unigram_bpc = float(-np.log2(p[self.val_y]).mean())

    def train_window(self, t: int) -> tuple[np.ndarray, int]:
        """训练步 t 的 (x0 一热 (W,256), y)。t 超出训练段则回绕（多 epoch）。"""
        n = len(self.train) - self.W - 1
        t = t % n
        b = self.train
        x = np.zeros((self.W, 256), dtype=np.float32)
        x[np.arange(self.W), b[t:t + self.W]] = 1.0
        return x, int(b[t + self.W])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# ----------------------------------------------------------------------
# LMPCN：µPC 适配的字节级语言 PCN
# ----------------------------------------------------------------------
class LMPCN:
    """E2 主网络。h = h1 = h2（梯子宽度），µPC 前乘子/步长锚定 h_ref。"""

    def __init__(self, cfg: E2Config, h: int, rng: np.random.Generator,
                 eta_w: float | None = None, iters: int | None = None):
        self.cfg = cfg
        self.h = h
        self.W, self.C = cfg.context, 256
        assert h % self.W == 0, "h 需 16 整除（锚点均衡）"
        self.per = h // self.W          # 每锚点单元数
        self.r = cfg.rf_blocks
        self.iters = iters if iters is not None else cfg.settle_iters
        ew = cfg.eta_w if eta_w is None else eta_w
        # ---- µPC 适配（锚定 h_ref，推导见 E2Config 文档）----
        self.s1 = (cfg.h_ref / h) ** 0.5            # W1 能量前乘子
        self.et1 = cfg.eta_inf * (h / cfg.h_ref) ** 0.5   # x1 活动步长补偿
        self.et2 = cfg.eta_inf
        self.eta_w1 = ew                              # ||e0|| 由 s1 稳住 → 恒定
        self.eta_w2 = ew * (cfg.h_ref / h) ** 0.5     # ||e1|| ∝ √h
        self.eta_w3 = ew * (cfg.h_ref / h) ** 0.5     # ||e2|| ∝ √h
        # ---- W1 块紧凑 (W, r·256, per)：每单元感受野 = 相邻 r 块 ----
        w1 = rng.normal(0.0, 1.0, (self.W, self.r * 256, self.per)
                        ).astype(np.float32)
        w1 /= np.maximum(np.linalg.norm(w1, axis=1, keepdims=True), 1e-8)
        # rf 越界的 pad 行清零（锚点 W-r..W-1 的感受野截断）
        pad = np.zeros((self.W, self.r * 256), dtype=bool)
        for b in range(self.W):
            e = min(b + self.r, self.W)
            pad[b, (e - b) * 256:] = True
        w1[pad] = 0.0
        self.pad = pad
        # x0rf 收集索引（含 pad 槽 d0）
        self.idx_rf = np.full((self.W, self.r * 256), self.W * 256,
                              dtype=np.int64)
        for b in range(self.W):
            e = min(b + self.r, self.W)
            self.idx_rf[b, :(e - b) * 256] = (np.arange(b * 256, e * 256))
        # 顶部散射索引（pred0 块 → x0 位置，pad 尾行丢弃）
        self._scatter = [(b * 256, min(b + self.r, self.W) * 256, b)
                         for b in range(self.W)]
        self.W1c = w1 * self.s1
        self.W1cT = np.ascontiguousarray(self.W1c.transpose(0, 2, 1))
        # ---- W2 (h×h, 随机扇入) / W3 (h×256, 随机扇入) ----
        m2 = np.zeros((h, h), dtype=bool)
        k2 = max(1, int(round(cfg.fan_in_frac * h)))
        for j in range(h):
            m2[rng.choice(h, size=k2, replace=False), j] = True
        m3 = np.zeros((h, self.C), dtype=bool)
        k3 = max(1, int(round(cfg.fan_in_frac * h)))
        for j in range(self.C):
            m3[rng.choice(h, size=k3, replace=False), j] = True
        self.m2, self.m3 = m2, m3
        self.W2 = _colnorm((rng.normal(0.0, 1.0, (h, h)) * m2).astype(np.float32))
        self.W3 = (_colnorm((rng.normal(0.0, 1.0, (h, self.C)) * m3)
                            .astype(np.float32)) * cfg.w3_scale)
        # ---- 状态与账本 ----
        self.x1g = np.zeros((self.W, self.per), dtype=np.float32)
        self.x2 = np.zeros(h, dtype=np.float32)
        # ---- 独立读出头（阶段 B §7.7 翻译器谱系）----
        # W3 是生成权重（钳制塑形 x2 类原型）；读出职责交给 W_out 线性头 +
        # bias：列幅度 ∝ 更新次数 ∝ 频率 => 承载 unigram 先验（LS 收敛
        # 读出会归一化列幅度、在类非均匀下丢先验——E2 与 E1 的差异点）。
        self.W_out = (rng.normal(0.0, 0.01, (h, self.C))).astype(np.float32)
        self.b_out = np.zeros(self.C, np.float32)
        self.learning = True
        self.tau = cfg.tau               # 读出温度（训练固定；评估经校准）
        self.n_params_struct = int(h * self.r * 256 + m2.sum() + m3.sum())
        self.n_params_dense = int(self.W * 256 * h + h * h + h * self.C)
        self.event_frac = 0.0           # 事件率（EMA，稀疏账本）
        self._ev_n = 0

    # ---- 规则 1：推断（训练钳制 / 评估自由 LS 读出）----
    def _infer(self, x0: np.ndarray, yoh: np.ndarray | None,
               block_mask: np.ndarray | None = None,
               clamp: bool = False, iters: int | None = None,
               free_out: bool | None = None) -> None:
        """clamp=True（训练）：x3 钳制目标 one-hot（信用分配塑形 x2）；
        clamp=False（评估/读出）：自由推断取 x2（类信息经 W_out 线性读出，
        判分不走 x3）。x3 收敛仅当 free_out=True（E1 谱系对照路径）。

        iters: 覆盖 self.iters（读出头自由推断用浅迭代省算力）。
        """
        cfg = self.cfg
        a, b_ = cfg.alpha, cfg.beta
        x0p = np.concatenate([x0.ravel(), np.zeros(1, np.float32)])
        if block_mask is not None:
            # 无效块（iPC 远端课程掩码，块粒度）：不可见块字节清零后再收集
            bm = np.repeat(block_mask, 256)
            x0p = x0p * np.concatenate([bm, np.ones(1, np.float32)])
        x0rf = x0p[self.idx_rf]
        x1g = np.zeros_like(self.x1g)
        x2 = np.zeros_like(self.x2)
        x3 = yoh.copy() if clamp else np.zeros(self.C, np.float32)
        W1c, W1cT = self.W1c, self.W1cT
        th = cfg.theta_event
        it = self.iters if iters is None else iters
        do_out = (not clamp) if free_out is None else free_out
        for _ in range(it):
            # e0（块紧凑）：pred0[b] = s1·W1c[b] @ x1g[b]
            pred0 = np.matmul(W1c, x1g[:, :, None])[:, :, 0]
            e0c = x0rf - pred0
            # e1 / e2（平坦方向；x3 = 钳制目标 / 自由读出变量）
            x1 = x1g.ravel()
            e1 = x1 - self.W2 @ x2
            e2 = x2 - self.W3 @ x3
            # Δx1（识别 = s1·W1ᵀe0，能量前乘子一致性）
            u1 = np.matmul(W1cT, e0c[:, :, None])[:, :, 0] * self.s1
            u1 = b_ * u1.reshape(self.W, self.per) - a * e1.reshape(
                self.W, self.per)
            u2 = b_ * (self.W2.T @ e1) - a * e2
            # 事件门控 + 阻尼 + 截断（不变量 3）
            g1 = np.abs(u1) > th
            g2 = np.abs(u2) > th
            x1g = np.clip(x1g + self.et1 * u1 * g1, 0.0, cfg.x_max)
            x2 = np.clip(x2 + self.et2 * u2 * g2, 0.0, cfg.x_max)
            if cfg.kwta_every_iter:
                # 每迭代 k-WTA（E1 谱系）：宽网络下增量小、保留旧子集，
                # 新单元难激活（死锁）——默认仅循环末一次，见 kwta_every_iter。
                x1g = _kwta2d(x1g, cfg.kwta_frac)
                x2 = _kwta(x2, cfg.kwta_frac)
            if do_out:
                # x3 自由变量（能量梯度 LS 读出，E1 谱系对照路径；E2 主
                # 判分走 W_out 线性读出，见 eval_batch）
                e2 = x2 - self.W3 @ x3
                x3 = np.clip(x3 + cfg.eta_out * (self.W3.T @ e2),
                             0.0, 1.0)
        # k-WTA（循环末一次：竞争选择活跃表征，保持整层稀疏，不变量 3）
        x1g = _kwta2d(x1g, cfg.kwta_frac)
        x2 = _kwta(x2, cfg.kwta_frac)
        # 事件率账本（EMA）
        self.event_frac += ((float(g1.mean() + g2.mean()) / 2)
                            - self.event_frac) * 0.01
        self._e0c, self._e1, self._e2 = e0c, e1, e2
        self._x1g, self._x2, self._x3 = x1g, x2, x3

    # ---- 规则 2：误差驱动局部学习（逐层，突触局部）----
    def _learn(self, yoh: np.ndarray) -> None:
        if not self.learning:
            return
        cfg = self.cfg
        x1g = self._x1g
        g1 = (x1g > cfg.theta_syn).astype(np.float32)
        g2 = (self._x2 > cfg.theta_syn).astype(np.float32)
        # dW1（块紧凑）：∂E/∂W1 = s1·e0 ⊗ x1
        dW1 = np.einsum("bi,bj->bij", self._e0c, x1g * g1) * self.s1
        self.W1c += (self.eta_w1 * dW1).astype(np.float32)
        # 列归一（每单元 rf 内单位范数）+ pad 行清零 + 前乘子
        n = np.maximum(np.linalg.norm(self.W1c, axis=1, keepdims=True), 1e-8)
        self.W1c = self.W1c / n
        self.W1c[self.pad] = 0.0
        self.W1c *= self.s1
        self.W1cT = np.ascontiguousarray(self.W1c.transpose(0, 2, 1))
        # dW2 / dW3（平坦，掩码 + 列归一，E1 同款）
        dW2 = np.outer(self._e1, self._x2 * g2)
        # dW3 = e2 ⊗ yoh：生成权重，只更新目标类列（E1 同款原型学习，
        # e2 为钳制收敛误差 x2−W3@yoh，把列 y 拉向类 y 的 x2 原型；
        # 评估经 W3ᵀ 线性读出判分）
        dW3 = np.outer(self._e2, yoh)
        self.W2 += (self.eta_w2 * dW2).astype(np.float32)
        self.W3 += (self.eta_w3 * dW3).astype(np.float32)
        self.W2 *= self.m2
        self.W3 *= self.m3
        self.W2 = _colnorm(self.W2)
        if self.cfg.w3_norm == "unit":
            # 列单位范数（E1 谱系）：抹掉频率幅度（unigram 先验缺失的根源，
            # 仅适用均匀类任务）
            self.W3 = _colnorm(self.W3) * self.cfg.w3_scale
        else:
            # clip：列幅度自由生长（dW3=e2⊗yoh 每步只更新目标类列，
            # 高频列累积更大幅度 => 读出承载 unigram 先验），防发散
            n = np.linalg.norm(self.W3, axis=0)
            cap = self.cfg.w3_norm_cap
            over = n > cap
            self.W3[:, over] *= cap / np.maximum(n[over], 1e-8)

    def train_step(self, x0: np.ndarray, y: int,
                   block_mask: np.ndarray | None = None) -> float:
        """钳制信用分配（塑形 x2 类结构）+ 核心局部学习 + 读出头 LMS。

        读出头在自由推断 x2 上训练（与评估同协议，E1 hebb_free 同思想）：
        浅迭代省算力；W_out/b_out 为机械外围翻译器（§7.7），列幅度 ∝
        频率承载 unigram 先验。
        """
        cfg = self.cfg
        yoh = np.zeros(self.C, np.float32)
        yoh[y] = 1.0
        self._infer(x0, yoh, block_mask, clamp=True)
        self._learn(yoh)
        # 读出头（自由推断 x2，浅迭代）
        self._infer(x0, None, block_mask, clamp=False,
                    iters=cfg.readout_iters)
        x2 = self._x2
        logit = self.W_out.T @ x2 + self.b_out
        p = _softmax(logit / cfg.readout_tau)
        err = p.copy()
        err[y] -= 1.0
        self.W_out -= (cfg.readout_lr * np.outer(x2, err)).astype(np.float32)
        self.b_out -= (cfg.readout_lr * err).astype(np.float32)
        return float(np.linalg.norm(self._e0c))

    # ---- 冻结评估：线性读出头（训练/评估同协议）----
    def eval_batch(self, X: np.ndarray, y: np.ndarray, tau: float,
                   block_mask: np.ndarray | None = None
                   ) -> tuple[float, float]:
        """返回 (BPC, acc)。X: (n, W, 256)。

        自由推断（深迭代）取 x2，W_out 线性读出 + bias（承载 unigram
        先验），softmax(·/τ) 计 BPC。
        """
        self.learning = False
        self.tau = tau
        nll = acc = 0.0
        for i in range(len(y)):
            self._infer(X[i].ravel(), None, block_mask)
            logit = self.W_out.T @ self._x2 + self.b_out
            p = _softmax(logit / tau)
            nll -= np.log2(max(p[y[i]], 1e-12))
            acc += float(p.argmax() == y[i])
        self.learning = True
        return float(nll / len(y)), float(acc / len(y))

    def fit_tau(self, X: np.ndarray, y: np.ndarray, taus: tuple) -> float:
        """读出温度 τ：校准切片上选最优（不参与在线学习，读出标量）。"""
        best, best_tau = np.inf, taus[0]
        for tau in taus:
            bpc, _ = self.eval_batch(X, y, tau)
            if bpc < best:
                best, best_tau = bpc, tau
        return float(best_tau)


def _kwta2d(xg: np.ndarray, frac: float) -> np.ndarray:
    """整层 k-WTA（跨锚点组竞争），保持分组形状。"""
    flat = _kwta(xg.ravel(), frac)
    return flat.reshape(xg.shape)


# ----------------------------------------------------------------------
# TwinMLP：参数量匹配的稠密 MLP + Adam（标准反传参照，不属 SR-PC 构造）
# ----------------------------------------------------------------------
class TwinMLP:
    """d0 -> g -> g -> 256 稠密 ReLU MLP，softmax CE，Adam。

    参数量（含偏置）≈ PCN 结构参数（同预算参照）。batch 训练，
    样本流与 PCN 相同（同样本位置、同样本数）。
    """

    def __init__(self, cfg: E2Config, target_params: int,
                 rng: np.random.Generator):
        d0 = cfg.context * 256
        g = int((-(d0 + 256) + np.sqrt((d0 + 256) ** 2
                                       + 4 * target_params)) / 2)
        self.g = g
        self.V1 = (rng.normal(0, 1, (g, d0)) / np.sqrt(d0)).astype(np.float32)
        self.V2 = (rng.normal(0, 1, (g, g)) / np.sqrt(g)).astype(np.float32)
        self.V3 = (rng.normal(0, 1, (256, g)) / np.sqrt(g)).astype(np.float32)
        self.b1 = np.zeros(g, np.float32)
        self.b2 = np.zeros(g, np.float32)
        self.b3 = np.zeros(256, np.float32)
        self.n_params = int(g * d0 + g * g + 256 * g + 2 * g + 256)
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)]
                      for k, v in self._params().items()}
        self.t = 0

    def _params(self) -> dict:
        return dict(V1=self.V1, V2=self.V2, V3=self.V3,
                    b1=self.b1, b2=self.b2, b3=self.b3)

    def train_batch(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """X: (B, d0) 一热；y: (B,)。返回 batch 平均 CE（nats）。"""
        B = len(y)
        z1 = X @ self.V1.T + self.b1          # (B, g)
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.V2.T + self.b2
        h2 = np.maximum(z2, 0.0)
        z3 = h2 @ self.V3.T + self.b3         # (B, 256)
        z3 -= z3.max(axis=1, keepdims=True)
        p = np.exp(z3)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
        # 反传（标准 BP，仅孪生参照）
        dz3 = p.copy()
        dz3[np.arange(B), y] -= 1.0
        dz3 /= B
        dV3 = dz3.T @ h2
        db3 = dz3.sum(0)
        dh2 = dz3 @ self.V3 * (z2 > 0)
        dV2 = dh2.T @ h1
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.V2 * (z1 > 0)
        dV1 = dh1.T @ X
        db1 = dh1.sum(0)
        # Adam
        self.t += 1
        b1a, b2a = 0.9, 0.999
        c = 1 - b1a ** self.t
        grads = dict(V1=dV1, V2=dV2, V3=dV3, b1=db1, b2=db2, b3=db3)
        for k, v in self._params().items():
            m, v_ = self._adam[k]
            m *= b1a
            m += (1 - b1a) * grads[k]
            v_ *= b2a
            v_ += (1 - b2a) * (grads[k] ** 2)
            mh = m / c
            vh = v_ / (1 - b2a ** self.t)
            v -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(loss)

    def eval_batch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        z1 = X @ self.V1.T + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.V2.T + self.b2
        h2 = np.maximum(z2, 0.0)
        z3 = h2 @ self.V3.T + self.b3
        z3 = z3 - z3.max(axis=1, keepdims=True)
        p = np.exp(z3)
        p /= p.sum(axis=1, keepdims=True)
        py = p[np.arange(len(y)), y]
        return float(-np.log2(py + 1e-12).mean()), float(
            (z3.argmax(1) == y).mean())


# ----------------------------------------------------------------------
# 训练/评估循环
# ----------------------------------------------------------------------
def train_lmpcn(cfg: E2Config, h: int, steps: int, corpus: ByteCorpus,
                eta_w: float | None = None, iters: int | None = None,
                seed: int = 0, ipc: bool = False,
                eval_points: bool = True) -> dict:
    """训练一个 LMPCN，返回曲线与终态评估（τ 用校准切片定）。"""
    rng = np.random.default_rng(seed * 977 + 5)
    m = LMPCN(cfg, h, rng, eta_w=eta_w, iters=iters)
    t0 = time.time()
    curve = []
    vx, vy = corpus.val_x, corpus.val_y
    cal_x, cal_y = vx[:cfg.tau_cal_windows], vy[:cfg.tau_cal_windows]
    rep_x, rep_y = (vx[cfg.tau_cal_windows:], vy[cfg.tau_cal_windows:])
    for t in range(steps):
        if ipc:
            bm = _ipc_mask(t, steps, cfg.context)
        else:
            bm = None
        x, y = corpus.train_window(t)
        m.train_step(x.ravel(), y, bm)
        if eval_points and (t + 1) % cfg.eval_every == 0 or t == steps - 1:
            tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
            bpc, acc = m.eval_batch(rep_x, rep_y, tau)
            curve.append(dict(step=t + 1, bpc=bpc, acc=acc, tau=float(tau),
                              event_rate=m.event_frac,
                              wall=round(time.time() - t0, 1)))
    tau = m.fit_tau(cal_x, cal_y, cfg.tau_grid)
    bpc, acc = m.eval_batch(rep_x, rep_y, tau)
    return dict(h=h, steps=steps, bpc=bpc, acc=acc, tau=float(tau),
                curve=curve, event_rate=m.event_frac,
                n_params_struct=m.n_params_struct,
                n_params_dense=m.n_params_dense,
                wall=round(time.time() - t0, 1),
                model=m)


def train_twin(cfg: E2Config, target_params: int, steps: int,
               corpus: ByteCorpus, seed: int = 0) -> dict:
    """孪生 BP 参照：同样本流、同样本数（batch 化）。

    评估与 PCN 同口径：同验证段（val_x[tau_cal_windows:] 起的窗口，
    与 LMPCN 校准后剩余段一致）。
    """
    rng = np.random.default_rng(seed * 977 + 9)
    tw = TwinMLP(cfg, target_params, rng)
    t0 = time.time()
    B = cfg.twin_batch
    rep_x = corpus.val_x[cfg.tau_cal_windows:]
    rep_y = corpus.val_y[cfg.tau_cal_windows:]
    curve = []
    for t0b in range(0, steps, B):
        n = min(B, steps - t0b)
        X = np.zeros((n, cfg.context, 256), np.float32)
        ys = np.zeros(n, np.int64)
        for i in range(n):
            x, y = corpus.train_window(t0b + i)
            X[i] = x
            ys[i] = y
        tw.train_batch(X.reshape(n, -1), ys, cfg.twin_lr)
        t = t0b + n
        if t % cfg.eval_every == 0 or t == steps:
            bpc, acc = tw.eval_batch(rep_x.reshape(len(rep_y), -1), rep_y)
            curve.append(dict(step=t, bpc=bpc, acc=acc,
                              wall=round(time.time() - t0, 1)))
    bpc, acc = tw.eval_batch(rep_x.reshape(len(rep_y), -1), rep_y)
    return dict(bpc=bpc, acc=acc, curve=curve, n_params=tw.n_params,
                wall=round(time.time() - t0, 1))


def _ipc_mask(t: int, steps: int, context: int) -> np.ndarray:
    """iPC 上下文课程：有效块数 4→8→16 三段展开（近端优先）。

    返回 (context,) bool：True = 该块可见。近端（最近）块始终可见，
    远端块按课程进度逐步放开。
    """
    frac = t / max(1, steps)
    n_on = 4 if frac < 1 / 3 else (8 if frac < 2 / 3 else context)
    bm = np.zeros(context, dtype=bool)
    bm[context - n_on:] = True
    return bm
