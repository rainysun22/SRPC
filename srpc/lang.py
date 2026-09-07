"""阶段 E1：字节级 UTF-8 词元前端 + 语言长程信用分配小任务（延迟文本关联）。

对应 docs/ROADMAP.md 阶段 E1 / docs/SRPC_DESIGN.md §9-2（自建词元前端，
机械外围不含现成预训练模型）。

**前端（ByteTokenizer）**：文本 → UTF-8 字节 → 256 维一热（正交基底），
确定性 I/O，与 ARC 符号编码（§7.7）同构；E2 语言流直接复用。
多语原生：任何 Unicode 文本的字节级表示无需词表。

**任务族（与 §8.5 延迟 XOR 早筛同谱系，v2 方案）**：字母表 A=16 个
ASCII 字母的 i.i.d. 均匀符号流，经 ByteTokenizer 编码；窗口 = 最近 W 步
字节一热拼接（延迟线属机械外围序列化，§2.2），远端块（窗口最前块）为
唯一任务相关块：
- **assoc（主验收任务）**：y_t = π(x_{t-Δ})，π = 固定随机置换 —— π 为
  双射且流均匀 => y 边际均匀，**零边际相关**（同 §8.5 承重墙判据），
  纯相关 Hebbian 一阶统计无信号；同时确定性映射信号强、在线单样本可学；
- **xorsum（边界记录）**：y_t = x_{t-Δ-1} ⊕ x_{t-Δ}（4 bit 逐位 XOR，
  16 类）。在线单样本学习不可达：BP 在线对照同样失败（parity 为 SQ-hard
  高频函数，在线更新期望梯度≈0），batch 上界 1.0（mini-batch SGD 可达）
  ——瓶颈为在线协议样本效率而非局部规则原理；修复排 F 阶段记忆回放。
  证据链详见 LangConfig v2 注释与 results_e1/report.md。

**网络（LangPCN）**：CreditPCN 的多类推广，同一套 7.3 局部规则：
    x0(窗口字节一热) -> x1(时间分块感受野) -> x2(随机扇入) -> x3(C 类 one-hot)
    ε0 = x0 − W1@x1；ε1 = x1 − W2@x2；ε2 = x2 − W3@x3
    dx1 = β·(W1ᵀ·ε0) − α·ε1；dx2 = β·(W2ᵀ·ε1) − α·ε2     # 规则 1
    dW ∝ ε ⊗ 突触前（error 臂）/ 突触后 ⊗ 突触前（hebb 臂）  # 规则 2
出生即稀疏（层 1 每单元一个时间块感受野 + 内部随机扇入 + k-WTA，不变量 3）。
评估双口径：自由推断（argmax + softmax BPC，E2 语言谱系主口径）与
钳制-比较子样本（§8.5 协议交叉验证）。
"""
from __future__ import annotations

import numpy as np

from .config import LangConfig
from .credit import _colnorm, _kwta, _orth_cols


# ----------------------------------------------------------------------
# E1a：字节级 UTF-8 词元化器（机械外围，§2.2 / §9-2）
# ----------------------------------------------------------------------
class ByteTokenizer:
    """确定性字节级词元化器：text ⇄ 256 维一热。

    词表 = 全部 256 个字节值（UTF-8 万国码原生覆盖，无需训练、无 OOV）。
    一热基底与 ARC 符号编码（§7.7）同构：符号 = 正交基底方向，
    核心网络在正交基底上做符号接地。decode 支持一热与 soft 输出
    （逐行 argmax，读方向与 §7.7 翻译器质检一致）。
    """

    vocab: int = 256

    def encode(self, text: str) -> np.ndarray:
        """text -> (n_bytes, 256) float32 一热。"""
        b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        x = np.zeros((len(b), self.vocab), dtype=np.float32)
        x[np.arange(len(b)), b] = 1.0
        return x

    def decode(self, x: np.ndarray) -> str:
        """(n, 256) 一热/soft -> text（逐行 argmax）。"""
        idx = np.asarray(x)
        if idx.ndim != 2 or idx.shape[1] != self.vocab:
            raise ValueError(f"decode 期望 (n, 256)，得到 {idx.shape}")
        return bytes(int(i) for i in idx.argmax(axis=1)).decode(
            "utf-8", errors="replace")


# ----------------------------------------------------------------------
# E1b：任务生成（延迟文本关联）
# ----------------------------------------------------------------------
def make_task(cfg: LangConfig, rng: np.random.Generator, n: int, delta: int,
              task: str, perm: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 (X 窗口, y 类索引, perm)。

    流 = 字母表 i.i.d. 均匀符号（文本形式），经 ByteTokenizer 编码为
    (T, 256) 字节一热；窗口最前块 = 最旧符号（与 credit.py 口径一致）。
      assoc : W = Δ+1，y_t = perm[流[t]]           （远端块 = 前 1 块）
      xorsum: W = Δ+2，y_t = 流[t] ⊕ 流[t+1]        （远端块 = 前 2 块）
    """
    a = len(cfg.alphabet)
    if task == "assoc":
        span = delta + 1
    elif task == "xorsum":
        span = delta + 2
    else:
        raise ValueError(f"未知任务 {task}")
    if perm is None:
        perm = rng.permutation(a)
    codes = rng.integers(0, a, n + span)
    text = "".join(cfg.alphabet[i] for i in codes)
    oh = ByteTokenizer().encode(text)              # (n+span, 256)
    X = np.empty((n, span * 256), dtype=np.float32)
    for t in range(n):
        X[t] = oh[t:t + span].ravel()
    if task == "assoc":
        y = perm[codes[:n]]
    else:
        y = codes[:n] ^ codes[1:n + 1]
    return X, y.astype(np.int64), perm


# ----------------------------------------------------------------------
# LangPCN：C 类局部误差驱动 PCN（credit.py 同规则，多类化）
# ----------------------------------------------------------------------
class LangPCN:
    """延迟文本关联上的 PCN。learn_mode='error'（信用分配）/ 'hebb'（纯相关对照）。"""

    def __init__(self, cfg: LangConfig, rng: np.random.Generator,
                 learn_mode: str = "error", n_blocks: int = 1):
        self.cfg = cfg
        self.mode = learn_mode
        self.C = cfg.n_classes
        self.n_blocks = n_blocks            # 窗口时间块数（= span）
        d0 = n_blocks * 256
        self.d0 = d0
        # 权重（有符号初始化，近正交列）
        self.W1 = _colnorm(rng.normal(0.0, 1.0, (d0, cfg.h1)).astype(np.float32))
        self.W2 = _colnorm(rng.normal(0.0, 1.0, (cfg.h1, cfg.h2)).astype(np.float32))
        self.W3 = rng.normal(0.0, 0.3, (cfg.h2, self.C)).astype(np.float32)
        # 出生即稀疏掩码（不变量 3）：层 1 = 每单元一个时间块（连续感受野，
        # 锚点随机 -> n_blocks 分之一单元看远端块）；内部层/输出层 = 随机扇入。
        m1 = np.zeros((d0, cfg.h1), dtype=bool)
        for j in range(cfg.h1):
            b = int(rng.integers(0, n_blocks))
            m1[b * 256:(b + 1) * 256, j] = True
        k2 = max(1, int(round(cfg.fan_in_frac * cfg.h1)))
        m2 = np.zeros((cfg.h1, cfg.h2), dtype=bool)
        for j in range(cfg.h2):
            m2[rng.choice(cfg.h1, size=k2, replace=False), j] = True
        k3 = max(1, int(round(cfg.fan_in_frac * cfg.h2)))
        m3 = np.zeros((cfg.h2, self.C), dtype=bool)
        for j in range(self.C):
            m3[rng.choice(cfg.h2, size=k3, replace=False), j] = True
        self.mask1, self.mask2, self.mask3 = m1, m2, m3
        self.W1 = _colnorm(self.W1 * m1)
        self.W2 = _colnorm(self.W2 * m2)
        # 先列归一再正交化：_orth_cols 对非单位范数列的顺序减法会按 ||w||^2
        # 过冲（每对放大 ~3x，16 列下指数爆炸溢出）；credit.py 的 2 列场景
        # 只减一次不触发，此处必须预归一（数值稳定性修复，方向不变）。
        self.W3 = _orth_cols(_colnorm(self.W3 * m3))   # 类原型近正交（§8.5）
        self.W1_init = self.W1.copy()
        self.x1 = np.zeros(cfg.h1, dtype=np.float32)
        self.x2 = np.zeros(cfg.h2, dtype=np.float32)
        self.learning = True
        self.iters = cfg.settle_iters
        self.hebb_free = cfg.hebb_free

    # 规则 1：局部推断（自由能梯度下降；事件门控 + k-WTA，不变量 3）
    def _infer(self, x0: np.ndarray, yoh: np.ndarray, free_out: bool) -> None:
        cfg = self.cfg
        a, b = cfg.alpha, cfg.beta
        x3 = np.zeros(self.C, dtype=np.float32) if free_out else yoh.copy()
        for _ in range(self.iters):
            e0 = x0 - self.W1 @ self.x1
            e1 = self.x1 - self.W2 @ self.x2
            e2 = self.x2 - self.W3 @ x3
            u1 = b * (self.W1.T @ e0) - a * e1
            u2 = b * (self.W2.T @ e1) - a * e2
            self.x1 = np.clip(self.x1 + cfg.eta_inf * u1 * (np.abs(u1) > cfg.theta_event),
                              0.0, cfg.x_max)
            self.x2 = np.clip(self.x2 + cfg.eta_inf * u2 * (np.abs(u2) > cfg.theta_event),
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
            # 分类比较口径：只含类相关项（e1+e2）——xorsum 类条件输入均值相同，
            # 重建项与类无关只稀释判别信号（§8.5 同理）。
            self._energy = 0.5 * float(np.dot(self._e1, self._e1)
                                       + np.dot(self._e2, self._e2))
        else:
            self._energy = 0.5 * float(np.dot(self._e0, self._e0)
                                       + np.dot(self._e1, self._e1)
                                       + np.dot(self._e2, self._e2))

    # 规则 2：局部学习（error 误差驱动 / hebb 纯相关）
    def _learn(self, x0: np.ndarray, yoh: np.ndarray) -> None:
        if not self.learning:
            return
        cfg = self.cfg
        g1 = self.x1 > cfg.theta_syn
        g2 = self.x2 > cfg.theta_syn
        if self.mode == "error":
            dW1 = np.outer(self._e0, self.x1 * g1)
            dW2 = np.outer(self._e1, self.x2 * g2)
            dW3 = np.outer(self._e2, yoh)
        else:
            dW1 = np.outer(x0, self.x1 * g1)
            dW2 = np.outer(self.x1, self.x2 * g2)
            dW3 = np.outer(self.x2, yoh)
        self.W1 += (cfg.eta_w * dW1).astype(np.float32)
        self.W2 += (cfg.eta_w * dW2).astype(np.float32)
        self.W3 += (cfg.eta_w * dW3).astype(np.float32)
        self.W1 *= self.mask1
        self.W2 *= self.mask2
        self.W3 *= self.mask3
        self.W1 = _colnorm(self.W1)
        self.W2 = _colnorm(self.W2)
        self.W3 = _colnorm(self.W3)

    def train_step(self, x0: np.ndarray, y: int) -> float:
        """reset -> 收敛（err 臂钳制 yoh；hebb 臂自由推断防类泄漏）-> 学习。"""
        yoh = np.zeros(self.C, dtype=np.float32)
        yoh[y] = 1.0
        self.x1[:] = 0.0
        self.x2[:] = 0.0
        self._infer(x0, yoh, free_out=self.mode == "hebb" and self.hebb_free)
        self._learn(x0, yoh)
        return float(np.linalg.norm(self._e0))

    def set_learning(self, flag: bool) -> None:
        self.learning = flag

    # 冻结评估：自由推断（主口径：acc + BPC）
    def predict_free(self, x0: np.ndarray) -> tuple[int, float]:
        """返回 (argmax 类, 目标侧 BPC 占位 —— 由调用方按真值算)。"""
        self.x1[:] = 0.0
        self.x2[:] = 0.0
        self._infer(x0, np.zeros(self.C, dtype=np.float32), free_out=True)
        p = np.exp(self._x3 - self._x3.max())
        p /= p.sum()
        return int(self._x3.argmax()), float(p.max())

    def predict_compare(self, x0: np.ndarray) -> int:
        """钳制-比较（§8.5 协议）：逐类钳制取自由能更低者。"""
        best_c, best_e = 0, np.inf
        for c in range(self.C):
            yoh = np.zeros(self.C, dtype=np.float32)
            yoh[c] = 1.0
            self.x1[:] = 0.0
            self.x2[:] = 0.0
            self._infer(x0, yoh, free_out=False)
            if self._energy < best_e:
                best_e, best_c = self._energy, c
        return best_c

    def distal_share(self, n_distal_blocks: int) -> float:
        """远端块（窗口最前 n_distal_blocks 块）的 W1 权重变化占比。"""
        dW = self.W1 - self.W1_init
        total = float(np.linalg.norm(dW)) + 1e-12
        distal = float(np.linalg.norm(dW[:n_distal_blocks * 256]))
        return distal / total


# ----------------------------------------------------------------------
# 探针运行器
# ----------------------------------------------------------------------
def run_lang_probe(cfg: LangConfig, seed: int, task: str, delta: int,
                   train_steps: int | None = None,
                   eval_steps: int | None = None,
                   compare_steps: int | None = None) -> dict:
    """单 (seed, task, delta) 探针：error/hebb 双臂训练 + 冻结评估。

    返回各臂 acc_free / bpc / acc_compare / distal_share 与学习曲线抽样。
    """
    train_steps = cfg.train_steps if train_steps is None else train_steps
    eval_steps = cfg.eval_steps if eval_steps is None else eval_steps
    compare_steps = cfg.compare_steps if compare_steps is None else compare_steps
    n_distal = 1 if task == "assoc" else 2
    span = delta + n_distal
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, perm = make_task(cfg, rng, train_steps + eval_steps, delta, task)

    out: dict = dict(seed=seed, task=task, delta=delta)
    for mode in ("error", "hebb"):
        m = LangPCN(cfg, np.random.default_rng(seed * 5000 + 17), mode,
                    n_blocks=span)
        if mode == "hebb":
            m.iters = cfg.hebb_settle_iters
        curve = []
        for t in range(train_steps):
            m.train_step(X[t], int(y[t]))
            if (t + 1) % max(1, train_steps // 8) == 0:
                curve.append(_quick_acc(m, X[train_steps:train_steps + 100],
                                        y[train_steps:train_steps + 100]))
        m.set_learning(False)
        # 自由推断主口径
        acc = bpc = 0.0
        for t in range(eval_steps):
            xt, yt = X[train_steps + t], int(y[train_steps + t])
            pred, pmax = m.predict_free(xt)
            acc += float(pred == yt)
            bpc += -np.log2(max(pmax, 1e-12))
        # 钳制-比较子样本交叉验证（复用 eval 段前 compare_steps 个样本）
        n_cmp = min(compare_steps, eval_steps)
        acc_c = 0.0
        for t in range(n_cmp):
            acc_c += float(m.predict_compare(X[train_steps + t])
                           == int(y[train_steps + t]))
        out[f"acc_{mode}"] = acc / eval_steps
        out[f"bpc_{mode}"] = bpc / eval_steps
        out[f"acc_cmp_{mode}"] = acc_c / n_cmp
        out[f"distal_{mode}"] = m.distal_share(n_distal)
        out[f"curve_{mode}"] = curve
    return out


def _quick_acc(m: LangPCN, X: np.ndarray, y: np.ndarray) -> float:
    """训练中快速抽查（自由推断，小子样本）。"""
    a = 0.0
    for t in range(len(y)):
        pred, _ = m.predict_free(X[t])
        a += float(pred == int(y[t]))
    return a / len(y)


def run_lang_screen(cfg: LangConfig | None = None, seeds=(0, 1, 2)) -> dict:
    """E1 主流程 v2：主线（assoc 验收扫描）+ 边界（在线信用分配上限记录）。

    - 主线：assoc @ Δ{1,4,8} × 双臂 × seeds——Δ=4 预注册判定
      （acc/gap/hebb 门/distal），Δ=8 跨度外推，Δ=1 近程对照；
    - 边界（记录不判收）：xorsum @ Δ{4,16}（在线组合信用分配边界，
      SQ-hard + BP 在线对照证据）、assoc @ Δ=16（远端表征稀释边界）。
    """
    cfg = cfg or LangConfig()
    grid = [("assoc", d) for d in (1, 4, 8, 16)] + [("xorsum", d) for d in (4, 16)]
    per = []
    for task, delta in grid:
        for s in seeds:
            # 边界档 compare 子样本缩样（16 类钳制-比较开销 ~C 倍）
            cmp_n = cfg.compare_steps if (task, delta) != ("xorsum", 16) else 30
            per.append(run_lang_probe(cfg, s, task, delta, compare_steps=cmp_n))
    mean = {}
    for task, delta in grid:
        rows = [p for p in per if p["task"] == task and p["delta"] == delta]
        mean[f"{task}_d{delta}"] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("acc_error", "acc_hebb", "bpc_error", "acc_cmp_error",
                      "distal_error", "distal_hebb")}
    # 验收判定（v2 预注册：主任务 assoc）
    m4, m8 = mean["assoc_d4"], mean["assoc_d8"]
    verdict = {
        "acc_assoc_d4": m4["acc_error"],
        "acc_hebb_d4": m4["acc_hebb"],
        "gap_d4": m4["acc_error"] - m4["acc_hebb"],
        "distal_assoc_d4": m4["distal_error"],
        "acc_assoc_d8": m8["acc_error"],
        "pass_acc": m4["acc_error"] >= cfg.acc_assoc_min,
        "pass_hebb": m4["acc_hebb"] <= cfg.acc_hebb_max,
        "pass_gap": (m4["acc_error"] - m4["acc_hebb"]) >= cfg.gap_min,
        "pass_distal": m4["distal_error"] >= cfg.distal_assoc_min,
        "pass_d8": m8["acc_error"] >= cfg.acc_assoc_d8_min,
    }
    verdict["pass_all"] = all(v for k, v in verdict.items() if k.startswith("pass_"))
    return dict(seeds=list(seeds), per_probe=per, metrics_mean=mean,
                verdict=verdict)
