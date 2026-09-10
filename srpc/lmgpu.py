"""阶段 E2 GPU 登顶跑：LMPCN 的 PyTorch 移植（仅加速，不改算法）。

对应 GPU_TASKS T1/T2。三条初衷强制（契约 1/2/3 条）：
    1. 局部规则：所有权重更新保持突触局部形式（误差驱动 LMS/列归一），
       本文件不构造任何 global gradient；
    2. 免反传：全程 torch.no_grad()，无 autograd 路径（孪生才是反传参照，
       孪生仍用 numpy 版 TwinMLP —— 与 pilot 同一实现口径，保证对照连续性）；
    3. 结构稀疏：出生定型掩码 + k-WTA + 事件门控 + 块紧凑 W1 原样迁移。

初始化与 numpy 版（srpc/lm.py LMPCN）逐位一致：以同 seed 的 np LMPCN 实例
为权重真源，复制其掩码/权重到 torch 张量（device 可 cpu/cuda）。训练/评估
的运算顺序与 numpy 版一一对应，仅浮点归约顺序可能引入 ~1e-7 级差异（文档
记录为 parity 容差，见报告 parity 节）。

用法（由 scripts/run_e2_summit.py 驱动）：
    from srpc.lmgpu import LMPCNg
"""
from __future__ import annotations

import numpy as np
import torch

from .config import E2Config
from .lm import LMPCN as _LMPCN_np   # 仅用于权重真源（同 seed 同初始化）

torch.set_grad_enabled(False)


# ----------------------------------------------------------------------
# torch 稀疏算子（与 srpc/credit.py 语义一致）
# ----------------------------------------------------------------------
def _colnorm_t(w: torch.Tensor) -> torch.Tensor:
    """逐列单位范数（axis=0，等价 np 版）。w 就地改？返回归一化视图。"""
    n = torch.linalg.vector_norm(w, dim=0, keepdim=True)
    return w / n.clamp_min(1e-8)


def _kwta_t(x: torch.Tensor, frac: float) -> torch.Tensor:
    """整层 k-WTA：保留最大 k 个原值，其余清零。k = round(frac·n)。"""
    n = x.numel()
    k = max(1, int(round(frac * n)))
    if k >= n:
        return x
    flat = x.reshape(-1)
    vals, idx = torch.topk(flat, k)
    out = torch.zeros_like(flat)
    out[idx] = vals
    return out.reshape(x.shape)


def _kwta2d_t(xg: torch.Tensor, frac: float) -> torch.Tensor:
    return _kwta_t(xg, frac)


# ----------------------------------------------------------------------
# LMPCN GPU 移植（训练循环与 numpy 版一致；iPC 掩码登顶跑不用）
# ----------------------------------------------------------------------
class LMPCNg:
    """同 srpc/lm.py LMPCN，张量落在 device。权重/mask 初始 = numpy 真源。

    对外接口（供登顶 runner 使用）：
        train_step(x0_np, y)   # x0_np (W,256) float32 numpy；y int
        eval_batch(X_np, y_np, tau, block_mask=None) -> (bpc, acc)
        fit_tau(X_np, y_np, taus) -> best_tau
    """

    def __init__(self, cfg: E2Config, h: int, rng: np.random.Generator,
                 eta_w: float | None = None, iters: int | None = None,
                 device: str = "cuda", mu_pc_exp: float = 0.5):
        self.cfg = cfg
        self.h = h
        self.W, self.C = cfg.context, 256
        self.device = torch.device(device)
        # ---- 权重真源：numpy 版同构造（同 rng 种子 -> 初始完全一致）----
        src = _LMPCN_np(cfg, h, rng, eta_w=eta_w, iters=iters,
                        mu_pc_exp=mu_pc_exp)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        self.per = src.per
        self.r = src.r
        self.s1 = float(src.s1)
        self.et1 = float(src.et1)
        self.et2 = float(src.et2)
        self.eta_w1 = float(src.eta_w1)
        self.eta_w2 = float(src.eta_w2)
        self.eta_w3 = float(src.eta_w3)
        self.iters = src.iters
        self.W1c = to(src.W1c)
        self.W1cT = to(src.W1cT)
        self.W2 = to(src.W2)
        self.W3 = to(src.W3)
        self.W_out = to(src.W_out)
        self.b_out = to(src.b_out)
        self.m2 = to(src.m2)
        self.m3 = to(src.m3)
        self.pad = to(src.pad)
        self.idx_rf = to(src.idx_rf)
        self.learning = True
        self.tau = cfg.tau
        self.n_params_struct = int(src.n_params_struct)
        self.n_params_dense = int(src.n_params_dense)
        # 事件率 EMA（GPU 标量，评估点才同步到 CPU）
        self._ev = torch.zeros((), dtype=torch.float32, device=self.device)
        # 稳定性修复计数器（W2 周期谱截断，见 _maybe_cap_w2）
        self._n_train = 0
        self.last_w2_smax = 0.0
        del src

    # ---- 规则 1：推断（训练钳制 / 评估自由读出）----
    def _infer(self, x0_flat_np: np.ndarray, yoh_np: np.ndarray | None,
               block_mask: np.ndarray | None = None,
               clamp: bool = False, iters: int | None = None,
               free_out: bool | None = None) -> None:
        cfg = self.cfg
        a, b_ = cfg.alpha, cfg.beta
        dev = self.device
        x0_flat_np = np.asarray(x0_flat_np).ravel()  # 与 numpy 版同（输入可能 2D）
        x0p = np.concatenate([x0_flat_np, np.zeros(1, np.float32)])
        if block_mask is not None:
            bm = np.repeat(block_mask, 256)
            x0p = x0p * np.concatenate([bm, np.ones(1, np.float32)])
        x0p_t = torch.as_tensor(x0p, device=dev)
        x0rf = x0p_t[self.idx_rf]
        x1g = torch.zeros((self.W, self.per), dtype=torch.float32, device=dev)
        x2 = torch.zeros(self.h, dtype=torch.float32, device=dev)
        if clamp:
            x3 = torch.as_tensor(yoh_np, device=dev)
        else:
            x3 = torch.zeros(self.C, dtype=torch.float32, device=dev)
        th = cfg.theta_event
        it = self.iters if iters is None else iters
        do_out = (not clamp) if free_out is None else free_out
        # 收缩步长：自由推断期把 x2/x3 更新步长压入收缩界，杜绝深迭代振荡/能量爆发
        scl = cfg.eta_inf_scl if do_out else 1.0
        et2_e = self.et2 * scl
        eta_out_e = cfg.eta_out * scl
        W1c, W1cT, W2, W3 = self.W1c, self.W1cT, self.W2, self.W3
        for _ in range(it):
            pred0 = torch.matmul(W1c, x1g.unsqueeze(-1)).squeeze(-1)
            e0c = x0rf - pred0
            x1 = x1g.reshape(-1)
            e1 = x1 - torch.mv(W2, x2)
            e2 = x2 - torch.mv(W3, x3)
            u1 = torch.matmul(W1cT, e0c.unsqueeze(-1)).squeeze(-1) * self.s1
            u1 = b_ * u1.reshape(self.W, self.per) - a * e1.reshape(
                self.W, self.per)
            u2 = b_ * torch.mv(W2.t(), e1) - a * e2
            g1 = u1.abs() > th
            g2 = u2.abs() > th
            x1g = (x1g + self.et1 * u1 * g1).clamp(0.0, cfg.x_max)
            x2 = (x2 + et2_e * u2 * g2).clamp(0.0, cfg.x_max)
            if cfg.kwta_every_iter:
                x1g = _kwta2d_t(x1g, cfg.kwta_frac)
                x2 = _kwta_t(x2, cfg.kwta_frac)
            if do_out:
                e2 = x2 - torch.mv(W3, x3)
                x3 = (x3 + eta_out_e * torch.mv(W3.t(), e2)).clamp(
                    0.0, 1.0)
        x1g = _kwta2d_t(x1g, cfg.kwta_frac)
        x2 = _kwta_t(x2, cfg.kwta_frac)
        # 事件率 EMA（GPU 标量；逐 step 更新，评估/结束时同步）
        self._ev = self._ev + ((g1.float().mean() + g2.float().mean()) * 0.5
                               - self._ev) * 0.01
        self._e0c, self._e1, self._e2 = e0c, e1, e2
        self._x1g, self._x2, self._x3 = x1g, x2, x3

    # ---- 规则 2：误差驱动局部学习（逐层，突触局部）----
    def _learn(self, yoh_np: np.ndarray) -> None:
        if not self.learning:
            return
        cfg = self.cfg
        x1g = self._x1g
        g1 = (x1g > cfg.theta_syn).to(x1g.dtype)
        g2 = (self._x2 > cfg.theta_syn).to(self._x2.dtype)
        dW1 = torch.einsum("bi,bj->bij", self._e0c, x1g * g1) * self.s1
        self.W1c += self.eta_w1 * dW1
        n = torch.linalg.vector_norm(self.W1c, dim=1, keepdim=True)
        if cfg.w1_norm == "unit":
            # 能量守恒：每列强制单位范数（赢者列会把弱输入行挤出至 0）
            self.W1c = self.W1c / n.clamp_min(cfg.w1_norm_eps)
        else:
            # clip：列范数仅截上限（≤1），允许弱列自由变弱、不把输入行挤出
            scale = torch.where(n > 1.0, 1.0 / n.clamp_min(1e-12),
                                torch.ones_like(n))
            self.W1c = self.W1c * scale
        self.W1c[self.pad] = 0.0
        self.W1c *= self.s1
        self.W1cT = self.W1c.transpose(1, 2).contiguous()
        dW2 = torch.outer(self._e1, self._x2 * g2)
        yoh = torch.as_tensor(yoh_np, device=self.device)
        dW3 = torch.outer(self._e2, yoh)
        self.W2 += self.eta_w2 * dW2
        self.W3 += self.eta_w3 * dW3
        self.W2 *= self.m2
        self.W3 *= self.m3
        self.W2 = _colnorm_t(self.W2)
        if cfg.w3_norm == "unit":
            self.W3 = _colnorm_t(self.W3) * cfg.w3_scale
        else:  # clip：列幅度自由生长（承载 unigram 先验），防发散
            n = torch.linalg.vector_norm(self.W3, dim=0)
            cap = cfg.w3_norm_cap
            over = n > cap
            self.W3[:, over] *= (cap / n[over].clamp_min(1e-8))

    def _maybe_cap_w2(self) -> None:
        """周期 W2 谱截断（对因稳定性修复，2026-09-08）。

        根因：超长预算 × 宽网络下 W2 列对齐塌缩，σmax(W2) 由健康 ~4.8 涨到
        ~37，自由推断（无 yoh 引导）收缩性丢失 -> x2 饱和死锁发散（见
        REPORT_E2_GPU_SUMMIT §5 与 diag 轨迹；s=33.0 万 σmax=4.81 健康，
        s=34.5 万 σmax=37.6 崩）。修复：每 cfg.w2_cap_every 步对 W2 顶奇异
        值谱截断至 cfg.w2_smax_cap。用 @cfg.w2_pow_iters 次幂迭代估计顶奇异
        三元组（O(n²) matvec，免每步 SVD），只削顶奇异值、保留其余谱与结构
        稀疏/列局部性。局部规则不变，不构造任何梯度；健康档 σmax≤~4.8<
        5，cap 不触发即零影响。失稳是单侧快速正反馈（1000 步内 5->37），必须
        持续压住 σmax（w2_cap_every≈1）防止表征在任何时刻被破坏。
        """
        cfg = self.cfg
        if not cfg.w2_cap:
            return
        self._n_train += 1
        if self._n_train % cfg.w2_cap_every != 0:
            return
        W = self.W2
        v = torch.randn(W.shape[1], dtype=torch.float32, device=W.device)
        for _ in range(cfg.w2_pow_iters):
            v = W.t() @ (W @ v)
            v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
        Wv = W @ v
        sig = torch.linalg.vector_norm(Wv)
        self.last_w2_smax = float(sig)
        if sig > cfg.w2_smax_cap:
            u = Wv / sig
            # 只削顶奇异值：(σmax−cap)·u·vᵀ 从 W2 中减去；就地写回保持图绑定
            self.W2.copy_(W - (sig - cfg.w2_smax_cap) * torch.outer(u, v))

    # ---- 训练步（与 numpy train_step 顺序一致）----
    def train_step(self, x0_np: np.ndarray, y: int) -> float:
        cfg = self.cfg
        yoh = np.zeros(self.C, np.float32)
        yoh[y] = 1.0
        self._infer(x0_np, yoh, clamp=True)
        self._learn(yoh)
        self._maybe_cap_w2()
        # 读出头（自由推断 x2 浅迭代 + LMS）
        self._infer(x0_np, None, clamp=False, iters=cfg.readout_iters)
        x2 = self._x2
        logit = torch.mv(self.W_out.t(), x2) + self.b_out
        p = torch.softmax(logit / cfg.readout_tau, dim=0)
        err = p.clone()
        err[y] -= 1.0
        self.W_out -= (cfg.readout_lr * torch.outer(x2, err))
        self.b_out -= (cfg.readout_lr * err)
        return float(torch.linalg.vector_norm(self._e0c))

    def train_step_ss(self, x0_np: np.ndarray, y: int) -> float:
        """scheduled-sampling 在线训练（H2 对因修复 v2）。

        先短自由推断得开环预测 p̂，钳制目标 t3=(1-ε)yoh+ε·p̂，clamp 推断与 W3
        学习都用 t3，使生成映射(x2→next-byte)与开环读出一致，缓解 teacher
        -forcing→free 塌陷。ε=cfg.ss_eps；0 时退化为原 train_step 行为。
        """
        cfg = self.cfg
        ss = cfg.ss_eps
        yoh = np.zeros(self.C, np.float32); yoh[y] = 1.0
        yoh_t = torch.as_tensor(yoh, device=self.device)
        # 1) 短自由推断得开环预测 p̂（复用给读出头）
        self._infer(x0_np, None, clamp=False, iters=cfg.readout_iters)
        x2f = self._x2
        logit = torch.mv(self.W_out.t(), x2f) + self.b_out
        p = torch.softmax(logit / cfg.readout_tau, dim=0)
        target_np = yoh
        if ss > 0.0:
            t3 = (1.0 - ss) * yoh_t + ss * p
            target_np = t3.cpu().numpy().astype(np.float32)
        # 2) 用混合目标钳制推断（塑造 x2 为开环可达的判别态）
        self._infer(x0_np, target_np, clamp=True, iters=self.iters)
        self._learn(target_np)
        self._maybe_cap_w2()
        # 3) 读出头 LMS（判别真标签，基于自由 x2）
        err = p - yoh_t
        self.W_out -= (cfg.readout_lr * torch.outer(x2f, err))
        self.b_out -= (cfg.readout_lr * err)
        return float(torch.linalg.vector_norm(self._e0c))

    # ---- 批量累积局部更新（H2 对因修复，2026-09-10）----
    # 根因（h2_conv 诊断）：SR-PC 在线单样本 Hebbian vs 孪生 batch=32+Adam，
    #  同样本数下相差甚大（1856@300k：acc 0.258/bpc 4.1 vs 孪生 0.42/更好）。
    #  收敛效率代差，非数据量差、非读出判别不足（CE 耦合 300k 验证无效，已弃）。
    # 文献：成批预测编码收敛至 BP（Salvatori et al.; Ha et al. ICLR'26 Meta-PCN）；
    #  E1 规划 F 阶段"记忆回放 -> batch 等效"（慢记忆/睡眠回放）。
    # 修复：把 B 个样本的 PC 误差外积（∂F/∂W = e_l ⊗ pre_l）批量累积并取平均，
    #  仍为局部规则、免反传、结构稀疏 + 列归一 + W2 谱截断全保留。B=32 与孪生同预算。
    @staticmethod
    def _bkwta_1d(x: torch.Tensor, frac: float) -> torch.Tensor:
        n = x.shape[1]
        k = max(1, int(round(frac * n)))
        if k >= n:
            return x
        vals, idx = torch.topk(x, k, dim=1)
        out = torch.zeros_like(x)
        out.scatter_(1, idx, vals)
        return out

    def _infer_batch(self, x0_np: np.ndarray, yoh_np: np.ndarray | None,
                     clamp: bool, iters: int) -> tuple:
        """批量推断 x0:(B,W,256) 或 (B, W*256)；yoh_np 仅 clamp 用。返回
        (x1g, x2, x3b, e0c, e1, e2, g1, g2)；x3 最后迭代的 e2 用于学习。"""
        cfg = self.cfg
        a, b_ = cfg.alpha, cfg.beta
        B = x0_np.shape[0]
        dev = self.device
        x0f = np.concatenate([np.asarray(x0_np).reshape(B, -1),
                              np.zeros((B, 1), np.float32)], axis=1)
        x0b = torch.as_tensor(np.ascontiguousarray(x0f), device=dev)   # (B, 4097)
        x0rf = x0b[:, self.idx_rf]                                     # (B, W, per)
        x1g = torch.zeros((B, self.W, self.per), dtype=torch.float32, device=dev)
        x2 = torch.zeros((B, self.h), dtype=torch.float32, device=dev)
        if clamp:
            x3b = torch.as_tensor(np.ascontiguousarray(yoh_np), device=dev)
        else:
            x3b = torch.zeros((B, self.C), dtype=torch.float32, device=dev)
        W1c, W1cT, W2, W3 = self.W1c, self.W1cT, self.W2, self.W3
        scl = cfg.eta_inf_scl if not clamp else 1.0
        et2_e = self.et2 * scl
        eta_out_e = cfg.eta_out * scl
        th = cfg.theta_event
        for _ in range(iters):
            pred0 = torch.einsum("hpq,bhq->bhp", W1c, x1g)
            e0c = x0rf - pred0                       # (B,W,per)
            x1 = x1g.reshape(B, -1)
            e1 = x1 - x2 @ W2.t()                    # (B,h)
            e2 = x2 - x3b @ W3.t()                   # (B,h)
            u1 = b_ * (torch.einsum("hpq,bhq->bhp", W1cT, e0c) * self.s1) \
                - a * e1.reshape(B, self.W, self.per)
            u2 = b_ * (e1 @ W2) - a * e2
            g1 = u1.abs() > th
            g2 = u2.abs() > th
            x1g = (x1g + self.et1 * u1 * g1).clamp(0.0, cfg.x_max)
            x2 = (x2 + et2_e * u2 * g2).clamp(0.0, cfg.x_max)
            if not clamp:
                e2 = x2 - x3b @ W3.t()
                x3b = (x3b + eta_out_e * (e2 @ W3)).clamp(0.0, 1.0)
        x1g = self._bkwta_1d(x1g.reshape(B, -1), cfg.kwta_frac).reshape(
            B, self.W, self.per)
        x2 = self._bkwta_1d(x2, cfg.kwta_frac)
        return x1g, x2, x3b, e0c, e1, e2, g1, g2

    def train_step_batch(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """批量累积局部更新（B 样本 PC 误差外积取平均后应用一次）。"""
        cfg = self.cfg
        X_np = np.asarray(X_np)
        y_np = np.asarray(y_np)
        B = X_np.shape[0]
        dev = self.device
        yoh = np.zeros((B, self.C), np.float32)
        yoh[np.arange(B), y_np] = 1.0
        yoh_t = torch.as_tensor(yoh, device=dev)
        # ---- 钳制信用分配（batch=32 同孪生学力）----
        x1g, x2, _x3, e0c, e1, e2, g1, g2 = self._infer_batch(
            X_np, yoh, clamp=True, iters=self.iters)
        g1s = (x1g > cfg.theta_syn).to(x1g.dtype)
        g2s = (x2 > cfg.theta_syn).to(x2.dtype)
        dW1 = torch.einsum("bhi,bhj->hij", e0c, x1g * g1s) * self.s1 / B
        dW2 = torch.einsum("bh,bs->hs", e1, x2 * g2s) / B
        dW3 = torch.einsum("bh,bc->hc", e2, yoh_t) / B
        self.W1c += self.eta_w1 * dW1
        n1 = torch.linalg.vector_norm(self.W1c, dim=1, keepdim=True)
        scale = torch.where(n1 > 1.0, 1.0 / n1.clamp_min(1e-12),
                            torch.ones_like(n1))
        self.W1c *= scale
        self.W1c[self.pad] = 0.0
        self.W1c *= self.s1
        self.W1cT = self.W1c.transpose(1, 2).contiguous()
        self.W2 += self.eta_w2 * dW2
        self.W3 += self.eta_w3 * dW3
        self.W2 *= self.m2
        self.W3 *= self.m3
        self.W2 = _colnorm_t(self.W2)
        n3 = torch.linalg.vector_norm(self.W3, dim=0)
        cap = cfg.w3_norm_cap
        over = n3 > cap
        self.W3[:, over] *= (cap / n3[over].clamp_min(1e-8))
        self._maybe_cap_w2()
        # ---- 读出头（自由推断 x2 + LMS，batch 平均）----
        x1g, x2f, _x3f, *_ = self._infer_batch(
            X_np, None, clamp=False, iters=cfg.readout_iters)
        logit = x2f @ self.W_out + self.b_out
        p = torch.softmax(logit / cfg.readout_tau, dim=1)
        err = p - torch.as_tensor(yoh, device=dev)
        self.W_out += -(cfg.readout_lr * torch.einsum("bh,bc->hc", x2f, err)) / B
        self.b_out += -(cfg.readout_lr * err.mean(dim=0))
        # 事件率 EMA（诊断）
        self._ev = self._ev + ((g1.float().mean() + g2.float().mean()) * 0.5
                               - self._ev) * 0.01

    # ---- 冻结评估：W_out 线性读出头（同 numpy 协议）----
    def eval_batch(self, X_np: np.ndarray, y_np: np.ndarray, tau: float,
                   block_mask: np.ndarray | None = None
                   ) -> tuple[float, float]:
        self.learning = False
        nll = acc = 0.0
        for i in range(len(y_np)):
            self._infer(X_np[i].ravel(), None, block_mask)
            logit = torch.mv(self.W_out.t(), self._x2) + self.b_out
            p = torch.softmax(logit / tau, dim=0)
            py = float(p[y_np[i]])
            nll -= np.log2(max(py, 1e-12))
            acc += float(p.argmax().item() == y_np[i])
        self.learning = True
        return float(nll / len(y_np)), float(acc / len(y_np))

    def fit_tau(self, X_np: np.ndarray, y_np: np.ndarray,
                taus: tuple) -> float:
        best, best_tau = np.inf, taus[0]
        for tau in taus:
            bpc, _ = self.eval_batch(X_np, y_np, tau)
            if bpc < best:
                best, best_tau = bpc, tau
        return float(best_tau)

    def event_rate(self) -> float:
        return float(self._ev.item())

    # ---- checkpoint（权重与元数据，torch 官方格式）----
    def state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.__dict__.items()
                if isinstance(v, torch.Tensor)}

    def load_state(self, sd: dict) -> None:
        for k, v in sd.items():
            setattr(self, k, v.to(self.device))
