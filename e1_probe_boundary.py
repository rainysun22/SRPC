"""E1 证据链探针（三）：(a) 冻结随机主干 + RLS 读出的可分性上限；
(b) 在线 PCN 的类数断点扫描（2/4/8/16 类跨块 XOR）。

结论（v2 记录）：
- (a) 随机主干特征对 16 类跨块 XOR 不可分（acc≈0.09 vs 机会 0.0625）
  ——判别需要主干塑形交互特征（信用分配），读出侧（RLS 二阶在线优化，
  阶段 B 已验证组件）无法替代；
- (b) 在线 PCN xorsum 类数断点：2/4/8/16 类 acc = 贴机会/0.38/0.16/0.07
  ——4 类（2-bit）有弱信号但 gap 不足；2 类 1-bit parity 为纯高频函数
  （所有低阶统计为零）反而最劣；与 SQ-hard 归因一致。
"""
import time

import numpy as np

from srpc.config import LangConfig
from srpc.lang import LangPCN, make_task


def features(m: LangPCN, X: np.ndarray) -> np.ndarray:
    """冻结主干自由推断下的 (x1, x2) 拼接特征。"""
    F = np.empty((len(X), m.x1.size + m.x2.size), dtype=np.float32)
    for i, x0 in enumerate(X):
        m.x1[:] = 0.0
        m.x2[:] = 0.0
        m._infer(x0, np.zeros(m.C, dtype=np.float32), free_out=True)
        F[i] = np.concatenate([m.x1, m.x2])
    return F


class RLS:
    """逐输出维独立 RLS（在线二阶优化，阶段 B 读出头同款）。"""

    def __init__(self, d: int, C: int, lam: float = 1e-2):
        self.W = np.zeros((C, d), dtype=np.float64)
        self.P = np.eye(d) / lam
        self.C = C

    def update(self, phi: np.ndarray, y: int):
        pred = self.W @ phi
        k = self.P @ phi
        denom = 1.0 + float(phi @ k)
        k /= denom
        e = np.zeros(self.C)
        e[y] = 1.0
        self.W += np.outer(e - pred, k)
        self.P -= np.outer(k, k) * denom

    def predict(self, phi: np.ndarray) -> int:
        return int((self.W @ phi).argmax())


def probe_oracle_rls(cfg, seed=0, delta=4, n_train=4000, n_eval=400):
    rng = np.random.default_rng(seed * 5000 + 13)
    X, y, _ = make_task(cfg, rng, n_train + n_eval, delta, "xorsum")
    m = LangPCN(cfg, np.random.default_rng(seed * 5000 + 17), "error",
                n_blocks=delta + 2)
    m.set_learning(False)                    # 冻结随机主干
    F = features(m, X)
    rls = RLS(F.shape[1], cfg.n_classes)
    for t in range(n_train):
        rls.update(F[t], int(y[t]))
    acc = float(np.mean([rls.predict(F[n_train + i]) == int(y[n_train + i])
                         for i in range(n_eval)]))
    return acc


def probe_classnum(cfg_base, seed=0, delta=4, steps=8000):
    """在线 PCN（error 臂，无累积）在 2/4/8/16 类跨块 XOR 上的精度。"""
    out = {}
    for C, alpha in ((2, "ab"), (4, "abcd"), (8, "abcdefgh"), (16, "abcdefghijklmnop")):
        cfg = type(cfg_base)(**{**cfg_base.__dict__, "alphabet": alpha,
                                "n_classes": C})
        rng = np.random.default_rng(seed * 5000 + 13)
        X, y, _ = make_task(cfg, rng, steps + 400, delta, "xorsum")
        m = LangPCN(cfg, np.random.default_rng(seed * 5000 + 17), "error",
                    n_blocks=delta + 2)
        for t in range(steps):
            m.train_step(X[t], int(y[t]))
        acc = float(np.mean([m.predict_free(X[steps + i])[0] == int(y[steps + i])
                             for i in range(400)]))
        out[C] = acc
    return out


if __name__ == "__main__":
    cfg = LangConfig()
    t0 = time.time()
    acc = probe_oracle_rls(cfg)
    print(f"(a) 冻结随机主干+RLS读出 16类 ({time.time()-t0:.0f}s): acc={acc:.3f} (机会0.0625)")
    t0 = time.time()
    out = probe_classnum(cfg)
    print(f"(b) 在线PCN类数断点 ({time.time()-t0:.0f}s): {out} (机会 1/C)")
