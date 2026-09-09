"""阶段 F1b-1：VSA/HRR 绑定层 组合查询自验证（纯 NumPy，CPU/GPU 均可）。

验证对象：srpc/vsa.py（循环卷积 bind/unbind + cleanup 记忆）。

两部分：
  1) 算子正确性：bind(unbind(bind(a,b),b)) ≈ a（解绑非负 + 余弦极高）；
     不同随机对叠加后按 role unbind 互相低干扰。
  2) 组合召回率：S = Σ_{i<k} role_i ⊗ filler_i；query(role_j) -> cleanup(S ⊛ role_j)
     应解回其真 filler。容量-精度曲线 = recall ~ (k 对, d 维)。
     F1b 口径容量上限 ~O(d/4-8)（Plate 1995）。

判据（对齐 ROADMAP F1：组合召回率 ≥ 阈值 / 容量-精度曲线）：
  - 低负载（k ≤ 容量）recall >= 0.95；
  - 中负载容量的分辨率（半高 P50）落在 d 量级区间。
"""
from __future__ import annotations
import time

import numpy as np

from srpc.vsa import VsaMemory, bind, unbind, gauss_vecs

DIMS = [128, 256, 512, 1024]
K_GRID = [1, 4, 8, 16, 32, 64]      # 绑定对数量（超负载压力）
SEED = 7


def test_operators(d: int = 512) -> dict:
    """bind/unbind 正确性 + 解绑正交性。"""
    rng = np.random.default_rng(SEED)
    a, b, c = (rng.standard_normal(d) for _ in range(3))
    # unbind(bind(a,b), b) ≈ a
    ahat = unbind(bind(a, b), b)
    cos_ab = float(np.dot(a, ahat) / (np.linalg.norm(a) * np.linalg.norm(ahat) + 1e-12))
    # bind(a,b) 与 bind(a,c) 在 b≠c 时近似不相关（a ⊗ b、a ⊗ c 的近正交）
    dab = bind(a, b); dac = bind(a, c)
    cos_bc = float(np.dot(dab, dac) / (np.linalg.norm(dab) * np.linalg.norm(dac) + 1e-12))
    return {"d": d, "unbind_cosine": round(cos_ab, 4),
            "bind_cross_cosine": round(cos_bc, 4)}


def capacity_curve(d: int, k_grid: list[int]) -> dict:
    """同一 d 下，随绑定对数 k 的组合召回率（即该维的容量-精度曲线）。"""
    n_filler = int(2 * max(k_grid))     # 词汇表存 2×最多对数，保证 query 有干扰项可辨
    fillers = gauss_vecs(d, n_filler, seed=100 + d)
    roles = gauss_vecs(d, n_filler, seed=200 + d)
    curve = {}
    for k in k_grid:
        if k > n_filler:
            continue
        m = VsaMemory(d=d, seed=SEED)
        for i in range(k):
            m.store(roles[i], fillers[i])
        curve[str(k)] = round(float(m.recall(roles[:k], fillers[:k])), 4)
    return {"d": d, "curve": curve}


def main() -> int:
    _t0 = time.time()
    ops = test_operators(DIMS[-1])
    print(f"[operators @d={ops['d']}] unbind_cosine={ops['unbind_cosine']} "
          f"(期望≈1, 正确性) bind_cross_cosine={ops['bind_cross_cosine']} (期望≈0, 正交)")

    caps = {}
    for d in DIMS:
        caps[d] = capacity_curve(d, K_GRID)

    # 汇总：每维在 K_GRID 上的召回表
    print("\n容量-精度曲线（组合召回率，行=维度 d，列=绑定对数 k）：")
    header = "d      " + "".join(f"{k:>9}" for k in K_GRID)
    print(header)
    for d in DIMS:
        c = caps[d]["curve"]
        row = "".join(f"{c.get(str(k), '  -   '):>9}" for k in K_GRID)
        print(f"{d:>6}" + row)

    # 判据集总
    low_ok = all(caps[d]["curve"]["8"] >= 0.95 for d in (512, 1024))
    # P50（召回 ≥0.5 的最大 k）估算：取 d 维下单超载端点
    p50 = {}
    for d in DIMS:
        over = next((k for k in K_GRID
                     if caps[d]["curve"].get(str(k), 0.0) < 0.5), K_GRID[-1])
        p50[d] = round(over if K_GRID[0] == 1 else over, 0)
    summary = {
        "operators": ops,
        "capacity_curves": caps,
        "low_load_recall8": {str(d): caps[d]["curve"]["8"] for d in DIMS},
        "criterion_low_load_pass": bool(low_ok),
        "p50_before_half": p50,
        "wall_s": round(time.time() - _t0, 1),
    }
    print(f"\n低负载(k=8)召回率: {summary['low_load_recall8']}")
    print(f"判据[低负载 recall8≥0.95 (d∈{512,1024})]: {summary['criterion_low_load_pass']}")
    print(f"P50(k 达半召前): {p50}")
    return 0 if low_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())