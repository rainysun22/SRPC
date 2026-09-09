"""G4a 诊断：无预钉时，PC 层级自组织出的 x_self 码是否可区分？

G1b 用 pin_xself_centroids 预钉不重叠码绕过"编码塌缩"。本脚本移除预钉，
直接观察 SRPCModel 的 PC 层级（规则1 稀疏推断 + 规则2 Hebbian + kWTA）在
纯自组织下产生的 x_self 码：
  - 是否塌缩：全零 / 全同 / 支撑重叠；
  - 解码分离度：冻结后对各状态独立重放，最近邻判读命中率、pairwise 距离。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns, train_wdyn
from srpc.config import ModelConfig
from srpc.model import SRPCModel


def build_model(seed: int) -> tuple[SRPCModel, np.random.Generator]:
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(d_obs=48, n_l1=24, n_l2=20, n_self=20,
                      inner_iters=3, fan_in_frac=0.75, kwta_frac=0.5,
                      theta_event=0.01)
    model = SRPCModel(cfg, n_actions=2,
                      rng=np.random.default_rng(seed + 1), self_loop=True)
    return model, rng


def stable_xs(model: SRPCModel, pat: np.ndarray, state_id: int,
              floods: int = 60) -> np.ndarray:
    """冻结模型下对状态重复重放，取收敛后 xs 的均值（消除推断抖动）。"""
    model.set_learning(False)
    acc = None
    for _ in range(floods):
        model.observe(pat, state_id)
        acc = model.xs if acc is None else acc + model.xs
    model.set_learning(True)
    return acc / floods


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def diagnose(seed: int) -> dict:
    task = TrapTree()
    model, rng = build_model(seed)
    patterns = encode_state_patterns(task, rng, model.cfg.d_obs)
    # 用训练期 Wdyn（含规则3 自省误差回拉）在线学一段，不钉码
    _, _diag = train_wdyn(model, task, patterns, 4000, rng)

    # 冻结测各状态稳态码
    codes = np.stack([stable_xs(model, patterns[v], v) for v in range(task.n_states)])
    norms = np.linalg.norm(codes, axis=1)
    support = (codes > 1e-6)          # (7,nself) 布尔支撑
    # 支撑重叠：两状态非零位置交集占比（对支撑更小的归一）
    overlaps = []
    for i in range(task.n_states):
        for j in range(i + 1, task.n_states):
            inter = np.logical_and(support[i], support[j]).sum()
            denom = min(support[i].sum(), support[j].sum()) or 1
            overlaps.append((inter / denom, i, j))
    # 最近邻解码命中（把 codes 当原型，重放判读）
    nn_hits = 0
    for v in range(task.n_states):
        c = stable_xs(model, patterns[v], v)
        d = np.linalg.norm(codes - c.reshape(1, -1), axis=1)
        nn_hits += int(np.argmin(d) == v)
    # pairwise 距离
    pw = [float(np.linalg.norm(codes[i] - codes[j]))
          for i in range(task.n_states) for j in range(i + 1, task.n_states)]
    return {
        "seed": seed,
        "state_norm": [round(float(n), 3) for n in norms],
        "zero_states": int((norms < 1e-6).sum()),
        "support_size_per_state": [int(s.sum()) for s in support],
        "max_support_overlap_frac": round(float(max((o[0] for o in overlaps), default=0.0)), 3),
        "mean_pairwise_dist": round(float(np.mean(pw)), 3),
        "min_pairwise_dist": round(float(np.min(pw)), 3),
        "cos_max_pair": round(float(max(cosine(codes[i], codes[j])
                for i in range(task.n_states) for j in range(i + 1, task.n_states))), 3),
        "nn_decode_hits": f"{nn_hits}/{task.n_states}",
        "decode_acc": round(nn_hits / task.n_states, 3),
    }


def main() -> int:
    out = [diagnose(s) for s in (0, 1, 2, 3, 4, 5)]
    print("=" * 70)
    print("G4a 诊断：无预钉 PC 层级自组织 x_self 码")
    print("=" * 70)
    for r in out:
        print(f" seed{r['seed']:>2} | norm={r['state_norm']} | 零态={r['zero_states']}"
              f" | 支撑每态={r['support_size_per_state']} | 最大支撑重叠={r['max_support_overlap_frac']}"
              f" | cos_max={r['cos_max_pair']} | minL2={r['min_pairwise_dist']}"
              f" | NN解码={r['nn_decode_hits']}")
    os.makedirs("/workspace/g4", exist_ok=True)
    with open("/workspace/g4/results_g4a_diag.json", "w") as f:
        json.dump({"diag": out}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())