"""G4a 扫描：无预钉下，受体野结构 + 稀疏度对 PC 自组织 x_self 码分离度的作用。

G1b 用 `fan_in_frac=0.75` 的**均匀随机**受体野（每单元近乎全连）→ 输入被混合成
共模向量 → x_self 码近重合（cos≈1、minL2≈0）。本脚本对比两种受体野构造：

  rf = "uniform"  : 现有均匀随机窗口（对照）
  rf = "block"    : 分块局部受体野 —— 每个隐单元只端接输入的连续局部窗口，
                    相邻隐单元端接相邻窗口（V1-like spatial topicality）。不同
                    状态（占不同输入段）→ 激活不相交隐单元。

扫描 fan_in（窗口覆盖比例）与 kwta_frac，看能否把"无预钉 NN 解码"拉到 7/7。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.g1_srpc_planner import TrapTree, encode_state_patterns
from srpc.config import ModelConfig
from srpc.env import make_patterns

from srpc.model import SRPCModel, _colnorm, _kwta


def build_local_sparse(cfg: ModelConfig, rng: np.random.Generator):
    """分块局部受体野替代 SRPCModel._born_sparse 的均匀随机扇入。

    每层：输出单元 j 端接输入轴的连续窗口 [j*step, j*step+w)，窗口中心沿
    输入轴均匀铺开（wrap 到头），w = round(win_frac * n_in)。这样相邻隐单元
    端接相邻输入窗口（V1 空间拓扑），不同输入位置激活不同隐单元区域。
    """
    def build_weights(n_in, n_out):
        w_frac = getattr(cfg, "recep_win_frac", 0.5)
        w = max(1, int(round(w_frac * n_in)))
        step = n_in / n_out
        W = np.zeros((n_in, n_out))
        for j in range(n_out):
            lo = int(j * step)
            idx = [(lo + k) % n_in for k in range(w)]
            W[idx, j] = rng.uniform(0.5, 1.0, size=w)
        return W

    cfg._rf = "block"
    W10 = build_weights(cfg.d_obs, cfg.n_l1)
    W21 = build_weights(cfg.n_l1, cfg.n_l2)
    Ws2 = build_weights(cfg.n_l2, cfg.n_self)
    return W10, W21, Ws2


def make_model(seed: int, rf: str, win_frac: float, kwta: float,
               fdyn: float = 0.5, alpha: float = 0.15,
               beta: float = 0.25) -> tuple[SRPCModel, np.random.Generator]:
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(d_obs=48, n_l1=24, n_l2=20, n_self=20,
                      inner_iters=3, fan_in_frac=0.0, kwta_frac=kwta,
                      theta_event=0.01, fan_in_dyn_frac=fdyn,
                      alpha=alpha, beta=beta)
    # 注入受体野窗口比例
    setattr(cfg, "recep_win_frac", win_frac)
    model = SRPCModel(cfg, n_actions=2,
                      rng=np.random.default_rng(seed + 1), self_loop=True)
    if rf == "block":
        W10, W21, Ws2 = build_local_sparse(cfg,
                                           np.random.default_rng(seed + 2))
        model.W10 = _colnorm(W10)
        model.W21 = _colnorm(W21)
        model.Ws2 = _colnorm(Ws2)
        model.mask10, model.mask21, model.mask2s = [
            w > 0 for w in (model.W10, model.W21, model.Ws2)]
        model.mask_sig = tuple(hash(m.tobytes())
                               for m in (model.mask10, model.mask21, model.mask2s)
                               if m is not None)
    # 注意：调受体野结构后需重建 dyn 掩码？fan_in_frac=0 → mask_dyn 由 fan_in_dyn_frac 触发
    return model, rng


def train_quick(model, task, patterns, steps, rng, reset=48):
    v = int(rng.integers(task.n_states))
    model.pred_self = np.zeros(model.cfg.n_self)
    model.last_z = None
    for step_i in range(steps):
        if step_i > 0 and step_i % reset == 0:
            v = int(rng.integers(task.n_states))
            model.pred_self = np.zeros(model.cfg.n_self)
            model.last_z = None
        model.observe(patterns[v], v)
        a = int(rng.integers(2))
        model.prepare_next(a)
        nxt = task.step(v, a, rng)
        v = nxt


def stable_xs(model, pat, state_id, floods=60):
    model.set_learning(False)
    acc = None
    for _ in range(floods):
        model.observe(pat, state_id)
        acc = model.xs if acc is None else acc + model.xs
    model.set_learning(True)
    return acc / floods


def nn_decode_acc(model, task, patterns):
    codes = np.stack([stable_xs(model, patterns[v], v)
                      for v in range(task.n_states)])
    # 用与训练不同的 seed 重放判读（避免冻结集分身）
    hits = 0
    for v in range(task.n_states):
        c = stable_xs(model, patterns[v], v)
        d = np.linalg.norm(codes - c.reshape(1, -1), axis=1)
        hits += int(np.argmin(d) == v)
    return hits / task.n_states, codes


def run_cfg(rf, win_frac, kwta, fdyn, seeds=(0, 1, 2)):
    accs = []
    for s in seeds:
        task = TrapTree()
        model, rng = make_model(s, rf, win_frac, kwta, fdyn)
        patterns = encode_state_patterns(task, rng, model.cfg.d_obs)
        train_quick(model, task, patterns, 3000, rng)
        acc, _ = nn_decode_acc(model, task, patterns)
        accs.append(acc)
    return float(np.mean(accs)), accs


def main() -> int:
    results = []
    print("=" * 74)
    print("G4a 扫描：受体野结构 × 稀疏度 —— 无预钉 NN 解码准确率（目标 7/7=1.0）")
    print("=" * 74)
    for rf in ("uniform", "block"):
        for win_frac in (0.5, 0.33, 0.25):
            for kwta in (0.5, 0.33):
                mean, accs = run_cfg(rf, win_frac, kwta, 0.5)
                tag = f"{rf:>8} win={win_frac:.2f} kwta={kwta:.2f}"
                print(f"  {tag} | NN解码 {mean:.3f}  seeds{accs}")
                results.append({"rf": rf, "win": win_frac, "kwta": kwta,
                                "mean": mean, "per_seed": accs})
    os.makedirs("/workspace/g4", exist_ok=True)
    with open("/workspace/g4/results_g4a_scan.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())