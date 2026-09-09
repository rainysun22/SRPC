"""G4：自组织编码版规划器 —— 无预钉、无 pin_codes，纯 PC 自组织 + 规则3 LMS 自学动力学。

承接 G1b 遗留问题（results_g1b_multi_seed.json 的 two deferred items）：
  G4-① 编码塌缩：G1b 用 `pin_xself_centroids` 结构钉码绕过塌缩。本版用**分块局部受体野**
        (block RF) + 推断权重平衡（alpha=0.05 弱顶层先验 / beta=0.5 强底层证据）让 x_self
        码在纯 Hebbian/竞争(kWTA) 自组织里自发可区分 → 彻底去掉 pin_codes。
  G4-② 动力学种子稳定：让吸收叶"持久→零误差"映射被稳定学到（叶自环残差→0），
        使陷阱信号对 DP 稳健（目标是 6/6 seed 前瞻>近视）。
        关键机制：**动作块掩码（per-action Wdyn）**。默认 last_z=[xs,a_onehot] 让所有
        动作共享同一个线性映射 pred=S@xs+A_a（S/A 跨动作状态共享 → 受限仿射容量，目标码
        靠种子运气才能到达 → seed2 陷阱反转 fail）。改成把 xs 放入第 a 个动作块
        z=[0.., xs_a块, ..]，Wdyn[:,a块] 即该动作独立转移矩阵 → 动作条件化鲁棒，6/6 pass。

与 G1b 唯一差异 = 编码方式（自组织 vs 结构钉码）+ 推断权重。规划、动力学学习、
判据完全一致（都是规则3 局部 LMS 自学 Wdyn + 前瞻 H DP 对照近视）。

判定（对齐 G1b，预注册）：
  ① 冻结下一状态判读命中率 ≥0.55（动力学自学达标）；
  ② 前瞻(H=3) 平均累计自误差 < 近视 greedy（>1e-3）；
  ③ 前瞻随 H 下降（H=3 < H=1）；
  ④ 近视损益可证伪（greedy−lookahead ≥0.05）。
  PASS = 四项全过；目标是 6/6 seed 全 PASS。

解码参考码 = 冻结（learning off）下 flooding 的稳态 x_self 码，与 G4a 验证的
"无预钉 NN 解码 7/7" 一致。

运行：python3 g4b_planner.py --seed N --all-seeds（写 g4/results_g4b_all_seeds.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srpc.config import ModelConfig
from srpc.env import make_patterns
from srpc.model import SRPCModel, _colnorm

from g1.g1_srpc_planner import (TrapTree, encode_state_patterns, _reset,
                                train_wdyn, run_policy, _learned_params)


# ----------------------------------------------------------------------------
# 自组织编码构建：分块局部受体野（g4a 验证最优结构）+ 推断权重平衡
# ----------------------------------------------------------------------------
def build_local_sparse_weights(n_in, n_out, win_frac, rng):
    """分块局部受体野：输出单元 j 端接输入轴连续窗口 [j*step, j*step+w)（wrap）。
    V1-like spatial topicality → 不同输入位置激活不相交隐单元 → 码天然可分离。"""
    w = max(1, int(round(win_frac * n_in)))
    step = n_in / n_out
    W = np.zeros((n_in, n_out))
    for j in range(n_out):
        lo = int(j * step)
        idx = [(lo + k) % n_in for k in range(w)]
        W[idx, j] = rng.uniform(0.5, 1.0, size=w)
    return W


def make_selforg_model(seed: int, n_l1: int = 60, nself: int = 64,
                       win_frac: float = 0.15, kwta: float = 0.33,
                       fdyn: float = 0.5, alpha: float = 0.05,
                       beta: float = 0.50, eta_dyn: float = 0.80,
                       dyn_decay: float = 1e-5, lms_norm: bool = False,
                       dyn_block: bool = False) -> tuple[SRPCModel, int]:
    """构建无 pin 的自组织编码 SRPCModel（block RF + 推断权重平衡）。

    nself/n_l1 放大 + win_frac 收窄：拉开枢纽状态(尤其 root/A-in)的码距，
    使 Wdyn 按动作可分地到达不同转移目标（重建"陷阱"）；eta_dyn 调大 +
    dyn_decay 调小：让吸收叶"持久→零误差"自环被稳定学到（叶自环残差→0）。
    """
    d_obs = 48
    cfg = ModelConfig(d_obs=d_obs, n_l1=n_l1, n_l2=nself, n_self=nself,
                      inner_iters=3, fan_in_frac=0.0, kwta_frac=kwta,
                      theta_event=0.01, fan_in_dyn_frac=fdyn,
                      alpha=alpha, beta=beta, eta_dyn=eta_dyn,
                      dyn_decay=dyn_decay)
    if lms_norm:
        setattr(cfg, "lms_norm", True)
    if dyn_block:
        setattr(cfg, "dyn_action_block", True)
    setattr(cfg, "recep_win_frac", win_frac)
    model = SRPCModel(cfg, n_actions=2,
                      rng=np.random.default_rng(seed + 1), self_loop=True)
    # 用 block RF 替换均匀随机扇入（fan_in_frac=0 时 _born_sparse 不会建 mask）
    rngb = np.random.default_rng(seed + 2)
    W10 = build_local_sparse_weights(cfg.d_obs, cfg.n_l1, win_frac, rngb)
    W21 = build_local_sparse_weights(cfg.n_l1, cfg.n_l2, win_frac, rngb)
    Ws2 = build_local_sparse_weights(cfg.n_l2, cfg.n_self, win_frac, rngb)
    model.W10 = _colnorm(W10)
    model.W21 = _colnorm(W21)
    model.Ws2 = _colnorm(Ws2)
    model.mask10, model.mask21, model.mask2s = [
        w > 0 for w in (model.W10, model.W21, model.Ws2)]
    model.mask_sig = tuple(hash(m.tobytes())
                           for m in (model.mask10, model.mask21, model.mask2s)
                           if m is not None)
    return model, d_obs


# ----------------------------------------------------------------------------
# 冻结评估（改用自组织稳态码作参考，而非 pin）
# ----------------------------------------------------------------------------
def _canon(model: SRPCModel) -> None:
    """规范重置：把 PC 感知状态清零，使 observe() 从干净初值沉降到该状态自有的
    规范码（消除跨步历史污染 —— PC 仅少量内迭代，不重置会让同一状态因"来路"
    不同而得到不同 live 码，把 Wdyn 误差目标变成噪声）。"""
    model.x1 = np.zeros(model.cfg.n_l1)
    model.x2 = np.zeros(model.cfg.n_l2)
    model.xs = np.zeros(model.cfg.n_self)


def stable_code(model: SRPCModel, pat, state_id, floods=60) -> np.ndarray:
    model.set_learning(False)
    acc = None
    for _ in range(floods):
        model.observe(pat, state_id)
        acc = model.xs if acc is None else acc + model.xs
    model.set_learning(True)
    return acc / floods


def eval_decode_frozen_selforg(model: SRPCModel, task: TrapTree, patterns,
                               n_meas: int = 400, seed: int = 3) -> dict:
    """冻结 Wdyn 的公平动力学评估，参考码 = 自组织稳态 x_self 码。"""
    rng = np.random.default_rng(seed)
    codes = np.stack([stable_code(model, patterns[v], v)
                      for v in range(task.n_states)])
    model.set_learning(False)
    cnt = np.zeros(task.n_states)
    hits = 0
    sq = 0.0
    for vv in range(task.n_states):
        for a in (0, 1):
            for _ in range(n_meas):
                _canon(model)             # 规范码：消除来路污染，使 xs(vv) 确定性可判读
                model.observe(patterns[vv], vv)
                model.prepare_next(a)
                pred = model.pred_self.copy()
                nxt = task.step(vv, a, rng)
                c = task.child[vv][a]
                # 期望下一码（用真转移 p：仅诊断 Wdyn 学到的量级，不喂给策略/规划）
                emix = task.p[vv][a] * codes[c] + (1.0 - task.p[vv][a]) * codes[vv]
                sq += float(np.linalg.norm(pred - emix) ** 2)
                cnt[vv] += 1.0
                d = np.linalg.norm(codes - pred.reshape(1, -1), axis=1)
                hits += int(np.argmin(d) == nxt)
    tot = cnt.sum()
    model.set_learning(True)
    norms = np.linalg.norm(codes, axis=1)
    # 码健康指标：最小成对距离 + 有无塌缩（零范数）
    sep = float(np.min([np.linalg.norm(codes[i] - codes[j])
                        for i in range(task.n_states)
                        for j in range(i + 1, task.n_states)]))
    zeros = int((norms < 1e-6).sum())
    return {
        "decode_acc": hits / tot,
        "expect_fit_rmse": float(np.sqrt(sq / tot)),
        "state_sep_min": sep,
        "zero_states": zeros,
    }


def train_selforg(model, task, patterns, n_steps, rng, reset=48, freeze_frac=0.5):
    """两阶段训练（G4-①② 协同收口）：

    阶段1（编码自组织）：**只**跑规则2(Hebbian) 长程自组织 x_self 码（关闭动力学，
        避免漂移码污染 Wdyn）；
    阶段2（动力学收敛）：冻结规则2、**只**跑规则3(LMS) 在静态码上按动作收敛。
        阶段2 采用**系统遍历 (state,action)**（非随机游走），保证每个转移——
        尤其低频入口 v0a1 / v1a1——都被 LMS 充分盯到，动作条件化不漏学：
        —— 使吸收叶"持久→零误差"自环被稳定学到（叶自环残差→0，G4-②目标）。
    """
    n_states = task.n_states
    p1 = int(freeze_frac * n_steps)     # 阶段1：编码自组织
    model.learn_perceptual = True
    model.learn_dynamics = False
    v = int(rng.integers(n_states))
    _reset(model)
    for step_i in range(n_steps):
        if step_i == p1:
            # 阶段2：冻结编码，专注动力学；系统遍历 (state,action)
            model.learn_perceptual = False
            model.learn_dynamics = True
        if step_i < p1:
            # ---- 阶段1：编码自组织（规则2），随机游走 ----
            if step_i > 0 and step_i % reset == 0:
                v = int(rng.integers(n_states))
                _reset(model)
            _canon(model)
            model.observe(patterns[v], v)
            a = int(rng.integers(2))
            model.prepare_next(a)
            nxt = task.step(v, a, rng)
            # 规范重置后再采集下一状态码：让 Wdyn 目标 = 确定性规范码(而非历史混合码)。
            # last_z 保留（含 xs(v) 与动作），observe(nxt) 内以 e_self=xs(nxt)−pred 更新 Wdyn。
            v = nxt
        else:
            # ---- 阶段2：系统遍历 (v,a)，每对观测 v→(a)→nxt，再观测 nxt 触发 Wdyn LMS ----
            idx = step_i - p1
            v = (idx // 2) % n_states
            a = idx % 2
            _reset(model)
            _canon(model)
            model.observe(patterns[v], v)
            model.prepare_next(a)
            nxt = task.step(v, a, rng)
            _canon(model)
            model.observe(patterns[nxt], nxt)
    model.learn_perceptual = True
    model.learn_dynamics = True


def _learned_params_c(model, task, patterns):
    """规范版经验转移参数（同 g1._learned_params，但每次观测前规范重置）。
    基于自学 Wdyn：Err[v][a]=平均真实自误差，P[v][a]=经验转移概率。
    规范重置 → 码确定性 → 估计稳定、DP 不被历史噪声误导。"""
    Err = np.zeros((task.n_states, 2))
    P = np.zeros((task.n_states, 2))
    naction = np.zeros((task.n_states, 2))
    model.set_learning(False)
    rng = np.random.default_rng(7)
    n_meas = 80
    for vv in range(task.n_states):
        for a in (0, 1):
            for _ in range(n_meas):
                _canon(model)
                model.observe(patterns[vv], vv)
                model.prepare_next(a)
                pred = model.pred_self.copy()
                nxt = task.step(vv, a, rng)
                _canon(model)
                model.observe(patterns[nxt], nxt)
                Err[vv][a] += float(np.linalg.norm(pred - model.xs))
                naction[vv][a] += 1.0
                if not task.is_leaf[vv] and nxt == task.child[vv][a]:
                    P[vv][a] += 1.0
    Err = Err / np.maximum(naction, 1e-9)
    P = P / np.maximum(naction, 1e-9)
    model.set_learning(True)
    return Err, P


def run_policy_c(model, task, patterns, policy, H, eps_steps, rng):
    """规范版策略执行（同 g1.run_policy）。每步在规范码上决策与测误差。"""
    Err, P = _learned_params_c(model, task, patterns)
    model.set_learning(False)
    J = np.zeros((task.n_states, H + 1))
    for h in range(1, H + 1):
        for v in range(task.n_states):
            if task.is_leaf[v]:
                J[v, h] = 0.0
                continue
            vals = []
            for a in (0, 1):
                c = task.child[v][a]
                future = P[v][a] * J[c, h - 1] + (1.0 - P[v][a]) * J[v, h - 1]
                vals.append(Err[v][a] + future)
            J[v, h] = min(vals)
    costs = []
    _reset(model)
    v = 0
    for _ in range(eps_steps):
        _canon(model)
        model.observe(patterns[v], v)
        if policy == "greedy":
            a = int(np.argmin(Err[v]))
        else:
            cand = [Err[v][aa]
                    + P[v][aa] * J[task.child[v][aa], H - 1]
                    + (1.0 - P[v][aa]) * J[v, H - 1]
                    for aa in (0, 1)]
            a = int(np.argmin(cand))
        model.prepare_next(a)
        pred = model.pred_self.copy()
        nxt = task.step(v, a, rng)
        _canon(model)
        model.observe(patterns[nxt], nxt)
        costs.append(float(np.linalg.norm(pred - model.xs)))
        v = nxt
    model.set_learning(True)
    return np.array(costs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--train_steps", type=int, default=4000)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--ep_horizon", type=int, default=40)
    args = ap.parse_args()

    seeds = list(range(6)) if args.all_seeds else [args.seed if args.seed is not None else 0]
    out = {"phase": "G4", "encoding": "selforg (block RF + alpha=0.05/beta=0.5) + 两阶段(systematic v,a sweep) + 动作块掩码(per-action Wdyn), no pin",
           "purpose": "无预钉下 x_self 码自发可区分 + 规则3 LMS 动力学种子稳定（动作块独立转移矩阵）→ 6/6 前瞻>近视",
           "per_seed": []}
    for sd in seeds:
        t0 = time.time()
        rng = np.random.default_rng(sd)
        task = TrapTree(rng=rng)
        model, d_obs = make_selforg_model(sd, dyn_block=True)
        patterns = encode_state_patterns(task, rng, d_obs)
        train_selforg(model, task, patterns, args.train_steps, rng)
        dec = eval_decode_frozen_selforg(model, task, patterns)
        dec_acc = dec["decode_acc"]

        # 用自学动力学做规划对比
        centroids = None  # 自组织版不需要外部质心
        g_rew, l_rew = [], []
        for e in range(args.episodes):
            rg = np.random.default_rng(100000 + sd * 10000 + e)
            cg = run_policy_c(model, task, patterns, "greedy", 1,
                              args.ep_horizon, rg)
            cl = run_policy_c(model, task, patterns, "lookahead", args.H,
                              args.ep_horizon, rg)
            g_rew.append(float(cg.sum()))
            l_rew.append(float(cl.sum()))
        g_mean, l_mean = float(np.mean(g_rew)), float(np.mean(l_rew))
        rel = (g_mean - l_mean) / (g_mean + 1e-9)

        j_1 = bool(dec_acc >= 0.55)
        j_2 = bool(l_mean < g_mean - 1e-3)
        j_3 = bool(l_mean < g_mean - 1e-3 and args.H >= 2)
        j_4 = bool((g_mean - l_mean) >= 0.05)
        per = {
            "seed": sd, "train_steps": args.train_steps, "episodes": args.episodes,
            "H": args.H, "ep_horizon": args.ep_horizon,
            "selforg_code": {"state_sep_min": round(dec["state_sep_min"], 4),
                             "zero_states": dec["zero_states"],
                             "expect_fit_rmse": round(dec["expect_fit_rmse"], 4)},
            "frozen_decode_acc": round(dec_acc, 4),
            "greedy": round(g_mean, 4), "lookahead": round(l_mean, 4),
            "rel_improv": round(rel, 4),
            "judges": {"dec_self_learned": j_1, "lookahead_lt_greedy": j_2,
                       "improve_with_H": j_3, "greedy_loss": j_4},
            "pass": all((j_1, j_2, j_3, j_4)),
        }
        out["per_seed"].append(per)
        print(f"seed {sd}: dec={dec_acc:.3f} greedy={g_mean:.3f} "
              f"lookahead={l_mean:.3f} rel={rel:+.1%} "
              f"sep_min={dec['state_sep_min']:.3f} zeros={dec['zero_states']} "
              f"-> {'PASS' if per['pass'] else 'FAIL'}  wall={time.time()-t0:.0f}s", flush=True)

    passes = sum(1 for p in out["per_seed"] if p["pass"])
    out["summary"] = {
        "pass_rate": f"{passes}/{len(out['per_seed'])}",
        "mean_rel_improv": round(float(np.mean([p['rel_improv'] for p in out['per_seed']])), 4),
    }
    os.makedirs("/workspace/g4", exist_ok=True)
    with open("/workspace/g4/results_g4b_all_seeds.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("=" * 70)
    print(f"G4 自组织编码 + 种子稳定规划：PASS {passes}/{len(out['per_seed'])}")
    print("写入 /workspace/g4/results_g4b_all_seeds.json")
    return 0 if passes == len(out["per_seed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())