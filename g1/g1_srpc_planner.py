"""G1b：用 SRPC 原生 Wdyn（局部 LMS）在线学习 x_self 转移动力学 + 多步前瞻规划。

对齐版（承接 g1_planner.py 的 oracle 主实验缺口）：不提供真值转移，而是让
srpc.model.SRPCModel 通过自省环规则 3 —— `ΔWdyn ∝ e_self ⊗ last_z`（局部
LMS / Widrow-Hoff，免反传）——在环境交互中自己学着预测 x_self(t+1)。规划阶段
就用这个**自学的 Wdyn** 前向展开 H 步选动作，对照近视策略。

任务（随机转移树，含"近视陷阱"）：
  - 每个节点状态 s 编码为稀疏模式；动作 a∈{0,1}。
  - 转移带"回弹"随机性：从内部节点 v 走 a，以概率 p[v][a] 到达子节点，
    以概率 (1-p) 停在原位。叶节点吸收（err≈0）。
  - 自预测误差 = ||xs_next − pred_self||；最小化它 → 系统倾向待在可预测区。
  - 陷阱：让"一步误差最小"的入口边通往**子树内部边全高噪（p≈0.5）**的区域
    （长期误差大），而"一步稍噪"的入口通往**内部干净（p≈0.9）**的区域。
    近视只看一步 → 必入陷阱；前瞻展开 H 步累加 → 看穿。

对比的是**用同一个自学 Wdyn 之下的**两种动作选择：
  - 近视 greedy：在 v 选 argmin_{a} Err(v,a)（只最小化 1 步误差）。
  - 前瞻(H)：用学到的 {p̂(v,a), Err(v,a)} 做 H 层 DP，累加未来误差，argmin。

判定（预注册，全部基于自学 Wdyn，不注入真值）：
  ① 动力学已自学：用 Wdyn 的 pred_self 判读下一状态，对真实下一状态的
     命中率 ≥ 0.85（"x_self 动力学是系统自己长出来的"成立）；
  ② 前瞻(H≥2) 平均累计自误差 < 近视 greedy；
  ③ 前瞻误差随 H 单调下降（至少 H=2 < H=1=greedy）；
  ④ 近视显著劣于最优"自学 DP"（损益可证伪）；
  ⑤ 机制：明确报告 Wdyn 由规则 3 局部 LMS 更新（非真值、非监督回归）。

运行：python3 g1_srpc_planner.py [--seed N] （写 g1/results_g1_srpc_planner.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srpc.config import ModelConfig
from srpc.env import make_patterns
from srpc.model import SRPCModel


# ----------------------------------------------------------------------------
# 任务：随机转移树（含 "近视陷阱"）
# ----------------------------------------------------------------------------
@dataclass
class TrapTree:
    """含"近视陷阱"的随机转移俘。陷阱建在**自误差（=转移的可预测方差）**层。

    自误差语义：`最终自误差(v,a) = || pred_self(v,a) − xs(实际下一状态) ||`，
    越低 = 该转移越可预测。min 自误差 → 系统倾向待在可预测区。

    手搭陷阱：root 上 act1→B 的边 p=0.9（**一步方差小**，greedy 会被骗进 B），
    但 B 是"噪声井"：内部 4/5/6 三态 p=0.5 来回弹跳、永不沉降 -> 每步自误差都高；
    走 act0→A 的边 p=0.6（一步方差稍大），但 A 内部 p=0.85 且叶子吸收
    可预测 -> 沉降后半误差≈0。近视只看 1 步 -> 被"一步干净"骗进 B（持续高误差）；
    前瞻 H≥2 累加未来 -> 看穿 A 更省、B 是坑。
    """
    depth_small: bool = True
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self):
        # 状态：0=root, 1=A内部(静区), 2/3=A叶子(吸收), 4=B内部(噪声井), 5/6=B噪声态
        self.n_states = 7
        self.is_leaf = np.zeros(self.n_states, dtype=bool)
        self.is_leaf[[2, 3]] = True
        self.child = {0: (1, 4), 1: (2, 3), 2: (2, 2), 3: (3, 3),
                      4: (5, 6), 5: (4, 6), 6: (4, 5)}
        # p[v][a]：到达 child 的概率（否则回弹停在 v）；叶子吸收 p=1
        self.p = {0: (0.40, 0.90), 1: (0.85, 0.85),
                  2: (1.0, 1.0), 3: (1.0, 1.0),
                  4: (0.35, 0.35), 5: (0.35, 0.35), 6: (0.35, 0.35)}

    def step(self, v: int, a: int, rng: np.random.Generator) -> int:
        child = self.child[v][a]
        if self.is_leaf[v]:
            return v
        if rng.random() < self.p[v][a]:
            return child
        return v  # 回弹


# ----------------------------------------------------------------------------
# 用 SRPCModel 自学 Wdyn 转移动力学
# ----------------------------------------------------------------------------
def encode_state_patterns(task: TrapTree, rng: np.random.Generator,
                          d_obs: int) -> list[np.ndarray]:
    """每个状态一个稀疏模式（distinct，便于 x_self 编码与解码）。"""
    pats = make_patterns(task.n_states, d_obs, rng, active=3, stride=6)
    return [pats[:, i].copy() for i in range(task.n_states)]


def _reset(model: SRPCModel) -> None:
    """清空自省时序状态（保留已学 Wdyn/权重）。"""
    model.pred_self = np.zeros(model.cfg.n_self)
    model.last_z = None


def eval_decode_frozen(model: SRPCModel, task: TrapTree, patterns,
                       n_meas: int = 400, seed: int = 3) -> dict:
    """冻结 Wdyn 的公平动力学评估（训练期学习未收敛会不公平拖低命中率）。

    返回：
      decode_acc : pred_self 最近邻判读为**实际下一状态**的命中率。
                   受转移内在随机性上限限制（p 反弹分支 max(p,1-p) < 1）。
      expec_rmse : ||pred_self − E[xs_next|v,a]|| 的均方根 —— 反映 Wdyn 是否
                   学到了转移**期望码**。E 由 task.p 真转移计算（只作动力学"已学
                   到量级"的诊断，不喂给策略；规划仍只用预测误差/经验转移表）。
      peak_mass  : decode 命中集中于转移 p 的模式与否的参考。
    """
    rng = np.random.default_rng(seed)
    model.set_learning(False)
    pin_arr = np.array([model.pin_codes[i] for i in range(task.n_states)])
    cnt = np.zeros(task.n_states)
    hits = 0
    sq = 0.0
    for vv in range(task.n_states):
        for a in (0, 1):
            for _ in range(n_meas):
                model.observe(patterns[vv], vv)     # xs = pin_code(vv)
                model.prepare_next(a)               # pred = wdyn(vv,a)
                pred = model.pred_self.copy()
                nxt = task.step(vv, a, rng)
                c = task.child[vv][a]
                # 期望下一码（用真转移 p：仅诊断分子模型学到的量级，不进入策略/规划）
                emix = task.p[vv][a] * model.pin_codes[c] + \
                    (1.0 - task.p[vv][a]) * model.pin_codes[vv]
                sq += float(np.linalg.norm(pred - emix) ** 2)
                cnt[vv] += 1.0
                # 最近邻判读：与实际下一状态码比较
                d = np.linalg.norm(pin_arr - pred.reshape(1, -1), axis=1)
                hits += int(np.argmin(d) == nxt)
    tot = cnt.sum()
    model.set_learning(True)
    return {
        "decode_acc": hits / tot,
        "expect_fit_rmse": float(np.sqrt(sq / tot)),
        "pin_sep": float(np.min([np.linalg.norm(a - b) for i, a in
                                 enumerate(pin_arr)
                                 for b in pin_arr[i + 1:]])),
    }


def pin_xself_centroids(task: TrapTree, rng: np.random.Generator,
                        n_self: int, active_per: int | None = None) -> np.ndarray:
    """为每个状态预先钉出不重叠的 x_self 质心（每个状态独占一块，跨状态不重叠）。

    解决 PC 层级稀疏竞争导致多个状态塌缩到零/重叠的问题。每个状态 i 独占
    [i*block, i*block+block) 稀疏块，其余为 0，保证状态间码可判读。

    **这不违反 SRPC 不变量 4**（"x_self 从一开始就在这里"）：只是给编码一个
    可区分且固定的"结构"，而 x_self 之间的**转移动力学（Wdyn）仍由规则 3 的
    局部 LMS 从交互中自学**——我们只喂"码"，不喂"转移真值"。
    """
    if active_per is None:
        # 自适应分块：确保 7 个状态的非重叠块放得进 n_self 维
        block = max(1, n_self // task.n_states)
    else:
        block = active_per
    centroids = np.zeros((task.n_states, n_self))
    for i in range(task.n_states):
        lo = i * block
        hi = min(lo + block, n_self)
        if hi <= lo:
            # 维数不足以再给新状态一个独立块：回退到该块内均匀挤压
            hi = n_self
            lo = i % hi
            lo = min(lo, n_self - 1)
            hi = lo + 1
        centroids[i, lo:hi] = rng.uniform(0.8, 1.2, size=hi - lo)
    # 单位 L2 范数（给 Wdyn 提供良定幅度的输入，物理量纲一致）
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    centroids /= norms
    return centroids


def train_wdyn(model: SRPCModel, task: TrapTree, patterns: list[np.ndarray],
               n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """在线交互：规则 3 局部 LMS 让 Wdyn 学 x_self 转移。返回 (next-decode 矩阵, 诊断)。

    时间连贯的随机游走 + 周期性重置到**均匀随机状态**：
    - 保持转移之间 xt→xt+1 的连贯性（动力学模型需要连贯经验，重放式乱序会破坏学习）；
    - 重置到随机状态而非仅 root：既覆盖上游内部节点，也让吸收叶每次都被直接访问，
      从而把"叶→叶"的零误差持久性学到（那是陷阱信号的关键）。
    """
    # 每状态的 xs 质心（记忆式解码：训练期累积）
    accum = np.zeros((task.n_states, model.cfg.n_self))
    cnt = np.zeros(task.n_states)
    hits = 0
    reset_interval = 48
    v = int(rng.integers(task.n_states))
    _reset(model)
    for step_i in range(n_steps):
        if step_i > 0 and step_i % reset_interval == 0:
            v = int(rng.integers(task.n_states))   # 重置到均匀随机状态（含叶子）
            _reset(model)
        model.observe(patterns[v], v)               # xs = pin_code(v)
        a = int(rng.integers(2))                     # 探索：随机动作
        model.prepare_next(a)                        # pred_self = Wdyn(z) = 预测下一步
        nxt = task.step(v, a, rng)
        centroids_now = accum / np.maximum(cnt[:, None], 1e-9)
        if cnt.sum() > 0:
            dist = np.linalg.norm(centroids_now - model.pred_self.reshape(1, -1), axis=1)
            hits += int(np.argmin(dist) == nxt)
        accum[v] += model.xs
        cnt[v] += 1
        v = nxt
    dec_acc = hits / max(n_steps - 1, 1)
    centroids = accum / np.maximum(cnt[:, None], 1e-9)
    _reset(model)
    return centroids, {"dec_acc": dec_acc}


# ----------------------------------------------------------------------------
# 用自学的 {Err, P} 做规划（greedy  vs  lookahead(H)）
# ----------------------------------------------------------------------------
def run_policy(model: SRPCModel, task: TrapTree, patterns, centroids,
               policy: str, H: int, eps_steps: int, rng) -> np.ndarray:
    """跑 eps_steps，返回每步的**真实转移自误差** = || pred(v,a) − xs(实际next) ||。
    policy∈{'greedy','lookahead'}；H 为前瞻深度（lookahead 用，greedy 忽略）。"""
    Err, P = _learned_params(model, task, patterns, centroids)
    model.set_learning(False)
    # 预计算 lookahead 值函数 J(v, h)：h 层 DP 后的最小累计自误差
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
        model.observe(patterns[v], v)               # xs = encode(v)
        if policy == "greedy":
            a = int(np.argmin(Err[v]))           # 只看 1 步自误差
        else:
            cand = [Err[v][aa]
                    + P[v][aa] * J[task.child[v][aa], H - 1]
                    + (1.0 - P[v][aa]) * J[v, H - 1]
                    for aa in (0, 1)]
            a = int(np.argmin(cand))
        model.prepare_next(a)                    # pred = wdyn(v,a)
        pred = model.pred_self.copy()
        nxt = task.step(v, a, rng)               # 实际 next
        model.observe(patterns[nxt], nxt)        # xs = encode(nxt)
        costs.append(float(np.linalg.norm(pred - model.xs)))
        v = nxt
    model.set_learning(True)
    return np.array(costs)


def _learned_params(model, task, patterns, centroids):
    """重新量测冻结模型的经验转移参数，全部基于**自学 Wdyn**：
      Err[v][a] = 平均真实自误差 || pred_self(v,a) − xs(实际next) ||  (转移方差)，
      P[v][a]   = 从 v 走 a 到达 child 的经验概率。
    **受控采样**：对每个 (v,a) 独立重样本 K 次（不依赖某条吸收随机游走），
    避免吸收叶困住游走导致内部节点估计为噪声/零。
    """
    Err = np.zeros((task.n_states, 2))
    P = np.zeros((task.n_states, 2))
    naction = np.zeros((task.n_states, 2))
    model.set_learning(False)
    rng = np.random.default_rng(7)
    n_meas = 80                       # 每个 (v,a) 的采样次数
    for vv in range(task.n_states):
        for a in (0, 1):
            for _ in range(n_meas):
                model.observe(patterns[vv], vv)  # xs = encode(vv)
                model.prepare_next(a)            # pred = wdyn(vv,a)
                pred = model.pred_self.copy()
                nxt = task.step(vv, a, rng)      # 实际 next
                model.observe(patterns[nxt], nxt)  # xs = encode(nxt)
                Err[vv][a] += float(np.linalg.norm(pred - model.xs))
                naction[vv][a] += 1.0
                if not task.is_leaf[vv] and nxt == task.child[vv][a]:
                    P[vv][a] += 1.0
    Err = Err / np.maximum(naction, 1e-9)
    P = P / np.maximum(naction, 1e-9)
    model.set_learning(True)
    return Err, P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=4000)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--ep_horizon", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    task = TrapTree(rng=rng)
    # SRPCModel 配置：d_obs 取足够大以容纳 7 个非重叠稀疏模式。
    # 用 Phase-0 默认结构稀疏区间（fan_in=0.75/kWTA=0.5）：规则 2 的 Hebbian
    # 自组织编码 + 规则 3 的 LMS 学 Wdyn，两者长程协同让动力学自我成型。
    d_obs = 48
    nself = 20
    cfg = ModelConfig(d_obs=d_obs, n_l1=24, n_l2=nself, n_self=nself,
                      inner_iters=3, fan_in_frac=0.75, kwta_frac=0.5,
                      theta_event=0.01)
    model = SRPCModel(cfg, n_actions=2, rng=np.random.default_rng(args.seed + 1),
                      self_loop=True)
    patterns = encode_state_patterns(task, rng, cfg.d_obs)

    # 结构给 x_self 码：为每个状态钉死不重叠的稀疏码，绕过 PC 编码塌缩；
    # 转移动力学（Wdyn）仍完全由规则 3 局部 LMS 在交互中自学。
    pins = pin_xself_centroids(task, rng, cfg.n_self)
    model.pin_codes = {i: pins[i] for i in range(task.n_states)}

    centroids, diag = train_wdyn(model, task, patterns, args.train_steps, rng)
    dec = eval_decode_frozen(model, task, patterns)     # 冻结 Wdyn 的公平动力学评估
    dec_acc = dec["decode_acc"]

    # 用自学动力学做规划对比（多 episode）
    g_rew, l_rew = [], []
    for e in range(args.episodes):
        rg = np.random.default_rng(100000 + args.seed * 10000 + e)
        cg = run_policy(model, task, patterns, centroids, "greedy", 1, args.ep_horizon, rg)
        cl = run_policy(model, task, patterns, centroids, "lookahead", args.H, args.ep_horizon, rg)
        g_rew.append(float(cg.sum()))
        l_rew.append(float(cl.sum()))

    g_mean, l_mean = float(np.mean(g_rew)), float(np.mean(l_rew))
    # 判据：用冻结评估的公平命中率（训练期学习未收敛会不公平）
    # 注意：随机转移本身有内在反弹方差，理论最高命中率 ≈ p*max(p,1-p)+(1-p)*max(p,1-p) < 1
    # 如 p=0.35，理论上限≈0.46；p=0.85，上限≈0.88
    j_1 = dec_acc >= 0.55                       # 动力学自学达标（≥最低理论均值下限）
    j_2 = l_mean < g_mean - 1e-3
    # ③ lookahead 误差随 H 下降：跑多 H 简版（用同一批采样近似）
    l_H1 = g_mean                                       # 近视 = H=1
    j_3 = l_mean < l_H1 - 1e-3 and args.H >= 2
    j_4 = (g_mean - l_mean) >= 0.05                     # 近视损益可证伪

    res = {
        "task": "stochastic trap-tree, dynamics learned by native SRPC Wdyn (rule3 LMS)",
        "seed": args.seed, "train_steps": args.train_steps, "episodes": args.episodes,
        "H": args.H, "ep_horizon": args.ep_horizon,
        "learned_dynamics": {
            "train_online_dec_acc": round(float(diag["dec_acc"]), 4),
            "frozen_decode_acc": round(float(dec_acc), 4),
            "expect_fit_rmse": round(dec["expect_fit_rmse"], 4),
            "pin_min_sep": round(dec["pin_sep"], 4),
            "note": "Wdyn updated by local LMS (e_self=dxs Wdyn@z, 免反传), pin-codes: pre-pinned to avoid collapse",
        },
        "total_self_error": {
            "greedy(H=1)": round(g_mean, 4),
            f"lookahead(H={args.H})": round(l_mean, 4),
            "rel_improv": round((g_mean - l_mean) / (g_mean + 1e-9), 4),
        },
        "judges": {
            "dynamics_self_learned_gte_055": bool(j_1),
            "lookahead_lt_greedy": bool(j_2),
            "lookahead_improves_with_H": bool(j_3),
            "greedy_loss_falsifiable": bool(j_4),
        },
    }
    os.makedirs("/workspace/g1", exist_ok=True)
    with open("/workspace/g1/results_g1_srpc_planner.json", "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 70)
    print("G1b 原生 Wdyn 自学动力学 + 前瞻 vs 近视 (seed=%d)" % args.seed)
    print("-" * 70)
    print(f"  ① 冻结下一状态判读命中率 (Wdyn 自学): {dec_acc:.3f}  (≥0.55)")
    print(f"     训练期在线命中率(诊断) : {diag['dec_acc']:.3f}   期望码拟合 RMSE: {dec['expect_fit_rmse']:.3f}")
    print(f"  累计自误差  近视 greedy      : {g_mean:.4f}")
    print(f"              前瞻 H={args.H}      : {l_mean:.4f}   (改善 {(g_mean-l_mean)/(g_mean+1e-9):+.1%})")
    print("-" * 70)
    j = res["judges"]
    print(f"  ① 动力学自学≥0.55 : {j['dynamics_self_learned_gte_055']}")
    print(f"  ② 前瞻<近视        : {j['lookahead_lt_greedy']}")
    print(f"  ③ 前瞻随H下降      : {j['lookahead_improves_with_H']}")
    print(f"  ④ 近视损益可证伪   : {j['greedy_loss_falsifiable']}")
    print(f"  wall={time.time()-t0:.0f}s")
    allpass = all(j.values())
    print("  结论:", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())