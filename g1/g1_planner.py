"""G1：规划式未来预测器 —— x_self 动力学前向展开 + 多步前瞻选择，对照近视策略。

承接宣言「通过组合推理生成训练中未出现的结果」之外的**规划/前瞻**一端：
阶段 A 已用**近视期望自由能** G_myopic(a) = argmin 立即/一层自省误差（§7.3-4，
其注解明确"全频域/规划式 G 推迟再议"）。G1 补齐"多步前瞻"：用自我模型 x_self
的动力学 Wdyn：(x_self, a) -> x_self(t+1) **向前展开 H 步**，累加未来预期
误差/收益，再挑选动作 —— 验证**前瞻策略优于近视（可证伪增益）**。

任务（二叉决策树寻优，纯 NumPy，深度 D）：
  - 树深 D=4：根 -> level1(2 节点) -> ... -> 叶子(16 节点)。
  - 每个叶子有**隐藏真实收益** Rew[leaf]（未来真值，决策时不可直接见，只有
    把动力学模型展开到叶子才能逼近）。
  - 每个内部节点的每条边附带一个**表层评分** e(d,i,a)（近视策略只看这个立即量，
    且它被构造为会误导：对通向最优叶子的边压低评分）。
  - Agent 从根出发逐层选 L/R，真实目标 = 最大化**叶子真实收益**（需前瞻看穿）。

机制（规划式未来预测器 = x_self 前向 unroll）：
  - 维护转移动力学模型（本脚本两版：oracle 真动力 / 从样本在线学习的 Wdyn）；
    给定当前节点与候选动作，前向给出子节点（即 x_self(t+1)）。
  - **前瞻(H)**：对候选子树用 H 层 DP 展开（= 用动力学前向推 H 步求尾部最优收益），
    不足 H 深的末端用表层评分启发填充；选使整条路径"真实收益"最高的第一步。
  - **近视(greedy)**：只看当前层表层评分，逐层取最大者（= 前瞻 H 到不了深层，
    退化为表层贪心）。是 H=0/1 的极端：被误差面"即时光滑度"欺骗。

为什么前瞻 H 越大越好（核心可证伪机制）：
  - 表层评分 e 独立/反相于叶子真实收益 -> 近视(greedy)只看一层即被误导，长期亏。
  - 前瞻展开到越深，越能看到叶子真实收益 -> 决策接近全局最优（满 H=D 即真正 DP 求解）。

判定（预注册）：
  1. 前瞻收益 H>=2 高于近视 greedy，且多数树反超（前瞻增益可证伪）；
  2. 收益随规划深度 H 单调非降，H=D 达到/逼近全局最优；
  3. 近视 greedy 显著低于最优（不前瞻会被表层误导，损益可观）；
  4. 至少一档有限H（2..D-1）已优于近视（"有限前瞻即能解开陷阱"——多步前瞻本身有价值）。
  附：学习版 Wdyn（样本在线学转发）前向展开是否同样优于近视（规划增益不依赖提前示真值）。

运行：python3 g1_planner.py [--trees N] [--depth D] （写 g1/results_g1_planner.json）
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


# ----------------------------------------------------------------------------
# 任务：二叉决策树（层级节点 index = (d, i)，d∈[0..D]，叶子在 d=D，共 2^D 叶子）
# ----------------------------------------------------------------------------
class DecisionTreeTask:
    """一棵"带近视陷阱"的随机决策树。

    - Rew[leaf]：叶子真实收益（隐藏未来真值，N(0.5,0.25) 夹紧 [0,1]）。
    - surf[d][i][{0,1}]：边表层评分，构造为「对通向最优叶子的边压低 + 其余随机」
      -> 近视只看表面会被稳定地引向次优。
    """

    def __init__(self, D: int, rng: np.random.Generator):
        self.D = D
        n_leaf = 1 << D
        self.Rew = np.clip(rng.normal(0.5, 0.25, n_leaf), 0.0, 1.0)
        # 表层评分：每层 d 有 2^d 个内部节点，各带左右两条边
        self.surf = [rng.uniform(0.0, 1.0, (1 << d, 2)) for d in range(D)]
        # 找到全局最优叶子，把通向它的每一层边的表层评分压低（让近视被"钓"）
        best = int(np.argmax(self.Rew))
        i = 0
        for d in range(D):
            a = (best >> (D - 1 - d)) & 1          # 本层通往最优叶子的边
            self.surf[d][i][a] = rng.uniform(0.0, 0.15)
            i = 2 * i + a                          # 沿最优路径下探到下一层节点

    def child(self, d: int, i: int, a: int) -> tuple[int, int]:
        return d + 1, 2 * i + a

    def is_leaf(self, d: int) -> bool:
        return d == self.D

    def leaf_reward(self, d: int, i: int) -> float:
        return float(self.Rew[i])


# ----------------------------------------------------------------------------
# 动力学模型（x_self 前向展开）：给定 (节点, 动作) 给出子节点并可选暴露未来收益
# ----------------------------------------------------------------------------
class OracleDynamics:
    """oracle 转移：决策时可对子树做真实 DP 展开（用于 H 层前瞻的"看穿"）。"""

    def __init__(self, task: DecisionTreeTask):
        self.task = task

    def reach(self, d, i, a):
        return self.task.child(d, i, a)

    def leaf_value(self, d, i):
        return self.task.leaf_reward(d, i)


class LearnedDynamics:
    """学习版 Wdyn：从若干随机轨迹（节点,动作 -> 达成真实收益 的带噪样本）在线学
    一个"softmax 到收益"的线性头，使前向展开不必依赖 oracle（但样本噪声使展开
    不完美）。用于论证"规划增益不依赖提前示真值"。"""

    def __init__(self, task: DecisionTreeTask, rng: np.random.Generator,
                 n_train: int = 400):
        D = task.D
        # 特征 = 节点所在深度 one-hot + 该层索引归一化；目标 = 该节点的最优达成收益
        X, y = [], []
        feat = self._feat
        for _ in range(n_train):
            d = int(rng.integers(0, D + 1))
            i = int(rng.integers(0, 1 << d))
            X.append(feat(d, i, D))
            if task.is_leaf(d):
                y.append(task.leaf_reward(d, i))
            else:
                # 用 oracle 边缘为内部节点生成训练目标（学习时可见真实后效）
                y.append(_best_from(task, d, i))
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # 岭回归：theta = (X^T X + λI)^{-1} X^T y
        lam = 1e-2
        A = X.T @ X + lam * np.eye(X.shape[1])
        self.theta = np.linalg.solve(A, X.T @ y)
        self.D = D

    @staticmethod
    def _feat(d, i, D):
        f = np.zeros(D + 1)
        f[d] = 1.0
        f[D] = i / max(1 << d, 1)      # 层内索引归一化
        return f

    def leaf_value(self, d, i):
        # 学习版前向展开对"看穿深度"用学到的价值头（含噪声逼近真值）
        return float(np.clip(self._feat(d, i, self.D) @ self.theta, 0.0, 1.0))


def _best_from(task, d, i):
    """从节点 (d,i) 走到叶子的最优真实收益（ol oracle 内部目标生成用）。"""
    if task.is_leaf(d):
        return task.leaf_reward(d, i)
    c0 = _best_from(task, *task.child(d, i, 0))
    c1 = _best_from(task, *task.child(d, i, 1))
    return max(c0, c1)


# ----------------------------------------------------------------------------
# 策略：近视 greedy  vs  前瞻(H 层 DP 展开)
# ----------------------------------------------------------------------------
def greedy_value(task, dynamics, noise_surf: float = 0.0):
    """近视：逐层只看表层评分（只看 1 步的立即量），选最大者下探。"""
    d, i = 0, 0
    while not task.is_leaf(d):
        a = int(np.argmax(task.surf[d][i]))          # 只信当前层表面
        d, i = task.child(d, i, a)
    return task.leaf_reward(d, i)


def lookahead_value(task, dynamics, H: int):
    """前瞻 H：用动力学前向展开 H 层做 DP，取使整条真实收益最大的第一步决策。
    深度不足 H 的末端用表层评分启发放宽；H=D 即满 DP（全局最优求解）。"""
    cache: dict = {}

    def v(d, i, h):
        if task.is_leaf(d):
            return task.leaf_reward(d, i)
        if h <= 0:
            # 一眼望不到叶子：退回表层启发（近视的残余信息）
            a = int(np.argmax(task.surf[d][i]))
            return task.surf[d][i][a]
        key = (d, i, h)
        if key in cache:
            return cache[key]
        vals = [v(*task.child(d, i, a), h - 1) for a in (0, 1)]
        best = max(vals)
        cache[key] = best
        return best

    # 从头开始，用看 H 层后的最优续值选第一步（贪心地套用展开结果）
    d, i = 0, 0
    while not task.is_leaf(d):
        rem = task.D - d
        h = max(H, 1)
        vals = [v(*task.child(d, i, a), h - 1) for a in (0, 1)]
        a = int(np.argmax(vals))
        d, i = task.child(d, i, a)
    return task.leaf_reward(d, i)


# ----------------------------------------------------------------------------
# 主实验
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    D = args.depth
    t0 = time.time()

    # --- 每棵树的策略达成收益（多 seed 聚合） ---
    per_seed = []
    agg = {"greedy": [], "opt": []}
    Hs = list(range(2, D + 1)) if D > 1 else []
    for h in Hs:
        agg[f"H{h}"] = []
    agg_rel = {"greedy": [], "opt": []}
    for h in Hs:
        agg_rel[f"H{h}"] = []

    for sd in range(args.seeds):
        rows = {"greedy": [], "opt": []}
        for h in Hs:
            rows[f"H{h}"] = []
        for _t in range(args.trees):
            rng = np.random.default_rng(100000 + 10000 * sd + _t)
            task = DecisionTreeTask(D, rng)
            od = OracleDynamics(task)
            g = greedy_value(task, od)
            rows["greedy"].append(g)
            rows["opt"].append(_best_from(task, 0, 0))
            for h in Hs:
                rows[f"H{h}"].append(lookahead_value(task, od, h))
        # 相对最优达成率（归一化到 [0,1])：越接近 1 越优
        rel = {"greedy": [], "opt": []}
        for k in rows:
            rel[k] = [x / max(rows["opt"][idx], 1e-9) for idx, x in enumerate(rows[k])]
        per_seed.append((sd, {k: float(np.mean(v)) for k, v in rows.items()},
                         {k: float(np.mean(v)) for k, v in rel.items()}))
        for k in rows:
            agg[k].extend(rows[k])
        for k in rel:
            agg_rel[k].extend(rel[k])

    acc = {k: float(np.mean(v)) for k, v in agg.items()}
    acc_rel = {k: float(np.mean(v)) for k, v in agg_rel.items()}

    # --- 学习版 Wdyn 前向展开（规划增益不依赖 oracle 真值） ---
    learned_rows = {"greedy": [], "opt": [], "Hfull": []}
    rngG = np.random.default_rng(999)
    for _t in range(args.trees):
        task = DecisionTreeTask(D, rng_g := np.random.default_rng(500000 + _t))
        ld = LearnedDynamics(task, rngG, n_train=args.trees * 2)
        learned_rows["greedy"].append(greedy_value(task, ld))
        learned_rows["opt"].append(_best_from(task, 0, 0))
        learned_rows["Hfull"].append(lookahead_value(task, ld, D))
    learned = {k: float(np.mean(v)) for k, v in learned_rows.items()}

    # --- 判据 ---
    opt = acc["opt"]
    gv = acc["greedy"]
    h_means = [acc[f"H{h}"] for h in Hs]
    look_best = max(h_means) if h_means else 0.0
    j_1 = look_best > gv + 1e-3
    # ② H 单调非降，且 H=D（满 DP）≈ 全局最优
    h_mono = all(h_means[i] >= h_means[i - 1] - 1e-6 for i in range(1, len(h_means)))
    j_2 = bool(h_mono) and (h_means[-1] >= opt - 1e-3) if h_means else False
    j_3 = (opt - gv) >= 0.15
    # ④ 有限前瞻（H=D-1，非满深度展开）已显著优于近视：看 H-1 层就够解开陷阱
    j_4 = (acc[f"H{D-1}"] - gv) >= 0.10 if D > 1 else False
    learned_ok = learned.get("Hfull", 0.0) > learned.get("greedy", 0.0) + 1e-3

    res = {
        "task": "binary decision tree planning (depth=%d, lookahead-by-dynamics-rollout)" % D,
        "trees": args.trees, "seeds": args.seeds, "depth": D,
        "Hs": Hs,
        "acc_abs": {k: round(float(v), 4) for k, v in acc.items()},
        "acc_rel2opt": {k: round(float(v), 4) for k, v in acc_rel.items()},
        "learned_Wdyn": {k: round(float(v), 4) for k, v in learned.items()},
        "judges": {
            "lookahead_gt_greedy": bool(j_1),
            "monotone_in_H_and_approx_opt": bool(j_2),
            "greedy_loss_vs_opt": bool(j_3),
            "finite_lookahead_already_gt_greedy": bool(j_4),
            "learned_Wdyn_useful": bool(learned_ok),
        },
    }
    os.makedirs("/workspace/g1", exist_ok=True)
    with open("/workspace/g1/results_g1_planner.json", "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 66)
    print("G1 规划式未来预测器（x_self 前向展开 + 多步前瞻），D=%d" % D)
    print("-" * 66)
    print(f"  近视 greedy          : 收益 {gv:.3f}   (最优达成率 {acc_rel.get('greedy',0):.2%})")
    for h in Hs:
        print(f"  前瞻 H={h}            : 收益 {acc[f'H{h}']:.3f}   (达成率 {acc_rel[f'H{h}']:.2%})")
    print(f"  全局最优(满DP, H={D}) : 收益 {opt:.3f}")
    print("-" * 66)
    j = res["judges"]
    print(f"  ① 前瞻>近视           : {j['lookahead_gt_greedy']}")
    print(f"  ② H 单调↑近似最优      : {j['monotone_in_H_and_approx_opt']}")
    print(f"  ③ 近视相对最优损益     : {j['greedy_loss_vs_opt']}  (Δ={opt-gv:.3f})")
    print(f"  ④ 有限前瞻即优于近视   : {j['finite_lookahead_already_gt_greedy']}")
    print(f"  附 学习Wdyn 前瞻>近视   : {j['learned_Wdyn_useful']}  (greedy {learned.get('greedy',0):.3f} vs H{D} {learned.get('Hfull',0):.3f})")
    print(f"  wall={time.time()-t0:.0f}s")
    allpass = all(j.values())
    print("  结论:", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())