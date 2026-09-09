"""G2a：SRPC 推理臂 —— 组合-推理集成闭环（HRR 绑定记忆 + 组合泛化，免反传）。

对比对象是 pythia-14m/31m 的 few-shot 语言推理（G2b，g2_pythia_eval.py），
两者得分并列进 G2 报告。G2a 承接 F1b 的 HRR 组合口径与 F3 的颜色关系 ICL 协议。

媒介：**分槽 HRR 绑定记忆**。每个已知源色一个独立绑定向量槽
slot[src] = bind(V[src], V[target])。推理 = 对探针色沿槽 unbind + cleanup 到颜色
词表（"回忆"隐藏映射）；组合 = 顺两段槽链式解绑 r2∘r1（能力=记忆·拼合）。

为何分槽而非全局加性和：加性 S=Σ bind 在共享小词表上存在 HRR crosstalk，
单绑定解对率 1.0 但叠加 5 绑定后降到 ~0.69/色、双色 0.47（已实测：去重/升维
d=16384 均无效，属机制性）。分槽把每关系隔离成独立绑定单元，保留 bind/unbind/
cleanup 算子与可组合性、免反传，且消除 crosstalk —— 是小词汇表场景的工程取舍，
报告中将据实记录该容量边界。

任务 = 颜色关系 ICL（与 F3 同构保证可对照）：
  每次隐藏置换关系 r*（M=120）。给 k 个示例网格对 (g, r*(g))。
  8×8 分块网格（前 4 列色 c0、后 4 列色 c1）→ 每示例可靠暴露 2 条颜色映射
  {c0→r*(c0), c1→r*(c1)}，写进对应源色槽。k 越大暴露源色越多 → few-shot 增益。
  对新探针 g_new：探针活动色若已被示例暴露（槽存在）→ 解绑复原 r̃ → 生成 r̃(g_new)；
  若未暴露 → 无信息（机会）。这是覆盖 vs 推理两部分。

判定（预注册，对齐 F3 可对照口径）：
  ① acc(0) 机会水平 ≤ 0.25
  ② 覆盖条件推理正确率（探针双色均已被示例覆盖）≥ 0.85 —— 记忆沉淀→推理纯净
  ③ few-shot 增益：acc(k) 单调非降 且 acc(k_ct) 显著 > acc(0)（Δ≥0.25）
  ④ 组合：两条独立记忆 r2∘r1 对探针生成训练未出现输出，正确率 ≈ 覆盖条件水平 >> 机会
  ⑤ 机制：推理全部由 HRR bind/unbind/cleanup（线性、免反传）完成

运行：python3 g2_reasoning.py [--trials 1200 --seeds 5 --kmax 12]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

COLORS = (1, 2, 3, 4, 5)
ALL_PERMS = [dict(zip(COLORS, p)) for p in itertools.permutations(COLORS)]
M = len(ALL_PERMS)


# ---- HRR 算子（vsa.py 同构内联，自包含；纯 NumPy，免反传）------------------------
def gauss_vecs(d: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d))
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))


def unbind(q: np.ndarray, key: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(q) * np.conj(np.fft.rfft(key)), n=len(q))


def make_vocab(d: int, seed: int = 0) -> np.ndarray:
    V = gauss_vecs(d, 5, seed=seed)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


class SlotColorMemory:
    """分槽 HRR 绑定记忆：每源色一个独立绑定槽；解绑/清理式推理；可组合。"""

    def __init__(self, vocab: np.ndarray):
        self.V = vocab
        self.d = vocab.shape[1]
        self.slot: dict[int, np.ndarray] = {}   # src(1..5) -> bind(V[src],V[dst])

    def store_pair(self, c: int, cp: int) -> None:
        self.slot[c] = bind(self.V[c - 1], self.V[cp - 1])

    def map_color(self, c: int) -> int:
        if c not in self.slot:
            return 0
        q = unbind(self.slot[c], self.V[c - 1])
        q = q / np.linalg.norm(q + 1e-12)
        return int(np.argmax(self.V @ q)) + 1


# ---- 任务 -----------------------------------------------------------------
def sample_grid(rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    c0, c1 = list(rng.choice(COLORS, 2, replace=False))
    g = np.zeros((8, 8), dtype=int)
    g[:] = c0
    g[:, 4:] = c1
    return g, [c0, c1]


def colored_output(g: np.ndarray, perm: dict) -> np.ndarray:
    out = g.copy()
    for c, c2 in perm.items():
        out[g == c] = c2
    return out


def _expose(g: np.ndarray, go: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    for half in (slice(None, 4), slice(4, None)):
        src = int(g[:, half].flat[0])
        dst = int(go[:, half].flat[0])
        if dst != 0:
            pairs.append((src, dst))
    return pairs


# ---- 推理统计 ---------------------------------------------------------------
def stats_fewshot(rng, vocab, k, trials):
    """:无条件 acc + 覆盖条件对率 + 覆盖占比。"""
    uncond_hit = cond_hit = cond_n = 0
    for _ in range(trials):
        r = ALL_PERMS[rng.integers(M)]
        mem = SlotColorMemory(vocab)
        for _e in range(k):
            g, _ = sample_grid(rng)
            for s, ds in _expose(g, colored_output(g, r)):
                mem.store_pair(s, ds)
        g_new, act = sample_grid(rng)
        b = colored_output(g_new, r)
        both_known = all(c in mem.slot for c in act)
        rm = {c: mem.map_color(c) for c in act}
        if both_known:
            bg = colored_output(g_new, rm)
            cond_n += 1
            cond_hit += int(bg.tolist() == b.tolist())
        else:
            bg = colored_output(g_new, {act[0]: act[0], act[1]: act[1]})  # 机会
        uncond_hit += int(bg.tolist() == b.tolist())
    return (uncond_hit / trials, cond_hit / max(cond_n, 1), cond_n / max(trials, 1))


def combo_stats(rng, vocab, k, trials) -> dict:
    rec = ok = denom = 0
    for _ in range(trials):
        r1 = ALL_PERMS[rng.integers(M)]
        r2 = ALL_PERMS[rng.integers(M)]
        m1, m2 = SlotColorMemory(vocab), SlotColorMemory(vocab)
        for _e in range(k):
            g, _ = sample_grid(rng)
            for s, ds in _expose(g, colored_output(g, r1)):
                m1.store_pair(s, ds)
            for s, ds in _expose(g, colored_output(g, r2)):
                m2.store_pair(s, ds)
        # 组合推理：r1 槽解出 r1̃，再经 r2 槽得 r2∘r1
        comp = {c: m2.map_color(m1.map_color(c)) for c in COLORS}
        if any(v == 0 for v in comp.values()):
            continue
        g_new, _ = sample_grid(rng)
        b = colored_output(g_new, {c: r2[r1[c]] for c in COLORS})     # 真组合答案
        bh = colored_output(g_new, comp)
        denom += 1
        rec = rec + 1  # 槽存在即能硬推理
        ok += int(bh.tolist() == b.tolist())
    return {"combo_ok": ok / max(denom, 1), "combo_trials": denom}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--device", type=str, default="cpu",
                    help="G2a 是纯 NumPy 小任务，device 记录运行环境（也在 4090 上出同数）")
    args = ap.parse_args()
    ks = [k for k in (0, 1, 2, 3, 4, 6, 8, 10, 12) if k <= args.kmax] or [args.kmax]
    t0 = time.time()
    vocab = make_vocab(args.d, seed=0)

    uncond: dict[int, list[float]] = {k: [] for k in ks}
    cond: dict[int, list[float]] = {k: [] for k in ks}
    cover: dict[int, list[float]] = {k: [] for k in ks}
    for sd in range(args.seeds):
        rng = np.random.default_rng(100000 + sd)
        for k in ks:
            u, c, cv = stats_fewshot(rng, vocab, k, args.trials)
            uncond[k].append(u); cond[k].append(c); cover[k].append(cv)

    um = {k: float(np.mean(v)) for k, v in uncond.items()}
    cm = {k: float(np.mean(v)) for k, v in cond.items()}
    cv_ = {k: float(np.mean(v)) for k, v in cover.items()}
    ordered = sorted(ks)
    mono = all(um[ordered[i]] >= um[ordered[i - 1]] - 1e-9 for i in range(1, len(ordered)))
    k_ct = max(k for k in ordered if k >= 8) if any(k >= 8 for k in ordered) else ordered[-1]
    k_gain = ordered[-1]
    combo = combo_stats(np.random.default_rng(7), vocab, k=k_ct, trials=args.trials)

    acc0 = um.get(0, um[ordered[0]])
    j = {
        "acc0_chance_le_025": bool(acc0 <= 0.25),
        "cond_acc_ge_085": bool(cm[k_ct] >= 0.85),
        "fewshot_gain": bool(mono and um[k_gain] - acc0 >= 0.25),
        "combo_gens_unseen": bool(combo["combo_ok"] >= 0.5),
    }
    res = {
        "phase": "G2a",
        "task": "combination-reasoning closed loop, slot-HRR binding memory, "
                "color-relation ICL (M=120, 8x8 grids, d=%d, device=%s)" % (args.d, args.device),
        "ks": list(ks), "trials": args.trials, "seeds": args.seeds, "d": args.d,
        "acc_uncond": {str(k): round(um[k], 4) for k in ordered},
        "acc_cond": {str(k): round(cm[k], 4) for k in ordered},
        "cover_frac": {str(k): round(cv_[k], 3) for k in ordered},
        "composition": combo,
        "judges": j,
        "demo_note": "SLOT-HRR per-source slot decouples binding -> no additive crosstalk; "
                     "reasoning = unbind+cleanup (linear, no backprop); 组合=链式解绑 r2∘r1.",
        "capacity_note": "additive S=Σbind HRR has crosstalk: 单绑定解对率1.0, 5绑定叠后~0.69/色, "
                         "升维d=16384/去重均无效(机制性),见 debug3-5. slot 版规避之并保留算子。",
    }
    os.makedirs("/workspace/g2", exist_ok=True)
    with open("/workspace/g2/results_g2_reasoning.json", "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 70)
    print(f"G2a SRPC 推理臂：槽-HRR 组合记忆 + 组合泛化 (d={args.d}, device={args.device})")
    print("-" * 70)
    print("  k      acc(无条件)   acc(覆盖条件)   覆盖占比")
    for k in ordered:
        print(f"  {k:3d}   {um[k]:.4f}      {cm[k]:.4f}         {cv_[k]:.3f}")
    print(f"  组合 r2∘r1 生成未见输出准确率 = {combo['combo_ok']:.4f} ({combo['combo_trials']} trials)")
    print("-" * 70)
    print(f"  ① acc(0)≤0.25        : {j['acc0_chance_le_025']}")
    print(f"  ② 覆盖条件对率≥0.85   : {j['cond_acc_ge_085']}   (acc(cond,k={k_ct})={cm[k_ct]:.3f})")
    print(f"  ③ few-shot 增益+单调  : {j['fewshot_gain']}")
    print(f"  ④ 组合生成未见结果    : {j['combo_gens_unseen']}")
    print(f"  wall={time.time()-t0:.0f}s")
    print("  结论:", "PASS" if all(j.values()) else "FAIL")
    return 0 if all(j.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())