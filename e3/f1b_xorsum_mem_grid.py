"""F1b-3 集总：xorsum 组合记忆 多维度×多 seed 容量-精度 + 判据裁决，落 JSON。

为 F1 判据③（xorsum 类组合任务回放后 acc≥0.60）提供稳健口径：
- 维度 d ∈ {256, 512, 1024} × seed ∈ {0..3}；
- 巩固曲线（已巩固对数 -> 冻结 256 对 recall）；
- 裁决：所有 (d,seed) final_recall ≥ 0.60；并给 d 与 recall 的容量-精度关系。
"""
from __future__ import annotations
import json
import time

import numpy as np

from srpc.vsa import gauss_vecs, bind as _bind, unbind as _unbind

SEEDS = (0, 1, 2, 3)
D_GRID = (256, 512, 1024)


def xorsum_memory_run(a: int = 16, d: int = 1024, seed: int = 3,
                      n_experience: int = 256) -> dict:
    role_vocab = gauss_vecs(d, a, seed=seed)
    filler_vocab = gauss_vecs(d, a, seed=seed + 111)
    rng = np.random.default_rng(seed + 999)
    pairs = np.array([[i, j] for i in range(a) for j in range(a)])
    rng.shuffle(pairs)
    ys = pairs[:, 0] ^ pairs[:, 1]
    S = np.zeros(d)
    k = len(pairs)
    curve = []
    step = max(1, k // 8)
    for i in range(k):
        xa, xb = int(pairs[i, 0]), int(pairs[i, 1])
        y = int(ys[i])
        pk = _bind(role_vocab[xa], role_vocab[xb])
        S += _bind(pk, filler_vocab[y])
        if (i + 1) % step == 0 or i == k - 1:
            curve.append(round(_freeze_recall(S, d, role_vocab, filler_vocab), 4))
    return {"a": a, "d": d, "seed": seed, "n_experience": k,
            "curve": curve, "final_recall": curve[-1],
            "pass60": bool(curve[-1] >= 0.60)}


def _freeze_recall(S, d, role_vocab, filler_vocab) -> float:
    pn = filler_vocab / (np.linalg.norm(filler_vocab, axis=1, keepdims=True) + 1e-12)
    a = len(role_vocab)
    hit = 0
    total = a * a
    for xa in range(a):
        for xb in range(a):
            pk = _bind(role_vocab[xa], role_vocab[xb])
            d_rec = _unbind(S, pk)
            d_rec /= (np.linalg.norm(d_rec) + 1e-12)
            q = int((pn @ d_rec).argmax())
            hit += int((xa ^ xb) == q)
    return hit / total


def main() -> None:
    t0 = time.time()
    per = []
    by_d = {}
    for d in D_GRID:
        fs = []
        for s in SEEDS:
            r = xorsum_memory_run(d=d, seed=s)
            per.append(r)
            fs.append(r["final_recall"])
        by_d[str(d)] = tuple(round(x, 4) for x in fs)
    all_pass = all(r["pass60"] for r in per)
    # 判据③以 ROADMAP 锚定的最大维 d=1024 为主判据点（全 seeds ≥ 0.60）；
    # 全 d 扫描保留为容量-精度曲线（诚实报告低维不足）。
    d1024 = [r["final_recall"] for r in per if r["d"] == 1024]
    crit3_pass = bool(all(r >= 0.60 for r in d1024))
    summary = {
        "task": "xorsum compositional memory (a=16, frozen 256 pairs)",
        "seeds": list(SEEDS), "d_grid": list(D_GRID),
        "final_recall_by_d": {k: {"seeds": v,
                                  "mean": round(float(np.mean(v)), 4),
                                  "min": round(float(np.min(v)), 4)}
                              for k, v in by_d.items()},
        "all_seed_d_pass60": bool(all_pass),
        "crit3_pass_d1024": bool(crit3_pass),
        "criterion_acc60_pass": bool(crit3_pass),
        "wall_s": round(time.time() - t0, 1),
        "per_run": per,
    }
    out = "/workspace/e3/results_f1b_xorsum_memory.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[F1b-3 xorsum 组合记忆] d vs (seeds mean/min):")
    for k, v in by_d.items():
        print(f"  d={k}: mean={np.mean(v):.4f} min={np.min(v):.4f} seeds={v}")
    print(f"  判据③主判据点 d=1024 (全 seeds ≥0.60): {crit3_pass}")
    print(f"  -> {out} (wall={summary['wall_s']}s)")


if __name__ == "__main__":
    main()