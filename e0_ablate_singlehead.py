#!/usr/bin/env python3
"""E0 补充消融：单头读出（遗忘压力最大化）× b1 零距离耦合下的记忆增益。

排除的反对意见："多条件头结构上免遗忘（只更新当前头），保留误差本来就不
依赖记忆 —— 记忆价值被读出头结构掩盖了。"

设计：head_idx 强制恒为 0（4 任务共用单头，顺序训练互相覆盖 = 遗忘压力
最大化）× ro_recon_mode="self"（读出源 = x_L = 记忆拉动的层，零传播距离）。
若记忆先验含增量信息，此设置下 mem 的保留误差应显著低于 no_mem。

判定：增益仍 <10% → 记忆先验对 ARC-lite（i.i.d. 输入→输出映射）无增量
信息；增益失败是信息性的（任务性质），不是架构性的（传播路径）→
记忆价值验证迁移 E3（语言流）/ F3（ICL）。
"""
import json
import sys
import time

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.runner_b import forget_stats, run_sequential

SEEDS = (0, 1, 2)


def hook_single_head(model):
    """所有条件共用读出头 0：顺序训练互相覆盖，遗忘压力最大化。"""
    model.head_idx = lambda: 0


def main() -> None:
    t0 = time.time()
    dcfg = DeepConfig(ro_recon_mode="self")   # b1：读出源 = x_L（记忆零距离）
    mcfg = MemoryConfig(d=dcfg.dims[-1])
    rows = {}
    for arm, with_mem in (("mem", True), ("no", False)):
        retains = []
        for s in SEEDS:
            r = run_sequential(s, with_mem, dcfg, mcfg, ArcConfig(),
                               CLConfig(), model_hook=hook_single_head)
            f = forget_stats(r["R"], r["diag"])
            retains.append(f["retain_mean"])
        rows[arm] = dict(retain=float(np.mean(retains)), per_seed=retains)
    rm, rn = rows["mem"]["retain"], rows["no"]["retain"]
    gain = (rn - rm) / (rn + 1e-8)
    out = dict(
        config="single-head (head_idx≡0) × b1 (ro_recon_mode='self'), 3 seeds",
        retain_mem=rm, retain_no=rn, gain=gain,
        per_seed_mem=rows["mem"]["per_seed"], per_seed_no=rows["no"]["per_seed"],
        verdict=("gain < 10% → 记忆先验无增量信息（信息性瓶颈，非架构性）"
                 if gain < 0.10 else "gain ≥ 10% → 记忆在遗忘压力下有价值，重审 E0 结论"),
    )
    with open("/workspace/results_phaseB/e0_ablate_singlehead.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
