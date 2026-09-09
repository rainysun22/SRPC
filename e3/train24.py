"""E3 放大档训练器（2M/每语种10万步）。

只训练、每段每档一定步数后打点保存 -> 4090 上 runs 由升级脚本跑、在 4090 上评估。
判据①②由 e3_eval24.py（评估脚本）计算，本训练器不 eval（避免 CPU/IO 与模型同机争用）。

用法：
  E3_H, E3_STEPS, E3_BP_FREE, E3_SEED, E3_CHK 环境变量。
  E3_BP_FREE=1 时 W1/W2/W3/W_out 全部 free-running（不自重学）。
"""
from __future__ import annotations
import json, os, sys, time
_cwd = os.path.dirname(os.path.abspath(__file__))
for _p in (_cwd, os.path.dirname(_cwd),
           "/root/srpc_e2/srpc_src", "/workspace"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from srpc.config import E2Config
from srpc.lmgpu import LMPCNg

sys.path.insert(0, _cwd)
from corpus import LANGS, build_train_segments, train_segment_windows

DEVICE = os.environ.get("E3_DEVICE", "cuda")
H = int(os.environ.get("E3_H", "1200"))
STEPS = int(os.environ.get("E3_STEPS", "100000"))
FREE = int(os.environ.get("E3_BP_FREE", "0"))
SEED = int(os.environ.get("E3_SEED", "0"))
OUT = os.environ.get("E3_OUT", os.path.join(_cwd, "ckpt_e3"))
os.makedirs(OUT, exist_ok=True)


def main() -> int:
    cfg = E2Config()
    # 启用 E2 验证过的 W2 谱截断稳化修复（失稳源于快速正反馈，必须每步钳制）
    cfg.w2_cap = True
    cfg.w2_cap_every = 1          # 每步都对顶奇异值钳制（开销可忽略，稳=赢）
    cfg.w2_smax_cap = 5.0         # 健康 σmax ~3-4.8，上限5.0完全留裕量不影响收敛
    rng = np.random.default_rng(SEED * 977 + 5)
    m = LMPCNg(cfg, H, rng, eta_w=0.01, iters=12, device=DEVICE)
    if FREE:
        # free-running：全部读出头冻结为常数（不自重学），权重也冻结（回看）
        m.learning = False
        m.W_out = m.W_out.clone().detach()
        m.b_out = m.b_out.clone().detach()
    segs = build_train_segments(seed=SEED, min_bytes=200000, max_words=90000)
    t0 = time.time()
    felt = {"h": H, "steps_per_lang": STEPS, "dev": DEVICE, "seed": SEED,
            "bp_free": bool(FREE),
            "w2_cap": cfg.w2_cap, "w2_cap_every": cfg.w2_cap_every,
            "w2_smax_cap": cfg.w2_smax_cap}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(felt, f)
    for gi, (lang, seg) in enumerate(segs):
        done = 0
        for x, y, _t in train_segment_windows(seg, cfg.context):
            if done >= STEPS:
                break
            m.train_step(x, y)
            done += 1
        tok = "free" if FREE else "learn"
        fn = os.path.join(OUT, f"lang_{lang}_{tok}.pt")
        torch_save = m.state_dict()
        torch_save["_lang_index"] = gi
        torch_save["_lang"] = lang
        torch_save["free"] = bool(FREE)
        import torch
        torch.save(torch_save, fn)
        print(f"  train {lang}: {done} steps [{tok}] -> {fn} "
              f"(wall {round(time.time()-t0,1)}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())