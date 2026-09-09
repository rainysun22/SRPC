"""E3 记忆闭环接入验证（2M final 模型）：nomem vs oracle-recall vs 自识别闭环 recall。

目标：把 E3 语言版慢记忆的【读出先验注入】从"依赖真实语种标签的 oracle recall"
落地为【闭环接入】——每窗口先用记忆固化的语种原型自识别当前语种（selfid），再注入
对应原型先验，全程不取标签。回答 README_SMOKE §4 的问题："记忆能否自己找到正确原型"。

对 final checkpoint（2M，cjk 段 train 完）冻结后同一模型、同一 tau：
  arm-A nomem        = 原读出头，无记忆（beta=0）
  arm-B oracle mem   = oracle recall（真实语种原型，β 扫）
  arm-C selfid mem   = 闭环：窗口字节直方图自识别语种 → recall（β 扫）
判读：
  ① 自识别闭环 acc（overall/switch/within）——能自识别多少窗口语种；
  ② C vs A 记忆增益（total/switch）——闭环不取标签下记忆仍是否有增量；
  ③ C 相对 B 的增益保留率 = gain_C/gain_B——闭环相对 oracle 打多少折；
  ④ 判据②在闭环口径下是否仍 ≥10%。

运行在 4090（CPU 串行逐样本 same 口径，~40-90s）。
"""
from __future__ import annotations
import json, os, sys, time
_cwd = os.path.dirname(os.path.abspath(__file__))
for _p in (_cwd, os.path.dirname(_cwd),
           "/root/srpc_e2/srpc_src", "/workspace"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from srpc.config import E2Config
from srpc.lmgpu import LMPCNg

sys.path.insert(0, _cwd)
from corpus import LANGS, build_train_segments, build_eval_stream
from memory_lm import LangSlowMem, eval_scores, selfid_acc, _agg

DEVICE = os.environ.get("E3_DEVICE", "cuda")
H = int(os.environ.get("E3_H", "1200"))
SEED = int(os.environ.get("E3_SEED", "0"))
CKPT_DIR = os.environ.get("E3_CKPT", os.path.join(_cwd, "ckpt_e3_2m_fix"))
BETA_GRID = (0.3, 0.5, 0.6, 0.8)
OUT_JSON = os.environ.get("E3_SELF_OUT", os.path.join(_cwd, "results_e3_2m_self.json"))
GAIN_TH = 0.10


def _load_final():
    cfg = E2Config()
    m = LMPCNg(cfg, H, np.random.default_rng(SEED * 977 + 5),
               eta_w=0.01, iters=12, device=DEVICE)
    sd = torch.load(os.path.join(CKPT_DIR, f"lang_{LANGS[-1]}_learn.pt"),
                    map_location=DEVICE)
    m.load_state({k: v for k, v in sd.items() if isinstance(v, torch.Tensor)})
    m.learning = False
    return m, cfg


def _mem_from_train(seed: int) -> LangSlowMem:
    mem = LangSlowMem(n_groups=3, C=256, device=DEVICE)
    for gi, (_, seg) in enumerate(build_train_segments(
            seed=seed, min_bytes=200000, max_words=90000)):
        mem.consolidate_segment(seg, gi)
    return mem


def main() -> int:
    _T0 = time.time()
    mf, cfg = _load_final()
    ev = build_eval_stream(seed=SEED, reps=3)
    X, y, groups, sw = ev["X"], ev["y"], ev["groups"], ev["switch"]
    sw_mask, within_mask = sw, ~sw
    mem = _mem_from_train(SEED)
    tau = mf.fit_tau(X[: min(120, len(y))], y[: min(120, len(y))], cfg.tau_grid)
    print(f"== E3 闭环接入: h={H} tau={tau:.3f} samples={len(y)} "
          f"switch={int(sw.sum())}", flush=True)

    # ---- 自识别闭环诊断：能自识别多少窗口的语种（overall/switch/within）----
    hid = selfid_acc(mem, X, groups)
    selfid = {
        "overall": float(hid.mean()),
        "switch": float(hid[sw_mask].mean()),
        "within": float(hid[within_mask].mean()),
    }
    print(f"  selfid acc: overall={selfid['overall']:.3f} "
          f"switch={selfid['switch']:.3f} within={selfid['within']:.3f}")

    # ---- arm-A 无记忆 ----
    nll0, acc0 = eval_scores(mf, X, y, tau, mem=None, beta=0.0)
    nomem = {"total": dict(zip(("bpc", "acc"), _agg(nll0, acc0))),
             "switch": dict(zip(("bpc", "acc"), _agg(nll0, acc0, sw_mask))),
             "within": dict(zip(("bpc", "acc"), _agg(nll0, acc0, within_mask)))}

    # ---- arm-B oracle / arm-C selfid（各自 β 扫）----
    def _arms(selfid_mode: bool) -> dict:
        out = {}
        for beta in BETA_GRID:
            nll, acc = eval_scores(mf, X, y, tau, mem, groups, beta,
                                   selfid=selfid_mode)
            out[str(beta)] = {"total": dict(zip(("bpc", "acc"), _agg(nll, acc))),
                              "switch": dict(zip(("bpc", "acc"),
                                                 _agg(nll, acc, sw_mask))),
                              "within": dict(zip(("bpc", "acc"),
                                                 _agg(nll, acc, within_mask)))}
        return out

    oracle = _arms(False)
    cloop = _arms(True)

    def _gain(arm):
        return {b: {"total": (nomem["total"]["bpc"] - arm[b]["total"]["bpc"])
                    / nomem["total"]["bpc"],
                    "switch": (nomem["switch"]["bpc"] - arm[b]["switch"]["bpc"])
                    / nomem["switch"]["bpc"]} for b in oracle}

    gain_o, gain_c = _gain(oracle), _gain(cloop)
    bo = max(gain_o, key=lambda b: max(gain_o[b].values()))
    bc = max(gain_c, key=lambda b: max(gain_c[b].values()))
    go = max(gain_o[bo].values())
    gc = max(gain_c[bc].values())
    retention = gc / (go + 1e-12) if go > 0 else float("nan")
    pass2_self = gc >= GAIN_TH

    def _fmt(name, arm, g):
        print(f"  {name:<7} {arm['total']['bpc']:6.3f} "
              f"{arm['switch']['bpc']:6.3f} {arm['within']['bpc']:6.3f} "
              f"| total_gain={g['total']:+.3f} switch_gain={g['switch']:+.3f}")

    print("\n  arm            total    switch   within  | gains")
    _fmt("nomem", nomem, {"total": 0.0, "switch": 0.0})
    _fmt(f"oracle@{bo}", oracle[bo], gain_o[bo])
    _fmt(f"selfid@{bc}", cloop[bc], gain_c[bc])

    res = {
        "host": {"h": H, "tau": float(tau), "ckpt": CKPT_DIR,
                 "beta_grid": list(BETA_GRID)},
        "selfid_acc": selfid,
        "nomem": {k: dict(arm) for k, arm in nomem.items()},
        "oracle_mem": oracle, "selfid_mem": cloop,
        "gains": {"oracle": gain_o[bo], "selfid": gain_c[bc],
                  "best_beta_oracle": bo, "best_beta_selfid": bc},
        "closed_loop": {"gain_oracle": float(go), "gain_selfid": float(gc),
                        "retention_of_oracle": float(retention),
                        "pass_crit2_selfid": bool(pass2_self),
                        "threshold": GAIN_TH},
        "verdict": selfid,
        "wall_s": round(time.time() - _T0, 2),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n=== 闭环判读 ===")
    print(f"  selfid acc: overall={selfid['overall']:.3f} "
          f"switch={selfid['switch']:.3f} within={selfid['within']:.3f}")
    print(f"  记忆增益: oracle(total/s,w)={go:.3f} selfid={gc:.3f} "
          f"retention={retention:.2f}")
    print(f"  判据②(闭环口径 selfid) gain={gc:.3f} ≥{GAIN_TH} "
          f"{'PASS' if pass2_self else 'FAIL'}")
    print(f"[json] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())