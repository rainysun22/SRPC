"""E3 冒烟机制验证（downscaled，h=768，cuda）：语种记忆先验是否降 BPC。

协议：
  1. corpus.build_train_segments → 顺序多语字节流（lang0→lang1→lang2）。
  2. 顺序训练 LMPCNg(h=768)：每语种训练其段（3k 步级），并把目标字节巩固进
     LangSlowMem 该语种原型（慢记忆 consolidation）。
  3. 冻结权重，评估两臂（同一冻结模型、同一 tau）：
       (a) 无记忆 = eval_scores(beta=0)（== 原 eval_batch 口径）
       (b) 有记忆 = eval_scores(coracle recall 语种原型, beta 扫描)
     在含语种切换边界的数据上，按 total / switch_tail / within 三个子集报 BPC/acc。
  4. 自识别诊断：用单 16 字节窗口直方图判语种（窗口可分性），对比整体 vs 边界；
     并报语种原型两两余弦距离（全局统计可分性）。
  5. 判据：with-memory 的 total_BPC 或 switch_tail_BPC 显著 < 无记忆 → 机制成立。

结果写 /workspace/e3/results_smoke.json（远程取回落盘）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# ---- sys.path 自举：兼容远程(4090:/root/srpc_e2/srpc_src)与本地(/workspace) ----
_cwd = os.path.dirname(os.path.abspath(__file__))
for _p in (_cwd, os.path.dirname(_cwd), "/root/srpc_e2/srpc_src", "/workspace"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from srpc.config import E2Config
    from srpc.lmgpu import LMPCNg
except Exception as _e:  # pragma: no cover
    raise SystemExit(f"cannot import srpc: {_e}")

sys.path.insert(0, _cwd)
from corpus import LANGS, build_train_segments, build_eval_stream, \
    train_segment_windows, lang_byte_hist
from memory_lm import LangSlowMem, eval_scores, _agg

DEVICE = os.environ.get("E3_DEVICE", "cuda")
H = int(os.environ.get("E3_H", "768"))
STEPS_PER_LANG = int(os.environ.get("E3_STEPS", "3200"))
BETA_GRID = (0.0, 0.3, 0.6, 1.0)
SEED = int(os.environ.get("E3_SEED", "0"))
OUT_JSON = os.environ.get("E3_OUT", os.path.join(_cwd, "results_smoke.json"))


def _switch_mask(grid):
    return grid["switch"]


def train_and_consolidate(cfg, mem):
    """顺序训练 LMPCNg(h)：lang0→lang1→lang2，每步巩固记忆原型。返回模型/步数。"""
    rng = np.random.default_rng(SEED * 977 + 5)
    m = LMPCNg(cfg, H, rng, eta_w=0.01, iters=12, device=DEVICE)
    segs = build_train_segments(seed=SEED, min_bytes=3300, max_words=2200)
    t0 = time.time()
    per_lang_steps = {}
    for gi, (lang, seg) in enumerate(segs):
        cnt = 0
        for x, y, t in train_segment_windows(seg, cfg.context):
            if cnt >= STEPS_PER_LANG:
                break
            m.train_step(x, y)
            mem.consolidate(y, gi)        # 慢记忆巩固：把目标字节记入当前语种原型
            cnt += 1
        per_lang_steps[lang] = cnt
        print(f"  train {lang}: {cnt} steps "
              f"(wall {round(time.time()-t0,1)}s)", flush=True)
    return m, segs, per_lang_steps


def _window_hist_and_selfid(X, mem):
    """用单窗口字节直方图判语种（窗口内可分性），对照原型（全局统计）。"""
    hist = X.sum(axis=1).astype(np.float64)          # (n,256) 窗口出现字节次数
    protos = np.stack([mem.counts[g] for g in range(3)]).astype(np.float64)
    protos /= protos.sum(1, keepdims=True) + 1e-12
    return hist, protos


def _classify_by_hist(hist, protos):
    """按余弦距离把窗口直方图归到最近语种原型。返回 (n,) 预测语种。"""
    hn = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-12)
    pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    d = 1.0 - hn @ pn.T                       # (n,3) 余弦距离
    return d.argmin(1)


def _proto_separation(protos):
    """语种原型两两余弦距离（全局统计可分性证据）。"""
    pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    return 1.0 - pn @ pn.T                    # (3,3) 余弦距离(=1-余弦相似)


def main() -> int:
    cfg = E2Config()
    mem = LangSlowMem(n_groups=3, C=256, device=DEVICE)
    _T0 = time.time()

    # ---- 1/2. 顺序训练 + 记忆巩固 ----
    print(f"== E3 smoke: h={H} device={DEVICE} steps/lang={STEPS_PER_LANG} ==",
          flush=True)
    m, segs, per_lang_steps = train_and_consolidate(cfg, mem)

    # ---- 3. 评估语料（含边界、多段多边界）----
    ev = build_eval_stream(seed=SEED, reps=3)
    X, y, groups, sw = ev["X"], ev["y"], ev["groups"], ev["switch"]
    sw_mask = sw
    within_mask = ~sw

    # ---- 冻结：tau 在无记忆校准子集上定 ----
    cal_n = min(120, len(y))
    tau = m.fit_tau(X[:cal_n], y[:cal_n], cfg.tau_grid)
    print(f"  calibrated tau = {tau:.3f}  eval samples = {len(y)} "
          f"(switch={int(sw.sum())})", flush=True)

    # ---- 两臂判分 ----
    res = {}
    # (a) 无记忆
    nll0, acc0 = eval_scores(m, X, y, tau, mem=None, beta=0.0)
    res["nomem"] = {
        "total": dict(zip(("bpc", "acc"), _agg(nll0, acc0))),
        "switch": dict(zip(("bpc", "acc"), _agg(nll0, acc0, sw_mask))),
        "within": dict(zip(("bpc", "acc"), _agg(nll0, acc0, within_mask))),
    }
    # (b) 有记忆（oracle recall 语种原型，beta 扫描）
    mem_arms = {}
    for beta in BETA_GRID:
        if beta == 0.0:
            continue
        nll, acc = eval_scores(m, X, y, tau, mem, groups, beta)
        mem_arms[str(beta)] = {
            "total": dict(zip(("bpc", "acc"), _agg(nll, acc))),
            "switch": dict(zip(("bpc", "acc"), _agg(nll, acc, sw_mask))),
            "within": dict(zip(("bpc", "acc"), _agg(nll, acc, within_mask))),
        }
    res["mem"] = mem_arms

    # ---- 4. 诊断：窗口可分性 vs 自识别 vs 全局原型分离 ----
    hist, protos = _window_hist_and_selfid(X, mem)
    pred_all = _classify_by_hist(hist, protos)
    acc_self_all = float((pred_all == groups).mean())
    acc_self_sw = float((pred_all[sw_mask] == groups[sw_mask]).mean())
    acc_self_within = float((pred_all[within_mask] == groups[within_mask]).mean())
    sep = _proto_separation(protos).tolist()
    res["diagnostics"] = {
        "selfid_from_16B_window_acc_overall": acc_self_all,
        "selfid_from_16B_window_acc_switch": acc_self_sw,
        "selfid_from_16B_window_acc_within": acc_self_within,
        "lang_proto_cos_dist_3x3": sep,
        "n_train_segments": len(segs),
        "train_steps_per_lang": per_lang_steps,
        "eval_samples": int(len(y)),
        "switch_samples": int(sw.sum()),
    }

    res["meta"] = {
        "h": H, "device": DEVICE, "tau": float(tau),
        "beta_grid": list(BETA_GRID), "seed": SEED,
        "wall_s": round(time.time() - _T0, 2),
    }

    # ---- 打印清晰两臂对比表 ----
    print("\n===== 两臂 BPC/acc 对比（无记忆 vs 有记忆，oracle recall）=====")
    print(f"{'arm':<8}{'subset':<8}{'bpc':>8}{'acc':>9}")
    head = res["nomem"]
    for sub in ("total", "switch", "within"):
        b, a = head[sub]["bpc"], head[sub]["acc"]
        print(f"{'nomem':<8}{sub:<8}{b:>8.4f}{a:>9.3f}")
    for beta_s, arm in mem_arms.items():
        for sub in ("total", "switch", "within"):
            b, a = arm[sub]["bpc"], arm[sub]["acc"]
            print(f"{'mem b'+beta_s:<8}{sub:<8}{b:>8.4f}{a:>9.3f}")
    print("\n===== 诊断 =====")
    print("16B窗口自识别语种 acc: overall=%.3f switch=%.3f within=%.3f"
          % (acc_self_all, acc_self_sw, acc_self_within))
    print("语种原型两两余弦距离 (3x3):")
    for row in sep:
        print("   ", " ".join(f"{x:.3f}" for x in row))

    # 判据
    nl0, ns0 = head["total"]["bpc"], head["switch"]["bpc"]
    verdict = {}
    for beta_s, arm in mem_arms.items():
        g_total = nl0 - arm["total"]["bpc"]
        g_switch = ns0 - arm["switch"]["bpc"]
        verdict[beta_s] = {"gain_total_bpc": g_total,
                           "gain_switch_bpc": g_switch,
                           "positive": bool(g_total > 0 or g_switch > 0)}
    res["verdict"] = verdict

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[json] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())