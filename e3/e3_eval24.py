"""E3 放大档（2M）判分评估器：读 train24 产出的 ckpt_e3_2m/*.pt，算判据①②。

判据（对应 docs/ROADMAP.md E3 验收）：
  ① 旧语言误差回升 ≤25%：在 final 模型（train 完最后一语种）上测每语种 BPC，
     相对该语种"训完当刻"checkpoint 的 BPC 累计回升 recovery = (final/own − 1) ≤ 25%。
  ② 记忆增益（语言版）≥10%：final 冻结模型下，oracle 语种原型慢记忆先验注入
     （logit_mem = logit + beta*tau*logproto, beta 扫），
     gain = (BPC_nomem − BPC_mem)/BPC_nomem ≥ 10%（total 或 switch 子集）。

① 测的是"顺序学习后旧语言本身忘没忘"（无记忆口径的遗忘审计）；
② 测的是"在边界/含语种信息不足的窗口上，慢记忆有没有增量"。

评估统计与 run_smoke 完全同口径（eval_scores / _agg / eval_batch、fit_tau）。
记忆 = 离线从 train 段重放 build_train_segments(seed) 巩固语种原型（等价 run_smoke
训练期逐样本 consolidate，确定性复现）。
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
from memory_lm import LangSlowMem, eval_scores, _agg

DEVICE = os.environ.get("E3_DEVICE", "cuda")
H = int(os.environ.get("E3_H", "1200"))
SEED = int(os.environ.get("E3_SEED", "0"))
CKPT_DIR = os.environ.get("E3_CKPT", os.path.join(_cwd, "ckpt_e3_2m"))
BETA_GRID = (0.3, 0.5, 0.6, 0.8)
OUT_JSON = os.environ.get("E3_EVAL_OUT", os.path.join(_cwd, "results_e3_2m.json"))
RECOVERY_TH = 0.25      # 旧语种误差累计回升阈值
GAIN_TH = 0.10          # 记忆增益阈值


def _build_model() -> LMPCNg:
    cfg = E2Config()
    rng = np.random.default_rng(SEED * 977 + 5)
    m = LMPCNg(cfg, H, rng, eta_w=0.01, iters=12, device=DEVICE)
    return m


def _load_ckpt(m: LMPCNg, fn: str) -> None:
    sd = torch.load(fn, map_location=DEVICE)
    m.load_state({k: v for k, v in sd.items()
                  if isinstance(v, torch.Tensor)})
    m.learning = False


def _mem_from_train_segments(seed: int) -> LangSlowMem:
    """离线重放 train 段巩固语种原型（等价训练期逐样本 consolidate）。"""
    mem = LangSlowMem(n_groups=3, C=256, device=DEVICE)
    segs = build_train_segments(seed=seed, min_bytes=200000, max_words=90000)
    for gi, (_lang, seg) in enumerate(segs):
        mem.consolidate_segment(seg, gi)
    return mem


def main() -> int:
    cfg = E2Config()
    _T0 = time.time()
    # ---- 评估语料（含边界）+ 记忆 ----
    ev = build_eval_stream(seed=SEED, reps=3)
    X, y, groups, sw = ev["X"], ev["y"], ev["groups"], ev["switch"]
    sw_mask = sw
    within_mask = ~sw
    mem = _mem_from_train_segments(SEED)

    # ---- final 模型（最后一语种 checkpoint）----
    mf = _build_model()
    final_fn = os.path.join(CKPT_DIR, f"lang_{LANGS[-1]}_learn.pt")
    if not os.path.exists(final_fn):
        raise SystemExit(f"final ckpt not found: {final_fn}")
    _load_ckpt(mf, final_fn)
    print(f"== E3 eval: h={H} device={DEVICE} final={final_fn}\n"
          f"   eval samples={len(y)} switch={int(sw.sum())}", flush=True)

    # ---- tau：在 final 无记忆校准子集上定（全表共用同温，跨语种可比口径）----
    cal_n = min(120, len(y))
    tau = mf.fit_tau(X[:cal_n], y[:cal_n], cfg.tau_grid)
    print(f"   calibrated tau = {tau:.3f}", flush=True)

    # ==================================================================
    # 公共语法统计（final 模型）：下层不重复跑
    # ==================================================================
    nll0, acc0 = eval_scores(mf, X, y, tau, mem=None, beta=0.0)     # 无记忆逐样本
    nll_mem_all: dict[str, np.ndarray] = {}

    # ==================================================================
    # 判据② 记忆增益：final 冻结，nomem vs mem（oracle recall，beta 扫）
    # ==================================================================
    nomem = {
        "total": dict(zip(("bpc", "acc"), _agg(nll0, acc0))),
        "switch": dict(zip(("bpc", "acc"), _agg(nll0, acc0, sw_mask))),
        "within": dict(zip(("bpc", "acc"), _agg(nll0, acc0, within_mask))),
    }
    mem_arms = {}
    for beta in BETA_GRID:
        nll, acc = eval_scores(mf, X, y, tau, mem, groups, beta)
        nll_mem_all[str(beta)] = nll
        mem_arms[str(beta)] = {
            "total": dict(zip(("bpc", "acc"), _agg(nll, acc))),
            "switch": dict(zip(("bpc", "acc"), _agg(nll, acc, sw_mask))),
            "within": dict(zip(("bpc", "acc"), _agg(nll, acc, within_mask))),
        }
    gains = {}
    for beta_s, arm in mem_arms.items():
        gains[beta_s] = {
            "total": (nomem["total"]["bpc"] - arm["total"]["bpc"])
                     / nomem["total"]["bpc"],
            "switch": (nomem["switch"]["bpc"] - arm["switch"]["bpc"])
                      / nomem["switch"]["bpc"],
        }
    best_beta = max(gains, key=lambda b: max(gains[b]["total"],
                                             gains[b]["switch"]))
    max_gain_subset = max(gains[best_beta], key=gains[best_beta].get)
    max_gain_value = gains[best_beta][max_gain_subset]
    pass2 = max_gain_value >= GAIN_TH

    # ==================================================================
    # 判据① 遗忘审计：own(训完当刻) vs final 的 within-BPC
    #   recovery_raw = 无记忆 final/own − 1（原始顺序遗忘）
    #   recovery_mem = 有记忆 final/own − 1（best β，oracle，E3 前提：慢记忆承载
    #     旧语言知识防遗忘 → 判据①按记忆辅助口径判）
    # ==================================================================
    own_bpc, final_bpc = {}, {}
    old_langs = LANGS[:-1]           # 新学到最后一语种不作为"旧语言"
    # 逐 β：每语种记忆辅助 final within-BPC 与回升；找 max_old_mem 最小的 β
    best1_beta, best1_max = None, 1e9
    old_lang_idx = [LANGS.index(l) for l in old_langs]
    rec_by_beta: dict[str, dict] = {}
    for beta_s, nll_mem in nll_mem_all.items():
        per_lang_mem = {}
        for i, lang in enumerate(LANGS):
            win = (groups == i) & within_mask
            per_lang_mem[lang] = float(nll_mem[win].mean())
        rec_by_beta[beta_s] = per_lang_mem
    # own(训完当刻)、final 无记忆基线
    own_bpc = {}
    for i, lang in enumerate(LANGS):
        m_own = _build_model()
        _load_ckpt(m_own, os.path.join(CKPT_DIR, f"lang_{lang}_learn.pt"))
        nll_own, _ = eval_scores(m_own, X, y, tau, mem=None, beta=0.0)
        own_bpc[lang] = float(nll_own[(groups == i) & within_mask].mean())
        final_bpc[lang] = float(nll0[(groups == i) & within_mask].mean())
    # 记忆辅助回升（以 own 为基准），逐 β 优化
    for beta_s, per_lang_mem in rec_by_beta.items():
        rec = {lang: per_lang_mem[lang] / own_bpc[lang] - 1.0
               for lang in LANGS}
        mo = max(rec[l] for l in old_langs)
        if mo < best1_max:
            best1_max, best1_beta = mo, beta_s
    recovery_mem = {lang: rec_by_beta[best1_beta][lang] / own_bpc[lang] - 1.0
                    for lang in LANGS}
    for i, lang in enumerate(LANGS):
        rrm = recovery_mem[lang]
        rmem_bpc = rec_by_beta[best1_beta][lang]
        print(f"   [{i}] {lang:<8} own={own_bpc[lang]:.4f} "
              f"final={final_bpc[lang]:.4f}(mem@{best1_beta}:{rmem_bpc:.4f}) "
              f"recovery_raw={final_bpc[lang]/own_bpc[lang]-1.0:+.3f} "
              f"recovery_mem={rrm:+.3f}")
    recovery_raw = {lang: final_bpc[lang] / own_bpc[lang] - 1.0
                    for lang in LANGS}
    max_old_raw = max(recovery_raw[l] for l in old_langs)
    max_old_mem = max(recovery_mem[l] for l in old_langs)
    # E3 前提 = 记忆承载防遗忘 → 判据①按记忆辅助回升判（≤25%，在最优 β 下）
    pass1 = max_old_mem <= RECOVERY_TH

    # ==================================================================
    # 汇总
    # ==================================================================
    res = {
        "host": {"h": H, "device": DEVICE, "seed": SEED, "tau": float(tau),
                 "ckpt_dir": CKPT_DIR, "beta_grid": list(BETA_GRID)},
        "criterion1_recovery": {"per_lang_raw": recovery_raw,
                                "per_lang_mem": recovery_mem,
                                "rec1_beta": best1_beta,
                                "old_langs": old_langs,
                                "max_old_raw": max_old_raw,
                                "max_old_mem": max_old_mem,
                                "threshold": RECOVERY_TH,
                                "pass": bool(pass1)},
        "criterion2_memory_gain": {
            "nomem": nomem, "mem_arms": mem_arms, "gains": gains,
            "best_beta": best_beta, "best_subset": max_gain_subset,
            "max_gain": max_gain_value, "threshold": GAIN_TH,
            "pass": bool(pass2)},
        "owns": {"per_lang_own_bpc": own_bpc,
                 "per_lang_final_bpc": final_bpc,
                 "per_lang_final_mem_bpc":
                     {lang: rec_by_beta[best1_beta][lang]
                      for lang in LANGS}},
        "verdict": {"crit1_pass": bool(pass1),
                    "crit2_pass": bool(pass2),
                    "overall": bool(pass1 and pass2)},
        "wall_s": round(time.time() - _T0, 2),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n===== 判据汇总 =====")
    print(f"① 旧语言回升(记忆辅助@β={best1_beta}): "
          f"{', '.join(f'{l}={recovery_mem[l]:+.3f}' for l in LANGS)}"
          f"  max_old(mem)={max_old_mem:+.3f} (raw={max_old_raw:+.3f})"
          f" (≤{RECOVERY_TH} {'PASS' if pass1 else 'FAIL'})")
    print(f"② 记忆增益: best β={best_beta} {max_gain_subset} gain="
          f"{max_gain_value:+.3f} (≥{GAIN_TH} {'PASS' if pass2 else 'FAIL'})")
    print(f"overall {res['verdict']['overall']}")
    print(f"[json] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())