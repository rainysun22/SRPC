"""F1a：结构化（n-gram）语言记忆容量-精度评估。

承接 E3 判据① FAIL（unigram 慢记忆不足以防遗忘：旧语言回升 38%>25%）。F1a 把
静态 per-language unigram 原型升级为**上下文相关的 n-gram 条件字节先验**（NgramLangMem，
见 memory_lm.py）——给定窗口尾部 order 字节预测下一字节，把旧语种被后训练覆盖的
词级/短程结构重新托起来。验证：记忆辅助回升能否压到 ≤25%（判据① 达标）。

容量轴 = n-gram 阶数 order ∈ {0,1,2,3}（0 = unigram 基线，应复现 E3 的 ~38%）。
对每阶：
  判据① recovery_mem（记忆辅助旧语言回升，best β 最小化 max_old）——主判据；
  判据② gain（记忆增益 ≥10%，total/switch）——回证记忆仍有增量不失效。
产【容量-精度】表 + JSON curve。

效率：窗口 logit 只对每个 checkpoint 算一遍（final + 3 own），阶数/β 扫描全在
numpy 层做（不加解析器/不复算 latent），远快于 E3 的逐臂 _infer。

运行在 4090：python3 f1a_eval24.py （读 ckpt_e3_2m_fix，写 results_f1a_2m.json）。
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
from memory_lm import LangSlowMem, NgramLangMem, _agg

DEVICE = os.environ.get("E3_DEVICE", "cuda")
H = int(os.environ.get("E3_H", "1200"))
SEED = int(os.environ.get("E3_SEED", "0"))
CKPT_DIR = os.environ.get("E3_CKPT", os.path.join(_cwd, "ckpt_e3_2m_fix"))
ORDERS = (0, 1, 2, 3)
BETA_GRID = (0.3, 0.5, 0.6, 0.8)
OUT_JSON = os.environ.get("E3_F1A_OUT", os.path.join(_cwd, "results_f1a_2m.json"))
RECOVERY_TH = 0.25
GAIN_TH = 0.10


def _build_model() -> LMPCNg:
    cfg = E2Config()
    rng = np.random.default_rng(SEED * 977 + 5)
    return LMPCNg(cfg, H, rng, eta_w=0.01, iters=12, device=DEVICE)


def _load_ckpt(m: LMPCNg, fn: str) -> None:
    sd = torch.load(fn, map_location=DEVICE)
    m.load_state({k: v for k, v in sd.items() if isinstance(v, torch.Tensor)})
    m.learning = False


def _window_logits(m: LMPCNg, X: np.ndarray) -> np.ndarray:
    """对每窗口跑一次 m._infer，返回 (n,256) 模型读出头 logit（softmax 前）。"""
    n = len(X)
    out = np.zeros((n, 256), np.float32)
    for i in range(n):
        m._infer(X[i].ravel(), None, None)
        logit = torch.mv(m.W_out.t(), m._x2) + m.b_out
        out[i] = logit.detach().cpu().numpy()
    return out


def _nll_nomem(logits: np.ndarray, y: np.ndarray, tau: float,
               mem, groups, win_bytes, beta: float) -> np.ndarray:
    """给定窗口 logit，输出逐样本 nll(bits)。beta=0 无记忆；否则注入 context 先验。"""
    n = len(y)
    nll = np.zeros(n, np.float64)
    for i in range(n):
        L = logits[i]
        if beta != 0.0:
            lp = mem.mem_logit(win_bytes[i], int(groups[i]))
            L = L + float(beta) * tau * lp.detach().cpu().numpy()
        logits_stab = L - L.max()
        p = np.exp(logits_stab / tau)
        p /= p.sum()
        py = float(p[y[i]])
        nll[i] = -np.log2(max(py, 1e-12))
    return nll


def main() -> int:
    _T0 = time.time()
    ev = build_eval_stream(seed=SEED, reps=3)
    X, y, groups, sw = ev["X"], ev["y"], ev["groups"], ev["switch"]
    sw_mask, within_mask = sw, ~sw
    win_bytes = X.argmax(axis=-1).astype(np.uint8)          # (n,16)
    segs = build_train_segments(seed=SEED, min_bytes=200000, max_words=90000)

    # ---- final + 各 own checkpoint 的窗口 logit（各算一次，全网络共享）----
    mfile = _build_model()
    _load_ckpt(mfile, os.path.join(CKPT_DIR, f"lang_{LANGS[-1]}_learn.pt"))
    tau = mfile.fit_tau(X[: min(120, len(y))], y[: min(120, len(y))],
                        E2Config().tau_grid)
    print(f"== F1a 容量-精度: h={H} tau={tau:.3f} samples={len(y)} "
          f"switch={int(sw.sum())} orders={ORDERS}", flush=True)
    logits_final = _window_logits(mfile, X)
    logits_own = {}
    for i, lang in enumerate(LANGS):
        m_own = _build_model()
        _load_ckpt(m_own, os.path.join(CKPT_DIR, f"lang_{lang}_learn.pt"))
        logits_own[lang] = _window_logits(m_own, X)
        print(f"  ckpt logits: {lang} own done (wall {round(time.time()-_T0,1)}s)",
              flush=True)

    old_langs = LANGS[:-1]
    # ---- nomem 不变基线（final / own，beta=0）----
    nll0 = _nll_nomem(logits_final, y, tau, None, groups, win_bytes, 0.0)
    acc0 = y == _argmax_rec(logits_final, tau)
    nomem = {"total": dict(zip(("bpc", "acc"), _agg(nll0, acc0))),
             "switch": dict(zip(("bpc", "acc"), _agg(nll0, acc0, sw_mask))),
             "within": dict(zip(("bpc", "acc"), _agg(nll0, acc0, within_mask)))}

    results = {}
    for order in ORDERS:
        mem = NgramLangMem(n_groups=3, C=256, order=order, device=DEVICE)
        mem.build_from_segments(segs)
        # ---- 判据② 记忆增益（oracle recall，β 扫）----
        gains, mem_nll = {}, {}
        for beta in BETA_GRID:
            nll_m = _nll_nomem(logits_final, y, tau, mem, groups, win_bytes, beta)
            mem_nll[str(beta)] = nll_m
            bpc_t = float(nll_m.mean())
            bpc_s = float(nll_m[sw_mask].mean())
            bpc_w = float(nll_m[within_mask].mean())
            gains[str(beta)] = {"total": (nll0.mean() - bpc_t) / nll0.mean(),
                                "switch": (nll0[sw_mask].mean() - bpc_s)
                                / nll0[sw_mask].mean()}
        best_beta = max(gains, key=lambda b: max(gains[b].values()))
        max_gain = max(gains[best_beta].values())
        # ---- 判据① 记忆辅助回升（best β 最小化 max_old）----
        own_bpc = {l: float(_nll_nomem(logits_own[l], y, tau, None, groups,
                                       win_bytes, 0.0)[(groups == LANGS.index(l))
                                                       & within_mask].mean())
                   for l in LANGS}
        final_bpc = {l: float(nll0[(groups == LANGS.index(l)) & within_mask].mean())
                     for l in LANGS}
        best1_beta, best1_max = None, 1e9
        rec = {}
        for beta_s, nll_m in mem_nll.items():
            r = {l: float(nll_m[(groups == LANGS.index(l)) & within_mask].mean())
                 / own_bpc[l] - 1.0 for l in LANGS}
            mo = max(r[l] for l in old_langs)
            if mo < best1_max:
                best1_max, best1_beta, rec = mo, beta_s, r
        results[str(order)] = {
            "recovery_mem": rec, "best1_beta": best1_beta,
            "max_old_mem": best1_max, "max_old_raw": max(
                final_bpc[l] / own_bpc[l] - 1.0 for l in old_langs),
            "threshold": RECOVERY_TH,
            "crit1_pass": bool(best1_max <= RECOVERY_TH),
            "gain": {"beta": best_beta, "max": max_gain, "all": gains[best_beta]},
            "crit2_pass": bool(max_gain >= GAIN_TH),
            "final_bpc": final_bpc, "own_bpc": own_bpc,
        }
        print(f"\n  --- order={order} (unigram)" if order == 0 else f"\n  --- order={order} (n-gram)",
              flush=True)
        for l in LANGS:
            print(f"    {l:<8} own={own_bpc[l]:.4f} final={final_bpc[l]:.4f} "
                  f"rec_mem={rec[l]:+.3f}", flush=True)
        print(f"    max_old_mem={best1_max:+.3f} (≤{RECOVERY_TH} "
              f"{'PASS' if best1_max <= RECOVERY_TH else 'FAIL'}) | "
              f"gain={max_gain:+.3f} ({best_beta}) "
              f"{'PASS' if max_gain >= GAIN_TH else 'FAIL'}",
              flush=True)

    # ---- 汇总图（capacity-精度）+ 落盘 ----
    curve = {o: results[str(o)] for o in ORDERS}
    summary = {
        "host": {"h": H, "tau": float(tau), "ckpt": CKPT_DIR,
                 "orders": list(ORDERS), "beta_grid": list(BETA_GRID)},
        "nomem": {"total": nomem["total"], "switch": nomem["switch"],
                  "within": nomem["within"]},
        "capacity_precision_curve": {o: {
            "max_old_mem": curve[o]["max_old_mem"],
            "crit1_pass": curve[o]["crit1_pass"],
            "gain": curve[o]["gain"], "crit2_pass": curve[o]["crit2_pass"]}
            for o in ORDERS},
        "first_pass_order": next((o for o in ORDERS
                                  if curve[o]["crit1_pass"]), None),
        "wall_s": round(time.time() - _T0, 2),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n=== capacity-精度曲线 (order -> max_old_mem / gain) ===")
    for o in ORDERS:
        c = curve[o]
        print(f"  order={o}: rec_mem={c['max_old_mem']:+.3f} "
              f"gain={c['gain']['max']:+.3f} "
              f"[{c['crit1_pass'] and 'P1 OK' or 'P1 x'} "
              f"{c['crit2_pass'] and 'P2 OK' or 'P2 x'}]")
    print(f"\n判据① 首达 order={summary['first_pass_order']}")
    print(f"[json] {OUT_JSON}")
    return 0


def _argmax_rec(logits: np.ndarray, tau: float) -> np.ndarray:
    L = logits - logits.max(1, keepdims=True)
    p = np.exp(L / tau)
    p /= p.sum(1, keepdims=True)
    return p.argmax(1)


if __name__ == "__main__":
    raise SystemExit(main())