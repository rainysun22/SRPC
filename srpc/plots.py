"""Phase-0 结果可视化（输出到 results/）。图表标签用英文以保证无中文字体环境可渲染。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _band(ax, runs, key, color, label):
    arr = np.stack([r["logs"][key] for r in runs])
    m, s = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(len(m))
    ax.plot(x, m, color=color, label=label, lw=1.6)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)


def fig_track1_curves(t1: dict, perturb_step: int, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    _band(ax, t1["on"]["runs"], "err_ema", "C0", "self-loop ON")
    _band(ax, t1["off"]["runs"], "err_ema", "C1", "self-loop OFF")
    ax.axvline(perturb_step, color="r", ls="--", lw=1,
               label="perturbation (global sensory remap)")
    ax.set_xlabel("interaction steps")
    ax.set_ylabel(r"prediction error EMA ($\|e_0\|^2$)")
    ax.set_title("Correction: error decreases with interaction + recovery")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1]
    recs = [[r["metrics"]["relearn_error"] for r in t1[t]["runs"]] for t in ("on", "off")]
    xs = np.arange(2)
    ax.bar(xs, [np.mean(r) for r in recs], yerr=[np.std(r) for r in recs],
           color=["C0", "C1"], capsize=4)
    ax.set_xticks(xs, ["self-loop ON", "self-loop OFF"])
    ax.set_ylabel("mean error in re-learning window")
    ax.set_title("Self-reflection gain: re-learning after perturbation (A/B)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_rfs(t1: dict, path: str) -> None:
    run = t1["on"]["runs"][0]
    pats = run["logs"]["patterns_pre"]
    W = run["logs"]["W10_pre"]
    pn = pats / np.linalg.norm(pats, axis=0, keepdims=True)
    wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    order = np.argsort(-(pn.T @ wn).max(axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].imshow(pats, aspect="auto", cmap="viridis")
    axes[0].set_title(f"true source patterns (D x {pats.shape[1]})")
    axes[0].set_xlabel("source id")
    axes[0].set_ylabel("sensory dim")
    axes[1].imshow(wn[:, order], aspect="auto", cmap="viridis")
    axes[1].set_title(f"self-organized L1 receptive fields (sorted)\n"
                      f"alignment {run['metrics']['rf_pre_mean']:.3f} "
                      f"(init {run['metrics']['rf_init_mean']:.3f})")
    axes[1].set_xlabel("L1 units (sorted)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_structure(t1: dict, t2: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, res, key, pkey, title in (
        (axes[0], t1, "nmi_zone", "nmi_zone_perm", "Track-1: concept layer vs spatial zone"),
        (axes[1], t2, "nmi_ctx", "nmi_ctx_perm", "Track-2: concept layer vs context"),
    ):
        on = res["on"]["metrics_mean"]
        off = res["off"]["metrics_mean"]
        ax.bar([0, 1, 2], [on[key], off[key], on[pkey]],
               color=["C0", "C1", "gray"],
               yerr=[res["on"]["metrics_std"][key], res["off"]["metrics_std"][key], 0],
               capsize=4)
        ax.set_xticks([0, 1, 2], ["self-loop ON", "self-loop OFF", "permuted baseline"])
        ax.set_ylabel("NMI")
        ax.set_title(title)
    fig.suptitle("Formation: concept grouping vs true contexts", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_combination(t2: dict, path: str) -> None:
    on = t2["on"]["metrics_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    groups = ("SR-PC\n(hier.+self)", "flat PC\n(same rules)", "lookup\n(no composition)")
    train_vals = [on["zs_srpc_train"], on["zs_flat_train"], on["zs_lookup_train"]]
    novel_vals = [on["zs_srpc_novel"], on["zs_flat_novel"], on["zs_lookup_novel"]]
    xs = np.arange(3)
    ax.bar(xs - 0.18, train_vals, 0.34, label="trained combos", color="C0")
    ax.bar(xs + 0.18, novel_vals, 0.34, label="held-out (novel) combos", color="C3")
    ax.set_xticks(xs, groups)
    ax.set_ylabel(r"zero-shot error ($\|e_0\|^2$)")
    ax.set_title("Composition: zero-shot generalization to novel combos")
    ax.legend(fontsize=9)

    ax = axes[1]
    for tag, color, label in (("on", "C0", "self-loop ON"), ("off", "C1", "self-loop OFF")):
        curves = np.stack([r["logs"]["fs_ema"] for r in t2[tag]["runs"]])
        m = curves.mean(axis=0)
        ax.plot(np.arange(len(m)), m, color=color, label=label, lw=1.6)
    ax.set_xlabel("online adaptation steps on novel combos")
    ax.set_ylabel(r"prediction error EMA ($\|e_0\|^2$)")
    ax.set_title("Self-reflection gain: few-shot adaptation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_events(t1: dict, path: str) -> None:
    run = t1["on"]["runs"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ev = run["logs"]["ev"]
    m = np.convolve(ev, np.ones(200) / 200, mode="valid")
    axes[0].plot(np.arange(len(m)), m, lw=1.4, color="C2")
    axes[0].set_xlabel("interaction steps")
    axes[0].set_ylabel("event-update fraction (3 levels)")
    axes[0].set_title("Sparse event-driven updates decline with learning")
    b = run["logs"]["boost"]
    mb = np.convolve(b, np.ones(200) / 200, mode="valid")
    axes[1].plot(np.arange(len(mb)), mb, lw=1.4, color="C4")
    axes[1].set_xlabel("interaction steps")
    axes[1].set_ylabel("precision boost (EMA)")
    axes[1].set_title("Self-loop precision modulation (fires on perturbation)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_all(t1: dict, t2: dict, perturb_step: int, out_dir: str) -> dict:
    import pathlib
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = dict(
        error_curves=str(out / "fig1_error_curves.png"),
        receptive_fields=str(out / "fig2_receptive_fields.png"),
        structure=str(out / "fig3_structure_nmi.png"),
        combination=str(out / "fig4_combination.png"),
        events=str(out / "fig5_event_rate.png"),
    )
    fig_track1_curves(t1, perturb_step, figs["error_curves"])
    fig_rfs(t1, figs["receptive_fields"])
    fig_structure(t1, t2, figs["structure"])
    fig_combination(t2, figs["combination"])
    fig_events(t1, figs["events"])
    return figs
