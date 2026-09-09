"""临时扫参：找能让 7 个 x_self 码分离的 SRPC 配置 + 输入几何。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_srpc_planner import TrapTree, _reset, train_wdyn
from srpc.config import ModelConfig
from srpc.model import SRPCModel
from srpc.env import make_patterns


def per_state_decode(model, task, patterns, N=60, seed=7):
    model.set_learning(False)
    accum = np.zeros((task.n_states, model.cfg.n_self))
    cnt = np.zeros(task.n_states)
    for _ in range(3):
        for vv in range(task.n_states):
            model.observe(patterns[vv]); accum[vv] += model.xs; cnt[vv] += 1
    centroids = accum / np.maximum(cnt[:, None], 1e-9)
    per = np.zeros(task.n_states)
    for vv in range(task.n_states):
        hit = 0
        for _ in range(N):
            model.observe(patterns[vv])
            d = np.linalg.norm(centroids - model.xs.reshape(1, -1), axis=1)
            hit += int(np.argmin(d) == vv)
        per[vv] = hit / N
    model.set_learning(True)
    return per, centroids


CFG_VARIANTS = {
    "v1_base": dict(fan_in_frac=0.75, kwta_frac=0.5, alpha=0.15, beta=0.25, inner_iters=3),
    "v4_kwta80": dict(fan_in_frac=0.75, kwta_frac=0.8, alpha=0.15, beta=0.25, inner_iters=3),
    "v5_more_self": dict(n_l2=32, n_self=32, fan_in_frac=0.75, kwta_frac=0.5, alpha=0.15, beta=0.25, inner_iters=3),
    "v6_no_kwta": dict(fan_in_frac=0.75, kwta_frac=0.0, alpha=0.15, beta=0.25, inner_iters=3),
    "v7_no_selfpull": dict(fan_in_frac=0.75, kwta_frac=0.5, alpha=0.0, beta=0.25, inner_iters=3),
}
GEO_VARIANTS = [
    ("g1_block8_96",  8, 8, 96, 40, 32, 32, 0.5),
    ("g2_block16_128",16,16,128, 48, 40, 40, 0.5),
    ("g3_stride14_96",6,14, 96, 40, 32, 32, 0.5),
    ("g4_wide_48",    8, 8, 96, 48, 48, 48, 0.6),
]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pick = ([p for p in [sys.argv[2]] if p not in ("", "_", "all")]
            if len(sys.argv) > 2 else None)
    mode = sys.argv[3] if len(sys.argv) > 3 else "cfg"
    task = TrapTree(rng=np.random.default_rng(seed))

    if mode == "geo":
        for name, active, stride, d_obs, n_l1, n_l2, n_self, kwt in GEO_VARIANTS:
            if pick and name not in pick:
                continue
            pats = make_patterns(task.n_states, d_obs, np.random.default_rng(seed),
                                 active=active, stride=stride)
            patterns = [pats[:, i].copy() for i in range(task.n_states)]
            cfg = ModelConfig(d_obs=d_obs, n_l1=n_l1, n_l2=n_l2, n_self=n_self,
                              inner_iters=4, fan_in_frac=0.75, kwta_frac=kwt,
                              theta_event=0.01)
            model = SRPCModel(cfg, n_actions=2, rng=np.random.default_rng(seed + 1),
                              self_loop=True)
            # 先训练 Wdyn + Hebbian（和 g1_srpc_planner 完全一样），再量
            centroids, diag = train_wdyn(model, task, patterns, 4000, np.random.default_rng(seed))
            # 训练完后再测每状态解码
            model.set_learning(False)
            per = np.zeros(task.n_states)
            for vv in range(task.n_states):
                hit = 0
                N = 60
                for _ in range(N):
                    model.observe(patterns[vv])
                    d = np.linalg.norm(centroids - model.xs.reshape(1, -1), axis=1)
                    hit += int(np.argmin(d) == vv)
                per[vv] = hit / N
            model.set_learning(True)
            print(f"[{name}] dec_acc={diag['dec_acc']:.3f} min={per.min():.2f} mean={per.mean():.2f} "
                  f"{'SEP' if per.min() > 0.85 else 'mix'}")
            print("   per-state:", np.round(per, 2).tolist())
            norms = [f"{np.linalg.norm(centroids[v]):.3f}" for v in range(task.n_states)]
            print("   cent-norms:", " ".join(norms))
        return

    pats = make_patterns(task.n_states, 48, np.random.default_rng(seed),
                         active=3, stride=6)
    patterns = [pats[:, i].copy() for i in range(task.n_states)]
    for name, ov in CFG_VARIANTS.items():
        if pick and name not in pick:
            continue
        cfg = ModelConfig(d_obs=48, n_l1=24, **ov)
        model = SRPCModel(cfg, n_actions=2, rng=np.random.default_rng(seed + 1),
                          self_loop=True)
        _reset(model)
        per, cen = per_state_decode(model, task, patterns)
        print(f"[{name}] min={per.min():.2f} mean={per.mean():.2f} "
              f"{'SEP' if per.min() > 0.85 else 'mix'}")
        print("   per-state:", np.round(per, 2).tolist())

        norms = [f"{np.linalg.norm(cen[v]):.3f}" for v in range(task.n_states)]
        print("   norms:   ", norms)


if __name__ == "__main__":
    main()