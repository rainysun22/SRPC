import sys, time
sys.path.insert(0, "/workspace")
from dataclasses import replace
import numpy as np
from srpc.config import ArcConfig, CLConfig, DeepConfig, MemoryConfig
from srpc.arc import ArcLite, TRANSFORMS
from srpc.runner_b import run_sequential

t0 = time.time()
dcfg = replace(DeepConfig(), trace_energy=True)
clcfg = replace(CLConfig(), steps_per_task=4000, eval_samples=24)
mcfg = MemoryConfig(d=dcfg.dims[-1])
res = run_sequential(0, with_memory=True, dcfg=dcfg, mcfg=mcfg, acfg=ArcConfig(), clcfg=clcfg)
model, arc = res["model"], ArcLite(ArcConfig(), np.random.default_rng(0*3000+2))

def calib(n_cal):
    rng = np.random.default_rng(0 * 913 + 7)
    model.set_learning(True)
    for i, name in enumerate(arc.train_names):
        cond = arc._cond(i)
        for _ in range(n_cal):
            s_in, s_out, _ = arc.sample(name)
            model.set_condition(cond)
            model.reset_states()
            model.prepare_next(action=None)
            model.observe(s_in)
            model.learn_readout(s_out)
    model.set_learning(False)
calib(4000)
print("calib done", round(time.time()-t0,1), "s")

# 对比 novel 组合：连续中间读出 vs 解码后中间读出
for task in arc.novel_names:
    i, j = arc.novel_combos[arc.novel_names.index(task)]
    c1, c2 = arc._cond(arc.train_names.index(i)), arc._cond(arc.train_names.index(j))
    rng = np.random.default_rng(7)
    ca_c, ca_d = [], []
    for _ in range(100):
        g_in = arc.sample_input()
        oh_in = arc._onehot(g_in).astype(float)
        g_true = TRANSFORMS[j](TRANSFORMS[i](g_in)).ravel()
        # 连续路径
        out1 = model.apply_transform(oh_in, c1)
        v = model.apply_transform(out1, c2)
        ca_c.append(float(np.mean(arc.decode_grid(v).ravel() == g_true)))
        # 解码路径
        out1d = arc.decode_grid(out1)   # 64, 值 0..3
        vd = model.apply_transform(arc._onehot(out1d).astype(float), c2)
        ca_d.append(float(np.mean(arc.decode_grid(vd).ravel() == g_true)))
    print(f"{task:14s} continuous_cell={np.mean(ca_c):.4f}  decode_cell={np.mean(ca_d):.4f}")
print("total", round(time.time()-t0,1), "s")
