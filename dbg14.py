#!/usr/bin/env python3
"""随机初值重启投票 × K：降低 compare 决策噪声（裕度薄时的鲁棒化）。"""
import sys
from dataclasses import replace

sys.path.insert(0, "/workspace")
import numpy as np

from srpc.config import CreditConfig
from srpc.credit import CreditPCN, _make_sequence


def spectral_of(m):
    W1, W2 = m.W1, m.W2
    h1, h2 = W2.shape
    H = np.zeros((h1 + h2, h1 + h2))
    H[:h1, :h1] = W1.T @ W1 + np.eye(h1)
    H[:h1, h1:] = -W2
    H[h1:, :h1] = -W2.T
    H[h1:, h1:] = W2.T @ W2 + np.eye(h2)
    return float(np.linalg.eigvalsh(H).max())


def predict_vote(m, x0, K, rng):
    votes = []
    for _ in range(K):
        m.x1[:] = rng.uniform(0, 0.5, m.x1.shape)
        m.x2[:] = rng.uniform(0, 0.5, m.x2.shape)
        best_c, best_e = 0, np.inf
        for c in (0, 1):
            yoh = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
            m.x1[:] = rng.uniform(0, 0.5, m.x1.shape)
            m.x2[:] = rng.uniform(0, 0.5, m.x2.shape)
            m._infer(x0, yoh, free_out=False)
            if m._energy < best_e:
                best_e, best_c = m._energy, c
        votes.append(best_c)
    return float(np.bincount(votes, minlength=2).argmax())


def run(cfg, seed, K, iters):
    rng = np.random.default_rng(seed * 3000 + 7)
    X, y = _make_sequence(cfg, rng, 10500, 4)
    Xtr, ytr = X[:10000], y[:10000]
    Xev, yev = X[10000:], y[10000:]
    m = CreditPCN(replace(cfg, delay=4), np.random.default_rng(seed * 3000 + 11), "error")
    lam = spectral_of(m)
    m.orth = False
    m.eta_inf = min(0.3, 0.6 * 1.2 / lam)
    m.iters = iters
    for t in range(10000):
        m.train_step(Xtr[t], float(ytr[t]))
    m.set_learning(False)
    vrng = np.random.default_rng(999)
    ok = 0
    for t in range(len(Xev)):
        ok += int(predict_vote(m, Xev[t], K, vrng) == float(yev[t]))
    return ok / len(Xev)


base = replace(CreditConfig(), h1=32, h2=16, energy_mode="full",
               alpha=1.0, beta=1.0)
print("=== 重启投票 K（iters=24）===")
for K in (1, 3, 5, 9):
    accs = [run(base, s, K, 24) for s in range(3)]
    print(f"K={K:2d} | " + " ".join(f"{a:.3f}" for a in accs)
          + f" | mean={np.mean(accs):.3f}")
