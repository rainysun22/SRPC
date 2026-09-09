"""阶段 F1b-3：xorsum 类组合任务的 VSA 绑定记忆（能力=记忆·拼合）。

背景（承接 E1 边界与 f1b_replay_{probe,batch}）：
  PCN 网络本体**在线**学 xorsum@16（4-bit parity）是 SQ-hard 不可达——E1 探针
  已证 BP 在线对照同样失败（期望梯度≈0），真 batch(512) SGD 可达 1.0；而
  类均值原型回放失败是**原理性**的（给定 y 时每个远端块的 marginal≡全局均匀，
  一阶统计零判别信息），真·成对累积批次回放也失败（结构稀疏组件本身无法在
  权重里吸收 parity）。

  => 修复不在"让网络在权重里硬学"，而在**记忆承担组合**：y = x_a ⊗ x_b 的
  组合查询由 VSA 绑定层完成。这正是不变量 #2 "能力=记忆·拼合"：知识住在记忆
  里（绑定叠加 S = Σ bind(role_pair, filler_y)），组合（unbind+cleanup）把旧
  经验固化后即可正确回答新查询。

协议（"回放/巩固" = 逐次 store 绑定对的成本摊开）：
  - 经验流逐个到达 (x_a, x_b, y)，每步把 bind(bind(v_a,v_b), filler_y) 叠加进 S
    （pair = 复合键 bind(v_a,v_b)，与 y 绑定 = 组合关系实例）；
  - 周期性评估 S 在**全 16×16=256 对的冻结集**上 recall（一批重放后查询）；
  - 曲线 = 已巩固绑定对数 -> recall；判据：巩固完成后 acc ≥ 0.60。

与 f1b_vsa_eval 的关系：那里测"role->filler 解绑-清洗"基元容量曲线；本脚本把
同一 VSA 绑定层套上 xorsum 组合数据生成，给 F1 判据③一个具体任务口径——
xorsum 组合查询不动网络权重，纯由记忆承担组合（能力=记忆的总成）。
"""
from __future__ import annotations
import time

import numpy as np
from srpc.vsa import gauss_vecs, bind as _bind, unbind as _unbind


def xorsum_memory_run(a: int = 16, d: int = 1024, seed: int = 3,
                      n_experience: int = 256) -> dict:
    """用 VSA 绑定层承担 xorsum 组合查询（纯记忆，不动网络权重）。"""
    role_vocab = gauss_vecs(d, a, seed=seed)            # 符号值 role 向量
    filler_vocab = gauss_vecs(d, a, seed=seed + 111)    # y 类 filler 向量
    rng = np.random.default_rng(seed + 999)
    # 真实经验流：随机 (x_a, x_b) -> y = x_a ⊕ x_b；覆盖到全 256 对
    if n_experience < a * a:
        pairs = rng.integers(0, a, size=(n_experience, 2))
    else:
        pairs = np.array([[i, j] for i in range(a) for j in range(a)])
        rng.shuffle(pairs)
    ys = pairs[:, 0] ^ pairs[:, 1]

    S = np.zeros(d)                      # 绑定叠加记忆本体
    k = len(pairs)
    curve = []
    step = max(1, k // 8)
    for i in range(k):
        xa, xb = int(pairs[i, 0]), int(pairs[i, 1])
        y = int(ys[i])
        pair_key = _bind(role_vocab[xa], role_vocab[xb])   # 复合键：配对
        S += _bind(pair_key, filler_vocab[y])              # 关系实例绑定
        if (i + 1) % step == 0 or i == k - 1:
            curve.append(round(_freeze_recall(S, d, role_vocab, filler_vocab), 4))
    return {"a": a, "d": d, "n_experience": k, "curve": curve,
            "final_recall": curve[-1], "pass60": bool(curve[-1] >= 0.60)}


def _freeze_recall(S, d, role_vocab, filler_vocab) -> float:
    pn = filler_vocab / (np.linalg.norm(filler_vocab, axis=1, keepdims=True) + 1e-12)
    a = len(role_vocab)
    hit = 0
    total = a * a
    for xa in range(a):
        for xb in range(a):
            pair_key = _bind(role_vocab[xa], role_vocab[xb])
            d_rec = _unbind(S, pair_key)
            d_rec /= (np.linalg.norm(d_rec) + 1e-12)
            q = int((pn @ d_rec).argmax())
            hit += int((xa ^ xb) == q)
    return hit / total


def main() -> int:
    t0 = time.time()
    res = xorsum_memory_run()
    print(f"[xorsum 组合记忆 @ a={res['a']} d={res['d']} exp={res['n_experience']}]")
    print(f"  巩固曲线(recall after n 对): {res['curve']}")
    print(f"  final_recall={res['final_recall']}  pass(≥0.60)={res['pass60']}")
    print(f"  wall={time.time()-t0:.0f}s")
    return 0 if res["pass60"] else 1


if __name__ == "__main__":
    raise SystemExit(main())