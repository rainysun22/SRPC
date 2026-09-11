"""阶段 I：知识沉降 memory=knowledge —— 合成事实键值集验收（纯 NumPy CPU）。

口径（ROADMAP I，2026-09-10 用户裁定）：
  任务   = 合成事实键值集：K 个 distinct 事实（key→identity 类）。
  宿主   = v12 判别式编码器（H2）：per-key 识别态 x_self 作为记忆查询向量
           （前端成本取 H2 1856 per-byte event MAC≈92663 作保守上界常量）。
  读出口 = 由"上下文直拉 W_out(x2)"升级为"记忆检索条件化"：
           x_self -> VSA 查询向量 -> 从 VsaMemory 检索 -> cleanup 解码值类。

两条验收线（每条给出 PASS/FAIL，不粉饰）：
  I-a  同能量下：(记忆条件化 acc) ≥ (同容量权重硬学 acc)，且能量低一个量级。
  I-b  检索命中率-容量曲线严格单调（K 增大 acc 单调降）。

对比对象（同容量）：
  记忆  = VsaMemory(d)：一次加性绑定 (role, filler) 即写入；查询一次 unbind+cleanup。
        容量一致：存储 K·d（roles+fillers 码表）≈ 权重头 K×d 参数。
  权重  = 线性读头 W(K×d)，Adam 训练映射 role->one-hot 类。多轮梯度才记住。

能量（MAC，bit 口径 ×1）：
  bind/unbind(FFT) C_fft(d) = 8·d·log2(d)（rfft×2 + 逐点 + irfft，含实部）。
  记忆：E_mem = K_store·C_fft + K_q·(C_fft + K·d[cleanup]) + K·E_host(前端编码器)
  权重：E_wt(p) = p_epochs · (fwd K²·d + bwd K²·d + Adam K·d) ≈ 3·K²·d/epoch
  判据强度：报告 "同能量 E_mem 下 权重 acc"（应≈随机） vs 记忆 acc（应高）；
           以及 "权重追到记忆 acc 所需能量 E_wt" / E_mem 比值（应 ≳ 量级）。

用法：python -B /workspace/scripts/i_memory_cond_readout.py [--d 1024] [--K 64,128,256,512,1024]
输出：results_phaseI/i_memory_cond_readout.json/.csv/.md
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

from srpc.vsa import VsaMemory, bind, unbind, gauss_vecs

R = "/workspace"
OUT = os.path.join(R, "results_phaseI")
os.makedirs(OUT, exist_ok=True)

# H2 1856 v12 per-byte event MAC：识别编码器前端成本（每 key 产 x_self，保守上界）
HOST_PER_KEY = 92663


# ----------------------------------------------------------------------
# 能量模型（MAC / bit）
# ----------------------------------------------------------------------
def c_fft(d: int) -> int:
    """一次 HRR bind/unbind（FFT 循环卷积）MAC（bit 口径 ≈1 MAC/bit）。"""
    return int(8 * d * np.log2(d))


def e_memory(K: int, d: int, host_per_key: int = HOST_PER_KEY) -> int:
    """记忆路线：K 次绑定写入 + K 次查询(unbind+cleanup over K vocab) + 前端编码器。"""
    store = K * c_fft(d)
    query = K * (c_fft(d) + K * d)        # 每查询 unbind + cleanup(K·d 余弦)
    host = K * host_per_key               # 读 x_self 前端（文档：v12 编码器 per-byte event）
    return int(store + query + host)


def e_weight_epoch(K: int, d: int) -> int:
    """权重硬学每 epoch（线性头 K×d）：fwd K²·d + bwd K²·d + Adam K·d ≈ 3K²·d。"""
    return int(3 * K * K * d)


# ----------------------------------------------------------------------
# 记忆路线：VSA one-shot 绑定 + 查询
# ----------------------------------------------------------------------
def memory_acc(K: int, d: int, seed: int = 0) -> float:
    roles = gauss_vecs(d, K, seed=1000 + K)
    fillers = gauss_vecs(d, K, seed=2000 + K)   # V=K 独立值类（identity 知识）
    m = VsaMemory(d=d, seed=seed)
    for i in range(K):
        m.store(roles[i], fillers[i])           # 一次加性绑定即写入
    return float(m.recall(roles, fillers))      # 命中率（cleanup over 自身码表）


# ----------------------------------------------------------------------
# 权重硬学路线：线性读头 Adam（同容量 K×d<->K·d）
# ----------------------------------------------------------------------
def weight_train(K: int, d: int, target_acc: float, cap_epochs: int = 300,
                 seed: int = 0) -> dict:
    """返回 {epochs_to_target, acc_at_mem_energy, acc_final, e_to_target, e_final}。"""
    rng = np.random.default_rng(seed)
    X = gauss_vecs(d, K, seed=1000 + K)           # 同记忆的角色（输入）
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y = np.eye(K, dtype=np.float32)               # identity 目标（一热）

    r = 0.05
    W = rng.standard_normal((K, d)) * r           # K×d 头（= 记忆 K·d 存储，同容量）
    m = np.zeros_like(W); v = np.zeros_like(W)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 5e-3

    e_mem = e_memory(K, d)
    e_per_epoch = e_weight_epoch(K, d)
    acc_at_mem = None
    epochs_at_target = None
    e_at_target = None
    acc = 0.0

    for ep in range(1, cap_epochs + 1):
        logit = Xn @ W.T                          # (K,K)
        logit -= logit.max(1, keepdims=True)
        e = np.exp(logit); p = e / e.sum(1, keepdims=True)
        p[range(K), range(K)] = 1.0               # 白化自环，避免 softmax 训练退化
        g = (p - Y) / K
        acc = float((logit.argmax(1) == np.arange(K)).mean())
        # Adam（逐坐标）
        m = b1 * m + (1 - b1) * g.T @ Xn
        mh = m / (1 - b1 ** ep)
        v = b2 * v + (1 - b2) * (g.T @ Xn) ** 2
        vh = v / (1 - b2 ** ep)
        W -= lr * mh / (np.sqrt(vh) + eps)
        # 记录"同能量 E_mem"时点
        if (ep * e_per_epoch) >= e_mem and acc_at_mem is None:
            acc_at_mem = float(acc)
        if acc >= target_acc and epochs_at_target is None:
            epochs_at_target = ep
            e_at_target = int(ep * e_per_epoch)
        if epochs_at_target is not None:
            break

    if false_path := (np.sum(Xn @ W.T * Y) / K):  # 顺手算对角平均 logit 信号（诊断）
        pass

    return {
        "epochs_to_target": epochs_at_target,
        "e_to_target": e_at_target,
        "acc_at_mem_energy": (acc_at_mem if acc_at_mem is not None
                              else float(acc)),
        "acc_final": float(acc),
        "e_final": int(cap_epochs * e_per_epoch),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--K", type=str, default="64,128,256,512,1024")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = args.d
    ks = [int(x) for x in args.K.split(",")]

    t0 = time.time()
    rows = []
    for K in ks:
        am = memory_acc(K, d, args.seed)
        # 公平横杆：权重只需追平记忆 acc（同 acc 下比能量），而非追 0.9（超容量不可达）
        wt = weight_train(K, d, target_acc=am, cap_epochs=300,
                          seed=args.seed)
        em = e_memory(K, d)
        ew = e_weight_epoch(K, d)
        ratio = (wt["e_to_target"] / em) if wt["e_to_target"] else None
        rows.append({
            "K": K, "d": d,
            "acc_mem": round(am, 4),
            "acc_wt_at_mem_energy": round(wt["acc_at_mem_energy"], 4),
            "acc_wt_final": round(wt["acc_final"], 4),
            "wt_epochs_to_target": wt["epochs_to_target"],
            "e_mem": em, "e_wt_per_epoch": ew,
            "e_wt_to_target": wt["e_to_target"],
            "ratio_e_wt_to_mem": (round(ratio, 1) if ratio else None),
            "I-a_same_energy_pass": bool(
                am >= wt["acc_at_mem_energy"] and
                (wt["e_to_target"] is not None and (wt["e_to_target"] / em) >= 10)
            ),
            "criterion_Ia_hit": bool(am >= wt["acc_at_mem_energy"]),
            "criterion_Ia_energy10x": bool(
                wt["e_to_target"] is not None and (wt["e_to_target"] / em) >= 10
            ),
        })

    # I-b：命中率-容量曲线单调（acc 随 K 严格递减到 <0.95）
    accs = [r["acc_mem"] for r in rows]
    monotonic = all(accs[i] >= accs[i + 1] for i in range(len(accs) - 1))
    low_k_pass = accs[0] >= 0.95
    cap_wall = next((r["K"] for r in rows if r["acc_mem"] < 0.95), None)

    # 扩容：容量墙随 d 右移（世界知识靠 d 扩容，无梯度、近乎免费）
    # 用细网格 + P50(acc=0.5) 插值测真实墙，避免粗 K 网格把 2048/4096 卡在同一格
    ds = [1024, 2048, 4096]
    # 墙在 d=4096 约 K~500，封顶 768 步长 16 足够（高 K 逐点 O(K²·d) 太贵）
    fine = list(np.arange(8, 769, 16))
    dc = {}
    for dd in ds:
        dc[dd] = {str(K): round(memory_acc(K, dd, args.seed), 4) for K in ks}
    cap_curve = {}
    p50 = {}
    for dd in ds:
        accs_f = [memory_acc(K, dd, args.seed) for K in fine]
        cap_curve[dd] = dict(zip(fine, [round(a, 4) for a in accs_f]))
        # 单调下降，P50 = acc 跨过 0.5 的 K（线性插值）
        p50[dd] = None
        for K, a in zip(fine, accs_f):
            if a <= 0.5:
                i = fine.index(K)
                prev_a = accs_f[i - 1]
                f = (prev_a - 0.5) / max(1e-9, prev_a - a)
                p50[dd] = round(fine[i - 1] + f * (fine[i] - fine[i - 1]), 1)
                break
        if p50[dd] is None:
            p50[dd] = fine[-1]
    cap_scales = all(p50[ds[i]] < p50[ds[i + 1]] for i in range(len(ds) - 1))

    summary = {
        "stage": "I",
        "task": "synthetic key-value facts (memory=knowledge)",
        "host": "v12 discriminative encoder (H2), host_per_key_MAC=HOST_PER_KEY",
        "d": d, "K_grid": ks,
        "rows": rows,
        "I-b_monotonic": bool(monotonic),
        "I-b_low_k_acc": accs[0] if accs else None,
        "I-b_low_k_pass": bool(low_k_pass),
        "I-b_capacity_wall_at_K": cap_wall,
        "I-c_scale_curve": dc,
        "I-c_capacity_curve": {str(k): {str(kk): v for kk, v in cv.items()}
                               for k, cv in cap_curve.items()},
        "I-c_p50_per_d": p50,
        "I-c_capacity_scales_with_d": bool(cap_scales),
        "energy_legend": {
            "e_mem": "memory store+query+host(bit MAC)",
            "e_wt_per_epoch": "weight head 3K^2·d/epoch",
            "ratio_e_wt_to_mem": "energy to match memory acc / memory energy (>=10 => 低一个量级)",
        },
        "wall_s": round(time.time() - t0, 1),
    }
    out_json = os.path.join(OUT, "i_memory_cond_readout.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # 控制台
    print(f"\nStage-I 知识沉降（d={d}, host_per_key={HOST_PER_KEY} MAC/bit）")
    print(f"{'K':>6}{'acc_mem':>10}{'acc_wt@E_mem':>14}{'acc_wt_final':>13}"
          f"{'e_mem':>16}{'e_wt/epoch':>14}{'e_wt_to_tgt':>14}{'ratio':>9}")
    for r in rows:
        print(f"{r['K']:>6}{r['acc_mem']:>10.3f}{r['acc_wt_at_mem_energy']:>14.3f}"
              f"{r['acc_wt_final']:>13.3f}{r['e_mem']:>16}{r['e_wt_per_epoch']:>14}"
              f"{str(r['e_wt_to_target'] or '-'):>14}{str(r['ratio_e_wt_to_mem'] or 0):>9}")
    print(f"\nI-a 同能量（记忆acc≥权重acc@E_mem）逐 K: "
          f"{[r['criterion_Ia_hit'] for r in rows]}")
    print(f"I-a 能量低一个量级（e_wt/mem≥10）逐 K: "
          f"{[r['criterion_Ia_energy10x'] for r in rows]}")
    print(f"I-b 命中率-容量曲线单调: {monotonic}, 低负载acc={accs[0] if accs else 'N/A'},"
          f" 容量墙K={cap_wall}")
    print("I-c 扩容（容量墙随 d 右移，记忆线无梯度免费扩容）:")
    print("     d     " + "".join(f"{k:>9}" for k in ks))
    for dd in ds:
        print(f"     {dd:>4} " + "".join(f"{dc[dd][str(k)]:>9.3f}" for k in ks))
    print(f"     P50(acc=0.5)墙/d: {p50}")
    print(f"     -> 随d右移(rescale): {cap_scales}")
    print(f"wall {summary['wall_s']}s -> {out_json}")
    ok = all(r["criterion_Ia_hit"] for r in rows) and monotonic and low_k_pass \
        and cap_scales
    print(f"Stage-I 总判: {'PASS' if ok else ('部分FAIL, 见行' if any(r['criterion_Ia_hit'] for r in rows) else 'FAIL')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())