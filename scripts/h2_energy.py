"""H2 能耗重锚对账 —— 同规模同任务 SR-PC(v12 局部Adam) vs BPTT 孪生，三口径累计 MAC @等能力。

口径（2026-09-10 用户裁定，与 run_h2.py / phase-C EnergyLedger 一致）：
  能力锚点 = SR-PC v12 达到的最高 acc（取 h2_adam 曲线最优步）；
           孪生按同 acc 在其 dense acc-vs-累计MAC 曲线上线性插值。
  三口径：
    struct = 结构稀疏硬件：保留结构突触全开（SR-PC 自身结构 MAC 上界）；
    event  = 事件驱动硬件：×实测活跃率（v12 eval 曲线 event_rate）——SR-PC 设计能量标；
    dense  = 稠密等价（孪生即稠密，用其原生 per-step MAC）。

网络几何（E2Config，与 lm.py/lmgpu.py 一致）：
  W=16（上下文块），C=256（字节一热），rf=2（W1c 感受野块数），fan_in=0.75（W2/W3 扇入）。
  per = h//16, w1_syn = W*(rf*256)*per, m2_syn = 0.75*h², m3_syn = 0.75*h*C, ro_syn = C*h。
  （W2 用 fan_in_frac=0.75 解析式；孪生 n_params 与 LMPCN n_params_struct 完全一致，交叉验证。）

SR-PC v12 判别前馈（无生成 x3）per-字节部署：
  fwd_x    = w1_syn + m2_syn                     # 编码器整链 x0->x1->x2
  per_byte = fwd_x + ro_syn                       # + 读头
SR-PC v12 per-训练步（rounds=1：前馈+读头credit+两部分主干credit+4张量局部Adam）：
  per_step ≈ 3·fwd_x + 3·ro_syn（保守：fwd + credit 重算 + 参数更新 ≈ 3× 前馈）
孪生 per-step = 4·n_params（fwd1 + bwd2 + Adam1，稠密）；per-byte = n_params。
累计 MAC = steps×per_step + eval_bytes×per_byte，eval_bytes = 200（同 run_h2 EVAL_N）。
比值 = 孪生累计MAC / SR-PC累计MAC（>1 即 SR-PC 更省）。
"""
from __future__ import annotations
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /workspace
PH = os.path.join(ROOT, "results_phaseH")
TWD = os.path.join(ROOT, "results_e2_gpu")
OUT = os.path.join(ROOT, "results_phaseH")
EVAL_BYTES = 200

# 网络几何常量（E2Config / lm.py）
W_, C_, RF = 16, 256, 2
FAN = 0.75


def sr_dims(h: int) -> dict:
    per = h // W_
    w1 = W_ * (RF * 256) * per            # W1c 结构突触（块紧凑感受野）
    m2 = FAN * h * h                      # W2 随机扇入突触
    ro = C_ * h                           # W_out 读头突触
    n_struct = int(w1 + m2 + ro)          # v12 判别路径结构参数
    return dict(per=per, w1_syn=int(w1), m2_syn=int(m2), ro_syn=ro,
                n_struct=n_struct)


def sr_costs(dims: dict, ev: float) -> dict:
    fwd_x = dims["w1_syn"] + dims["m2_syn"]
    ro = dims["ro_syn"]
    perbyte_struct = fwd_x + ro
    perbyte_event = (fwd_x + ro) * ev
    perstep_struct = 3 * fwd_x + 3 * ro
    perstep_event = 3 * (fwd_x + ro) * ev
    return dict(perbyte_struct=int(perbyte_struct), perbyte_event=int(perbyte_event),
                perstep_struct=int(perstep_struct), perstep_event=int(perstep_event))


def best_point(curve):
    """返回 SR-PC 能力锚点 (acc, step)。取曲线最优 acc（并列取最小步）。"""
    bp = max(curve, key=lambda r: (r["acc"], -r["step"]))
    return bp["acc"], bp["step"]


def mean_event_rate(curve):
    return float(statistics.mean([r["event_rate"] for r in curve]))


def twin_pts(twinj):
    """(acc, cum_dense_mac) 排序点：cum = step × 4·n_params。"""
    n = int(twinj["n_params"])
    per_step = 4 * n
    pts = sorted(((r["acc"], r["step"] * per_step) for r in twinj["curve"]),
                 key=lambda p: p[0])
    return n, per_step, pts


def twin_cum_to_acc(pts, target_acc):
    """插值孪生达到 target_acc 所需累计 dense MAC（线性，最近区间）。"""
    if target_acc <= pts[0][0]:
        return pts[0][1]
    if target_acc >= pts[-1][0]:
        return pts[-1][1]
    for (a0, m0), (a1, m1) in zip(pts, pts[1:]):
        if a0 <= target_acc <= a1:
            f = (target_acc - a0) / max(1e-9, a1 - a0)
            return m0 + f * (m1 - m0)
    return pts[-1][1]


def main():
    rows = []
    for h in (768, 1200, 1856):
        sr = json.load(open(os.path.join(PH, f"h2_adam_{h}_lr0.0003_r1_c2.5.json")))
        tw = json.load(open(os.path.join(TWD, f"twin_{h}.json")))
        dims = sr_dims(h)
        ev = mean_event_rate(sr["curve"])
        cs = sr_costs(dims, ev)
        s_acc, s_step = best_point(sr["curve"])
        # SR-PC 累计 MAC @能力锚点
        sr_struct = s_step * cs["perstep_struct"] + EVAL_BYTES * cs["perbyte_struct"]
        sr_event = s_step * cs["perstep_event"] + EVAL_BYTES * cs["perbyte_event"]
        # 孪生同能力累计 dense MAC
        n, twin_ps, pts = twin_pts(tw)
        twin_cum = twin_cum_to_acc(pts, s_acc)
        twin_bytes = EVAL_BYTES * n
        tw_cum = twin_cum + twin_bytes
        rows.append(dict(h=h, best_acc=round(s_acc, 4), best_step=s_step,
                         ev=round(ev, 4),
                         n_sr_struct=dims["n_struct"],
                         n_twin=n,
                         sr_perbyte_struct=int(cs["perbyte_struct"]),
                         sr_perbyte_event=int(cs["perbyte_event"]),
                         twin_perbyte=int(n),
                         sr_cum_struct=int(sr_struct), sr_cum_event=int(sr_event),
                         twin_cum=int(tw_cum),
                         ratio_ev=round(tw_cum / sr_event, 2),
                         ratio_struct=round(tw_cum / sr_struct, 2)))

    hdr = ["h", "best_acc", "best_step", "ev",
           "n_sr_struct", "n_twin",
           "sr_perbyte_struct", "sr_perbyte_event", "twin_perbyte",
           "sr_cum_struct", "sr_cum_event", "twin_cum",
           "ratio_struct", "ratio_ev"]
    print("H2 能耗重锚对账（SR-PC v12 vs 孪生，@等能力）\n")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[k]) for k in hdr))

    with open(os.path.join(OUT, "h2_energy_v12.csv"), "w") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved -> {OUT}/h2_energy_v12.csv")
    return rows


if __name__ == "__main__":
    main()