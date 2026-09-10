# H2 能耗重锚对账 —— SR-PC(v12) vs BPTT 孪生，三口径累计 MAC @等能力

> 日期：2026-09-10 · 口径裁定：全链路累计 MAC @等能力，三档 h∈{768,1200,1856}。
> 机制：H2 最终采纳 **v12 完全判别式局部 Adam（幸存集 straight-through credit + 4 张量逐突触局部 Adam）**，
> 结果见 `results_phaseH/h2_adam_*.json`；孪生曲线见 `results_e2_gpu/twin_*.json`。

## 1. 结论速览

在**事件驱动口径**（SR-PC 的设计能量标，kWTA/事件门控 + 实测活跃率）下，@等能力 SR-PC 的
**累计 MAC 仅为同规模孪生的 1/32 ~ 1/102**（结构口径为 1/1.04 ~ 1/2.67，即持平到近 3 倍更省）。

| h | SR-PC 最优acc | 活跃率 ev | 累计MAC(event) | 孪生累计MAC(dense) | event比值 | struct比值 |
|---|:---:|:---:|---:|---:|:---:|:---:|
| 768  | 0.388 | 0.0435 |  4.04e10 | 1.31e12 | **32.5×** | 1.41× |
| 1200 | 0.407 | 0.0261 |  3.29e10 | 3.37e12 | **102.3×** | 2.67× |
| 1856 | 0.382 | 0.0231 |  8.34e10 | 3.74e12 | **44.8×** | 1.04× |

- 部署 per-字节（MAC/bit）：SR-PC event 每字节 44.9k~92.7k MAC，孪生 dense 每字节 0.98M~3.89M MAC
  → **22×~42× 更省**（逐字节，768/1200/1856 依次 22×/37×/42×）。
- 核心：SR-PC 是**结构稀疏 + 事件激活稀疏**（KPWTA/事件门控），v12 判别前馈落在 ~2.3%~4.4%
  活跃率；孪生是稠密逐字节全量。等能力锚点下，省的是"能"的来源是激活稀疏，
  而**参数规模恒等**（n_struct = 孪生 n_params，见 §3 交叉验证）。

## 2. 口径与记账

网络几何（E2Config）：`W=16 上下文块, C=256 字节一热, rf=2, fan_in=0.75`；
`per=h//16, w1_syn=16·(2·256)·per, m2_syn=0.75·h², ro_syn=256·h`。

- SR-PC v12 判别前馈（无生成 x3）：`fwd_x = w1_syn + m2_syn`；`per_byte_struct = fwd_x + ro_syn`。
- SR-PC per-训练步（rounds=1，前馈+读头credit+主干credit+4 张量局部Adam）：
  `per_step = 3·fwd_x + 3·ro_syn`（≈3× 前馈，保守含 credit 重算 + 参数更新）。
- 事件口径 × 实测活跃率 `ev`（取该档 v12 eval 曲线 event_rate 均值）。
- 孪生 dense：`per_step = 4·n_params`（fwd 1 + bwd 2 + Adam 1）；`per_byte = n_params`。
- 累计 = `steps×per_step + eval_bytes(200)×per_byte`。
- **等能力锚点** = SR-PC v12 曲线最优 acc；孪生按其 dense acc-vs-累计MAC 曲线线性插值到同 acc。
- 比值 = 孪生累计 MAC / SR-PC 累计 MAC（>1 即 SR-PC 更省）。

## 3. 同规模（参数恒等）交叉验证

`n_struct = w1_syn + m2_syn + m3_syn`（m3 = 0.75·h·C）与孪生 `n_params` 逐位一致：

| h | SR-PC n_struct | 孪生 n_params |
|---:|---:|---:|
| 768  | 983040  | 983040  |
| 1200 | 1924800 | 1924800 |
| 1856 | 3890176 | 3890176 |

→ "同能力下 MAC/bit 显著低于**同规模**孪生"中的"同规模"由参数恒等保证，
对比不是靠缩小模型，而是靠算法（局部免反传 + 结构/事件稀疏）拿到的。

## 4. 能力对齐说明（诚实边界）

- 能力锚点取 SR-PC v12 已完成 300k 步预算下的最优 acc（0.38~0.41）；
- 孪生在**相同训练步**那一点的 acc 略高（如 1856：SR 0.382 vs 孪生 0.422@300k），
  但达到 SR 这个 acc 时孪生需 ~240k 步×4·n_params（dense），累计 MAC 仍远高于 SR event 口径。
- SR-PC 目前 v12 跑 300k 步即达标（未跑满孪生 1.04M）；若按"追平孪生终值 acc≈0.49"
  需要继续扩展训练步数——但每步 MAC(event) 极低，扩展步数对 event 累计 MAC 影响可控。
- **结论成立域**：事件驱动/结构稀疏硬件（阶段 C EnergyLedger 的设计假设）。
  在"所有结构突触全开"的保守 struct 口径下，SR-PC 为持平到 ~2.7× 更省，不构成反超——这是保守下限。

## 5. 文件

- 数据：[h2_energy_v12.csv](h2_energy_v12.csv)
- 脚本：`scripts/h2_energy.py`