# GPU 任务清单（沙箱外环境执行）

> 沙箱现实：3 核 CPU / 5GB 内存 / 无 GPU / 纯 NumPy。本清单记录**超出沙箱算力的任务**，
> 由用户在外部 GPU 环境执行后回传结果。触发条件未到的不跑；清单随阶段推进更新。
> 契约与不变量见文末 —— 移植版必须保持"局部规则、免反传、结构稀疏"三条初衷不变。
>
> **口径更新（2026-09-08）**：本机 GTX 1660 Ti 6GB 已可用，GPU 任务改为由 AI
> **自行调用**（用户指令），登顶跑已在本机后台启动（results_e2_gpu/）；触发条件、
> 契约与不变量不变。本清单保留为任务登记与执行口径。

## T1：E2 缩放曲线登顶跑（8M / 15M）—— ✅ 完成（2026-09-08，RTX 4090）

> 结果与发散发现见 results_e2_gpu/REPORT_E2_GPU_SUMMIT.md（0.98M/1.92M 健康终态 BPC 3.972/3.930；3.89M/8.01M/15.03M 分别在 step≈33/33/45 万突发发散——已定位为 W2 σmax 谱发散引发的自由推断失稳。**对因修复（每步 W2 幂迭代谱截断 cap=5.0）已验证：3.89M 档全预算稳定收敛终态 BPC 3.884、8.01M/15.03M 越崩溃点终态 4.043/4.072，判据 1 重判通过**，见报告 §6）。**收口清帐（2026-09-09，报告 §7）**：失稳裁决为**算法级真实失稳**（numpy 原版与 torch 移植 W2 更新规则逐行一致，机制在规则本身非实现相关，§7.2）；full-epoch 步数差 1003838 vs 1039838 = 语料版本字节量差非切分 bug（§7.1）；收敛性诊断（§5.4）已由 §6.1 完成，根治设计目标 = 自由推断收缩性约束 σmax(W2)≤5（§7.3）。剩余 8.01M/15.03M 全预算补跑等均为可选。

- **触发**：✅ 已触发——E2 梯子（1M/2M/4M）CPU pilot 已跑完（results_e2/report.md），
  判据 2/3/4 PASS；**判据 1（4M 单调性 FAIL）需全预算裁决**（pilot 150k 步 ≈15% epoch 下
  4M 回退与 BP 孪生同步，疑数据预算瓶颈）。
- **内容**（2026-09-08 落地口径，见 results_e2_gpu/engineering.json 与报告）：
  tinyshakespeare **全 epoch**（n≈1,003,838 步）梯子 {768,1200,1856,2832,4032} =
  {1M,2M,4M,8M,15M} × PCN + 同规模孪生；超参 = 锚点迁移值（iters=12 / eta_w=0.01 /
  τ 校准），iPC 关，与 pilot 同协议仅放大预算。
  - **torch 移植**：`srpc/lmgpu.py`（LMPCNg，权重真源 = numpy 同 seed，无 autograd，
    局部规则/掩码/k-WTA/事件门控逐算子对应）+ `srpc/lmgpu_graph.py`（LMPCNgG：
    CUDA-Graph 捕获整训练步，就地更新，host 调度开销移除，实测 ~10×，parity max|ΔW|
    3k 步 7.7e-4）。
  - 运行：`scripts/run_e2_summit.py`（单档，--resume 断点续跑）/ `scripts/run_e2_summit_all.py`
    （顺序全流程 + 自动报告 `scripts/build_e2_summit_report.py`）。
- **输出**：BPC / acc 曲线（逐 eval 落盘 JSON）+ 终态；判据 1 单调性按尾部噪声带裁决；
  与孪生并列对照见报告 results_e2_gpu/report.md。

## T2：E2 BPTT 孪生对照（同规模）—— ✅ 完成（2026-09-08，部分）

- 已完成 0.98M/1.92M/3.89M 三档全预算（终态 BPC 2.586/2.495/2.420，全程稳定）；
  8.01M/15.03M 档未跑完（收尾中断）。结果见 results_e2_gpu/twin_{h}.json 与
  results_e2_gpu/REPORT_E2_GPU_SUMMIT.md。

- 与 T1 同批：同数据、同参数量、同样本流（batch 32）的 numpy TwinMLP（CPU，与 pilot
  同一实现口径，保证对照连续性）；全 epoch 每档一跑，逐档 JSON 落盘。

## T3：G2 同规模 LLM 推理评测

- **触发**：G2 启动前（W9 前）
- **内容**：pythia-14m / pythia-31m 在统一基准上推理评测（CPU 可跑小模型但慢，GPU 提效）
- **估算**：< 1 GPU·时
- **说明**：须标注 Pile 训练域差异 caveat；主对照仍是 T2 孪生

## T4：G3 组合泛化规模测试（可选）

- **触发**：G3 且组合增益单调性在 CPU 规模下成立
- **内容**：片段重组规模扩大（更多变换/更深组合），压力测试
- **估算**：视规模定，先 CPU 探边界再决定是否上 GPU

## 契约与不变量（移植版强制）

1. **局部规则**：所有权重更新保持突触局部形式（Hebbian/LMS/RLS），移植版不得引入 global gradient
2. **免反传**：无 autograd 反传路径；PyTorch 版仅用其张量运算加速，`loss.backward()` 只允许出现在 T2 孪生基准
3. **结构稀疏**：出生定型掩码 + k-WTA 激活原样移植（`srpc/config.py` 结构超参为唯一真源）
4. **口径一致**：三口径 MAC 记账（event/struct/dense）随移植版输出，能效对比用同一口径
